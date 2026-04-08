from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig


ADAPTER_CONFIG_FILENAME = "adapter_config.json"
DEFAULT_QLORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "w1",
    "w2",
    "w3",
)


@dataclass
class PolicyOutputs:
    lm_loss: Optional[torch.Tensor]
    lm_logits: torch.Tensor


@dataclass
class ValueOutputs:
    value_logits: torch.Tensor


def _resolve_hidden_size(config) -> int:
    candidates = [
        "hidden_size",
        "n_embd",
        "model_dim",
        "d_model",
        "hidden_sizes",
    ]
    for attr in candidates:
        if hasattr(config, attr):
            val = getattr(config, attr)
            if isinstance(val, (list, tuple)) and val:
                return int(val[-1])
            if isinstance(val, int):
                return int(val)

    for nested_attr in ["text_config", "model_config", "llm_config"]:
        if hasattr(config, nested_attr):
            nested = getattr(config, nested_attr)
            try:
                return _resolve_hidden_size(nested)
            except ValueError:
                pass

    raise ValueError("Could not infer hidden size from model config.")


def _maybe_local_path(path_like: str) -> Optional[Path]:
    try:
        path = Path(path_like).expanduser()
    except (TypeError, OSError, ValueError):
        return None
    return path if path.exists() else None


def _adapter_config_path(backbone_name: str) -> Optional[Path]:
    local_path = _maybe_local_path(backbone_name)
    if local_path is None:
        return None
    candidate = local_path / ADAPTER_CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def _load_adapter_metadata(backbone_name: str) -> Optional[dict]:
    config_path = _adapter_config_path(backbone_name)
    if config_path is None:
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_policy_backbone_name(backbone_name: str) -> str:
    adapter_metadata = _load_adapter_metadata(backbone_name)
    if adapter_metadata is None:
        return backbone_name

    base_model_name = adapter_metadata.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name.strip():
        raise ValueError(
            f"Adapter checkpoint '{backbone_name}' is missing base_model_name_or_path in {ADAPTER_CONFIG_FILENAME}."
        )
    return base_model_name


def _load_backbone(
    backbone_name: str,
    torch_dtype: Optional[torch.dtype],
    device_map: Optional[str | dict],
    trust_remote_code: bool,
    max_memory: Optional[dict] = None,
    offload_folder: Optional[str] = None,
    offload_state_dict: bool = False,
    attn_implementation: Optional[str] = None,
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    if device_map == "cpu":
        return AutoModelForCausalLM.from_pretrained(
            backbone_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to("cpu")

    return AutoModelForCausalLM.from_pretrained(
        backbone_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_state_dict=offload_state_dict,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )


def _build_bnb_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_policy_backbone(
    backbone_name: str,
    torch_dtype: Optional[torch.dtype],
    device_map: Optional[str | dict],
    trust_remote_code: bool,
    max_memory: Optional[dict] = None,
    offload_folder: Optional[str] = None,
    offload_state_dict: bool = False,
    *,
    use_qlora: bool = False,
    qlora_trainable: bool = False,
    qlora_compute_dtype: Optional[torch.dtype] = None,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
    attn_implementation: Optional[str] = None,
) -> AutoModelForCausalLM:
    adapter_metadata = _load_adapter_metadata(backbone_name)
    base_backbone_name = _resolve_policy_backbone_name(backbone_name)
    config = AutoConfig.from_pretrained(base_backbone_name, trust_remote_code=trust_remote_code)
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    should_quantize = use_qlora or adapter_metadata is not None
    quantization_config = None
    if should_quantize:
        quantization_config = _build_bnb_config(qlora_compute_dtype or torch.bfloat16)
        if device_map is None and torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}

    backbone = AutoModelForCausalLM.from_pretrained(
        base_backbone_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_state_dict=offload_state_dict,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )

    if adapter_metadata is not None:
        if use_qlora and qlora_trainable:
            backbone = prepare_model_for_kbit_training(backbone)
        return PeftModel.from_pretrained(
            backbone,
            backbone_name,
            is_trainable=qlora_trainable,
        )

    if not use_qlora:
        return backbone

    backbone = prepare_model_for_kbit_training(backbone)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=not qlora_trainable,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(lora_target_modules or DEFAULT_QLORA_TARGET_MODULES),
    )
    return get_peft_model(backbone, peft_config)


class PolicyModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str | dict] = None,
        trust_remote_code: bool = True,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        use_qlora: bool = False,
        qlora_trainable: bool = False,
        qlora_compute_dtype: Optional[torch.dtype] = None,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = _load_policy_backbone(
            backbone_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            use_qlora=use_qlora,
            qlora_trainable=qlora_trainable,
            qlora_compute_dtype=qlora_compute_dtype,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            attn_implementation=attn_implementation,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> PolicyOutputs:
        device = next(self.backbone.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if labels is not None and labels.device != device:
            labels = labels.to(device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return PolicyOutputs(
            lm_loss=outputs.loss,
            lm_logits=outputs.logits,
        )

    def save(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)


class ValueModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str | dict] = None,
        trust_remote_code: bool = True,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = _load_backbone(
            backbone_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            attn_implementation=attn_implementation,
        )
        hidden_size = _resolve_hidden_size(self.backbone.config)
        self.value_head = nn.Linear(hidden_size, 2)
        if torch_dtype is not None:
            self.value_head = self.value_head.to(dtype=torch_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ValueOutputs:
        device = next(self.backbone.parameters()).device
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        last_index = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        value_logits = self.value_head(pooled)
        return ValueOutputs(value_logits=value_logits)

    def save(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)
        torch.save(self.value_head.state_dict(), f"{output_dir}/value_head.pt")

    def load_value_head(self, checkpoint_dir: str) -> None:
        state = torch.load(f"{checkpoint_dir}/value_head.pt", map_location="cpu")
        self.value_head.load_state_dict(state)
