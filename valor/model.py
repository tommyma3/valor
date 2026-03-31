from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


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


def _load_backbone(
    backbone_name: str,
    torch_dtype: Optional[torch.dtype],
    device_map: Optional[str | dict],
    trust_remote_code: bool,
    max_memory: Optional[dict] = None,
    offload_folder: Optional[str] = None,
    offload_state_dict: bool = False,
) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
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
    )


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
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> PolicyOutputs:
        # Ensure tensors are on the same device as the model
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
        )
        hidden_size = _resolve_hidden_size(self.backbone.config)
        self.value_head = nn.Linear(hidden_size, 2)
        # Ensure value_head uses the same dtype as backbone
        if torch_dtype is not None:
            self.value_head = self.value_head.to(dtype=torch_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ValueOutputs:
        # Ensure tensors are on the same device as the model
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
