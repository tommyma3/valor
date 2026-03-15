from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class ModelOutputs:
    lm_loss: Optional[torch.Tensor]
    lm_logits: torch.Tensor
    value_logits: torch.Tensor


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str | dict] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(backbone_name, trust_remote_code=trust_remote_code)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = config.hidden_size
        self.value_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutputs:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        last_index = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        value_logits = self.value_head(pooled)
        return ModelOutputs(
            lm_loss=outputs.loss,
            lm_logits=outputs.logits,
            value_logits=value_logits,
        )

    def save(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)
        torch.save(self.value_head.state_dict(), f"{output_dir}/value_head.pt")

    def load_value_head(self, checkpoint_dir: str) -> None:
        state = torch.load(f"{checkpoint_dir}/value_head.pt", map_location="cpu")
        self.value_head.load_state_dict(state)
