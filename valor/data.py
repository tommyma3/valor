from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from valor.prompts import State, Action, format_state_prompt, format_action


REQUIRED_STATE_FIELDS = [
    "question",
    "memory",
    "prev_tool_query",
    "prev_tool_result",
]
REQUIRED_ACTION_FIELDS = [
    "action_think",
    "action_memory_update",
    "action_tool_query",
]


@dataclass
class PolicyExample:
    prompt: str
    target: str


@dataclass
class ValueExample:
    prompt: str
    value_label: int


class PolicyDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        for field in REQUIRED_STATE_FIELDS:
            if field not in record:
                raise KeyError(f"Missing required state field: {field}")
        for field in REQUIRED_ACTION_FIELDS:
            if field not in record:
                raise KeyError(f"Missing required action field: {field}")

        state = State(
            question=record["question"],
            memory=record["memory"],
            prev_tool_query=record["prev_tool_query"],
            prev_tool_result=record["prev_tool_result"],
        )
        action = Action(
            think=record["action_think"],
            memory_update=record["action_memory_update"],
            tool_query=record["action_tool_query"],
        )

        return {
            "state": state,
            "action": action,
            "advantage_label": record.get("advantage_label"),
        }


class ValueDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        for field in REQUIRED_STATE_FIELDS:
            if field not in record:
                raise KeyError(f"Missing required state field: {field}")

        value_label = record.get("value_label")
        if value_label is None:
            raise KeyError("Missing value_label for value training.")

        state = State(
            question=record["question"],
            memory=record["memory"],
            prev_tool_query=record["prev_tool_query"],
            prev_tool_result=record["prev_tool_result"],
        )

        return {
            "state": state,
            "value_label": int(value_label),
        }


def _prompt_for_record(
    state: State,
    include_advantage: bool,
    advantage_label: Optional[int],
    indicator_drop_prob: float,
) -> str:
    if include_advantage and advantage_label is not None and indicator_drop_prob > 0.0:
        drop = torch.rand(1).item() < indicator_drop_prob
    else:
        drop = False

    prompt = format_state_prompt(
        state,
        include_advantage=include_advantage and not drop,
        advantage_label=advantage_label,
    )
    return prompt


def collate_policy(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    include_advantage: bool,
    indicator_drop_prob: float,
) -> Dict[str, torch.Tensor]:
    prompts: List[str] = []
    targets: List[str] = []
    for item in batch:
        prompt = _prompt_for_record(
            item["state"],
            include_advantage=include_advantage,
            advantage_label=item.get("advantage_label"),
            indicator_drop_prob=indicator_drop_prob,
        )
        prompts.append(prompt)
        targets.append(format_action(item["action"]))

    full_text = [p + t for p, t in zip(prompts, targets)]
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = enc.input_ids.clone()
    prompt_lens = prompt_enc.attention_mask.sum(dim=1)

    valid_batches = []
    for i, length in enumerate(prompt_lens.tolist()):
        if length < max_length:
            labels[i, :length] = -100
            valid_batches.append(i)
        else:
            print(f"WARNING: Example {i} has prompt length {length} >= max_length {max_length}, max_length may be too small")
            # Keep the example but mask everything (will be skipped by loss check)
            labels[i, :] = -100

    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels": labels,
    }

    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "labels": labels,
    }


def collate_value(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    prompts: List[str] = [
        format_state_prompt(item["state"], include_advantage=False)
        for item in batch
    ]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    value_labels = torch.tensor([item["value_label"] for item in batch], dtype=torch.long)
    return {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "value_labels": value_labels,
    }
