from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from valor.browsecomp_prompting import (
    BrowseCompPromptState,
    build_browsecomp_messages,
    build_browsecomp_prompt,
    format_browsecomp_target,
)


REQUIRED_STATE_FIELDS = [
    "question",
    "memory",
    "prev_tool_query",
    "prev_tool_result",
]

REQUIRED_POLICY_FIELDS = [
    "action_memory_update",
    "action_tool_query",
]


@dataclass
class BrowseCompPolicyExample:
    state: BrowseCompPromptState
    report: str
    tool_call: str | None
    answer: str | None
    advantage_label: int | None


@dataclass
class BrowseCompValueExample:
    state: BrowseCompPromptState
    value_label: int


def _build_state(record: Dict[str, Any]) -> BrowseCompPromptState:
    for field in REQUIRED_STATE_FIELDS:
        if field not in record:
            raise KeyError(f"Missing required state field: {field}")
    return BrowseCompPromptState(
        question=str(record.get("question", "")),
        last_report=str(record.get("memory", "")),
        last_tool_call=str(record.get("prev_tool_query", "")),
        last_tool_response=str(record.get("prev_tool_result", "")),
    )


def _extract_report(record: Dict[str, Any]) -> str:
    report = str(record.get("action_memory_update", "")).strip()
    if report:
        return report
    think = str(record.get("action_think", "")).strip()
    if think:
        return think
    memory = str(record.get("memory", "")).strip()
    return memory if memory else "No report."


def _extract_answer(record: Dict[str, Any]) -> str:
    final_answer = str(record.get("final_answer", "")).strip()
    if final_answer:
        return final_answer
    memory_update = str(record.get("action_memory_update", "")).strip()
    if memory_update:
        return memory_update
    think = str(record.get("action_think", "")).strip()
    return think


def _split_tool_or_answer(record: Dict[str, Any]) -> tuple[str | None, str | None]:
    raw_tool_call = str(record.get("action_tool_query", "")).strip()
    if raw_tool_call and raw_tool_call != "<NO_TOOL_CALL>":
        return raw_tool_call, None
    return None, _extract_answer(record)


class BrowseCompPolicyDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> BrowseCompPolicyExample:
        record = self.records[idx]
        for field in REQUIRED_POLICY_FIELDS:
            if field not in record:
                raise KeyError(f"Missing required policy field: {field}")
        state = _build_state(record)
        tool_call, answer = _split_tool_or_answer(record)
        return BrowseCompPolicyExample(
            state=state,
            report=_extract_report(record),
            tool_call=tool_call,
            answer=answer,
            advantage_label=record.get("advantage_label"),
        )


class BrowseCompValueDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> BrowseCompValueExample:
        record = self.records[idx]
        value_label = record.get("value_label")
        if value_label is None:
            raise KeyError("Missing value_label for value training.")
        return BrowseCompValueExample(
            state=_build_state(record),
            value_label=int(value_label),
        )


def _render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> str:
    messages = build_browsecomp_messages(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered_messages = []
    for message in messages:
        rendered_messages.append(f"{message['role']}: {message['content']}")
    rendered_messages.append("assistant:")
    return "\n\n".join(rendered_messages)


def _tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> List[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]


def _pad_sequences(
    *,
    sequences: List[List[int]],
    pad_value: int,
) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        padded.append(sequence + [pad_value] * (max_len - len(sequence)))
    return torch.tensor(padded, dtype=torch.long)


def _prompt_for_policy_example(
    example: BrowseCompPolicyExample,
    *,
    tools_prompt: str,
    date_to_use: str | None,
    include_advantage: bool,
    indicator_drop_prob: float,
) -> str:
    if include_advantage and example.advantage_label is not None and indicator_drop_prob > 0.0:
        drop = torch.rand(1).item() < indicator_drop_prob
    else:
        drop = False

    return build_browsecomp_prompt(
        example.state,
        tools_prompt=tools_prompt,
        date_to_use=date_to_use,
        advantage_label=(None if drop or not include_advantage else example.advantage_label),
    )


def collate_browsecomp_policy(
    batch: List[BrowseCompPolicyExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    *,
    tools_prompt: str,
    date_to_use: str | None,
    include_advantage: bool,
    indicator_drop_prob: float,
) -> Dict[str, torch.Tensor]:
    input_ids_batch: List[List[int]] = []
    attention_masks: List[List[int]] = []
    labels_batch: List[List[int]] = []

    for example in batch:
        prompt = _prompt_for_policy_example(
            example,
            tools_prompt=tools_prompt,
            date_to_use=date_to_use,
            include_advantage=include_advantage,
            indicator_drop_prob=indicator_drop_prob,
        )
        rendered_prompt = _render_chat_prompt(tokenizer, prompt)
        target = format_browsecomp_target(
            report=example.report,
            tool_call=example.tool_call,
            answer=example.answer,
        )

        prompt_ids = _tokenize_text(tokenizer, rendered_prompt)
        target_ids = _tokenize_text(tokenizer, target)
        input_ids = (prompt_ids + target_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        for idx in range(prompt_len):
            labels[idx] = -100

        input_ids_batch.append(input_ids)
        attention_masks.append(attention_mask)
        labels_batch.append(labels)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for BrowseComp policy collation.")

    input_ids_tensor = _pad_sequences(sequences=input_ids_batch, pad_value=pad_token_id)
    attention_mask_tensor = _pad_sequences(sequences=attention_masks, pad_value=0)
    labels_tensor = _pad_sequences(sequences=labels_batch, pad_value=-100)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
    }


def collate_browsecomp_value(
    batch: List[BrowseCompValueExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    *,
    tools_prompt: str,
    date_to_use: str | None,
) -> Dict[str, torch.Tensor]:
    input_ids_batch: List[List[int]] = []
    attention_masks: List[List[int]] = []
    value_labels: List[int] = []

    for example in batch:
        prompt = build_browsecomp_prompt(
            example.state,
            tools_prompt=tools_prompt,
            date_to_use=date_to_use,
            advantage_label=None,
        )
        rendered_prompt = _render_chat_prompt(tokenizer, prompt)
        input_ids = _tokenize_text(tokenizer, rendered_prompt)[:max_length]
        attention_mask = [1] * len(input_ids)

        input_ids_batch.append(input_ids)
        attention_masks.append(attention_mask)
        value_labels.append(example.value_label)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is required for BrowseComp value collation.")

    input_ids_tensor = _pad_sequences(sequences=input_ids_batch, pad_value=pad_token_id)
    attention_mask_tensor = _pad_sequences(sequences=attention_masks, pad_value=0)
    value_labels_tensor = torch.tensor(value_labels, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "value_labels": value_labels_tensor,
    }
