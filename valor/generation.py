from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from valor.model import PolicyModel


STRICT_FORMAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that must follow the required output format exactly."
)


@dataclass
class GenerationResult:
    rendered_prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def build_chat_messages(
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_local_completion(
    model: PolicyModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device | None = None,
    system_prompt: str | None = None,
) -> GenerationResult:
    rendered_prompt = tokenizer.apply_chat_template(
        build_chat_messages(prompt, system_prompt=system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    if device is not None:
        encoded = encoded.to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.backbone.generate(
            **encoded,
            **generation_kwargs,
        )

    prompt_len = int(encoded["input_ids"].shape[-1])
    completion_ids = generated[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    total_tokens = int(generated.shape[-1])
    completion_tokens = max(total_tokens - prompt_len, 0)

    return GenerationResult(
        rendered_prompt=rendered_prompt,
        completion=completion,
        prompt_tokens=prompt_len,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
