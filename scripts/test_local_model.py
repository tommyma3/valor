import argparse
import json
import re
from datetime import date
from typing import Any

import torch
from transformers import AutoTokenizer

from prompts import browsecomp_initial_instruction_prompt
from valor.model import PolicyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a local policy model checkpoint.")
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or HF model id.")
    parser.add_argument(
        "--mode",
        choices=["simple", "browsecomp"],
        default="simple",
        help="Prompt mode. 'simple' checks basic text generation, 'browsecomp' checks structured tool-call output.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-memory", default=None, help="Per-GPU memory limit or JSON dict.")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--question",
        default="What is 2 + 2? Reply with one short sentence.",
        help="Question used in the test prompt.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype: str, device: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype]


def _parse_max_memory(value: str | None, gpu_count: int) -> dict[int, str] | dict | None:
    if not value:
        return None
    raw = value.strip()
    if raw.startswith("{"):
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("--max-memory JSON must be an object.")
        return loaded
    return {idx: raw for idx in range(gpu_count)}


def _format_browsecomp_prompt(question: str) -> str:
    tools = (
        'search: Query the local BrowseComp corpus.\n'
        'parameters: {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}'
    )
    return browsecomp_initial_instruction_prompt.format(
        date_to_use=date.today().isoformat(),
        question=question,
        tools=tools,
    )


def _format_simple_prompt(question: str) -> str:
    return question


def _build_messages(mode: str, prompt: str) -> list[dict[str, str]]:
    if mode == "browsecomp":
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that must follow the required XML format exactly.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    return [{"role": "user", "content": prompt}]


def _filter_thinking_sections(text: str) -> str:
    filtered = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    filtered = re.sub(r"^.*?</think>", "", filtered, flags=re.DOTALL | re.IGNORECASE)
    return filtered.strip()


def _extract_sections(text: str) -> tuple[str, str, str]:
    filtered_text = _filter_thinking_sections(text)
    strict_pattern = (
        r"^\s*<report>(?P<report>.*?)</report>\s*"
        r"(?:<answer>(?P<answer>.*?)</answer>|<tool_call>(?P<tool_call>.*?)</tool_call>)\s*$"
    )
    match = re.search(strict_pattern, filtered_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return (
            (match.group("report") or "").strip(),
            (match.group("answer") or "").strip(),
            (match.group("tool_call") or "").strip(),
        )

    relaxed_pattern = re.compile(
        r"<report>(?P<report>.*?)</report>\s*"
        r"<(?P<kind>answer|tool_call)>(?P<body>.*?)</(?P=kind)>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(relaxed_pattern.finditer(filtered_text))
    if not matches:
        return "", "", ""

    terminal_matches = [m for m in matches if not filtered_text[m.end() :].strip()]
    match = terminal_matches[-1] if terminal_matches else matches[-1]
    report = (match.group("report") or "").strip()
    kind = str(match.group("kind") or "").strip().lower()
    body = (match.group("body") or "").strip()
    if kind == "answer":
        return report, body, ""
    return report, "", body


def main() -> None:
    args = parse_args()

    if args.mode == "browsecomp":
        prompt = _format_browsecomp_prompt(args.question)
    else:
        prompt = _format_simple_prompt(args.question)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
    torch_dtype = _resolve_dtype(args.dtype, args.device)
    max_memory = _parse_max_memory(args.max_memory, gpu_count)

    model = PolicyModel(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        max_memory=max_memory,
        offload_folder=args.offload_folder,
        offload_state_dict=args.offload_state_dict,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device_map is None and not getattr(model.backbone, "is_loaded_in_4bit", False):
        model.to(device)
    model.eval()

    messages = _build_messages(args.mode, prompt)
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    if args.device_map is None:
        encoded = encoded.to(device)

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        generated = model.backbone.generate(
            **encoded,
            **generation_kwargs,
        )

    prompt_len = encoded["input_ids"].shape[-1]
    completion_ids = generated[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    print("=== Prompt ===")
    print(prompt)

    print("\n=== Rendered Prompt ===")
    print(rendered_prompt)

    print("\n=== Completion ===")
    print(completion)

    print("\n=== Stats ===")
    print(
        json.dumps(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "prompt_tokens": int(prompt_len),
                "total_tokens": int(generated.shape[-1]),
                "completion_tokens": int(generated.shape[-1] - prompt_len),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.mode == "browsecomp":
        report, answer, tool_call = _extract_sections(completion)
        print("\n=== Parsed Sections ===")
        print(
            json.dumps(
                {
                    "report": report,
                    "answer": answer,
                    "tool_call": tool_call,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
