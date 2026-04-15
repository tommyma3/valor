from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from valor.generation import build_chat_messages, generate_local_completion
from valor.model import PolicyModel
from valor.prompts import State, format_state_prompt, parse_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one VALOR inference step with the positive advantage indicator."
    )
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint directory or HF model id.")
    parser.add_argument("--question", default="")
    parser.add_argument("--memory", default="")
    parser.add_argument("--prev-tool-query", default="")
    parser.add_argument("--prev-tool-result", default="")
    parser.add_argument(
        "--state-json",
        default=None,
        help="Optional JSON file with question/memory/prev_tool_query/prev_tool_result fields.",
    )
    parser.add_argument("--tools", default="", help="Optional tool description block injected into the prompt.")
    parser.add_argument("--date", default=None, help="Optional date injected into the prompt (YYYY-MM-DD).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--vllm-url",
        "--sglang-url",
        dest="vllm_url",
        default="",
        help="Optional vLLM OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--vllm-model",
        "--sglang-model",
        dest="vllm_model",
        default="",
        help="Model name for the vLLM server. Defaults to --checkpoint.",
    )
    parser.add_argument(
        "--vllm-api-key",
        "--sglang-api-key",
        dest="vllm_api_key",
        default=os.getenv("VLLM_API_KEY", os.getenv("SGLANG_API_KEY", "")),
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def _load_state(args: argparse.Namespace) -> State:
    if args.state_json is None:
        return State(
            question=args.question,
            memory=args.memory,
            prev_tool_query=args.prev_tool_query,
            prev_tool_result=args.prev_tool_result,
        )

    state_path = Path(args.state_json).expanduser().resolve()
    with state_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {state_path}.")
    return State(
        question=str(payload.get("question", "")),
        memory=str(payload.get("memory", "")),
        prev_tool_query=str(payload.get("prev_tool_query", "")),
        prev_tool_result=str(payload.get("prev_tool_result", "")),
    )


def _vllm_chat(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": build_chat_messages(prompt),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def main() -> None:
    args = parse_args()
    state = _load_state(args)
    prompt = format_state_prompt(
        state,
        include_advantage=True,
        advantage_label=1,
        tools=args.tools,
        date_to_use=args.date,
    )

    if args.vllm_url:
        completion = _vllm_chat(
            args.vllm_url,
            args.vllm_model or args.checkpoint,
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            api_key=args.vllm_api_key,
            timeout=args.timeout,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = torch.bfloat16 if args.device == "cuda" else None
        model = PolicyModel(
            args.checkpoint,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if not getattr(model.backbone, "is_loaded_in_4bit", False):
            model.to(device)
        model.eval()

        completion = generate_local_completion(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        ).completion

    print("=== Prompt ===")
    print(prompt)
    print("=== Completion ===")
    print(completion.strip())
    print("=== Parsed Action ===")

    try:
        action = parse_action(completion)
    except ValueError:
        print("Could not parse a VALOR action from the completion.")
        return

    print("THINK:")
    print(action.think)
    print("MEMORY:")
    print(action.memory_update)
    print("TOOL:")
    print(action.tool_query)


if __name__ == "__main__":
    main()
