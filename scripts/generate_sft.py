import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import requests

from valor.io_utils import read_jsonl, write_jsonl
from valor.prompts import State, format_state_prompt, parse_action
from valor.system_prompts import render_system_prompt


DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT data with Kimi models.")
    parser.add_argument("--input", help="Input JSONL with states.")
    parser.add_argument("--output", help="Output JSONL with actions.")
    parser.add_argument("--model", help="Kimi model id (e.g., a K2.5 model).")
    parser.add_argument("--api-key", default=os.getenv("MOONSHOT_API_KEY"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def request_json(
    base_url: str,
    api_key: str,
    path: str,
    payload: Dict[str, Any] | None = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if payload is None:
        resp = requests.get(url, headers=headers, timeout=timeout)
    else:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def list_models(base_url: str, api_key: str, timeout: int) -> None:
    data = request_json(base_url, api_key, "/models", timeout=timeout)
    models = [m.get("id") for m in data.get("data", [])]
    for model in sorted(m for m in models if m):
        print(model)


def build_messages(state: State) -> List[Dict[str, str]]:
    system_prompt = render_system_prompt(question=state.question)
    prompt = format_state_prompt(state, include_system_prompt=False)
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": prompt},
    ]


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set MOONSHOT_API_KEY or pass --api-key.")

    if args.list_models:
        list_models(args.base_url, args.api_key, args.timeout)
        return

    if not args.input or not args.output or not args.model:
        raise SystemExit("--input, --output, and --model are required unless --list-models is used.")

    records = read_jsonl(args.input)
    outputs = []
    for record in records:
        state = State(
            question=record.get("question", ""),
            memory=record.get("memory", ""),
            prev_tool_query=record.get("prev_tool_query", ""),
            prev_tool_result=record.get("prev_tool_result", ""),
        )
        messages = build_messages(state)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        data = request_json(
            args.base_url,
            args.api_key,
            "/chat/completions",
            payload=payload,
            timeout=args.timeout,
        )

        content = data["choices"][0]["message"]["content"]
        try:
            action = parse_action(content)
            record.update(
                {
                    "action_think": action.think,
                    "action_memory_update": action.memory_update,
                    "action_tool_query": action.tool_query,
                    "action_valid": True,
                }
            )
        except ValueError:
            record.update(
                {
                    "action_raw": content,
                    "action_valid": False,
                }
            )

        record["sft_model"] = args.model
        usage = data.get("usage")
        if usage:
            record["sft_usage"] = usage

        outputs.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, outputs)


if __name__ == "__main__":
    main()
