import argparse
import json
import os
from datetime import date
from typing import Any

import requests

from prompts import browsecomp_initial_instruction_prompt
from valor.generation import STRICT_FORMAT_SYSTEM_PROMPT, build_chat_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a vLLM/OpenAI-compatible endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="vLLM base URL.")
    parser.add_argument("--model", required=True, help="Model name to send to vLLM.")
    parser.add_argument(
        "--mode",
        choices=["simple", "browsecomp"],
        default="simple",
        help="Prompt mode. 'simple' checks basic chat output, 'browsecomp' checks structured tool-call output.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", os.getenv("SGLANG_API_KEY", "")),
        help="Optional API key.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--question",
        default="What is 2 + 2? Reply with one short sentence.",
        help="Question used in the test prompt.",
    )
    parser.add_argument(
        "--raw-response",
        action="store_true",
        help="Print the full JSON response body.",
    )
    return parser.parse_args()


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


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


def _extract_choice_fields(choice: dict[str, Any]) -> dict[str, Any]:
    message = choice.get("message")
    if not isinstance(message, dict):
        return {}

    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    tool_calls = message.get("tool_calls")

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason"),
    }


def _print_models(base_url: str, headers: dict[str, str], timeout: int) -> None:
    models_url = base_url.rstrip("/") + "/v1/models"
    response = requests.get(models_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    print("=== /v1/models ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _chat(
    base_url: str,
    headers: dict[str, str],
    timeout: int,
    model: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    chat_url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }
    response = requests.post(chat_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    headers = _headers(args.api_key)

    _print_models(args.url, headers, args.timeout)

    if args.mode == "browsecomp":
        prompt = _format_browsecomp_prompt(args.question)
        messages = build_chat_messages(
            prompt,
            system_prompt=STRICT_FORMAT_SYSTEM_PROMPT,
        )
    else:
        prompt = _format_simple_prompt(args.question)
        messages = build_chat_messages(prompt)

    response = _chat(
        base_url=args.url,
        headers=headers,
        timeout=args.timeout,
        model=args.model,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Parsed Choice Fields ===")
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        print("No choices found in response.")
    else:
        for idx, choice in enumerate(choices):
            print(f"-- choice {idx} --")
            print(json.dumps(_extract_choice_fields(choice), ensure_ascii=False, indent=2))

    if args.raw_response:
        print("\n=== Raw Response ===")
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
