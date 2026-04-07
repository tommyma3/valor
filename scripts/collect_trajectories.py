import argparse
import os
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.io_utils import read_jsonl, write_jsonl
from valor.model import PolicyModel
from valor.prompts import State, format_state_prompt, parse_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect trajectories with the policy model.")
    parser.add_argument("--states", required=True, help="Input states jsonl.")
    parser.add_argument("--output", required=True, help="Output trajectories jsonl.")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint directory or HF id.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sglang-url", default="", help="SGLang OpenAI-compatible base URL.")
    parser.add_argument("--sglang-model", default="", help="Model name for SGLang server.")
    parser.add_argument("--sglang-api-key", default=os.getenv("SGLANG_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def _sglang_chat(
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
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.states)
    use_sglang = bool(args.sglang_url)

    if use_sglang:
        model_name = args.sglang_model or args.checkpoint
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

    outputs = []
    for record in tqdm(records, desc="generate"):
        state = State(
            question=record.get("question", ""),
            memory=record.get("memory", ""),
            prev_tool_query=record.get("prev_tool_query", ""),
            prev_tool_result=record.get("prev_tool_result", ""),
        )
        prompt = format_state_prompt(state)

        if use_sglang:
            completion = _sglang_chat(
                args.sglang_url,
                model_name,
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                api_key=args.sglang_api_key,
                timeout=args.timeout,
            )
        else:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.backbone.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            completion = text[len(prompt):]

        try:
            action = parse_action(completion)
            record.update(
                {
                    "action_think": action.think,
                    "action_memory_update": action.memory_update,
                    "action_tool_query": action.tool_query,
                }
            )
        except ValueError:
            record["action_raw"] = completion.strip()

        outputs.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, outputs)


if __name__ == "__main__":
    main()
