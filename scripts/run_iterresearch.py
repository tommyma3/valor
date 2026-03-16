import argparse
import sys
from datetime import date
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import initial_instruction_prompt
from valor.model import PolicyValueModel
from valor.system_prompts import build_tools_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IterResearch prompt with a local model.")
    parser.add_argument("--model-path", default="model/Qwen3.5-35B-A3B")
    parser.add_argument("--question", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tools_text = build_tools_prompt()
    prompt = initial_instruction_prompt.format(
        date_to_use=args.date or date.today().isoformat(),
        question=args.question,
        tools=tools_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyValueModel(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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

    print("=== Prompt ===")
    print(prompt)
    print("=== Completion ===")
    print(completion.strip())


if __name__ == "__main__":
    main()
