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
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for multi-GPU inference (e.g., auto, balanced, balanced_low_0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def report(step: int, total: int, message: str) -> None:
        print(f"[Progress {step}/{total}] {message}")

    total_steps = 9
    step = 1
    report(step, total_steps, "Building tools prompt.")
    tools_text = build_tools_prompt()

    step += 1
    report(step, total_steps, "Formatting instruction prompt.")
    prompt = initial_instruction_prompt.format(
        date_to_use=args.date or date.today().isoformat(),
        question=args.question,
        tools=tools_text,
    )

    step += 1
    report(step, total_steps, "Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    step += 1
    gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
    device_map = args.device_map
    if device_map is None and gpu_count > 1:
        device_map = "auto"
    if device_map is not None and args.device != "cuda":
        device_map = None
    report(
        step,
        total_steps,
        f"Resolved device map: {device_map or 'none'} (GPU count: {gpu_count}).",
    )

    step += 1
    report(step, total_steps, "Loading model.")
    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyValueModel(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    step += 1
    report(step, total_steps, "Preparing device.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device_map is None:
        model.to(device)
    model.eval()

    step += 1
    report(step, total_steps, "Tokenizing prompt.")
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        step += 1
        report(step, total_steps, "Generating completion.")
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

    step += 1
    report(step, total_steps, "Printing results.")
    print("=== Prompt ===")
    print(prompt)
    print("=== Completion ===")
    print(completion.strip())


if __name__ == "__main__":
    main()
