import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.io_utils import read_jsonl, write_jsonl
from valor.model import PolicyValueModel
from valor.prompts import State, format_state_prompt, parse_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect trajectories with the policy model.")
    parser.add_argument("--states", required=True, help="Input states jsonl.")
    parser.add_argument("--output", required=True, help="Output trajectories jsonl.")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint directory.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.states)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyValueModel(
        args.checkpoint,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
