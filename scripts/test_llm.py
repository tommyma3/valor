import argparse

import torch
from transformers import AutoTokenizer

from valor.model import PolicyModel
from valor.prompts import State, format_state_prompt, parse_action


def _resolve_advantage_label(raw_value: str) -> int | None:
    normalized = raw_value.strip().lower()
    if normalized == "positive":
        return 1
    if normalized == "negative":
        return 0
    if normalized == "none":
        return None
    raise ValueError(f"Unsupported advantage label: {raw_value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick LLM sanity test.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory.")
    parser.add_argument("--question", default="Summarize the key idea in one sentence.")
    parser.add_argument("--memory", default="")
    parser.add_argument("--prev-tool-query", default="")
    parser.add_argument("--prev-tool-result", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--advantage-label",
        choices=["positive", "negative", "none"],
        default="none",
        help="Condition generation on an advantage indicator. Use 'positive' to inspect the improved policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    advantage_label = _resolve_advantage_label(args.advantage_label)

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

    state = State(
        question=args.question,
        memory=args.memory,
        prev_tool_query=args.prev_tool_query,
        prev_tool_result=args.prev_tool_result,
    )
    prompt = format_state_prompt(
        state,
        include_advantage=advantage_label is not None,
        advantage_label=advantage_label,
    )

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

    try:
        action = parse_action(completion)
    except ValueError:
        print("=== Parsed Action ===")
        print("Could not parse structured action from completion.")
        return

    print("=== Parsed Action ===")
    print("THINK:")
    print(action.think)
    print("MEMORY:")
    print(action.memory_update)
    print("TOOL:")
    print(action.tool_query)


if __name__ == "__main__":
    main()
