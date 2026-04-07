import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import PolicyDataset, collate_policy
from valor.io_utils import read_jsonl
from valor.model import DEFAULT_QLORA_TARGET_MODULES, PolicyModel
from valor.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy with QLoRA advantage conditioning.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl with advantage labels.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint.")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before optimizer step.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Compute dtype used inside bitsandbytes 4-bit kernels.",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_QLORA_TARGET_MODULES),
        help="Comma-separated LoRA target module suffixes. Defaults cover attention and MLP/expert projections.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing on the quantized backbone.",
    )
    return parser.parse_args()


def _parse_lora_target_modules(raw_value: str) -> list[str]:
    modules = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not modules:
        raise ValueError("--lora-target-modules must contain at least one module name.")
    return modules



def _resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]



def _resolve_device_map(device: str, raw_device_map: str | None):
    if raw_device_map is not None:
        stripped = raw_device_map.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
        return stripped

    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}
    return None



def _count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Policy QLoRA training requires CUDA and bitsandbytes on the remote server.")

    records = read_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PolicyDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )

    compute_dtype = _resolve_compute_dtype(args.bnb_4bit_compute_dtype)
    device_map = _resolve_device_map(args.device, args.device_map)
    lora_target_modules = _parse_lora_target_modules(args.lora_target_modules)

    print(f"Loading policy backbone with QLoRA from {args.backbone}...")
    model = PolicyModel(
        args.backbone,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
        use_qlora=True,
        qlora_trainable=True,
        qlora_compute_dtype=compute_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    if not args.disable_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after QLoRA initialization.")

    trainable_count, total_count = _count_trainable_parameters(model)
    print(
        "Trainable parameters: "
        f"{trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / max(total_count, 1):.4f}%)"
    )
    print(f"LoRA target modules: {', '.join(lora_target_modules)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        foreach=False,
        eps=1e-7,
    )

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(enumerate(loader), desc=f"epoch {epoch + 1}", total=len(loader))
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in progress:
            if batch is None:
                continue

            base_batch = collate_policy(
                batch,
                tokenizer,
                args.max_length,
                include_advantage=False,
                indicator_drop_prob=0.0,
            )

            if (base_batch["labels"] == -100).all():
                print(f"WARNING: all labels masked at batch {batch_idx}, skipping")
                continue

            outputs = model(
                input_ids=base_batch["input_ids"],
                attention_mask=base_batch["attention_mask"],
                labels=base_batch["labels"],
            )
            loss = outputs.lm_loss
            if loss is None:
                raise RuntimeError("Policy model returned no language-model loss.")
            loss = loss.float()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf detected in base loss at batch {batch_idx}, skipping")
                continue

            if args.alpha > 0:
                indicator_batch = collate_policy(
                    batch,
                    tokenizer,
                    args.max_length,
                    include_advantage=True,
                    indicator_drop_prob=args.indicator_drop_prob,
                )
                indicator_outputs = model(
                    input_ids=indicator_batch["input_ids"],
                    attention_mask=indicator_batch["attention_mask"],
                    labels=indicator_batch["labels"],
                )
                indicator_loss = indicator_outputs.lm_loss
                if indicator_loss is None:
                    raise RuntimeError("Indicator-conditioned forward pass returned no loss.")
                indicator_loss = indicator_loss.float()
                if torch.isnan(indicator_loss) or torch.isinf(indicator_loss):
                    print(f"WARNING: NaN/Inf detected in indicator loss at batch {batch_idx}, ignoring")
                else:
                    loss = loss + args.alpha * indicator_loss

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += scaled_loss.item()
            accumulated_batches += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                if torch.isnan(grad_norm):
                    print("WARNING: NaN gradients detected, skipping optimizer step")
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    accumulated_batches = 0
                    continue

                optimizer.step()
                optimizer.zero_grad()
                avg_loss = accumulated_loss / accumulated_batches if accumulated_batches > 0 else 0.0
                progress.set_postfix(loss=avg_loss)
                accumulated_loss = 0.0
                accumulated_batches = 0

        if accumulated_batches > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if not torch.isnan(grad_norm):
                optimizer.step()
            optimizer.zero_grad()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved QLoRA policy adapter to {output_dir}")


if __name__ == "__main__":
    main()
