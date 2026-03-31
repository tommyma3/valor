import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import PolicyDataset, collate_policy
from valor.io_utils import read_jsonl
from valor.model import PolicyModel
from valor.utils import set_seed

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy with advantage conditioning.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl with advantage labels.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint.")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Number of steps to accumulate gradients (simulate larger batch size)")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="Path to DeepSpeed config file (enables DeepSpeed training)")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="Local rank for distributed training (passed by DeepSpeed)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = read_jsonl(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PolicyDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,  # Return list as-is, collate manually in training loop
    )

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyModel(
        args.backbone,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save memory
    model.backbone.gradient_checkpointing_enable()

    # Initialize DeepSpeed if config provided
    use_deepspeed = False
    device = None
    if args.deepspeed is not None:
        if not DEEPSPEED_AVAILABLE:
            print("WARNING: DeepSpeed requested but not installed. Install with: pip install deepspeed")
            print("Continuing without DeepSpeed...")
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad), lr=args.lr, foreach=False, eps=1e-7
            )
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            if args.device_map is None:
                model.to(device)
        else:
            print(f"Initializing DeepSpeed with config: {args.deepspeed}")
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=[p for p in model.parameters() if p.requires_grad],
                config=args.deepspeed,
                optimizer=None,
                mpu=None,
                dist_init_required=True,
            )
            use_deepspeed = True
            # DeepSpeed handles all device placement, don't manually move tensors
    else:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr, foreach=False, eps=1e-7
        )
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if args.device_map is None:
            model.to(device)

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(enumerate(loader), desc=f"epoch {epoch+1}", total=len(loader))

        # Initialize gradient accumulation
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in progress:
            # Skip if batch is None or contains NaN
            if batch is None:
                continue

            # Debug: Check batch content
            if batch_idx < 5:
                print(f"DEBUG Batch {batch_idx}:")
                print(f"  Batch keys: {batch[0].keys() if batch else 'Empty'}")
                if batch and 'state' in batch[0]:
                    state = batch[0]['state']
                    print(f"  State question length: {len(state.question)}")
                if 'advantage_label' in batch[0]:
                    print(f"  Advantage label: {batch[0]['advantage_label']}")

            base_batch = collate_policy(
                batch,
                tokenizer,
                args.max_length,
                include_advantage=False,
                indicator_drop_prob=0.0,
            )

            # Debug: Check collated batch
            if batch_idx < 5:
                print(f"  After collate_policy:")
                print(f"    input_ids shape: {base_batch['input_ids'].shape}")
                print(f"    labels min/max: {base_batch['labels'].min()}/{base_batch['labels'].max()}")
                print(f"    Number of -100 in labels: {(base_batch['labels'] == -100).sum().item()}")
                print(f"    Number of valid labels: {(base_batch['labels'] != -100).sum().item()}")

                # Check if all labels are -100
                if (base_batch['labels'] == -100).all():
                    print(f"    ERROR: All labels are -100! This will cause NaN loss.")
                    continue

            # Move to device if not using DeepSpeed
            # DeepSpeed handles device placement automatically
            if device is not None:
                base_batch = {k: v.to(device) for k, v in base_batch.items()}

            outputs = model(
                input_ids=base_batch["input_ids"],
                attention_mask=base_batch["attention_mask"],
                labels=base_batch["labels"],
            )
            loss = outputs.lm_loss

            # Debug: Print loss value
            if batch_idx < 5:
                print(f"    Loss value: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
                print(f"    Loss type: {type(loss)}")

            # Check for NaN in loss before continuing
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf detected in base loss at batch {batch_idx}, skipping batch")
                if batch_idx < 10:
                    print(f"  Debug info for NaN batch:")
                    print(f"    input_ids min/max: {base_batch['input_ids'].min()}/{base_batch['input_ids'].max()}")
                    print(f"    attention_mask sum: {base_batch['attention_mask'].sum()}")
                continue

            if args.alpha > 0:
                indicator_batch = collate_policy(
                    batch,
                    tokenizer,
                    args.max_length,
                    include_advantage=True,
                    indicator_drop_prob=args.indicator_drop_prob,
                )
                if device is not None:
                    indicator_batch = {k: v.to(device) for k, v in indicator_batch.items()}

                # Move to device if needed
                if device is not None:
                    indicator_batch = {k: v.to(device) for k, v in indicator_batch.items()}

                indicator_outputs = model(
                    input_ids=indicator_batch["input_ids"],
                    attention_mask=indicator_batch["attention_mask"],
                    labels=indicator_batch["labels"],
                )
                indicator_loss = indicator_outputs.lm_loss

                # Cast to float32 for stability if using bfloat16
                if indicator_loss.dtype == torch.bfloat16:
                    indicator_loss = indicator_loss.float()

                # Check for NaN in indicator loss
                if torch.isnan(indicator_loss) or torch.isinf(indicator_loss):
                    print(f"WARNING: NaN/Inf in indicator loss at batch {batch_idx}")
                else:
                    loss = loss + args.alpha * indicator_loss

            if use_deepspeed:
                # DeepSpeed handles gradient accumulation and optimizer internally
                model.backward(loss)
                model.step()
                progress.set_postfix(loss=loss.item())
            else:
                # Manual gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()
                accumulated_batches += 1

                # Only update weights every gradient_accumulation_steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Check for NaN in gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm):
                        print(f"WARNING: NaN gradients at step, skipping optimizer step")
                        optimizer.zero_grad()
                        accumulated_loss = 0.0
                        accumulated_batches = 0
                        continue

                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update progress with average loss over accumulated steps
                    avg_loss = accumulated_loss / accumulated_batches if accumulated_batches > 0 else 0.0
                    progress.set_postfix(loss=avg_loss)

                    # Reset for next accumulation
                    accumulated_loss = 0.0
                    accumulated_batches = 0

        # Handle any remaining accumulated gradients at end of epoch (non-DeepSpeed only)
        if not use_deepspeed and accumulated_batches > 0:
            print(f"Processing final {accumulated_batches} accumulated batches")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isnan(grad_norm):
                optimizer.step()
                optimizer.zero_grad()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model differently based on training method
    if use_deepspeed:
        model.save_checkpoint(str(output_dir))
    else:
        model.save(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
