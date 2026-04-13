import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import PolicyDataset, collate_policy
from valor.io_utils import read_jsonl
from valor.model import PolicyModel
from valor.rl_utils import load_json, save_json, utc_now_iso
from valor.utils import set_seed


TRAINING_STATE_FILENAME = "training_state.json"
OPTIMIZER_STATE_FILENAME = "optimizer.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy with advantage conditioning.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl with advantage labels.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint.")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-9B")
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
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for policy training (e.g. auto, balanced, balanced_low_0, sequential, or JSON).",
    )
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Per-GPU memory limit (e.g. 20GiB) or JSON dict for max_memory when sharding the model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model parameter/activation dtype used during policy training.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing on the policy backbone.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention backend for policy training. Defaults to sdpa to avoid unstable model-specific kernels.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a resumable checkpoint every N optimizer steps. Set to 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume policy training from the latest checkpoint in --output.",
    )

    args = parser.parse_args()
    args.data = str(Path(args.data).expanduser().resolve())
    args.output = str(Path(args.output).expanduser().resolve())
    return args


def _resolve_compute_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def _resolve_device_map(
    device: str,
    raw_device_map: str | None,
    gpu_count: int,
    backbone_name: str,
) -> tuple[str | dict | None, str | None]:
    del backbone_name
    if raw_device_map is not None:
        stripped = raw_device_map.strip()
        if stripped.startswith("{"):
            return json.loads(stripped), None
        if stripped == "auto" and gpu_count > 1:
            return "balanced_low_0", "cuda:0"
        return stripped, None

    if device == "cuda":
        return {"": 0}, None
    if device.startswith("cuda:"):
        return {"": int(device.split(":", maxsplit=1)[1])}, None
    return None, None


def _parse_max_memory(value: str | None, gpu_count: int) -> dict[int, str] | dict | None:
    if not value:
        return None
    raw = value.strip()
    if raw.startswith("{"):
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("--max-memory JSON must be an object.")
        return loaded
    return {idx: raw for idx in range(gpu_count)}


def _build_loader(dataset: PolicyDataset, batch_size: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: batch,
    )


def _count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def _build_config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": args.data,
        "backbone": args.backbone,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_length": args.max_length,
        "alpha": args.alpha,
        "indicator_drop_prob": args.indicator_drop_prob,
        "seed": args.seed,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_memory": args.max_memory,
        "torch_dtype": args.torch_dtype,
        "disable_gradient_checkpointing": bool(args.disable_gradient_checkpointing),
        "attn_implementation": args.attn_implementation,
    }


def _save_optimizer_state(optimizer: torch.optim.Optimizer, output_dir: Path) -> None:
    optimizer_state_path = output_dir / OPTIMIZER_STATE_FILENAME
    tmp_path = optimizer_state_path.with_suffix(optimizer_state_path.suffix + ".tmp")
    torch.save(optimizer.state_dict(), tmp_path)
    tmp_path.replace(optimizer_state_path)


def _save_training_checkpoint(
    *,
    output_dir: Path,
    model: PolicyModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _save_optimizer_state(optimizer, output_dir)
    save_json(output_dir / TRAINING_STATE_FILENAME, state)


def _load_training_state(output_dir: Path) -> dict[str, Any] | None:
    state = load_json(output_dir / TRAINING_STATE_FILENAME)
    if state is not None and not isinstance(state, dict):
        raise ValueError(f"Expected object in {output_dir / TRAINING_STATE_FILENAME}.")
    return state


def _validate_resume_state(state: dict[str, Any], config_snapshot: dict[str, Any]) -> None:
    saved_config = state.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint state is missing a valid config snapshot.")

    mismatches: list[str] = []
    for key, current_value in config_snapshot.items():
        if key not in saved_config:
            continue
        saved_value = saved_config.get(key)
        if saved_value != current_value:
            mismatches.append(f"{key}: saved={saved_value!r}, current={current_value!r}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Resume config mismatch for policy checkpoint: {mismatch_text}")


def _load_optimizer_state(optimizer: torch.optim.Optimizer, output_dir: Path) -> None:
    optimizer_state_path = output_dir / OPTIMIZER_STATE_FILENAME
    if not optimizer_state_path.is_file():
        raise FileNotFoundError(f"Missing optimizer state for resume: {optimizer_state_path}")
    optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
    optimizer.load_state_dict(optimizer_state)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.data)
    dataset = PolicyDataset(records)

    compute_dtype = _resolve_compute_dtype(args.torch_dtype)
    gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
    config_snapshot = _build_config_snapshot(args)

    resume_state = None
    model_source = args.backbone
    tokenizer_source = args.backbone
    if args.resume:
        resume_state = _load_training_state(output_dir)
        if resume_state is None:
            raise FileNotFoundError(
                f"--resume was set but no checkpoint state exists at {output_dir / TRAINING_STATE_FILENAME}"
            )
        _validate_resume_state(resume_state, config_snapshot)
        model_source = str(output_dir)
        tokenizer_source = str(output_dir)

    device_map, io_device = _resolve_device_map(args.device, args.device_map, gpu_count, model_source)
    max_memory = _parse_max_memory(args.max_memory, gpu_count)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_implementation = None if args.attn_implementation == "auto" else args.attn_implementation

    print(
        f"Loading policy backbone from {model_source} "
        f"(dtype: {args.torch_dtype}, attention: {args.attn_implementation}, "
        f"device_map: {device_map}, max_memory: {max_memory}, io_device: {io_device})..."
    )
    model = PolicyModel(
        model_source,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
        max_memory=max_memory,
        attn_implementation=attn_implementation,
        io_device=io_device,
    )

    model_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device_map is None:
        model.to(model_device)

    if not args.disable_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for policy training.")

    trainable_count, total_count = _count_trainable_parameters(model)
    print(
        "Trainable parameters: "
        f"{trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / max(total_count, 1):.4f}%)"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        foreach=False,
    )
    if resume_state is not None:
        _load_optimizer_state(optimizer, output_dir)

    model.train()
    start_epoch = int(resume_state.get("next_epoch", 0)) if resume_state is not None else 0
    start_batch_idx = int(resume_state.get("next_batch_idx", 0)) if resume_state is not None else 0
    optimizer_step = int(resume_state.get("optimizer_step", 0)) if resume_state is not None else 0

    if start_epoch >= args.epochs:
        print("Training is already complete according to the checkpoint state.")
        return

    if resume_state is not None:
        print(
            "Resuming policy training from "
            f"epoch {start_epoch + 1}, batch {start_batch_idx + 1}, optimizer step {optimizer_step}."
        )

    for epoch in range(start_epoch, args.epochs):
        loader = _build_loader(dataset, args.batch_size, args.seed, epoch)
        total_batches = len(loader)
        epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
        progress = tqdm(enumerate(loader), desc=f"epoch {epoch + 1}", total=total_batches)
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in progress:
            if batch_idx < epoch_start_batch_idx:
                continue
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
                optimizer_step += 1
                avg_loss = accumulated_loss / accumulated_batches if accumulated_batches > 0 else 0.0
                progress.set_postfix(loss=avg_loss)
                accumulated_loss = 0.0
                accumulated_batches = 0

                if args.checkpoint_every > 0 and optimizer_step % args.checkpoint_every == 0:
                    checkpoint_state = {
                        "version": 1,
                        "updated_at": utc_now_iso(),
                        "optimizer_step": optimizer_step,
                        "next_epoch": epoch,
                        "next_batch_idx": batch_idx + 1,
                        "config": config_snapshot,
                    }
                    _save_training_checkpoint(
                        output_dir=output_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        state=checkpoint_state,
                    )

        if accumulated_batches > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if not torch.isnan(grad_norm):
                optimizer.step()
                optimizer_step += 1
            optimizer.zero_grad()

            if args.checkpoint_every > 0 and optimizer_step % args.checkpoint_every == 0:
                checkpoint_state = {
                    "version": 1,
                    "updated_at": utc_now_iso(),
                    "optimizer_step": optimizer_step,
                    "next_epoch": epoch + 1,
                    "next_batch_idx": 0,
                    "config": config_snapshot,
                }
                _save_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    state=checkpoint_state,
                )

        start_batch_idx = 0

    final_state = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "optimizer_step": optimizer_step,
        "next_epoch": args.epochs,
        "next_batch_idx": 0,
        "completed": True,
        "config": config_snapshot,
    }
    _save_training_checkpoint(
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        state=final_state,
    )
    print(f"Saved policy checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
