import argparse
import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import PolicyDataset, collate_policy
from valor.io_utils import read_jsonl
from valor.model import PolicyModel
from valor.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train policy with advantage conditioning.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl with advantage labels.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint.")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Number of steps to accumulate gradients (simulate larger batch size)")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="Local rank for distributed training")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    records = read_jsonl(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset with distributed sampler
    dataset = PolicyDataset(records)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=lambda batch: batch,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )

    torch_dtype = torch.bfloat16

    # Load model on CPU first, then wrap with FSDP
    if is_main_process:
        print(f"Loading model {args.backbone} on CPU...")
    model = PolicyModel(
        args.backbone,
        torch_dtype=torch_dtype,
        device_map=None,  # Load on CPU
        trust_remote_code=True,
    )

    # Enable gradient checkpointing before FSDP wrapping
    model.backbone.gradient_checkpointing_enable()

    # Set device
    device = torch.device(f"cuda:{local_rank}")

    # Move model to GPU - FSDP will shard across GPUs
    if world_size > 1:
        print(f"Rank {rank}: Moving model to GPU...")
    model = model.to(device)

    # Wrap with FSDP for distributed training
    if world_size > 1:
        if is_main_process:
            print("Wrapping model with FSDP...")

        # Define mixed precision policy
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Get transformer layer class for auto-wrap
        # For Qwen models, we need to identify the transformer layer type
        # This is typically the decoder layer class
        transformer_layer_cls = None
        for name, module in model.named_modules():
            if 'layers' in name.lower() or 'decoder' in name.lower():
                # Get the layer class from the first child
                for child_name, child_module in module.named_children():
                    transformer_layer_cls = type(child_module)
                    break
                if transformer_layer_cls:
                    break

        if transformer_layer_cls is None:
            # Fallback: try to find any Layer-like class
            for name, module in model.named_modules():
                if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
                    transformer_layer_cls = type(module)
                    break

        if is_main_process:
            print(f"Using transformer layer class: {transformer_layer_cls}")

        # Wrap with FSDP using ModuleWrapPolicy
        if transformer_layer_cls:
            auto_wrap_policy = ModuleWrapPolicy({transformer_layer_cls})
        else:
            # Fallback to size-based policy if we can't identify transformer layers
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=1e6,  # Wrap modules with at least 1M params
            )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
            device_id=device,
            limit_all_gathers=True,
        )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        foreach=False,
        eps=1e-7,
    )

    model.train()
    for epoch in range(args.epochs):
        if world_size > 1:
            loader.sampler.set_epoch(epoch)

        if is_main_process:
            progress = tqdm(enumerate(loader), desc=f"epoch {epoch+1}", total=len(loader))
        else:
            progress = enumerate(loader)

        # Initialize gradient accumulation
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in progress:
            # Skip if batch is None or contains NaN
            if batch is None:
                continue

            # Debug: Check batch content (only on main process)
            if is_main_process and batch_idx < 5:
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
            if is_main_process and batch_idx < 5:
                print(f"  After collate_policy:")
                print(f"    input_ids shape: {base_batch['input_ids'].shape}")
                print(f"    labels min/max: {base_batch['labels'].min()}/{base_batch['labels'].max()}")
                print(f"    Number of -100 in labels: {(base_batch['labels'] == -100).sum().item()}")
                print(f"    Number of valid labels: {(base_batch['labels'] != -100).sum().item()}")

                # Check if all labels are -100
                if (base_batch['labels'] == -100).all():
                    print(f"    ERROR: All labels are -100! This will cause NaN loss.")
                    continue

            # Move inputs to GPU - FSDP handles parameter movement but not inputs
            base_batch = {k: v.to(device) for k, v in base_batch.items()}

            if is_main_process and batch_idx < 3:
                print(f"  Input device after to(device): {base_batch['input_ids'].device}")

            outputs = model(
                input_ids=base_batch["input_ids"],
                attention_mask=base_batch["attention_mask"],
                labels=base_batch["labels"],
            )
            loss = outputs.lm_loss

            # Debug: Print loss value
            if is_main_process and batch_idx < 5:
                print(f"    Loss value: {loss.item() if isinstance(loss, torch.Tensor) else loss}")

            # Check for NaN in loss before continuing
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process:
                    print(f"WARNING: NaN/Inf detected in base loss at batch {batch_idx}, skipping batch")
                continue

            if args.alpha > 0:
                indicator_batch = collate_policy(
                    batch,
                    tokenizer,
                    args.max_length,
                    include_advantage=True,
                    indicator_drop_prob=args.indicator_drop_prob,
                )
                # Move inputs to GPU
                indicator_batch = {k: v.to(device) for k, v in indicator_batch.items()}

                if is_main_process and batch_idx < 3:
                    print(f"  Indicator input device: {indicator_batch['input_ids'].device}")

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
                    if is_main_process:
                        print(f"WARNING: NaN/Inf in indicator loss at batch {batch_idx}")
                else:
                    loss = loss + args.alpha * indicator_loss

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
                    if is_main_process:
                        print(f"WARNING: NaN gradients at step, skipping optimizer step")
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    accumulated_batches = 0
                    continue

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Update progress with average loss over accumulated steps
                if is_main_process:
                    avg_loss = accumulated_loss / accumulated_batches if accumulated_batches > 0 else 0.0
                    progress.set_postfix(loss=avg_loss)

                # Reset for next accumulation
                accumulated_loss = 0.0
                accumulated_batches = 0

        # Handle any remaining accumulated gradients at end of epoch
        if accumulated_batches > 0:
            if is_main_process:
                print(f"Processing final {accumulated_batches} accumulated batches")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isnan(grad_norm):
                optimizer.step()
                optimizer.zero_grad()

    # Save model (only on main process)
    if is_main_process:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FSDP model
        # Need to gather full state dict from all shards
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp.api import StateDictType

        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        state_dict = model.state_dict()

        # Save backbone
        if hasattr(model, 'module'):
            model.module.backbone.save_pretrained(output_dir, state_dict=state_dict)
        else:
            model.backbone.save_pretrained(output_dir, state_dict=state_dict)

        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
