#!/usr/bin/env python3
"""Debug NaN loss issue in policy training."""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from valor.data import PolicyDataset, collate_policy
from valor.model import PolicyModel

def debug_data_file(data_path, max_examples=10):
    """Check the data file for issues."""
    print(f"\n=== Debugging data file: {data_path} ===")
    data = [json.loads(line) for line in open(data_path)]

    # Check lengths
    prompt_lengths = []
    action_lengths = []
    total_lengths = []

    for i, d in enumerate(data[:max_examples]):
        question_len = len(d.get('question', ''))
        memory_len = len(d.get('memory', ''))
        prev_query_len = len(d.get('prev_tool_query', ''))
        prev_result_len = len(d.get('prev_tool_result', ''))

        action_think_len = len(d.get('action_think', ''))
        action_mem_len = len(d.get('action_memory_update', ''))
        action_tool_len = len(d.get('action_tool_query', ''))

        prompt_len = question_len + memory_len + prev_query_len + prev_result_len
        action_len = action_think_len + action_mem_len + action_tool_len

        prompt_lengths.append(prompt_len)
        action_lengths.append(action_len)
        total_lengths.append(prompt_len + action_len)

        if i < 5:  # Print first 5 examples
            print(f"\nExample {i}:")
            print(f"  Prompt chars: {prompt_len}")
            print(f"  Action chars: {action_len}")
            print(f"  Total chars: {prompt_len + action_len}")
            print(f"  Advantage label: {d.get('advantage_label')}")
            print(f"  Action fields present: {all(k in d for k in ['action_think', 'action_memory_update', 'action_tool_query'])}")

    print(f"\n=== Length Statistics ===")
    print(f"Max prompt length: {max(prompt_lengths)}")
    print(f"Max action length: {max(action_lengths)}")
    print(f"Max total length: {max(total_lengths)}")
    print(f"Mean total length: {sum(total_lengths) / len(total_lengths):.2f}")

    # Check for overly long examples
    too_long = sum(1 for l in total_lengths if l > 1500)
    print(f"Examples with >1500 chars: {too_long}/{len(total_lengths)}")

def debug_collate(data_path, max_length=2048, num_batches=5):
    """Debug the collation process."""
    print(f"\n=== Debugging collation (max_length={max_length}) ===")

    records = [json.loads(line) for line in open(data_path)]
    dataset = PolicyDataset(records)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    print(f"Dataset size: {len(dataset)}")

    for i in range(min(num_batches, len(dataset))):
        print(f"\n--- Batch {i} ---")
        item = dataset[i]

        # Check the raw item
        print(f"Item keys: {item.keys()}")
        print(f"Advantage label: {item.get('advantage_label')}")

        # Try collating
        try:
            batch = collate_policy([item], tokenizer, max_length, False, 0.0)

            print(f"Input shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Labels min/max: {batch['labels'].min()}/{batch['labels'].max()}")

            num_valid = (batch['labels'] != -100).sum().item()
            num_ignored = (batch['labels'] == -100).sum().item()
            print(f"Valid labels: {num_valid}")
            print(f"Ignored labels (-100): {num_ignored}")

            if num_valid == 0:
                print("ERROR: No valid labels! This batch will cause NaN loss.")
                print(f"This likely means the prompt is too long for max_length={max_length}")

        except Exception as e:
            print(f"Error during collation: {e}")
            import traceback
            traceback.print_exc()

def debug_model_forward(data_path, max_length=2048, device='cuda'):
    """Debug model forward pass."""
    print(f"\n=== Debugging model forward pass ===")

    records = [json.loads(line) for line in open(data_path)]
    dataset = PolicyDataset(records)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    print(f"Loading model to {device}...")
    model = PolicyModel(
        "Qwen/Qwen3.5-35B-A3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.backbone.gradient_checkpointing_enable()
    model.train()

    print(f"Model training mode: {model.training}")
    print(f"Model dtype: {model.backbone.dtype}")

    for i in range(min(3, len(dataset))):
        print(f"\n--- Testing batch {i} ---")
        item = dataset[i]
        batch = collate_policy([item], tokenizer, max_length, False, 0.0)

        # Check if batch has valid labels
        num_valid = (batch['labels'] != -100).sum().item()
        if num_valid == 0:
            print("Skipping batch with no valid labels")
            continue

        # Move to device
        device_batch = {k: v.to(device) for k, v in batch.items()}

        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    labels=device_batch["labels"],
                )
                loss = outputs.lm_loss

                print(f"Loss: {loss.item():.4f}")
                print(f"Loss dtype: {loss.dtype}")
                print(f"Loss is NaN: {torch.isnan(loss).item()}")
                print(f"Loss is inf: {torch.isinf(loss).item()}")

        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    data_path = "./runs/iter_001_checkpoints/trajectories_adv.jsonl"

    print("Checking if data file exists...")
    if Path(data_path).exists():
        print(f"✓ Found: {data_path}")
    else:
        print(f"✗ Not found: {data_path}")
        exit(1)

    # Run debug checks
    debug_data_file(data_path)
    debug_collate(data_path, max_length=2048)
    debug_model_forward(data_path, max_length=2048, device='cuda')

    print("\n=== Summary ===")
    print("If you see 'No valid labels' warnings, reduce --max-length or remove long examples")
    print("If loss is NaN in forward pass, try: --max-length 1536 --lr 1e-5")
