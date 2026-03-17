import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
    )

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyModel(
        args.backbone,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    if args.device_map is None:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = None

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch+1}")
        for batch in progress:
            base_batch = collate_policy(
                batch,
                tokenizer,
                args.max_length,
                include_advantage=False,
                indicator_drop_prob=0.0,
            )
            if device is not None:
                base_batch = {k: v.to(device) for k, v in base_batch.items()}

            outputs = model(
                input_ids=base_batch["input_ids"],
                attention_mask=base_batch["attention_mask"],
                labels=base_batch["labels"],
            )
            loss = outputs.lm_loss

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

                indicator_outputs = model(
                    input_ids=indicator_batch["input_ids"],
                    attention_mask=indicator_batch["attention_mask"],
                    labels=indicator_batch["labels"],
                )
                loss = loss + args.alpha * indicator_outputs.lm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=loss.item())

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
