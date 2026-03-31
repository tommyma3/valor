import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import ValueDataset, collate_value
from valor.io_utils import read_jsonl
from valor.model import ValueModel
from valor.trajectory import compute_returns, ensure_value_labels
from valor.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the value model for VALOR.")
    parser.add_argument("--data", required=True, help="Path to transitions jsonl.")
    parser.add_argument("--output", required=True, help="Output directory for checkpoint.")
    parser.add_argument("--backbone", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = read_jsonl(args.data)
    compute_returns(records)
    ensure_value_labels(records)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ValueDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_value(batch, tokenizer, args.max_length),
    )

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = ValueModel(
        args.backbone,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device_map is None:
        model.to(device)

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )

    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch+1}")
        for batch in progress:
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = torch.nn.functional.cross_entropy(
                outputs.value_logits, batch["value_labels"]
            )
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
