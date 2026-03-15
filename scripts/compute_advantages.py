import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from valor.data import ValueDataset, collate_value
from valor.io_utils import read_jsonl, write_jsonl
from valor.model import PolicyValueModel
from valor.trajectory import compute_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute value predictions and advantages.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl.")
    parser.add_argument("--value-model", required=True, help="Checkpoint dir for value model.")
    parser.add_argument("--output", required=True, help="Output jsonl with advantages.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data)
    compute_returns(records)

    tokenizer = AutoTokenizer.from_pretrained(args.value_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ValueDataset(
        [
            {
                **record,
                "value_label": 0,
            }
            for record in records
        ]
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_value(batch, tokenizer, args.max_length),
    )

    torch_dtype = torch.bfloat16 if args.device == "cuda" else None
    model = PolicyValueModel(
        args.value_model,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    model.load_value_head(args.value_model)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    values = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="value"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = torch.softmax(outputs.value_logits, dim=-1)
            value_pred = probs[:, 1].detach().cpu().tolist()
            values.extend(value_pred)

    for record, value in zip(records, values):
        ret = float(record.get("return", 0.0))
        record["value_pred"] = value
        record["advantage"] = ret - value
        record["advantage_label"] = 1 if (ret - value) > args.epsilon else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)


if __name__ == "__main__":
    main()
