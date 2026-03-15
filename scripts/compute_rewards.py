import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from valor.io_utils import read_jsonl, write_jsonl


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary rewards.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl.")
    parser.add_argument("--output", required=True, help="Output jsonl with rewards.")
    parser.add_argument("--trajectory-field", default="trajectory_id")
    parser.add_argument("--timestep-field", default="t")
    parser.add_argument("--final-answer-field", default="final_answer")
    parser.add_argument("--gold-answer-field", default="gold_answer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = record.get(args.trajectory_field, str(idx))
        grouped[key].append(record)

    for traj in grouped.values():
        if args.timestep_field in traj[0]:
            traj.sort(key=lambda r: r[args.timestep_field])
        for record in traj:
            record["reward"] = 0
        final = traj[-1]
        pred = normalize(str(final.get(args.final_answer_field, "")))
        gold = normalize(str(final.get(args.gold_answer_field, "")))
        final["reward"] = 1 if pred and pred == gold else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)


if __name__ == "__main__":
    main()
