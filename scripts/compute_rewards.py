import argparse
from pathlib import Path

from valor.io_utils import read_jsonl, write_jsonl
from valor.rollout_data import (
    assign_terminal_binary_rewards,
    assign_terminal_binary_rewards_from_correctness,
    load_browsecomp_eval_summary_correctness,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary rewards.")
    parser.add_argument("--data", required=True, help="Trajectories jsonl.")
    parser.add_argument("--output", required=True, help="Output jsonl with rewards.")
    parser.add_argument(
        "--eval-summary",
        default=None,
        help=(
            "Optional BrowseComp evaluation_summary.json. If provided, terminal rewards are "
            "assigned from per_query_metrics[*].correct instead of final_answer vs gold_answer."
        ),
    )
    parser.add_argument("--trajectory-field", default="trajectory_id")
    parser.add_argument("--timestep-field", default="t")
    parser.add_argument("--final-answer-field", default="final_answer")
    parser.add_argument("--gold-answer-field", default="gold_answer")
    args = parser.parse_args()
    if args.eval_summary is not None:
        args.eval_summary = str(Path(args.eval_summary).expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data)
    if args.eval_summary:
        correctness_by_query_id = load_browsecomp_eval_summary_correctness(
            Path(args.eval_summary)
        )
        assign_terminal_binary_rewards_from_correctness(
            records,
            correctness_by_query_id,
            trajectory_field=args.trajectory_field,
            timestep_field=args.timestep_field,
        )
    else:
        assign_terminal_binary_rewards(
            records,
            trajectory_field=args.trajectory_field,
            timestep_field=args.timestep_field,
            final_answer_field=args.final_answer_field,
            gold_answer_field=args.gold_answer_field,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)


if __name__ == "__main__":
    main()
