from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from valor.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare state/QA JSONL files from WebShaper for collect_trajectories.py."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Local WebShaper JSONL path (one object per line).",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Local WebShaper JSON path (array of objects, or {data:[...]}).",
    )
    parser.add_argument(
        "--hf-dataset",
        default="Alibaba-NLP/WebShaper",
        help="Hugging Face dataset id when local input is not provided.",
    )
    parser.add_argument(
        "--hf-split",
        default="main",
        help="Hugging Face split name.",
    )

    parser.add_argument("--states-out", type=Path, required=True)
    parser.add_argument("--qa-out", type=Path, required=True)

    parser.add_argument("--id-field", default="id")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument(
        "--keep-fields",
        default="formalization,urls",
        help="Comma-separated extra fields copied into both outputs when present.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--strip-wrapping-quotes",
        action="store_true",
        help="Strip one pair of wrapping single/double quotes from question text.",
    )

    args = parser.parse_args()

    if args.input_jsonl is not None:
        args.input_jsonl = args.input_jsonl.expanduser().resolve()
    if args.input_json is not None:
        args.input_json = args.input_json.expanduser().resolve()

    if args.input_jsonl is not None and args.input_json is not None:
        raise ValueError("Use only one of --input-jsonl or --input-json.")

    args.states_out = args.states_out.expanduser().resolve()
    args.qa_out = args.qa_out.expanduser().resolve()

    return args


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl is not None:
        if not args.input_jsonl.is_file():
            raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
        rows = read_jsonl(args.input_jsonl)
        return [row for row in rows if isinstance(row, dict)]

    if args.input_json is not None:
        if not args.input_json.is_file():
            raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
        with args.input_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            raise ValueError(
                "--input-json must be either a list of objects or an object containing a list under 'data'."
            )
        return [row for row in rows if isinstance(row, dict)]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "`datasets` is required for loading from Hugging Face. Install with: uv pip install datasets"
        ) from exc

    dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    return [row for row in dataset if isinstance(row, dict)]


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [normalize_answer(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "output", "value", "final_answer"):
            if key in value:
                normalized = normalize_answer(value.get(key))
                if normalized:
                    return normalized
    return ""


def clean_question(text: str, strip_wrapping_quotes: bool) -> str:
    cleaned = text.strip()
    if not strip_wrapping_quotes:
        return cleaned
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1].strip()
    return cleaned


def safe_id(raw_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", raw_id.strip())
    return sanitized.strip("_")


def prepare_outputs(args: argparse.Namespace, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    keep_fields = [field.strip() for field in args.keep_fields.split(",") if field.strip()]

    states: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []

    seen_ids: set[str] = set()
    stats = {"input": len(records), "kept": 0, "skipped": 0}

    for idx, record in enumerate(records):
        if args.max_samples is not None and stats["kept"] >= args.max_samples:
            break

        question = clean_question(
            str(record.get(args.question_field, "")),
            strip_wrapping_quotes=args.strip_wrapping_quotes,
        )
        answer = normalize_answer(record.get(args.answer_field))
        if not question or not answer:
            stats["skipped"] += 1
            continue

        raw_id = str(record.get(args.id_field, "")).strip()
        if not raw_id:
            raw_id = f"webshaper_{idx:07d}"

        trajectory_id = safe_id(raw_id)
        if not trajectory_id:
            trajectory_id = f"webshaper_{idx:07d}"
        if trajectory_id in seen_ids:
            trajectory_id = f"{trajectory_id}_{idx:07d}"
        seen_ids.add(trajectory_id)

        state_item: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "t": 0,
            "question": question,
            "memory": "",
            "prev_tool_query": "",
            "prev_tool_result": "",
            "gold_answer": answer,
            "source": "webshaper",
            "source_id": raw_id,
        }

        qa_item: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "question": question,
            "gold_answer": answer,
            "source": "webshaper",
            "source_id": raw_id,
        }

        for field in keep_fields:
            if field in record:
                state_item[field] = record[field]
                qa_item[field] = record[field]

        states.append(state_item)
        qa_rows.append(qa_item)
        stats["kept"] += 1

    return states, qa_rows, stats


def main() -> None:
    args = parse_args()
    records = load_records(args)
    states, qa_rows, stats = prepare_outputs(args, records)

    if not states:
        raise ValueError("No valid records produced. Check field names and input content.")

    write_jsonl(args.states_out, states)
    write_jsonl(args.qa_out, qa_rows)

    print(f"Input records: {stats['input']}")
    print(f"Kept records: {stats['kept']}")
    print(f"Skipped records: {stats['skipped']}")
    print(f"States JSONL: {args.states_out}")
    print(f"QA JSONL: {args.qa_out}")


if __name__ == "__main__":
    main()
