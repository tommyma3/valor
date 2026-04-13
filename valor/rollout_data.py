from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valor.io_utils import read_jsonl, write_jsonl
from valor.rl_utils import extract_final_answer, normalize_text, safe_query_id


@dataclass
class QAPair:
    query_id: str
    query: str
    answer: str


def _log(logger: logging.Logger | None, message: str, *args: Any) -> None:
    if logger is not None:
        logger.info(message, *args)


def normalize_answer_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [normalize_answer_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "output", "value", "final_answer"):
            if key in value:
                normalized = normalize_answer_value(value.get(key))
                if normalized:
                    return normalized
    return ""


def load_browsecomp_qa_pairs(queries_tsv: Path, answers_jsonl: Path) -> dict[str, QAPair]:
    queries: dict[str, str] = {}
    with queries_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            query_id = row[0].strip()
            query = row[1].strip()
            if query_id and query:
                queries[query_id] = query

    answers: dict[str, str] = {}
    for record in read_jsonl(answers_jsonl):
        query_id = str(record.get("query_id", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if query_id and answer:
            answers[query_id] = answer

    qa_pairs: dict[str, QAPair] = {}
    for query_id, query in queries.items():
        answer = answers.get(query_id)
        if answer is None:
            continue
        qa_pairs[query_id] = QAPair(query_id=query_id, query=query, answer=answer)
    return qa_pairs


def load_webshaper_qa_pairs(
    dataset_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    id_field: str | None,
    logger: logging.Logger | None = None,
) -> dict[str, QAPair]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for loading WebShaper data. "
            "Install it with: uv pip install datasets"
        ) from exc

    _log(logger, "Loading dataset from Hugging Face: %s (split=%s)", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    qa_pairs: dict[str, QAPair] = {}
    skipped = 0
    for idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            skipped += 1
            continue

        query = str(row.get(question_field, "")).strip()
        answer = normalize_answer_value(row.get(answer_field))
        if not query or not answer:
            skipped += 1
            continue

        raw_query_id = str(row.get(id_field, "")).strip() if id_field else ""
        if not raw_query_id:
            raw_query_id = f"webshaper_{idx:07d}"

        query_id = safe_query_id(raw_query_id) or f"webshaper_{idx:07d}"
        if query_id in qa_pairs:
            query_id = f"{query_id}_{idx:07d}"

        qa_pairs[query_id] = QAPair(query_id=query_id, query=query, answer=answer)

    _log(
        logger,
        "Loaded WebShaper QA pairs: kept=%d skipped=%d (missing/invalid fields)",
        len(qa_pairs),
        skipped,
    )
    return qa_pairs


def build_transition_dataset_from_rollouts(
    rollout_dir: Path,
    qa_pairs: dict[str, QAPair],
    output_path: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    trace_dir = rollout_dir / "traces"
    if not trace_dir.is_dir():
        raise FileNotFoundError(
            f"Trace directory not found: {trace_dir}. Ensure rollouts were run with --save-traces."
        )

    run_by_query_id: dict[str, dict[str, Any]] = {}
    for run_path in rollout_dir.glob("run_*.json"):
        try:
            with run_path.open("r", encoding="utf-8") as f:
                run_obj = json.load(f)
        except Exception:
            continue

        query_id = str(run_obj.get("query_id", "")).strip()
        if query_id:
            run_by_query_id[query_id] = run_obj

    transitions: list[dict[str, Any]] = []
    num_trajectories = 0
    num_completed = 0

    for trace_path in sorted(trace_dir.glob("trace_*.json")):
        with trace_path.open("r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        query_id = str(trace_obj.get("query_id", "")).strip()
        if not query_id:
            continue

        qa_pair = qa_pairs.get(query_id)
        if qa_pair is None:
            continue

        steps = trace_obj.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

        run_obj = run_by_query_id.get(query_id, {})
        final_answer = extract_final_answer(run_obj)
        if not final_answer:
            for step in reversed(steps):
                answer = str(step.get("answer", "")).strip()
                if answer:
                    final_answer = answer
                    break

        memory = ""
        prev_tool_query = ""
        prev_tool_result = ""
        trajectory_records: list[dict[str, Any]] = []

        for step_idx, step in enumerate(steps):
            report = str(step.get("report", "")).strip()
            tool_name = str(step.get("tool_name", "")).strip()
            tool_params = step.get("tool_params", {})
            raw_tool_call = str(step.get("tool_call", "")).strip()

            if tool_name:
                action_tool_query = json.dumps(
                    {"tool": tool_name, "parameters": tool_params},
                    ensure_ascii=False,
                )
            elif raw_tool_call:
                action_tool_query = raw_tool_call
            else:
                action_tool_query = "<NO_TOOL_CALL>"

            action_think = report if report else "No report."
            action_memory_update = report if report else (memory if memory else "<empty>")

            trajectory_records.append(
                {
                    "trajectory_id": query_id,
                    "t": step_idx,
                    "question": qa_pair.query,
                    "memory": memory,
                    "prev_tool_query": prev_tool_query,
                    "prev_tool_result": prev_tool_result,
                    "action_think": action_think,
                    "action_memory_update": action_memory_update,
                    "action_tool_query": action_tool_query,
                }
            )

            if report:
                memory = report

            tool_output = str(step.get("tool_output", "")).strip()
            tool_error = str(step.get("tool_error", "")).strip()
            observed = tool_output if tool_output else tool_error
            if observed or action_tool_query != "<NO_TOOL_CALL>":
                prev_tool_query = action_tool_query
                prev_tool_result = observed

        if not trajectory_records:
            continue

        trajectory_records[-1]["final_answer"] = final_answer
        trajectory_records[-1]["gold_answer"] = qa_pair.answer

        status = str(run_obj.get("status", trace_obj.get("status", ""))).strip().lower()
        if status == "completed":
            num_completed += 1

        transitions.extend(trajectory_records)
        num_trajectories += 1

    if not transitions:
        raise ValueError(f"No transitions were created from rollout traces in {trace_dir}.")

    write_jsonl(output_path, transitions)
    stats = {
        "num_trajectories": num_trajectories,
        "num_completed": num_completed,
        "num_transitions": len(transitions),
        "output": str(output_path),
    }
    _log(
        logger,
        "Built dataset: trajectories=%d completed=%d transitions=%d",
        num_trajectories,
        num_completed,
        len(transitions),
    )
    return stats


def assign_terminal_binary_rewards(
    records: list[dict[str, Any]],
    *,
    trajectory_field: str = "trajectory_id",
    timestep_field: str = "t",
    final_answer_field: str = "final_answer",
    gold_answer_field: str = "gold_answer",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records):
        key = str(record.get(trajectory_field, idx))
        grouped.setdefault(key, []).append(record)

    for traj in grouped.values():
        if traj and timestep_field in traj[0]:
            traj.sort(key=lambda row: row[timestep_field])
        for record in traj:
            record["reward"] = 0
        final_record = traj[-1]
        pred = normalize_text(str(final_record.get(final_answer_field, "")))
        gold = normalize_text(str(final_record.get(gold_answer_field, "")))
        final_record["reward"] = 1 if pred and pred == gold else 0

    return records
