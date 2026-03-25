"""Collect trajectories using SGLang for RL training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valor.rl_utils import (
    configure_logger,
    extract_final_answer,
    load_json,
    load_query_ids,
    normalize_text,
    read_jsonl,
    safe_query_id,
    save_json,
    utc_now_iso,
    write_jsonl,
    write_query_ids,
    write_queries_tsv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], logger: logging.Logger, cwd: Path | None = None) -> None:
    """Run a command and log its output."""
    logger.info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[cmd] %s", line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


@dataclass
class QAPair:
    query_id: str
    query: str
    answer: str


def load_browsecomp_qa_pairs(queries_tsv: Path, answers_jsonl: Path) -> dict[str, QAPair]:
    """Load QA pairs from BrowseComp format."""
    queries: dict[str, str] = {}
    with queries_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid = row[0].strip()
            query = row[1].strip()
            if qid and query:
                queries[qid] = query

    answers_raw = read_jsonl(answers_jsonl)
    answers: dict[str, str] = {}
    for rec in answers_raw:
        qid = str(rec.get("query_id", "")).strip()
        answer = str(rec.get("answer", "")).strip()
        if qid and answer:
            answers[qid] = answer

    qa_pairs: dict[str, QAPair] = {}
    for qid, query in queries.items():
        answer = answers.get(qid)
        if answer is None:
            continue
        qa_pairs[qid] = QAPair(query_id=qid, query=query, answer=answer)
    return qa_pairs


def normalize_answer_field(value: Any) -> str:
    """Normalize answer field from various formats."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [normalize_answer_field(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "output", "value", "final_answer"):
            if key in value:
                normalized = normalize_answer_field(value.get(key))
                if normalized:
                    return normalized
    return ""


def load_webshaper_qa_pairs(
    dataset_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    id_field: str | None,
    logger: logging.Logger,
) -> dict[str, QAPair]:
    """Load QA pairs from WebShaper dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for loading WebShaper data. "
            "Install it with: uv pip install datasets"
        ) from exc

    logger.info("Loading dataset from Hugging Face: %s (split=%s)", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    qa_pairs: dict[str, QAPair] = {}
    skipped = 0

    for idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            skipped += 1
            continue

        query = str(row.get(question_field, "")).strip()
        answer = normalize_answer_field(row.get(answer_field))
        if not query or not answer:
            skipped += 1
            continue

        raw_query_id = ""
        if id_field:
            raw_query_id = str(row.get(id_field, "")).strip()
        if not raw_query_id:
            raw_query_id = f"webshaper_{idx:07d}"

        query_id = safe_query_id(raw_query_id)
        if not query_id:
            query_id = f"webshaper_{idx:07d}"
        if query_id in qa_pairs:
            query_id = f"{query_id}_{idx:07d}"

        qa_pairs[query_id] = QAPair(query_id=query_id, query=query, answer=answer)

    logger.info(
        "Loaded WebShaper QA pairs: kept=%d skipped=%d (missing/invalid fields)",
        len(qa_pairs),
        skipped,
    )
    return qa_pairs


def build_rollout_command(
    args: argparse.Namespace,
    model_path: str,
    output_dir: Path,
    queries_tsv: Path,
    query_id_file: Path | None,
) -> list[str]:
    """Build command for running rollouts."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_browsecomp_plus.py"),
        "--browsecomp-root",
        str(args.browsecomp_root),
        "--queries",
        str(queries_tsv),
        "--output-dir",
        str(output_dir),
        "--model-path",
        model_path,
        "--searcher-type",
        args.searcher_type,
        "--index-path",
        str(args.index_path),
        "--max-steps",
        str(args.max_steps),
        "--format-retries",
        str(args.format_retries),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--query-template",
        args.query_template,
        "--agent-prompt-template",
        args.agent_prompt_template,
        "--checkpoint-every",
        str(args.checkpoint_every),
"--device",
        args.device,
        "--save-traces",
    ]

    if query_id_file is not None:
        cmd.extend(["--query-id-file", str(query_id_file)])
    if args.max_queries is not None:
        cmd.extend(["--max-queries", str(args.max_queries)])

    if args.sglang_url:
        cmd.extend(["--sglang-url", args.sglang_url])
        if args.sglang_model:
            cmd.extend(["--sglang-model", args.sglang_model])
        if args.sglang_api_key:
            cmd.extend(["--sglang-api-key", args.sglang_api_key])
        if args.sglang_timeout is not None:
            cmd.extend(["--sglang-timeout", str(args.sglang_timeout)])

    if args.device_map is not None:
        cmd.extend(["--device-map", args.device_map])
    if args.dtype is not None:
        cmd.extend(["--dtype", args.dtype])
    if args.max_memory is not None:
        cmd.extend(["--max-memory", args.max_memory])
    if args.offload_folder is not None:
        cmd.extend(["--offload-folder", args.offload_folder])
    if args.offload_state_dict:
        cmd.append("--offload-state-dict")

    if args.search_top_k is not None:
        cmd.extend(["--k", str(args.search_top_k)])
    if args.snippet_max_tokens is not None:
        cmd.extend(["--snippet-max-tokens", str(args.snippet_max_tokens)])
    if args.snippet_tokenizer is not None:
        cmd.extend(["--snippet-tokenizer", args.snippet_tokenizer])
    if args.get_document:
        cmd.append("--get-document")

    if args.searcher_type in {"faiss", "reasonir"}:
        if not args.retriever_model_name:
            raise ValueError("--retriever-model-name is required for faiss/reasonir searcher.")
        cmd.extend(["--model-name", args.retriever_model_name])
        if args.searcher_attn_implementation != "auto":
            cmd.extend(["--attn-implementation", args.searcher_attn_implementation])
        if args.searcher_device != "auto":
            cmd.extend(["--searcher-device", args.searcher_device])
        if args.searcher_cuda_device != 0:
            cmd.extend(["--searcher-cuda-device", str(args.searcher_cuda_device)])
        if args.normalize:
            cmd.append("--normalize")
        if args.pooling is not None:
            cmd.extend(["--pooling", args.pooling])
        if args.searcher_torch_dtype is not None:
            cmd.extend(["--torch-dtype", args.searcher_torch_dtype])
        if args.dataset_name is not None:
            cmd.extend(["--dataset-name", args.dataset_name])
        if args.task_prefix is not None:
            cmd.extend(["--task-prefix", args.task_prefix])
        if args.searcher_max_length is not None:
            cmd.extend(["--max-length", str(args.searcher_max_length)])

    if args.hf_token is not None:
        cmd.extend(["--hf-token", args.hf_token])
    if args.hf_home is not None:
        cmd.extend(["--hf-home", args.hf_home])

    return cmd


def build_dataset_from_rollouts(
    rollout_dir: Path,
    qa_pairs: dict[str, QAPair],
    output_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Convert rollout traces to RL transitions dataset."""
    trace_dir = rollout_dir / "traces"
    if not trace_dir.is_dir():
        raise FileNotFoundError(
            f"Trace directory not found: {trace_dir}. Ensure rollouts were run with --save-traces."
        )

    run_by_qid: dict[str, dict[str, Any]] = {}
    for run_path in rollout_dir.glob("run_*.json"):
        try:
            with run_path.open("r", encoding="utf-8") as f:
                run_obj = json.load(f)
            qid = str(run_obj.get("query_id", "")).strip()
            if qid:
                run_by_qid[qid] = run_obj
        except Exception:
            continue

    transitions: list[dict[str, Any]] = []
    num_trajectories = 0
    num_completed = 0

    for trace_path in sorted(trace_dir.glob("trace_*.json")):
        with trace_path.open("r", encoding="utf-8") as f:
            trace_obj = json.load(f)

        query_id = str(trace_obj.get("query_id", "")).strip()
        if not query_id:
            continue
        qa = qa_pairs.get(query_id)
        if qa is None:
            continue

        steps = trace_obj.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            continue

        run_obj = run_by_qid.get(query_id, {})
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

        for t, step in enumerate(steps):
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
                    "t": t,
                    "question": qa.query,
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

        if len(trajectory_records) == 0:
            continue

        trajectory_records[-1]["final_answer"] = final_answer
        trajectory_records[-1]["gold_answer"] = qa.answer

        status = str(run_obj.get("status", trace_obj.get("status", ""))).strip().lower()
        if status == "completed":
            num_completed += 1

        transitions.extend(trajectory_records)
        num_trajectories += 1

    if len(transitions) == 0:
        raise ValueError(
            f"No transitions were created from rollout traces in {trace_dir}."
        )

    write_jsonl(output_path, transitions)
    stats = {
        "num_trajectories": num_trajectories,
        "num_completed": num_completed,
        "num_transitions": len(transitions),
        "output": str(output_path),
    }
    logger.info(
        "Built dataset: trajectories=%d completed=%d transitions=%d",
        num_trajectories,
        num_completed,
        len(transitions),
    )
    return stats


def compute_rewards(
    trajectories_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Compute binary rewards for trajectories."""
    logger.info("Computing rewards from %s", trajectories_path)
    records = read_jsonl(trajectories_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = record.get("trajectory_id", str(idx))
        grouped[key].append(record)

    for traj in grouped.values():
        if "t" in traj[0]:
            traj.sort(key=lambda r: r["t"])
        for record in traj:
            record["reward"] = 0
        final = traj[-1]
        pred = normalize_text(str(final.get("final_answer", "")))
        gold = normalize_text(str(final.get("gold_answer", "")))
        final["reward"] = 1 if pred and pred == gold else 0

    write_jsonl(output_path, records)
    logger.info("Rewards computed: %s", output_path)


def load_state(path: Path) -> dict[str, Any] | None:
    """Load state from JSON file."""
    return load_json(path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Save state to JSON file."""
    save_json(path, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect trajectories using SGLang for RL training."
    )
    parser.add_argument("--browsecomp-root", type=Path, required=True, help="BrowseComp repository root")
    parser.add_argument("--queries-tsv", type=Path, required=True, help="Queries TSV file")
    parser.add_argument("--answers-jsonl", type=Path, required=True, help="Answers JSONL file")
    parser.add_argument("--query-id-file", type=Path, help="File with query IDs to process (optional, uses all queries if not provided)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    parser.add_argument("--model-path", required=True, help="Model path for rollouts (SGLang or local)")
    parser.add_argument("--searcher-type", choices=["bm25", "faiss", "reasonir", "custom"], required=True)
    parser.add_argument("--index-path", required=True, help="Path to search index")
    parser.add_argument("--retriever-model-name", default=None, help="Retriever model name for faiss/reasonir")
    parser.add_argument(
        "--searcher-attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help="Attention backend for FAISS/ReasonIR retriever loading.",
    )
    parser.add_argument(
        "--searcher-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for FAISS/ReasonIR retriever encoder.",
    )
    parser.add_argument(
        "--searcher-cuda-device",
        type=int,
        default=0,
        help="CUDA device index for retriever when --searcher-device uses CUDA.",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--pooling", default="eos")
    parser.add_argument("--searcher-torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--dataset-name", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--task-prefix", default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:")
    parser.add_argument("--searcher-max-length", type=int, default=8192)

    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--snippet-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--get-document", action="store_true")

    parser.add_argument("--query-template",
        choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
    )
    parser.add_argument("--agent-prompt-template", choices=["browsecomp", "default"], default="browsecomp")

    parser.add_argument("--max-steps", type=int, default=24, help="Max steps per trajectory")
    parser.add_argument("--format-retries", type=int, default=1, help="Retries for invalid step format")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=None, help="Maximum number of queries to process (optional)")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")

    parser.add_argument("--sglang-url", default="", help="SGLang URL for rollouts")
    parser.add_argument("--sglang-model", default="", help="Model name for SGLang")
    parser.add_argument("--sglang-api-key", default=None, help="API key for SGLang")
    parser.add_argument("--sglang-timeout", type=int, default=120, help="SGLang timeout in seconds")

    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-home", default=None)

    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.browsecomp_root = args.browsecomp_root.expanduser().resolve()
    args.queries_tsv = args.queries_tsv.expanduser().resolve()
    args.answers_jsonl = args.answers_jsonl.expanduser().resolve()
    if args.query_id_file:
        args.query_id_file = args.query_id_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.queries_tsv.is_file():
        raise FileNotFoundError(f"Queries TSV not found: {args.queries_tsv}")
    if not args.answers_jsonl.is_file():
        raise FileNotFoundError(f"Answers JSONL not found: {args.answers_jsonl}")
    if args.query_id_file and not args.query_id_file.is_file():
        raise FileNotFoundError(f"Query ID file not found: {args.query_id_file}")

    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_dir, "collect_trajectories")

    if args.hf_token:
        import os
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home

    logger.info("Loading QA pairs from %s", args.queries_tsv)
    qa_pairs = load_browsecomp_qa_pairs(args.queries_tsv, args.answers_jsonl)
    if not qa_pairs:
        raise ValueError("No QA pairs loaded")

    if args.query_id_file:
        query_ids = [qid for qid in load_query_ids(args.query_id_file) if qid in qa_pairs]
        if not query_ids:
            raise ValueError("No valid query IDs found")
    else:
        query_ids = list(qa_pairs.keys())
        logger.info(f"No query ID file provided, using all {len(query_ids)} queries from dataset")

    # Limit number of queries if max_queries is specified
    if args.max_queries is not None:
        original_count = len(query_ids)
        query_ids = query_ids[:args.max_queries]
        logger.info(f"Limited queries from {original_count} to {len(query_ids)} (max_queries={args.max_queries})")

    queries_tsv = args.output_dir / "queries.tsv"
    write_queries_tsv(queries_tsv, query_ids, {qid: qa.query for qid, qa in qa_pairs.items()})

    state_path = args.output_dir / "state.json"
    if args.resume and state_path.is_file():
        state = load_state(state_path)
        if state and state.get("completed", False):
            logger.info("Found completed state, skipping trajectory collection")
            return
        logger.info("Resuming trajectory collection")

    logger.info("Starting trajectory collection for %d queries", len(query_ids))
    logger.info("Model: %s", args.model_path)
    logger.info("SGLang URL: %s", args.sglang_url or "<not using SGLang>")

    rollout_dir = args.output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    rollout_cmd = build_rollout_command(
        args,
        model_path=args.model_path,
        output_dir=rollout_dir,
        queries_tsv=queries_tsv,
        query_id_file=args.query_id_file,
    )
    run_command(rollout_cmd, logger, cwd=REPO_ROOT)

    transitions_path = args.output_dir / "trajectories.jsonl"
    dataset_stats = build_dataset_from_rollouts(
        rollout_dir,
        qa_pairs,
        transitions_path,
        logger,
    )

    rewarded_path = args.output_dir / "trajectories_rewarded.jsonl"
    compute_rewards(transitions_path, rewarded_path, logger)
    logger.info("Computed rewards: %s", rewarded_path)

    state = {
        "completed": True,
        "timestamp": utc_now_iso(),
        "args": vars(args),
        "dataset_stats": dataset_stats,
    }
    save_state(state_path, state)
    logger.info("Trajectory collection completed successfully")


if __name__ == "__main__":
    main()
