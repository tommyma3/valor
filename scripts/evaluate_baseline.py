"""Baseline evaluation of a model on BrowseComp-Plus."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from valor.rl_utils import configure_logger, extract_final_answer, normalize_text, read_jsonl

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


def load_browsecomp_qa_pairs(queries_tsv: Path, answers_jsonl: Path) -> dict[str, Any]:
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

    qa_pairs: dict[str, Any] = {}
    for qid, query in queries.items():
        answer = answers.get(qid)
        if answer is None:
            continue
        qa_pairs[qid] = {"query": query, "answer": answer}
    return qa_pairs


def compute_em_score(rollout_dir: Path, query_ids: list[str], qa_pairs: dict[str, Any]) -> dict[str, Any]:
    """Compute exact match score from rollouts."""
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

    total = len(query_ids)
    correct = 0
    completed = 0

    for qid in query_ids:
        qa = qa_pairs.get(qid)
        if qa is None:
            continue

        run_obj = run_by_qid.get(qid)
        if run_obj is None:
            continue

        status = str(run_obj.get("status", "")).strip().lower()
        if status == "completed":
            completed += 1

        pred = extract_final_answer(run_obj)
        if normalize_text(pred) == normalize_text(qa["answer"]):
            correct += 1

    accuracy = (100.0 * correct / total) if total > 0 else 0.0
    completion = (100.0 * completed / total) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "completed": completed,
        "accuracy_em": accuracy,
        "completion_rate": completion,
    }


def build_rollout_command(
    args: argparse.Namespace,
    model_path: str,
    output_dir: Path,
    queries_tsv: Path,
    query_id_file: Path,
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
        "--query-id-file",
        str(query_id_file),
    ]

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


def run_official_eval(
    args: argparse.Namespace,
    rollout_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Run official BrowseComp evaluation if enabled."""
    if not args.official_eval:
        return None

    eval_root = output_dir / "official_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(args.browsecomp_root / "scripts_evaluation" / "evaluate_run.py"),
        "--input_dir",
        str(rollout_dir),
        "--ground_truth",
        str(args.answers_jsonl),
        "--eval_dir",
        str(eval_root),
        "--model",
        args.official_eval_model,
        "--tensor_parallel_size",
        str(args.official_eval_tensor_parallel_size),
    ]

    run_command(cmd, logger, cwd=REPO_ROOT)

    summary_files = sorted(eval_root.rglob("evaluation_summary.json"))
    if not summary_files:
        logger.warning("Official eval enabled but summary file not found under %s", eval_root)
        return None

    summary_path = summary_files[-1]
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return {
        "summary_path": str(summary_path),
        "accuracy_percent": summary.get("Accuracy (%)"),
        "recall_percent": summary.get("Recall (%)"),
        "calibration_error_percent": summary.get("Calibration Error (%)"),
    }


def load_query_ids(path: Path) -> list[str]:
    """Load query IDs from text file."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                ids.append(qid)
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline evaluation of a model on BrowseComp-Plus."
    )
    parser.add_argument("--browsecomp-root", type=Path, required=True)
    parser.add_argument("--queries-tsv", type=Path, required=True)
    parser.add_argument("--answers-jsonl", type=Path, required=True)
    parser.add_argument("--query-id-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="Model to evaluate")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--searcher-type", choices=["bm25", "faiss", "reasonir", "custom"], required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--retriever-model-name", default=None)
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

    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint-every", type=int, default=50)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")

    parser.add_argument("--sglang-url", default="", help="SGLang URL")
    parser.add_argument("--sglang-model", default="", help="Model name for SGLang")
    parser.add_argument("--sglang-api-key", default=None)
    parser.add_argument("--sglang-timeout", type=int, default=120)

    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--official-eval-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--official-eval-tensor-parallel-size", type=int, default=1)

    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-home", default=None)

    args = parser.parse_args()
    args.browsecomp_root = args.browsecomp_root.expanduser().resolve()
    args.queries_tsv = args.queries_tsv.expanduser().resolve()
    args.answers_jsonl = args.answers_jsonl.expanduser().resolve()
    args.query_id_file = args.query_id_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.queries_tsv.is_file():
        raise FileNotFoundError(f"Queries TSV not found: {args.queries_tsv}")
    if not args.answers_jsonl.is_file():
        raise FileNotFoundError(f"Answers JSONL not found: {args.answers_jsonl}")
    if not args.query_id_file.is_file():
        raise FileNotFoundError(f"Query ID file not found: {args.query_id_file}")

    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_dir, "evaluate_baseline")

    if args.hf_token:
        import os
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home

    logger.info("Loading QA pairs and query IDs")
    qa_pairs = load_browsecomp_qa_pairs(args.queries_tsv, args.answers_jsonl)
    query_ids = load_query_ids(args.query_id_file)

    # Run rollouts
    rollout_dir = args.output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    # Write subset of queries for evaluation
    queries_subset = {}
    for qid in query_ids:
        if qid in qa_pairs:
            queries_subset[qid] = qa_pairs[qid]["query"]

    queries_tsv = args.output_dir / "queries.tsv"
    with open(queries_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid, query in queries_subset.items():
            writer.writerow([qid, query])

    logger.info("Running rollouts with model: %s", args.model_path)
    rollout_cmd = build_rollout_command(
        args,
        model_path=args.model_path,
        output_dir=rollout_dir,
        queries_tsv=queries_tsv,
        query_id_file=args.query_id_file,
    )
    run_command(rollout_cmd, logger, cwd=REPO_ROOT)

    # Compute EM score
    em_score = compute_em_score(rollout_dir, query_ids, qa_pairs)
    logger.info(
        "EM Score | accuracy=%.2f%% | completion=%.2f%%",
        em_score["accuracy_em"],
        em_score["completion_rate"],
    )

    # Run official evaluation if enabled
    official_score = run_official_eval(args, rollout_dir, args.output_dir, logger)
    if official_score:
        logger.info(
            "Official Score | accuracy=%s%% | recall=%s%%",
            official_score.get("accuracy_percent"),
            official_score.get("recall_percent"),
        )

    # Save results
    results = {
        "model_path": args.model_path,
        "em_score": em_score,
        "official_score": official_score,
    }
    results_path = args.output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Evaluation completed. Results saved to: %s", results_path)


if __name__ == "__main__":
    main()
