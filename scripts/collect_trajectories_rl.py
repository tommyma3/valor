"""Collect trajectories using vLLM for RL training."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from valor.rl_utils import (
    configure_logger,
    load_json,
    load_query_ids,
    read_jsonl,
    save_json,
    utc_now_iso,
    write_jsonl,
    write_query_ids,
    write_queries_tsv,
)
from valor.rollout_data import (
    QAPair,
    assign_terminal_binary_rewards,
    build_transition_dataset_from_rollouts,
    load_browsecomp_qa_pairs as shared_load_browsecomp_qa_pairs,
    load_webshaper_qa_pairs as shared_load_webshaper_qa_pairs,
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


def load_browsecomp_qa_pairs(queries_tsv: Path, answers_jsonl: Path) -> dict[str, QAPair]:
    """Load QA pairs from BrowseComp format."""
    return shared_load_browsecomp_qa_pairs(queries_tsv, answers_jsonl)

def load_webshaper_qa_pairs(
    dataset_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    id_field: str | None,
    logger: logging.Logger,
) -> dict[str, QAPair]:
    """Load QA pairs from WebShaper dataset."""
    return shared_load_webshaper_qa_pairs(
        dataset_name=dataset_name,
        split=split,
        question_field=question_field,
        answer_field=answer_field,
        id_field=id_field,
        logger=logger,
    )


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
        "--advantage-label",
        args.advantage_label,
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

    if args.vllm_url:
        cmd.extend(["--vllm-url", args.vllm_url])
        if args.vllm_model:
            cmd.extend(["--vllm-model", args.vllm_model])
        if args.vllm_api_key:
            cmd.extend(["--vllm-api-key", args.vllm_api_key])
        if args.vllm_timeout is not None:
            cmd.extend(["--vllm-timeout", str(args.vllm_timeout)])

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
    return build_transition_dataset_from_rollouts(rollout_dir, qa_pairs, output_path, logger)


def compute_rewards(
    trajectories_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Compute binary rewards for trajectories."""
    logger.info("Computing rewards from %s", trajectories_path)
    records = read_jsonl(trajectories_path)
    assign_terminal_binary_rewards(records)
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
        description="Collect trajectories using vLLM for RL training."
    )
    parser.add_argument("--browsecomp-root", type=Path, required=True, help="BrowseComp repository root")
    parser.add_argument("--queries-tsv", type=Path, required=True, help="Queries TSV file")
    parser.add_argument("--answers-jsonl", type=Path, required=True, help="Answers JSONL file")
    parser.add_argument("--query-id-file", type=Path, help="File with query IDs to process (optional, uses all queries if not provided)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    parser.add_argument("--model-path", required=True, help="Model path for rollouts (vLLM or local)")
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
    parser.add_argument(
        "--advantage-label",
        choices=["positive", "negative", "none"],
        default="none",
        help="Condition each BrowseComp state on an advantage indicator.",
    )

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

    parser.add_argument("--vllm-url", "--sglang-url", dest="vllm_url", default="", help="vLLM URL for rollouts")
    parser.add_argument("--vllm-model", "--sglang-model", dest="vllm_model", default="", help="Model name for vLLM")
    parser.add_argument("--vllm-api-key", "--sglang-api-key", dest="vllm_api_key", default=None, help="API key for vLLM")
    parser.add_argument("--vllm-timeout", "--sglang-timeout", dest="vllm_timeout", type=int, default=120, help="vLLM timeout in seconds")

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
    logger.info("vLLM URL: %s", args.vllm_url or "<not using vLLM>")

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
