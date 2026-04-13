from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
from valor.rl_utils import (
    append_jsonl,
    configure_logger,
    extract_final_answer,
    load_json,
    load_query_ids,
    normalize_text,
    read_jsonl,
    save_json,
    utc_now_iso,
    write_query_ids,
)
from valor.rollout_data import (
    QAPair,
    assign_terminal_binary_rewards,
    build_transition_dataset_from_rollouts,
    load_browsecomp_qa_pairs as shared_load_browsecomp_qa_pairs,
    load_webshaper_qa_pairs as shared_load_webshaper_qa_pairs,
)


def run_command(cmd: list[str], logger: logging.Logger, cwd: Path | None = None) -> None:
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
    return shared_load_browsecomp_qa_pairs(queries_tsv, answers_jsonl)

def load_webshaper_qa_pairs(
    dataset_name: str,
    split: str,
    question_field: str,
    answer_field: str,
    id_field: str | None,
    logger: logging.Logger,
) -> dict[str, QAPair]:
    return shared_load_webshaper_qa_pairs(
        dataset_name=dataset_name,
        split=split,
        question_field=question_field,
        answer_field=answer_field,
        id_field=id_field,
        logger=logger,
    )


def write_queries_tsv(path: Path, query_ids: list[str], qa_pairs: dict[str, QAPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid in query_ids:
            qa = qa_pairs.get(qid)
            if qa is None:
                continue
            writer.writerow([qid, qa.query])

def build_rollout_command(
    args: argparse.Namespace,
    model_path: str,
    output_dir: Path,
    queries_tsv: Path,
    query_id_file: Path,
    save_traces: bool,
) -> list[str]:
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
        args.index_path,
        "--max-steps",
        str(args.rollout_max_steps),
        "--format-retries",
        str(args.rollout_format_retries),
        "--max-new-tokens",
        str(args.rollout_max_new_tokens),
        "--temperature",
        str(args.rollout_temperature),
        "--top-p",
        str(args.rollout_top_p),
        "--query-template",
        args.query_template,
        "--agent-prompt-template",
        args.agent_prompt_template,
        "--checkpoint-every",
        str(args.rollout_checkpoint_every),
        "--device",
        args.rollout_device,
        "--query-id-file",
        str(query_id_file),
    ]

    if args.rollout_vllm_url:
        cmd.extend(["--vllm-url", args.rollout_vllm_url])
        if args.rollout_vllm_model:
            cmd.extend(["--vllm-model", args.rollout_vllm_model])
        if args.rollout_vllm_api_key:
            cmd.extend(["--vllm-api-key", args.rollout_vllm_api_key])
        if args.rollout_vllm_timeout is not None:
            cmd.extend(["--vllm-timeout", str(args.rollout_vllm_timeout)])

    if args.rollout_device_map is not None:
        cmd.extend(["--device-map", args.rollout_device_map])
    if args.rollout_dtype is not None:
        cmd.extend(["--dtype", args.rollout_dtype])
    if args.rollout_max_memory is not None:
        cmd.extend(["--max-memory", args.rollout_max_memory])
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

    if save_traces:
        cmd.append("--save-traces")

    return cmd


def build_dataset_from_rollouts(
    rollout_dir: Path,
    qa_pairs: dict[str, QAPair],
    output_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    return build_transition_dataset_from_rollouts(rollout_dir, qa_pairs, output_path, logger)


def compute_em_score(
    rollout_dir: Path,
    eval_query_ids: list[str],
    qa_pairs: dict[str, QAPair],
) -> dict[str, Any]:
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

    total = len(eval_query_ids)
    correct = 0
    completed = 0

    for qid in eval_query_ids:
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
        if normalize_text(pred) == normalize_text(qa.answer):
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


def maybe_run_official_eval(
    args: argparse.Namespace,
    rollout_dir: Path,
    iter_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    if not args.official_eval:
        return None

    eval_root = iter_dir / "official_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(args.browsecomp_root / "scripts_evaluation" / "evaluate_run.py"),
        "--input_dir",
        str(rollout_dir),
        "--ground_truth",
        str(args.eval_answers_jsonl),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL training loop for VALOR with WebShaper training data and BrowseComp-Plus evaluation."
    )
    parser.add_argument("--browsecomp-root", type=Path, required=True)
    parser.add_argument(
        "--eval-queries-tsv",
        "--queries-tsv",
        dest="eval_queries_tsv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--eval-answers-jsonl",
        "--answers-jsonl",
        dest="eval_answers_jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument(
        "--train-qa-source",
        choices=["webshaper", "browsecomp"],
        default="webshaper",
        help="Source of RL training QA pairs.",
    )
    parser.add_argument("--train-queries-tsv", type=Path, default=None)
    parser.add_argument("--train-answers-jsonl", type=Path, default=None)
    parser.add_argument("--webshaper-dataset", default="Alibaba-NLP/WebShaper")
    parser.add_argument("--webshaper-split", default="main")
    parser.add_argument("--webshaper-question-field", default="question")
    parser.add_argument("--webshaper-answer-field", default="answer")
    parser.add_argument(
        "--webshaper-id-field",
        default="id",
        help="Field used as query id. Set empty string to auto-generate ids.",
    )

    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--policy-init-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--value-init-model", default="Qwen/Qwen3.5-9B")

    parser.add_argument("--searcher-type", choices=["bm25", "faiss", "reasonir", "custom"], required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--retriever-model-name", default=None)
    parser.add_argument(
        "--searcher-attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help="Attention backend for FAISS/ReasonIR retriever loading.",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--pooling", default="eos")
    parser.add_argument("--searcher-torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--dataset-name", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument(
        "--task-prefix",
        default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    )
    parser.add_argument("--searcher-max-length", type=int, default=8192)

    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--snippet-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--get-document", action="store_true")

    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
    )
    parser.add_argument(
        "--agent-prompt-template",
        choices=["browsecomp", "default"],
        default="browsecomp",
    )

    parser.add_argument("--rollout-max-steps", type=int, default=24)
    parser.add_argument("--rollout-format-retries", type=int, default=1)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=768)
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument("--rollout-top-p", type=float, default=0.9)
    parser.add_argument("--rollout-checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--rollout-vllm-url",
        "--rollout-sglang-url",
        dest="rollout_vllm_url",
        default="",
        help="If set, rollout generation uses a vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--rollout-vllm-model",
        "--rollout-sglang-model",
        dest="rollout_vllm_model",
        default="",
        help="Model name sent to vLLM. Defaults to current --model-path each iteration.",
    )
    parser.add_argument(
        "--rollout-vllm-api-key",
        "--rollout-sglang-api-key",
        dest="rollout_vllm_api_key",
        default=None,
        help="Optional API key for vLLM endpoint.",
    )
    parser.add_argument(
        "--rollout-vllm-timeout",
        "--rollout-sglang-timeout",
        dest="rollout_vllm_timeout",
        type=int,
        default=120,
        help="Timeout seconds for each vLLM generation request.",
    )

    parser.add_argument("--rollout-device", default="cuda")
    parser.add_argument("--rollout-device-map", default=None)
    parser.add_argument("--rollout-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--rollout-max-memory", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")

    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--policy-device-map", default=None)
    parser.add_argument("--value-device-map", default=None)

    parser.add_argument("--value-batch-size", type=int, default=2)
    parser.add_argument("--value-epochs", type=int, default=1)
    parser.add_argument("--value-lr", type=float, default=2e-5)
    parser.add_argument("--value-max-length", type=int, default=2048)

    parser.add_argument("--policy-batch-size", type=int, default=1)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=2e-5)
    parser.add_argument("--policy-max-length", type=int, default=2048)
    parser.add_argument("--policy-alpha", type=float, default=1.0)
    parser.add_argument("--policy-indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--policy-torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--train-query-id-file", type=Path, default=None)
    parser.add_argument("--eval-query-id-file", type=Path, default=None)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-eval-queries", type=int, default=None)

    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-home", default=None)

    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--official-eval-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--official-eval-tensor-parallel-size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    args.browsecomp_root = args.browsecomp_root.expanduser().resolve()
    if args.eval_queries_tsv is None:
        args.eval_queries_tsv = args.browsecomp_root / "topics-qrels" / "queries.tsv"
    else:
        args.eval_queries_tsv = args.eval_queries_tsv.expanduser().resolve()

    if args.eval_answers_jsonl is None:
        args.eval_answers_jsonl = args.browsecomp_root / "data" / "browsecomp_plus_decrypted.jsonl"
    else:
        args.eval_answers_jsonl = args.eval_answers_jsonl.expanduser().resolve()

    if args.train_query_id_file is not None:
        args.train_query_id_file = args.train_query_id_file.expanduser().resolve()
    if args.eval_query_id_file is not None:
        args.eval_query_id_file = args.eval_query_id_file.expanduser().resolve()

    if args.train_qa_source == "browsecomp":
        if args.train_queries_tsv is None:
            args.train_queries_tsv = args.eval_queries_tsv
        else:
            args.train_queries_tsv = args.train_queries_tsv.expanduser().resolve()

        if args.train_answers_jsonl is None:
            args.train_answers_jsonl = args.eval_answers_jsonl
        else:
            args.train_answers_jsonl = args.train_answers_jsonl.expanduser().resolve()

    args.output_root = args.output_root.expanduser().resolve()

    return args


def select_query_ids(
    args: argparse.Namespace,
    train_qa_pairs: dict[str, QAPair],
    eval_qa_pairs: dict[str, QAPair],
) -> tuple[list[str], list[str]]:
    train_all_ids = sorted(train_qa_pairs.keys())
    eval_all_ids = sorted(eval_qa_pairs.keys())

    if args.train_query_id_file is not None:
        train_ids = [qid for qid in load_query_ids(args.train_query_id_file) if qid in train_qa_pairs]
    else:
        train_ids = list(train_all_ids)

    if args.eval_query_id_file is not None:
        eval_ids = [qid for qid in load_query_ids(args.eval_query_id_file) if qid in eval_qa_pairs]
    else:
        eval_ids = list(eval_all_ids)

    if args.max_train_queries is not None:
        train_ids = train_ids[: args.max_train_queries]
    if args.max_eval_queries is not None:
        eval_ids = eval_ids[: args.max_eval_queries]

    return train_ids, eval_ids


def load_state(path: Path) -> dict[str, Any] | None:
    return load_json(path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    save_json(path, state)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_root)

    if args.hf_token:
        import os

        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        import os

        os.environ["HF_HOME"] = args.hf_home

    if not args.eval_queries_tsv.is_file():
        raise FileNotFoundError(f"eval queries TSV not found: {args.eval_queries_tsv}")
    if not args.eval_answers_jsonl.is_file():
        raise FileNotFoundError(f"eval answers JSONL not found: {args.eval_answers_jsonl}")

    eval_qa_pairs = load_browsecomp_qa_pairs(args.eval_queries_tsv, args.eval_answers_jsonl)
    if not eval_qa_pairs:
        raise ValueError("No eval QA pairs loaded from BrowseComp-Plus files.")

    if args.train_qa_source == "webshaper":
        train_qa_pairs = load_webshaper_qa_pairs(
            dataset_name=args.webshaper_dataset,
            split=args.webshaper_split,
            question_field=args.webshaper_question_field,
            answer_field=args.webshaper_answer_field,
            id_field=args.webshaper_id_field.strip() or None,
            logger=logger,
        )
    else:
        assert args.train_queries_tsv is not None
        assert args.train_answers_jsonl is not None
        if not args.train_queries_tsv.is_file():
            raise FileNotFoundError(f"train queries TSV not found: {args.train_queries_tsv}")
        if not args.train_answers_jsonl.is_file():
            raise FileNotFoundError(f"train answers JSONL not found: {args.train_answers_jsonl}")
        train_qa_pairs = load_browsecomp_qa_pairs(args.train_queries_tsv, args.train_answers_jsonl)

    if not train_qa_pairs:
        raise ValueError("No training QA pairs loaded.")

    train_ids, eval_ids = select_query_ids(args, train_qa_pairs, eval_qa_pairs)
    if not train_ids:
        raise ValueError("No training query ids selected.")
    if not eval_ids:
        raise ValueError("No eval query ids selected.")

    splits_dir = args.output_root / "splits"
    train_ids_file = splits_dir / "train_ids.txt"
    eval_ids_file = splits_dir / "eval_ids.txt"
    train_queries_tsv = splits_dir / "train_queries.tsv"
    eval_queries_tsv = splits_dir / "eval_queries.tsv"

    write_query_ids(train_ids_file, train_ids)
    write_query_ids(eval_ids_file, eval_ids)
    write_queries_tsv(train_queries_tsv, train_ids, train_qa_pairs)
    write_queries_tsv(eval_queries_tsv, eval_ids, eval_qa_pairs)

    logger.info("Loaded train QA pairs (%s): %d", args.train_qa_source, len(train_qa_pairs))
    logger.info("Loaded eval QA pairs (browsecomp): %d", len(eval_qa_pairs))
    logger.info("Train queries: %d | Eval queries: %d", len(train_ids), len(eval_ids))

    if args.rollout_vllm_url:
        logger.info(
            "vLLM rollout mode enabled | url=%s | model_override=%s",
            args.rollout_vllm_url,
            args.rollout_vllm_model or "<none>",
        )
        if not args.rollout_vllm_model and args.num_iters > 1:
            logger.warning(
                "vLLM model name is not fixed. The script will send each iteration's policy checkpoint path as model id; ensure your vLLM server can resolve/load it."
            )

    state_path = args.output_root / "training_state.json"
    metrics_history_path = args.output_root / "metrics_history.jsonl"

    start_iter = 1
    current_policy_model = args.policy_init_model
    current_value_model = args.value_init_model
    history: list[dict[str, Any]] = []

    if args.resume:
        state = load_state(state_path)
        if state is not None:
            start_iter = int(state.get("current_iteration", 0)) + 1
            current_policy_model = str(state.get("current_policy_model", current_policy_model))
            current_value_model = str(state.get("current_value_model", current_value_model))
            old_history = state.get("history", [])
            if isinstance(old_history, list):
                history = old_history
            logger.info(
                "Resuming from iteration %d with policy=%s value=%s",
                start_iter,
                current_policy_model,
                current_value_model,
            )

    if start_iter == 1:
        baseline_dir = args.output_root / "iter_000_baseline"
        eval_rollout_dir = baseline_dir / "eval_rollouts"
        eval_rollout_dir.mkdir(parents=True, exist_ok=True)

        eval_cmd = build_rollout_command(
            args,
            model_path=current_policy_model,
            output_dir=eval_rollout_dir,
            queries_tsv=eval_queries_tsv,
            query_id_file=eval_ids_file,
            save_traces=False,
        )
        run_command(eval_cmd, logger, cwd=REPO_ROOT)

        em_score = compute_em_score(eval_rollout_dir, eval_ids, eval_qa_pairs)
        official_score = maybe_run_official_eval(args, eval_rollout_dir, baseline_dir, logger)

        baseline_metric = {
            "timestamp": utc_now_iso(),
            "iteration": 0,
            "stage": "baseline_eval",
            "policy_model": current_policy_model,
            "score_em": em_score,
            "score_official": official_score,
        }
        append_jsonl(metrics_history_path, baseline_metric)
        history.append(baseline_metric)
        logger.info(
            "Baseline score | EM accuracy=%.2f%% | completion=%.2f%%",
            em_score["accuracy_em"],
            em_score["completion_rate"],
        )

    for iteration in range(start_iter, args.num_iters + 1):
        iter_dir = args.output_root / f"iter_{iteration:03d}"
        train_rollout_dir = iter_dir / "train_rollouts"
        eval_rollout_dir = iter_dir / "eval_rollouts"
        data_dir = iter_dir / "data"
        ckpt_dir = iter_dir / "checkpoints"

        train_rollout_dir.mkdir(parents=True, exist_ok=True)
        eval_rollout_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Iteration %d/%d ===", iteration, args.num_iters)
        logger.info("Current policy model: %s", current_policy_model)
        logger.info("Current value model: %s", current_value_model)

        # 1) Rollout training trajectories.
        rollout_train_cmd = build_rollout_command(
            args,
            model_path=current_policy_model,
            output_dir=train_rollout_dir,
            queries_tsv=train_queries_tsv,
            query_id_file=train_ids_file,
            save_traces=True,
        )
        run_command(rollout_train_cmd, logger, cwd=REPO_ROOT)

        # 2) Convert rollouts to RL transitions with gold answers.
        transitions_path = data_dir / "trajectories.jsonl"
        dataset_stats = build_dataset_from_rollouts(
            train_rollout_dir,
            train_qa_pairs,
            transitions_path,
            logger,
        )

        # 3) Reward assignment.
        rewarded_path = data_dir / "trajectories_rewarded.jsonl"
        reward_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compute_rewards.py"),
            "--data",
            str(transitions_path),
            "--output",
            str(rewarded_path),
            "--trajectory-field",
            "trajectory_id",
            "--timestep-field",
            "t",
            "--final-answer-field",
            "final_answer",
            "--gold-answer-field",
            "gold_answer",
        ]
        run_command(reward_cmd, logger, cwd=REPO_ROOT)

        # 4) Value training (Qwen3.5-9B backbone / checkpoint continuation).
        value_ckpt = ckpt_dir / "value"
        value_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_value.py"),
            "--data",
            str(rewarded_path),
            "--output",
            str(value_ckpt),
            "--backbone",
            current_value_model,
            "--batch-size",
            str(args.value_batch_size),
            "--epochs",
            str(args.value_epochs),
            "--lr",
            str(args.value_lr),
            "--max-length",
            str(args.value_max_length),
            "--device",
            args.train_device,
            "--seed",
            str(args.seed + iteration),
        ]
        if args.value_device_map is not None:
            value_cmd.extend(["--device-map", args.value_device_map])
        run_command(value_cmd, logger, cwd=REPO_ROOT)

        # 5) Advantage computation.
        adv_path = data_dir / "trajectories_adv.jsonl"
        adv_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compute_advantages.py"),
            "--data",
            str(rewarded_path),
            "--value-model",
            str(value_ckpt),
            "--output",
            str(adv_path),
            "--batch-size",
            str(args.value_batch_size),
            "--max-length",
            str(args.value_max_length),
            "--device",
            args.train_device,
        ]
        run_command(adv_cmd, logger, cwd=REPO_ROOT)

        # 6) Policy training (Qwen3.5-9B backbone / checkpoint continuation).
        policy_ckpt = ckpt_dir / "policy"
        policy_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_policy.py"),
            "--data",
            str(adv_path),
            "--output",
            str(policy_ckpt),
            "--backbone",
            current_policy_model,
            "--batch-size",
            str(args.policy_batch_size),
            "--epochs",
            str(args.policy_epochs),
            "--lr",
            str(args.policy_lr),
            "--max-length",
            str(args.policy_max_length),
            "--device",
            args.train_device,
            "--alpha",
            str(args.policy_alpha),
            "--indicator-drop-prob",
            str(args.policy_indicator_drop_prob),
            "--torch-dtype",
            args.policy_torch_dtype,
            "--seed",
            str(args.seed + iteration),
        ]
        if args.policy_device_map is not None:
            policy_cmd.extend(["--device-map", args.policy_device_map])
        run_command(policy_cmd, logger, cwd=REPO_ROOT)

        # 7) Evaluate updated policy on BrowseComp-Plus query set.
        rollout_eval_cmd = build_rollout_command(
            args,
            model_path=str(policy_ckpt),
            output_dir=eval_rollout_dir,
            queries_tsv=eval_queries_tsv,
            query_id_file=eval_ids_file,
            save_traces=False,
        )
        run_command(rollout_eval_cmd, logger, cwd=REPO_ROOT)

        em_score = compute_em_score(eval_rollout_dir, eval_ids, eval_qa_pairs)
        official_score = maybe_run_official_eval(args, eval_rollout_dir, iter_dir, logger)

        metric = {
            "timestamp": utc_now_iso(),
            "iteration": iteration,
            "policy_model_before": current_policy_model,
            "value_model_before": current_value_model,
            "policy_checkpoint": str(policy_ckpt),
            "value_checkpoint": str(value_ckpt),
            "dataset": dataset_stats,
            "score_em": em_score,
            "score_official": official_score,
        }
        append_jsonl(metrics_history_path, metric)
        history.append(metric)

        logger.info(
            "Iteration %d score | EM accuracy=%.2f%% | completion=%.2f%%",
            iteration,
            em_score["accuracy_em"],
            em_score["completion_rate"],
        )

        current_policy_model = str(policy_ckpt)
        current_value_model = str(value_ckpt)

        state = {
            "updated_at": utc_now_iso(),
            "current_iteration": iteration,
            "current_policy_model": current_policy_model,
            "current_value_model": current_value_model,
            "history": history,
            "config": {
                "policy_init_model": args.policy_init_model,
                "value_init_model": args.value_init_model,
                "num_iters": args.num_iters,
                "train_qa_source": args.train_qa_source,
                "webshaper_dataset": args.webshaper_dataset if args.train_qa_source == "webshaper" else None,
                "webshaper_split": args.webshaper_split if args.train_qa_source == "webshaper" else None,
                "eval_queries_tsv": str(args.eval_queries_tsv),
                "eval_answers_jsonl": str(args.eval_answers_jsonl),
            },
        }
        save_state(state_path, state)

    logger.info("Training loop finished. Final policy checkpoint: %s", current_policy_model)


if __name__ == "__main__":
    main()


