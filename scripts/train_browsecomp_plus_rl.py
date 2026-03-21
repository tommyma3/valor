from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def safe_query_id(query_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in query_id)


def configure_logger(output_root: Path) -> logging.Logger:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"train_loop_{stamp}.log"

    logger = logging.getLogger("valor.browsecomp_rl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Training loop logs: %s", log_path)
    return logger


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object JSON on line {line_num} in {path}")
            records.append(obj)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


@dataclass
class QAPair:
    query_id: str
    query: str
    answer: str


def load_qa_pairs(queries_tsv: Path, answers_jsonl: Path) -> dict[str, QAPair]:
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


def load_query_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                ids.append(qid)
    return ids


def write_query_ids(path: Path, query_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(qid)
            f.write("\n")


def extract_final_answer(run_record: dict[str, Any]) -> str:
    result = run_record.get("result", [])
    if not isinstance(result, list):
        return ""
    for item in reversed(result):
        if isinstance(item, dict) and item.get("type") == "output_text":
            return str(item.get("output", "")).strip()
    return ""


def build_rollout_command(
    args: argparse.Namespace,
    model_path: str,
    output_dir: Path,
    query_id_file: Path,
    save_traces: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_browsecomp_plus.py"),
        "--browsecomp-root",
        str(args.browsecomp_root),
        "--queries",
        str(args.queries_tsv),
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

    if args.rollout_sglang_url:
        cmd.extend(["--sglang-url", args.rollout_sglang_url])
        if args.rollout_sglang_model:
            cmd.extend(["--sglang-model", args.rollout_sglang_model])
        if args.rollout_sglang_api_key:
            cmd.extend(["--sglang-api-key", args.rollout_sglang_api_key])
        if args.rollout_sglang_timeout is not None:
            cmd.extend(["--sglang-timeout", str(args.rollout_sglang_timeout)])

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
            # Fallback to trace answer on the last step containing one.
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
        "Built RL dataset: trajectories=%d completed=%d transitions=%d",
        num_trajectories,
        num_completed,
        len(transitions),
    )
    return stats


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL training loop for VALOR on BrowseComp-Plus QA pairs."
    )
    parser.add_argument("--browsecomp-root", type=Path, required=True)
    parser.add_argument("--queries-tsv", type=Path, default=None)
    parser.add_argument("--answers-jsonl", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)

    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--policy-init-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--value-init-model", default="Qwen/Qwen3.5-9B")

    parser.add_argument("--searcher-type", choices=["bm25", "faiss", "reasonir", "custom"], required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--retriever-model-name", default=None)
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
    parser.add_argument("--rollout-max-new-tokens", type=int, default=768)
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument("--rollout-top-p", type=float, default=0.9)
    parser.add_argument("--rollout-checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--rollout-sglang-url",
        default="",
        help="If set, rollout generation uses SGLang OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--rollout-sglang-model",
        default="",
        help="Model name sent to SGLang. Defaults to current --model-path each iteration.",
    )
    parser.add_argument(
        "--rollout-sglang-api-key",
        default=None,
        help="Optional API key for SGLang endpoint.",
    )
    parser.add_argument(
        "--rollout-sglang-timeout",
        type=int,
        default=120,
        help="Timeout seconds for each SGLang generation request.",
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
    if args.queries_tsv is None:
        args.queries_tsv = args.browsecomp_root / "topics-qrels" / "queries.tsv"
    else:
        args.queries_tsv = args.queries_tsv.expanduser().resolve()

    if args.answers_jsonl is None:
        args.answers_jsonl = args.browsecomp_root / "data" / "browsecomp_plus_decrypted.jsonl"
    else:
        args.answers_jsonl = args.answers_jsonl.expanduser().resolve()

    args.output_root = args.output_root.expanduser().resolve()

    return args


def select_query_ids(args: argparse.Namespace, qa_pairs: dict[str, QAPair]) -> tuple[list[str], list[str]]:
    all_ids = sorted(qa_pairs.keys())

    if args.train_query_id_file is not None:
        train_ids = [qid for qid in load_query_ids(args.train_query_id_file) if qid in qa_pairs]
    else:
        train_ids = list(all_ids)

    if args.eval_query_id_file is not None:
        eval_ids = [qid for qid in load_query_ids(args.eval_query_id_file) if qid in qa_pairs]
    else:
        eval_ids = list(all_ids)

    if args.max_train_queries is not None:
        train_ids = train_ids[: args.max_train_queries]
    if args.max_eval_queries is not None:
        eval_ids = eval_ids[: args.max_eval_queries]

    return train_ids, eval_ids


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_root)

    if not args.queries_tsv.is_file():
        raise FileNotFoundError(f"queries TSV not found: {args.queries_tsv}")
    if not args.answers_jsonl.is_file():
        raise FileNotFoundError(f"answers JSONL not found: {args.answers_jsonl}")

    qa_pairs = load_qa_pairs(args.queries_tsv, args.answers_jsonl)
    if not qa_pairs:
        raise ValueError("No QA pairs loaded from BrowseComp-Plus files.")

    train_ids, eval_ids = select_query_ids(args, qa_pairs)
    if not train_ids:
        raise ValueError("No training query ids selected.")
    if not eval_ids:
        raise ValueError("No eval query ids selected.")

    splits_dir = args.output_root / "splits"
    train_ids_file = splits_dir / "train_ids.txt"
    eval_ids_file = splits_dir / "eval_ids.txt"
    write_query_ids(train_ids_file, train_ids)
    write_query_ids(eval_ids_file, eval_ids)

    logger.info("Loaded QA pairs: %d", len(qa_pairs))
    logger.info("Train queries: %d | Eval queries: %d", len(train_ids), len(eval_ids))

    if args.rollout_sglang_url:
        logger.info(
            "SGLang rollout mode enabled | url=%s | model_override=%s",
            args.rollout_sglang_url,
            args.rollout_sglang_model or "<none>",
        )
        if not args.rollout_sglang_model and args.num_iters > 1:
            logger.warning(
                "SGLang model name is not fixed. The script will send each iteration's policy checkpoint path as model id; ensure your SGLang server can resolve/load it."
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
            query_id_file=eval_ids_file,
            save_traces=False,
        )
        run_command(eval_cmd, logger, cwd=REPO_ROOT)

        em_score = compute_em_score(eval_rollout_dir, eval_ids, qa_pairs)
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
            query_id_file=train_ids_file,
            save_traces=True,
        )
        run_command(rollout_train_cmd, logger, cwd=REPO_ROOT)

        # 2) Convert rollouts to RL transitions with gold answers.
        transitions_path = data_dir / "trajectories.jsonl"
        dataset_stats = build_dataset_from_rollouts(
            train_rollout_dir,
            qa_pairs,
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

        # 6) Policy training (Qwen3.5-35B-A3B backbone / checkpoint continuation).
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
            query_id_file=eval_ids_file,
            save_traces=False,
        )
        run_command(rollout_eval_cmd, logger, cwd=REPO_ROOT)

        em_score = compute_em_score(eval_rollout_dir, eval_ids, qa_pairs)
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
            },
        }
        save_state(state_path, state)

    logger.info("Training loop finished. Final policy checkpoint: %s", current_policy_model)


if __name__ == "__main__":
    main()
