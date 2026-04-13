
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import pkgutil
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import initial_instruction_prompt, instruction_prompt, last_instruction_prompt
from valor.system_prompts import build_tools_prompt


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def safe_query_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


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


def run_command(cmd: list[str], logger: logging.Logger, env: dict[str, str] | None = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[cmd] %s", line.rstrip("\n"))

    code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with code {code}: {' '.join(cmd)}")


def configure_logger(output_root: Path) -> logging.Logger:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"webshaper_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("valor.webshaper_rl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_path)
    return logger


@dataclass
class QAPair:
    query_id: str
    query: str
    answer: str


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [normalize_answer(v) for v in value]
        return " ".join(p for p in parts if p).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "output", "value", "final_answer"):
            if key in value:
                val = normalize_answer(value.get(key))
                if val:
                    return val
    return ""


def clean_question(text: str, strip_quotes: bool) -> str:
    q = text.strip()
    if strip_quotes and len(q) >= 2 and q[0] == q[-1] and q[0] in {'"', "'"}:
        return q[1:-1].strip()
    return q


def load_webshaper_qa_pairs(args: argparse.Namespace, logger: logging.Logger) -> dict[str, QAPair]:
    if args.input_jsonl is not None:
        rows = read_jsonl(args.input_jsonl)
        logger.info("Loaded %d records from %s", len(rows), args.input_jsonl)
    else:
        from datasets import load_dataset

        ds = load_dataset(args.webshaper_dataset, split=args.webshaper_split)
        rows = [row for row in ds if isinstance(row, dict)]
        logger.info("Loaded %d records from HF dataset %s[%s]", len(rows), args.webshaper_dataset, args.webshaper_split)

    qa: dict[str, QAPair] = {}
    skipped = 0
    for idx, row in enumerate(rows):
        q = clean_question(str(row.get(args.question_field, "")), args.strip_wrapping_quotes)
        a = normalize_answer(row.get(args.answer_field))
        if not q or not a:
            skipped += 1
            continue

        raw_id = str(row.get(args.id_field, "")).strip() or f"webshaper_{idx:07d}"
        qid = safe_query_id(raw_id) or f"webshaper_{idx:07d}"
        if qid in qa:
            qid = f"{qid}_{idx:07d}"

        qa[qid] = QAPair(query_id=qid, query=q, answer=a)

    logger.info("Prepared QA pairs: kept=%d skipped=%d", len(qa), skipped)
    return qa

def load_query_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def write_query_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in ids:
            f.write(qid)
            f.write("\n")


def select_splits(args: argparse.Namespace, qa_pairs: dict[str, QAPair]) -> tuple[list[str], list[str]]:
    all_ids = sorted(qa_pairs.keys())

    if args.train_query_id_file is not None:
        train_ids = [x for x in load_query_ids(args.train_query_id_file) if x in qa_pairs]
    else:
        train_ids = list(all_ids)

    if args.eval_query_id_file is not None:
        eval_ids = [x for x in load_query_ids(args.eval_query_id_file) if x in qa_pairs]
    else:
        rng = random.Random(args.seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        eval_size = int(len(shuffled) * args.eval_ratio)
        if len(shuffled) > 1 and eval_size == 0:
            eval_size = 1
        eval_ids = shuffled[:eval_size]
        eval_set = set(eval_ids)
        train_ids = [x for x in shuffled if x not in eval_set]

    if args.max_train_queries is not None:
        train_ids = train_ids[: args.max_train_queries]
    if args.max_eval_queries is not None:
        eval_ids = eval_ids[: args.max_eval_queries]

    return train_ids, eval_ids


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return "" if not m else m.group(1).strip()


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def normalize_tool_call(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = payload.get("tool") or payload.get("name") or payload.get("tool_name")
    params = (
        payload.get("parameters")
        or payload.get("params")
        or payload.get("arguments")
        or payload.get("args")
        or payload.get("input")
    )

    if name is None and len(payload) == 1:
        name = next(iter(payload.keys()))
        params = payload[name]

    if name is None:
        raise ValueError("Tool name missing in tool_call JSON")
    if params is None:
        params = {k: v for k, v in payload.items() if k not in {"tool", "name", "tool_name"}}
    if not isinstance(params, dict):
        raise ValueError("Tool parameters must be an object")

    return str(name), params


def sglang_chat(base_url: str, model: str, prompt: str, temperature: float, top_p: float, max_tokens: int, api_key: str, timeout: int) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"]).strip()


def load_tool_registry() -> dict[str, Any]:
    import tools as tools_pkg

    registry: dict[str, Any] = {}
    for attr in dir(tools_pkg):
        obj = getattr(tools_pkg, attr)
        if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "parameters"):
            registry[str(getattr(obj, "name"))] = obj

    if hasattr(tools_pkg, "__path__"):
        for mod_info in pkgutil.iter_modules(tools_pkg.__path__):
            try:
                module = importlib.import_module(f"{tools_pkg.__name__}.{mod_info.name}")
            except Exception:
                continue
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "parameters"):
                    registry[str(getattr(obj, "name"))] = obj

    return {k.lower(): v for k, v in registry.items()}


class ToolRuntime:
    def __init__(self, registry: dict[str, Any], max_output_chars: int):
        self.registry = registry
        self.instances: dict[str, Any] = {}
        self.init_errors: dict[str, str] = {}
        self.max_output_chars = max_output_chars

    def _norm_output(self, output: Any) -> str:
        if isinstance(output, tuple):
            output = output[0] if output else ""
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False)
            except Exception:
                text = str(output)
        if self.max_output_chars > 0 and len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n...[truncated]"
        return text

    def execute(self, name: str, params: dict[str, Any]) -> tuple[str, str]:
        key = name.strip().lower()
        if key not in self.registry:
            raise ValueError(f"Unknown tool '{name}'. Available: {', '.join(sorted(self.registry.keys()))}")

        if key not in self.instances and key not in self.init_errors:
            try:
                self.instances[key] = self.registry[key]()
            except Exception as exc:
                self.init_errors[key] = str(exc)

        if key in self.init_errors:
            raise RuntimeError(f"Failed to initialize tool '{key}': {self.init_errors[key]}")

        output = self.instances[key].call(params)
        return key, self._norm_output(output)

def rollout(args: argparse.Namespace, model_name: str, output_dir: Path, query_ids: list[str], qa_pairs: dict[str, QAPair], save_traces: bool, logger: logging.Logger) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    if save_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    tools_prompt = build_tools_prompt()
    registry = load_tool_registry()
    completed = 0

    for i, qid in enumerate(query_ids, start=1):
        qa = qa_pairs[qid]
        runtime = ToolRuntime(registry, args.max_tool_output_chars)

        steps: list[dict[str, Any]] = []
        result: list[dict[str, Any]] = []
        last_report = ""
        last_action = ""
        last_obs = ""
        final_answer = ""
        status = "incomplete"

        logger.info("Rollout %d/%d | %s", i, len(query_ids), qid)

        for step in range(1, args.rollout_max_steps + 1):
            if step == 1:
                prompt = initial_instruction_prompt.format(
                    date_to_use=args.rollout_date or date.today().isoformat(),
                    question=qa.query,
                    tools=tools_prompt,
                )
            else:
                prompt = instruction_prompt.format(
                    date_to_use=args.rollout_date or date.today().isoformat(),
                    question=qa.query,
                    tools=tools_prompt,
                    report=last_report,
                    action=last_action,
                    observation=last_obs,
                )

            completion = sglang_chat(
                args.sglang_url,
                model_name,
                prompt,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                max_tokens=args.rollout_max_new_tokens,
                api_key=args.sglang_api_key or "",
                timeout=args.sglang_timeout,
            )

            report = extract_tag(completion, "report")
            answer = extract_tag(completion, "answer")
            tool_call = extract_tag(completion, "tool_call")
            trace_item: dict[str, Any] = {"step": step, "report": report, "answer": answer, "tool_call": tool_call, "completion": completion}

            if report:
                result.append({"type": "reasoning", "tool_name": None, "arguments": None, "output": report})
                last_report = report

            if answer:
                final_answer = answer.strip()
                result.append({"type": "output_text", "tool_name": None, "arguments": None, "output": final_answer})
                status = "completed"
                steps.append(trace_item)
                break

            if not tool_call:
                last_action = ""
                last_obs = "[Tool Error] Missing <tool_call> tag."
                trace_item["tool_error"] = last_obs
                steps.append(trace_item)
                continue

            payload = safe_json_loads(tool_call)
            if payload is None:
                last_action = tool_call
                last_obs = "[Tool Error] Could not parse tool_call JSON."
                trace_item["tool_error"] = last_obs
                steps.append(trace_item)
                continue

            try:
                tool_name, tool_params = normalize_tool_call(payload)
                canonical, tool_out = runtime.execute(tool_name, tool_params)
                last_action = json.dumps({"tool": canonical, "parameters": tool_params}, ensure_ascii=False)
                last_obs = tool_out
                trace_item["tool_name"] = canonical
                trace_item["tool_params"] = tool_params
                trace_item["tool_output"] = tool_out
                result.append({"type": "tool_call", "tool_name": canonical, "arguments": tool_params, "output": tool_out})
            except Exception as exc:
                last_action = tool_call
                last_obs = f"[Tool Error] {exc}"
                trace_item["tool_error"] = last_obs
                result.append({"type": "tool_call", "tool_name": "tool_error", "arguments": payload, "output": last_obs})

            steps.append(trace_item)

        if args.force_final_answer and not final_answer:
            prompt = last_instruction_prompt.format(
                date_to_use=args.rollout_date or date.today().isoformat(),
                question=qa.query,
                report=last_report,
                action=last_action,
                observation=last_obs,
            )
            completion = sglang_chat(
                args.sglang_url,
                model_name,
                prompt,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                max_tokens=args.rollout_max_new_tokens,
                api_key=args.sglang_api_key or "",
                timeout=args.sglang_timeout,
            )
            report = extract_tag(completion, "report")
            answer = extract_tag(completion, "answer")
            if report:
                result.append({"type": "reasoning", "tool_name": None, "arguments": None, "output": report})
            if answer:
                final_answer = answer.strip()
                status = "completed"
                result.append({"type": "output_text", "tool_name": None, "arguments": None, "output": final_answer})
            steps.append({"step": args.rollout_max_steps + 1, "report": report, "answer": answer, "tool_call": "", "completion": completion, "forced_final": True})

        run_record = {
            "metadata": {"generator": "train_webshaper_rl", "generated_at": utc_now_iso(), "model": model_name},
            "query_id": qid,
            "query": qa.query,
            "result": result,
            "status": status,
        }
        with (output_dir / f"run_{safe_query_id(qid)}.json").open("w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        if save_traces:
            with (trace_dir / f"trace_{safe_query_id(qid)}.json").open("w", encoding="utf-8") as f:
                json.dump({"query_id": qid, "status": status, "final_answer": final_answer, "steps": steps}, f, ensure_ascii=False, indent=2)

        if status == "completed":
            completed += 1

    return {"total": len(query_ids), "completed": completed, "failed": len(query_ids) - completed}


def extract_final_answer(run_record: dict[str, Any]) -> str:
    result = run_record.get("result", [])
    if not isinstance(result, list):
        return ""
    for item in reversed(result):
        if isinstance(item, dict) and item.get("type") == "output_text":
            return str(item.get("output", "")).strip()
    return ""


def build_transitions(rollout_dir: Path, qa_pairs: dict[str, QAPair], output_path: Path) -> dict[str, Any]:
    run_by_qid: dict[str, dict[str, Any]] = {}
    for path in rollout_dir.glob("run_*.json"):
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        qid = str(obj.get("query_id", "")).strip()
        if qid:
            run_by_qid[qid] = obj

    transitions: list[dict[str, Any]] = []
    completed = 0
    trajectories = 0
    for trace_path in sorted((rollout_dir / "traces").glob("trace_*.json")):
        with trace_path.open("r", encoding="utf-8") as f:
            trace = json.load(f)
        qid = str(trace.get("query_id", "")).strip()
        if not qid or qid not in qa_pairs:
            continue
        qa = qa_pairs[qid]
        steps = trace.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue

        run_obj = run_by_qid.get(qid, {})
        final_answer = extract_final_answer(run_obj) or str(trace.get("final_answer", "")).strip()

        memory = ""
        prev_q = ""
        prev_r = ""
        row_list: list[dict[str, Any]] = []
        for t, step in enumerate(steps):
            report = str(step.get("report", "")).strip()
            tool_name = str(step.get("tool_name", "")).strip()
            tool_params = step.get("tool_params", {})
            raw_tool_call = str(step.get("tool_call", "")).strip()
            if tool_name:
                tool_query = json.dumps({"tool": tool_name, "parameters": tool_params}, ensure_ascii=False)
            elif raw_tool_call:
                tool_query = raw_tool_call
            else:
                tool_query = "<NO_TOOL_CALL>"

            row_list.append({
                "trajectory_id": qid,
                "t": t,
                "question": qa.query,
                "memory": memory,
                "prev_tool_query": prev_q,
                "prev_tool_result": prev_r,
                "action_think": report or "No report.",
                "action_memory_update": report or (memory if memory else "<empty>"),
                "action_tool_query": tool_query,
            })

            if report:
                memory = report
            observed = str(step.get("tool_output", "")).strip() or str(step.get("tool_error", "")).strip()
            if observed or tool_query != "<NO_TOOL_CALL>":
                prev_q = tool_query
                prev_r = observed

        row_list[-1]["final_answer"] = final_answer
        row_list[-1]["gold_answer"] = qa.answer

        if str(run_obj.get("status", trace.get("status", ""))).strip().lower() == "completed":
            completed += 1
        transitions.extend(row_list)
        trajectories += 1

    if not transitions:
        raise ValueError("No transitions built from rollouts.")
    write_jsonl(output_path, transitions)
    return {"num_trajectories": trajectories, "num_completed": completed, "num_transitions": len(transitions)}

def compute_em(rollout_dir: Path, eval_ids: list[str], qa_pairs: dict[str, QAPair]) -> dict[str, Any]:
    run_by_qid: dict[str, dict[str, Any]] = {}
    for path in rollout_dir.glob("run_*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            qid = str(obj.get("query_id", "")).strip()
            if qid:
                run_by_qid[qid] = obj
        except Exception:
            continue

    total = len(eval_ids)
    correct = 0
    completed = 0
    for qid in eval_ids:
        qa = qa_pairs.get(qid)
        run = run_by_qid.get(qid)
        if qa is None or run is None:
            continue
        if str(run.get("status", "")).strip().lower() == "completed":
            completed += 1
        if normalize_text(extract_final_answer(run)) == normalize_text(qa.answer):
            correct += 1

    return {
        "total": total,
        "correct": correct,
        "completed": completed,
        "accuracy_em": (100.0 * correct / total) if total else 0.0,
        "completion_rate": (100.0 * completed / total) if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VALOR with value+policy RL on WebShaper (no BrowseComp dependencies).")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--input-jsonl", type=Path, default=None)
    p.add_argument("--webshaper-dataset", default="Alibaba-NLP/WebShaper")
    p.add_argument("--webshaper-split", default="main")
    p.add_argument("--question-field", default="question")
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--id-field", default="id")
    p.add_argument("--strip-wrapping-quotes", action="store_true")

    p.add_argument("--train-query-id-file", type=Path, default=None)
    p.add_argument("--eval-query-id-file", type=Path, default=None)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--max-train-queries", type=int, default=None)
    p.add_argument("--max-eval-queries", type=int, default=None)

    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--policy-init-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--value-init-model", default="Qwen/Qwen3.5-9B")

    p.add_argument("--sglang-url", required=True)
    p.add_argument("--sglang-api-key", default=None)
    p.add_argument("--rollout-sglang-model", default="")
    p.add_argument("--sglang-timeout", type=int, default=120)
    p.add_argument("--rollout-max-steps", type=int, default=8)
    p.add_argument("--rollout-max-new-tokens", type=int, default=768)
    p.add_argument("--rollout-temperature", type=float, default=0.0)
    p.add_argument("--rollout-top-p", type=float, default=0.9)
    p.add_argument("--rollout-date", default=None)
    p.add_argument("--no-force-final-answer", action="store_true", help="Disable forced final-answer generation when no <answer> is produced in rollout steps.")
    p.add_argument("--max-tool-output-chars", type=int, default=12000)

    p.add_argument("--train-device", default="cuda")
    p.add_argument("--value-batch-size", type=int, default=2)
    p.add_argument("--value-epochs", type=int, default=1)
    p.add_argument("--value-lr", type=float, default=2e-5)
    p.add_argument("--value-max-length", type=int, default=2048)
    p.add_argument("--policy-batch-size", type=int, default=1)
    p.add_argument("--policy-epochs", type=int, default=1)
    p.add_argument("--policy-lr", type=float, default=2e-5)
    p.add_argument("--policy-max-length", type=int, default=2048)
    p.add_argument("--policy-alpha", type=float, default=1.0)
    p.add_argument("--policy-indicator-drop-prob", type=float, default=0.1)
    p.add_argument("--policy-device-map", default=None)
    p.add_argument("--policy-torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-home", default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    args.force_final_answer = not args.no_force_final_answer
    if args.input_jsonl is not None:
        args.input_jsonl = args.input_jsonl.expanduser().resolve()
    if args.train_query_id_file is not None:
        args.train_query_id_file = args.train_query_id_file.expanduser().resolve()
    if args.eval_query_id_file is not None:
        args.eval_query_id_file = args.eval_query_id_file.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_root)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    qa_pairs = load_webshaper_qa_pairs(args, logger)
    train_ids, eval_ids = select_splits(args, qa_pairs)
    if not train_ids or not eval_ids:
        raise ValueError("Train/eval split is empty.")

    split_dir = args.output_root / "splits"
    write_query_ids(split_dir / "train_ids.txt", train_ids)
    write_query_ids(split_dir / "eval_ids.txt", eval_ids)

    logger.info("Train queries=%d Eval queries=%d", len(train_ids), len(eval_ids))

    current_policy = args.policy_init_model
    current_value = args.value_init_model
    metrics_path = args.output_root / "metrics_history.jsonl"
    env = os.environ.copy()

    base_model_for_rollout = args.rollout_sglang_model.strip()

    for it in range(1, args.num_iters + 1):
        iter_dir = args.output_root / f"iter_{it:03d}"
        train_rollout_dir = iter_dir / "train_rollouts"
        eval_rollout_dir = iter_dir / "eval_rollouts"
        data_dir = iter_dir / "data"
        ckpt_dir = iter_dir / "checkpoints"
        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        rollout_model = base_model_for_rollout or current_policy
        logger.info("=== Iteration %d/%d | rollout model: %s ===", it, args.num_iters, rollout_model)

        train_rollout_stats = rollout(args, rollout_model, train_rollout_dir, train_ids, qa_pairs, True, logger)

        transitions = data_dir / "trajectories.jsonl"
        dataset_stats = build_transitions(train_rollout_dir, qa_pairs, transitions)

        rewarded = data_dir / "trajectories_rewarded.jsonl"
        run_command([
            sys.executable, str(REPO_ROOT / "scripts" / "compute_rewards.py"),
            "--data", str(transitions), "--output", str(rewarded),
            "--trajectory-field", "trajectory_id", "--timestep-field", "t",
            "--final-answer-field", "final_answer", "--gold-answer-field", "gold_answer",
        ], logger, env)

        value_ckpt = ckpt_dir / "value"
        run_command([
            sys.executable, str(REPO_ROOT / "scripts" / "train_value.py"),
            "--data", str(rewarded), "--output", str(value_ckpt), "--backbone", current_value,
            "--batch-size", str(args.value_batch_size), "--epochs", str(args.value_epochs), "--lr", str(args.value_lr),
            "--max-length", str(args.value_max_length), "--device", args.train_device,
        ], logger, env)

        adv = data_dir / "trajectories_adv.jsonl"
        run_command([
            sys.executable, str(REPO_ROOT / "scripts" / "compute_advantages.py"),
            "--data", str(rewarded), "--value-model", str(value_ckpt), "--output", str(adv),
            "--batch-size", str(args.value_batch_size), "--max-length", str(args.value_max_length), "--device", args.train_device,
        ], logger, env)

        policy_ckpt = ckpt_dir / "policy"
        policy_cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "train_policy.py"),
            "--data", str(adv), "--output", str(policy_ckpt), "--backbone", current_policy,
            "--batch-size", str(args.policy_batch_size), "--epochs", str(args.policy_epochs), "--lr", str(args.policy_lr),
            "--max-length", str(args.policy_max_length), "--device", args.train_device,
            "--alpha", str(args.policy_alpha), "--indicator-drop-prob", str(args.policy_indicator_drop_prob),
            "--torch-dtype", args.policy_torch_dtype,
        ]
        if args.policy_device_map is not None:
            policy_cmd.extend(["--device-map", args.policy_device_map])
        run_command(policy_cmd, logger, env)

        eval_rollout_stats = rollout(args, rollout_model if base_model_for_rollout else str(policy_ckpt), eval_rollout_dir, eval_ids, qa_pairs, False, logger)
        score = compute_em(eval_rollout_dir, eval_ids, qa_pairs)

        append_jsonl(metrics_path, {
            "timestamp": utc_now_iso(), "iteration": it,
            "policy_model_before": current_policy, "value_model_before": current_value,
            "policy_checkpoint": str(policy_ckpt), "value_checkpoint": str(value_ckpt),
            "train_rollout": train_rollout_stats, "eval_rollout": eval_rollout_stats,
            "dataset": dataset_stats, "score_em": score,
        })

        logger.info("Iteration %d | EM=%.2f%% | completion=%.2f%%", it, score["accuracy_em"], score["completion_rate"])

        current_policy = str(policy_ckpt)
        current_value = str(value_ckpt)

    logger.info("Finished. Final policy checkpoint: %s", current_policy)


if __name__ == "__main__":
    main()



