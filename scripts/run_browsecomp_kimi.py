"""
Run BrowseComp-Plus benchmark using Moonshot AI API with Kimi-K2.5 model.

Uses the IterResearch paradigm with <report>, <tool_call>, and <answer> tags.
Supports checkpointing, parallel processing, and all BrowseComp-Plus searchers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import (
    browsecomp_initial_instruction_prompt,
    browsecomp_instruction_prompt,
)
from valor.utils import set_seed


# Moonshot API configuration
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
MOONSHOT_MODEL = "kimi-k2.5"

QUERY_TEMPLATE_CHOICES = [
    "QUERY_TEMPLATE",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
]

SEARCHER_CHOICES = ["bm25", "faiss", "reasonir", "custom"]

SEARCH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Query string for retrieval."}
    },
    "required": ["query"],
}

GET_DOCUMENT_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "docid": {"type": "string", "description": "Document id to fetch."}
    },
    "required": ["docid"],
}


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _filter_thinking_sections(text: str) -> str:
    """Remove thinking sections from the text.

    Handles:
    1. Text between <think> and </think>
    2. Text before </think> if <think> is missing
    """
    think_pattern = r"<think>.*?</think>"
    filtered = re.sub(think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    think_end_pattern = r"^.*?</think>"
    filtered = re.sub(think_end_pattern, "", filtered, flags=re.DOTALL | re.IGNORECASE)
    return filtered.strip()


def _extract_sections_strict(text: str) -> tuple[str, str, str]:
    """Parse exactly two top-level blocks: <report> + one of <answer>/<tool_call>."""
    filtered_text = _filter_thinking_sections(text)
    pattern = (
        r"^\s*<report>(?P<report>.*?)</report>\s*"
        r"(?:<answer>(?P<answer>.*?)</answer>|<tool_call>(?P<tool_call>.*?)</tool_call>)\s*$"
    )
    match = re.search(pattern, filtered_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return "", "", ""
    report = (match.group("report") or "").strip()
    answer = (match.group("answer") or "").strip()
    tool_call = (match.group("tool_call") or "").strip()
    return report, answer, tool_call


def _extract_sections_relaxed(text: str) -> tuple[str, str, str]:
    """Extract the last contiguous <report> + (<answer>|<tool_call>) pair."""
    filtered_text = _filter_thinking_sections(text)
    pair_pattern = re.compile(
        r"<report>(?P<report>.*?)</report>\s*"
        r"<(?P<kind>answer|tool_call)>(?P<body>.*?)</(?P=kind)>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(pair_pattern.finditer(filtered_text))
    if not matches:
        return "", "", ""

    terminal_matches = [m for m in matches if not filtered_text[m.end() :].strip()]
    match = terminal_matches[-1] if terminal_matches else matches[-1]

    report = (match.group("report") or "").strip()
    kind = str(match.group("kind") or "").strip().lower()
    body = (match.group("body") or "").strip()

    if kind == "answer":
        return report, body, ""
    if kind == "tool_call":
        return report, "", body
    return "", "", ""


def _extract_sections(text: str) -> tuple[str, str, str]:
    """Prefer strict parse; fall back to relaxed tail parse for noisy completions."""
    report, answer, tool_call = _extract_sections_strict(text)
    if report or answer or tool_call:
        return report, answer, tool_call
    return _extract_sections_relaxed(text)


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _normalize_tool_call(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = payload.get("tool") or payload.get("name") or payload.get("tool_name")
    params = (
        payload.get("parameters")
        or payload.get("params")
        or payload.get("arguments")
        or payload.get("args")
        or payload.get("input")
    )

    if tool_name is None and len(payload) == 1:
        tool_name = next(iter(payload.keys()))
        params = payload[tool_name]

    if tool_name is None:
        raise ValueError("Tool name not found in tool_call JSON.")
    if params is None:
        params = {
            k: v for k, v in payload.items() if k not in {"tool", "name", "tool_name"}
        }
    if not isinstance(params, dict):
        raise ValueError("Tool parameters must be a JSON object.")

    return str(tool_name), params


def _step_format_error(step_idx: int, answer_text: str, tool_call_text: str) -> str:
    has_answer = bool(answer_text.strip())
    has_tool = bool(tool_call_text.strip())

    if step_idx == 1:
        if has_answer:
            return "Initial step must not include <answer>; it must output <tool_call> only."
        if not has_tool:
            return "Initial step is missing <tool_call>."
    else:
        if has_answer == has_tool:
            return "Each non-initial step must include exactly one of <answer> or <tool_call>."

    if has_answer:
        lowered = answer_text.lower()
        if any(tag in lowered for tag in ("<report>", "</report>", "<tool_call>", "</tool_call>", "<answer>", "</answer>")):
            return "<answer> must not contain nested tag blocks."

    if has_tool and _safe_json_loads(tool_call_text) is None:
        return "<tool_call> must contain a single valid JSON object."

    return ""


def _format_retry_suffix(error: str) -> str:
    return (
        "\n\nFORMAT CORRECTION (highest priority): "
        f"{error}\n"
        "Regenerate now.\n"
        "Return only the required XML-like blocks.\n"
        "Do not output 'Thinking Process' or any text outside tags.\n"
        "If uncertain, output <tool_call> with valid JSON."
    )


def _configure_logging(output_dir: Path) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{run_stamp}.log"

    logger = logging.getLogger("valor.browsecomp_kimi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging to %s", log_path)
    return logger


def _generate_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int = 120,
) -> str:
    """Generate completion using Moonshot AI API."""
    url = f"{MOONSHOT_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        print(f"Moonshot API Error: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Payload model: {model}")
        print(f"Payload max_tokens: {max_tokens}")
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def _read_queries_tsv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            query_id = row[0].strip()
            query = row[1].strip()
            if query_id and query:
                rows.append((query_id, query))
    return rows


def _safe_query_id(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", query_id)


def _load_existing_query_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not output_dir.exists():
        return completed
    for path in output_dir.glob("run_*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            query_id = data.get("query_id")
            if query_id and str(data.get("status", "")) == "completed":
                completed.add(str(query_id))
        except Exception:
            continue
    return completed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class CheckpointState:
    created_at: str
    updated_at: str
    total_queries: int
    completed_query_ids: list[str]
    failed_query_ids: list[str]
    run_args: dict[str, Any]


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> CheckpointState | None:
        if not self.path.is_file():
            return None
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return CheckpointState(
            created_at=str(raw.get("created_at", _utc_now_iso())),
            updated_at=str(raw.get("updated_at", _utc_now_iso())),
            total_queries=int(raw.get("total_queries", 0)),
            completed_query_ids=[str(x) for x in raw.get("completed_query_ids", [])],
            failed_query_ids=[str(x) for x in raw.get("failed_query_ids", [])],
            run_args=dict(raw.get("run_args", {})),
        )

    def save(self, state: CheckpointState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "total_queries": state.total_queries,
            "completed_query_ids": sorted(set(state.completed_query_ids)),
            "failed_query_ids": sorted(set(state.failed_query_ids)),
            "run_args": state.run_args,
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)


class BrowseCompToolRuntime:
    """Runtime for executing search and get_document tools."""

    def __init__(
        self,
        searcher: Any,
        *,
        k: int,
        include_get_document: bool,
        snippet_max_tokens: int,
        snippet_tokenizer: Any | None,
    ) -> None:
        self.searcher = searcher
        self.k = k
        self.include_get_document = include_get_document
        self.snippet_max_tokens = snippet_max_tokens
        self.snippet_tokenizer = snippet_tokenizer
        self.retrieved_docids: set[str] = set()
        self.tool_call_counts: dict[str, int] = {}

    def build_tools_prompt(self) -> str:
        entries = [
            {
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": SEARCH_TOOL_PARAMETERS,
            }
        ]
        if self.include_get_document:
            entries.append(
                {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": GET_DOCUMENT_TOOL_PARAMETERS,
                }
            )

        sections: list[str] = []
        for entry in entries:
            sections.append(
                f"{entry['name']}: {entry['description']}\n"
                f"parameters: {json.dumps(entry['parameters'], ensure_ascii=False)}"
            )
        return "\n\n".join(sections)

    def execute(self, tool_name: str, params: dict[str, Any]) -> tuple[str, str]:
        normalized = tool_name.strip().lower()
        if normalized in {"search", "local_knowledge_base_retrieval"}:
            query = params.get("query")
            if query is None:
                query = params.get("user_query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search requires a non-empty 'query' string.")
            output = self._search(query)
            canonical_name = "search"
        elif normalized == "get_document" and self.include_get_document:
            docid = params.get("docid")
            if docid is None:
                raise ValueError("get_document requires 'docid'.")
            output = self._get_document(str(docid))
            canonical_name = "get_document"
        else:
            supported = ["search"] + (["get_document"] if self.include_get_document else [])
            raise ValueError(f"Unknown tool '{tool_name}'. Supported: {', '.join(supported)}")

        self.tool_call_counts[canonical_name] = self.tool_call_counts.get(canonical_name, 0) + 1
        return canonical_name, output

    def _truncate_snippet(self, text: str) -> str:
        if (
            self.snippet_tokenizer is None
            or self.snippet_max_tokens <= 0
            or not isinstance(text, str)
        ):
            return text
        token_ids = self.snippet_tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.snippet_max_tokens:
            return text
        trimmed = token_ids[: self.snippet_max_tokens]
        return self.snippet_tokenizer.decode(trimmed, skip_special_tokens=True)

    def _search(self, query: str) -> str:
        candidates = self.searcher.search(query, self.k)
        results: list[dict[str, Any]] = []
        for cand in candidates:
            docid = str(cand.get("docid", ""))
            text = str(cand.get("text", cand.get("snippet", "")))
            snippet = self._truncate_snippet(text)
            item: dict[str, Any] = {"docid": docid, "snippet": snippet}
            if "score" in cand and cand["score"] is not None:
                item["score"] = cand["score"]
            results.append(item)
            if docid:
                self.retrieved_docids.add(docid)
        return json.dumps(results, ensure_ascii=False, indent=2)

    def _get_document(self, docid: str) -> str:
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, indent=2)


def _prepare_browsecomp_imports(root: Path) -> Callable[[str, str | None], str]:
    if not root.is_dir():
        raise FileNotFoundError(f"BrowseComp-Plus root does not exist: {root}")

    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    def _identity(query: str, query_template: str | None = None) -> str:
        del query_template
        return query

    try:
        from search_agent.prompts import format_query as browsecomp_format_query  # type: ignore
        formatter = browsecomp_format_query
    except Exception:
        formatter = _identity

    return formatter


def _load_searcher_class(searcher_type: str, browsecomp_root: Path) -> Any:
    import importlib
    import importlib.util

    mapping: dict[str, tuple[str, str]] = {
        "bm25": ("bm25_searcher.py", "BM25Searcher"),
        "faiss": ("faiss_searcher.py", "FaissSearcher"),
        "reasonir": ("faiss_searcher.py", "ReasonIrSearcher"),
        "custom": ("custom_searcher.py", "CustomSearcher"),
    }

    if searcher_type not in mapping:
        raise ValueError(
            f"Unknown searcher type '{searcher_type}'. Supported: {', '.join(SEARCHER_CHOICES)}"
        )

    module_name_std_map = {
        "bm25": "searcher.searchers.bm25_searcher",
        "faiss": "searcher.searchers.faiss_searcher",
        "reasonir": "searcher.searchers.faiss_searcher",
        "custom": "searcher.searchers.custom_searcher",
    }
    class_name = mapping[searcher_type][1]
    module_name_std = module_name_std_map[searcher_type]
    try:
        module = importlib.import_module(module_name_std)
        return getattr(module, class_name)
    except Exception:
        pass

    searchers_dir = browsecomp_root / "searcher" / "searchers"
    base_path = searchers_dir / "base.py"
    target_filename, class_name = mapping[searcher_type]
    target_path = searchers_dir / target_filename

    spec_base = importlib.util.spec_from_file_location("searcher.searchers.base", base_path)
    module_base = importlib.util.module_from_spec(spec_base)  # type: ignore
    sys.modules["searcher.searchers.base"] = module_base
    spec_base.loader.exec_module(module_base)  # type: ignore

    spec_target = importlib.util.spec_from_file_location(
        f"searcher.searchers.{searcher_type}", target_path
    )
    module_target = importlib.util.module_from_spec(spec_target)  # type: ignore
    sys.modules[f"searcher.searchers.{searcher_type}"] = module_target
    spec_target.loader.exec_module(module_target)  # type: ignore
    return getattr(module_target, class_name)


def _read_query_filter(path: str | None) -> set[str] | None:
    if path is None:
        return None
    query_ids: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                query_ids.add(qid)
    return query_ids


def process_single_query(
    query_id: str,
    raw_query: str,
    formatted_query: str,
    agent_question: str,
    args: argparse.Namespace,
    runtime: BrowseCompToolRuntime,
    api_key: str,
    logger: logging.Logger,
    output_dir: Path,
    trace_dir: Path | None,
) -> dict[str, Any]:
    """Process a single query using the IterResearch paradigm with Moonshot API."""

    tools_text = runtime.build_tools_prompt()
    date_to_use = args.date or datetime.now().strftime("%Y-%m-%d")

    last_report = ""
    last_action = ""
    last_observation = ""

    steps: list[dict[str, Any]] = []
    result_items: list[dict[str, Any]] = []
    status = "incomplete"

    for step in range(1, args.max_steps + 1):
        trace_item: dict[str, Any] = {"step": step}

        if step == 1:
            prompt = browsecomp_initial_instruction_prompt.format(
                date_to_use=date_to_use,
                question=agent_question,
                tools=tools_text,
            )
        else:
            prompt = browsecomp_instruction_prompt.format(
                date_to_use=date_to_use,
                question=agent_question,
                tools=tools_text,
                report=last_report,
                action=last_action,
                observation=last_observation,
            )

        completion = ""
        format_error = ""
        report_text = ""
        answer_text = ""
        tool_call_text = ""

        for retry in range(args.format_retries + 1):
            try:
                completion = _generate_completion(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.api_timeout,
                )
            except requests.RequestException as exc:
                logger.error("Query %s | API request failed (step %d, retry %d): %s", query_id, step, retry, exc)
                if retry == args.format_retries:
                    status = "error"
                    trace_item["error"] = f"API request failed: {exc}"
                    steps.append(trace_item)
                    break
                continue

            report_text, answer_text, tool_call_text = _extract_sections(completion)
            format_error = _step_format_error(step, answer_text, tool_call_text)

            if not format_error:
                break

            if retry < args.format_retries:
                prompt += _format_retry_suffix(format_error)

        if status == "error":
            break

        trace_item["prompt"] = prompt
        trace_item["completion"] = completion
        trace_item["report"] = report_text
        trace_item["answer"] = answer_text
        trace_item["tool_call"] = tool_call_text

        if format_error:
            error_payload = {
                "tool": "error",
                "parameters": {"message": "Invalid tool call."}
            }
            tool_call_text = json.dumps(error_payload, ensure_ascii=False)
            if not last_report:
                last_report = "Starting analysis."
                trace_item["report"] = last_report
                result_items.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": last_report,
                })
            trace_item["tool_call"] = tool_call_text
            trace_item["format_error"] = format_error
            trace_item["format_recovered"] = True
            answer_text = ""
            trace_item["answer"] = ""

        if answer_text:
            result_items.append({
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": answer_text.strip(),
            })
            status = "completed"
            steps.append(trace_item)
            break

        if not tool_call_text:
            last_action = ""
            last_observation = "[Tool Error] Missing <tool_call> tag."
            trace_item["tool_error"] = last_observation
            steps.append(trace_item)
            continue

        if tool_call_text == '{"tool": "error", "parameters": {"message": "Invalid tool call."}}':
            last_action = tool_call_text
            last_observation = ""
            trace_item["tool_error"] = "Invalid tool call from format error recovery"
            result_items.append({
                "type": "tool_call",
                "tool_name": "error",
                "arguments": {"message": "Invalid tool call."},
                "output": "",
            })
            steps.append(trace_item)
            continue

        tool_payload = _safe_json_loads(tool_call_text)
        if tool_payload is None:
            last_action = tool_call_text
            last_observation = "[Tool Error] Could not parse tool_call JSON."
            trace_item["tool_error"] = last_observation
            result_items.append({
                "type": "tool_call",
                "tool_name": "invalid_tool_call",
                "arguments": tool_call_text,
                "output": last_observation,
            })
            steps.append(trace_item)
            continue

        try:
            tool_name, tool_params = _normalize_tool_call(tool_payload)
            canonical_tool, tool_output = runtime.execute(tool_name, tool_params)
            last_action = json.dumps(
                {"tool": canonical_tool, "parameters": tool_params},
                ensure_ascii=False,
            )
            last_observation = tool_output
            result_items.append({
                "type": "tool_call",
                "tool_name": canonical_tool,
                "arguments": tool_params,
                "output": tool_output,
            })
            trace_item["tool_name"] = canonical_tool
            trace_item["tool_params"] = tool_params
            trace_item["tool_output"] = tool_output
        except Exception as exc:
            last_action = tool_call_text
            last_observation = f"[Tool Error] {exc}"
            trace_item["tool_error"] = last_observation
            result_items.append({
                "type": "tool_call",
                "tool_name": "tool_error",
                "arguments": tool_payload,
                "output": last_observation,
            })

        last_report = report_text
        steps.append(trace_item)

    run_record = {
        "metadata": {
            "model": args.model,
            "searcher_type": args.searcher_type,
            "query_template": args.query_template,
            "k": args.k,
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "generation_backend": "moonshot_api",
            "generation_model": args.model,
        },
        "query_id": query_id,
        "tool_call_counts": runtime.tool_call_counts,
        "status": status,
        "retrieved_docids": sorted(runtime.retrieved_docids),
        "result": result_items,
    }

    run_path = output_dir / f"run_{_safe_query_id(query_id)}.json"
    _write_json(run_path, run_record)

    if trace_dir and args.save_traces:
        trace_path = trace_dir / f"trace_{_safe_query_id(query_id)}.json"
        _write_json(
            trace_path,
            {
                "query_id": query_id,
                "query": raw_query,
                "formatted_query": formatted_query,
                "agent_question": agent_question,
                "steps": steps,
                "status": status,
            },
        )

    return {
        "query_id": query_id,
        "status": status,
        "tool_call_counts": runtime.tool_call_counts.copy(),
        "retrieved_docids": len(runtime.retrieved_docids),
    }


def run_experiment(args: argparse.Namespace, format_query: Callable[[str, str | None], str]) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    set_seed(args.seed)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY is not set in environment")

    browsecomp_root = Path(args.browsecomp_root).expanduser().resolve()
    queries_path = (
        Path(args.queries).expanduser().resolve()
        if args.queries is not None
        else browsecomp_root / "topics-qrels" / "queries.tsv"
    )
    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries TSV not found: {queries_path}")

    query_rows = _read_queries_tsv(queries_path)
    query_filter = _read_query_filter(args.query_id_file)
    if query_filter is not None:
        query_rows = [(qid, q) for (qid, q) in query_rows if qid in query_filter]
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]

    if len(query_rows) == 0:
        raise ValueError("No queries selected. Check --queries / --query-id-file.")

    logger.info("Loaded %d queries from %s", len(query_rows), queries_path)

    checkpoint_path = output_dir / "checkpoint_state.json"
    checkpoint_store = CheckpointStore(checkpoint_path)
    existing_state = checkpoint_store.load()

    existing_query_ids = _load_existing_query_ids(output_dir)
    if existing_state is not None:
        existing_query_ids.update(existing_state.completed_query_ids)

    if args.overwrite:
        existing_query_ids.clear()

    pending_rows = [(qid, q) for (qid, q) in query_rows if qid not in existing_query_ids]
    logger.info(
        "Pending queries: %d (skipping %d already processed).",
        len(pending_rows),
        len(query_rows) - len(pending_rows),
    )
    if not pending_rows:
        logger.info("Nothing to run.")
        return

    searcher = args._searcher_class(args)
    logger.info("Initialized searcher type: %s", args.searcher_type)

    snippet_tokenizer = None
    if args.snippet_max_tokens > 0 and args.snippet_tokenizer:
        try:
            from transformers import AutoTokenizer
            snippet_tokenizer = AutoTokenizer.from_pretrained(args.snippet_tokenizer, trust_remote_code=True)
        except Exception as exc:
            logger.warning(
                "Could not load snippet tokenizer '%s' (%s). Using full snippets.",
                args.snippet_tokenizer,
                exc,
            )

    if args.query_template == "QUERY_TEMPLATE" and not args.get_document:
        logger.warning(
            "QUERY_TEMPLATE asks for get_document usage, but --get-document is disabled."
        )
    if args.query_template != "QUERY_TEMPLATE" and args.get_document:
        logger.info(
            "get_document tool enabled while using %s query template.",
            args.query_template,
        )

    trace_dir: Path | None = None
    if args.save_traces:
        trace_dir = output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_state = CheckpointState(
        created_at=_utc_now_iso(),
        updated_at=_utc_now_iso(),
        total_queries=len(query_rows),
        completed_query_ids=list(existing_query_ids) if existing_state is None else existing_state.completed_query_ids,
        failed_query_ids=list(existing_state.failed_query_ids) if existing_state else [],
        run_args={k: str(v) for k, v in vars(args).items() if not k.startswith("_")},
    )

    run_args_snapshot = checkpoint_state.run_args
    processed_since_save = 0

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    logger.info("Tools registered: %s", ", ".join(tools_registered))

    def _process_row(row: tuple[str, str]) -> dict[str, Any]:
        query_id, raw_query = row
        formatted_query = format_query(raw_query, args.query_template)
        agent_question = formatted_query

        runtime = BrowseCompToolRuntime(
            searcher=searcher,
            k=args.k,
            include_get_document=args.get_document,
            snippet_max_tokens=args.snippet_max_tokens,
            snippet_tokenizer=snippet_tokenizer,
        )

        return process_single_query(
            query_id=query_id,
            raw_query=raw_query,
            formatted_query=formatted_query,
            agent_question=agent_question,
            args=args,
            runtime=runtime,
            api_key=api_key,
            logger=logger,
            output_dir=output_dir,
            trace_dir=trace_dir,
        )

    if args.num_workers <= 1:
        with tqdm(pending_rows, desc="Queries", unit="query") as pbar:
            for row in pbar:
                result = _process_row(row)
                if result["status"] == "completed":
                    checkpoint_state.completed_query_ids.append(result["query_id"])
                else:
                    checkpoint_state.failed_query_ids.append(result["query_id"])

                processed_since_save += 1
                if processed_since_save >= max(1, args.checkpoint_every):
                    checkpoint_state.updated_at = _utc_now_iso()
                    checkpoint_store.save(checkpoint_state)
                    processed_since_save = 0
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_workers) as executor,
            tqdm(total=len(pending_rows), desc="Queries", unit="query") as pbar,
        ):
            futures = {executor.submit(_process_row, row): row for row in pending_rows}

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result["status"] == "completed":
                    checkpoint_state.completed_query_ids.append(result["query_id"])
                else:
                    checkpoint_state.failed_query_ids.append(result["query_id"])

                processed_since_save += 1
                if processed_since_save >= max(1, args.checkpoint_every):
                    checkpoint_state.updated_at = _utc_now_iso()
                    checkpoint_store.save(checkpoint_state)
                    processed_since_save = 0

    checkpoint_state.updated_at = _utc_now_iso()
    checkpoint_store.save(checkpoint_state)

    completed = len(set(checkpoint_state.completed_query_ids))
    failed = len(set(checkpoint_state.failed_query_ids))
    logger.info("Finished. Completed=%d Failed/Incomplete=%d", completed, failed)


def _build_arg_parser(argv: list[str] | None = None) -> tuple[argparse.Namespace, Callable[[str, str | None], str]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--browsecomp-root", required=True)
    bootstrap.add_argument("--searcher-type", required=True, choices=SEARCHER_CHOICES)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    browsecomp_root = Path(bootstrap_args.browsecomp_root).expanduser().resolve()
    searcher_type = str(bootstrap_args.searcher_type)
    format_query = _prepare_browsecomp_imports(browsecomp_root)
    searcher_class = _load_searcher_class(searcher_type, browsecomp_root)

    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus benchmark using Moonshot AI API with Kimi-K2.5 model."
    )
    parser.add_argument(
        "--browsecomp-root",
        required=True,
        help="Path to BrowseComp-Plus repository root.",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Path to queries TSV file. Defaults to <browsecomp-root>/topics-qrels/queries.tsv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write run files.",
    )
    parser.add_argument(
        "--model",
        default=MOONSHOT_MODEL,
        help=f"Moonshot model name (default: {MOONSHOT_MODEL}).",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for Moonshot API requests (default: 120).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Max tokens to generate per step (default: 768).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Maximum iterations per query (default: 24).",
    )
    parser.add_argument(
        "--format-retries",
        type=int,
        default=1,
        help="Retry generation when output format/tool_call JSON is invalid (default: 1).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date injected into prompt (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--searcher-type",
        choices=SEARCHER_CHOICES,
        required=True,
        help="Type of searcher to use.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k results returned by search tool (default: 5).",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Truncate snippets to this many tokens (0 disables truncation, default: 512).",
    )
    parser.add_argument(
        "--snippet-tokenizer",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer for snippet truncation (default: Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="Register get_document tool in addition to search.",
    )
    parser.add_argument(
        "--query-template",
        choices=QUERY_TEMPLATE_CHOICES,
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="BrowseComp-Plus query template (default: QUERY_TEMPLATE_NO_GET_DOCUMENT).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap for number of queries.",
    )
    parser.add_argument(
        "--query-id-file",
        default=None,
        help="Optional file containing query ids (one per line) to run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run queries even if run files already exist.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save checkpoint_state.json every N processed queries (default: 20).",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save detailed per-query reasoning/tool traces.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing queries (default: 1).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token forwarded to searcher process.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="HF home forwarded to searcher process.",
    )

    searcher_class.parse_args(parser)
    args = parser.parse_args(argv)
    args._searcher_class = searcher_class
    return args, format_query


def main(argv: list[str] | None = None) -> None:
    args, format_query = _build_arg_parser(argv)
    run_experiment(args, format_query)


if __name__ == "__main__":
    main()
