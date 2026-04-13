from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import (
    browsecomp_initial_instruction_prompt,
    browsecomp_instruction_prompt,
    initial_instruction_prompt,
    instruction_prompt,
)
from valor.generation import (
    STRICT_FORMAT_SYSTEM_PROMPT,
    build_chat_messages,
    generate_local_completion,
)
from valor.model import PolicyModel
from valor.utils import set_seed


QUERY_TEMPLATE_CHOICES = [
    "QUERY_TEMPLATE",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
]

AGENT_PROMPT_TEMPLATE_CHOICES = ["browsecomp", "default"]
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
    # Remove content between <think> and </think>
    think_pattern = r"<think>.*?</think>"
    filtered = re.sub(think_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove text before </think> if <think> is missing
    think_end_pattern = r"^.*?</think>"
    filtered = re.sub(think_end_pattern, "", filtered, flags=re.DOTALL | re.IGNORECASE)

    return filtered.strip()


def _extract_sections_strict(text: str) -> tuple[str, str, str]:
    """Parse exactly two top-level blocks: <report> + one of <answer>/<tool_call>."""
    # First filter out thinking sections
    filtered_text = _filter_thinking_sections(text)

    # Then parse the remaining text
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
    """Extract the last contiguous <report> + (<answer>|<tool_call>) pair.

    This path is for noisy completions (e.g., chain-of-thought text before tags).
    It intentionally avoids a single anchored regex that can bind an early
    `<answer>` example and consume until the final `</answer>`.
    """
    # First filter out thinking sections
    filtered_text = _filter_thinking_sections(text)

    pair_pattern = re.compile(
        r"<report>(?P<report>.*?)</report>\s*"
        r"<(?P<kind>answer|tool_call)>(?P<body>.*?)</(?P=kind)>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    matches = list(pair_pattern.finditer(filtered_text))
    if not matches:
        return "", "", ""

    # Prefer a pair that reaches the end (ignoring whitespace). Otherwise fall
    # back to the last complete pair in the text.
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


def _fallback_search_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}", question)
    if not tokens:
        return question.strip()[:220]

    stop = {
        "the", "and", "for", "with", "from", "that", "this", "were", "was",
        "into", "what", "which", "where", "when", "book", "author", "award",
        "city", "born", "first", "second", "later", "above", "same", "over",
        "more", "than", "year", "years", "2010s", "2000s",
    }

    picked: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        tl = t.lower()
        if tl in stop or tl in seen:
            continue
        seen.add(tl)
        picked.append(t)
        if len(picked) >= 12:
            break

    if not picked:
        picked = tokens[:12]
    return " ".join(picked)[:220]


def _resolve_dtype(dtype: str | None, device: str) -> torch.dtype | None:
    if dtype is None:
        return torch.bfloat16 if device == "cuda" else None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _parse_max_memory(value: str | None, gpu_count: int) -> dict[int, str] | dict | None:
    if not value:
        return None
    raw = value.strip()
    if raw.startswith("{"):
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("--max-memory JSON must be an object.")
        return loaded
    return {idx: raw for idx in range(gpu_count)}


def _configure_logging(output_dir: Path) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{run_stamp}.log"

    logger = logging.getLogger("valor.browsecomp_plus")
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
    model: PolicyModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    return generate_local_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        system_prompt=STRICT_FORMAT_SYSTEM_PROMPT,
    ).completion


def _vllm_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    api_key: str,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": build_chat_messages(
            prompt,
            system_prompt=STRICT_FORMAT_SYSTEM_PROMPT,
        ),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content")
    if content is None:
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content.strip()
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "".join(text_parts).strip()
    return str(content).strip()


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
    def __init__(
        self,
        searcher: Any,
        *,
        k: int,
        include_get_document: bool,
        snippet_max_tokens: int,
        snippet_tokenizer: AutoTokenizer | None,
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
    import types

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

    # Fast path: standard import works when dependencies are all available.
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

    # Fallback: load target module directly from file to avoid executing
    # searcher/searchers/__init__.py, which imports BM25 unconditionally.
    searchers_dir = browsecomp_root / "searcher" / "searchers"
    base_path = searchers_dir / "base.py"
    target_filename, class_name = mapping[searcher_type]
    target_path = searchers_dir / target_filename

    if not target_path.is_file():
        raise FileNotFoundError(f"Searcher module file not found: {target_path}")

    searcher_pkg_name = "searcher"
    searchers_pkg_name = "searcher.searchers"

    if searcher_pkg_name not in sys.modules:
        pkg = types.ModuleType(searcher_pkg_name)
        pkg.__path__ = [str(browsecomp_root / "searcher")]
        sys.modules[searcher_pkg_name] = pkg

    if searchers_pkg_name not in sys.modules:
        pkg = types.ModuleType(searchers_pkg_name)
        pkg.__path__ = [str(searchers_dir)]
        sys.modules[searchers_pkg_name] = pkg

    base_module_name = "searcher.searchers.base"
    if base_module_name not in sys.modules:
        base_spec = importlib.util.spec_from_file_location(base_module_name, base_path)
        if base_spec is None or base_spec.loader is None:
            raise RuntimeError(f"Failed to create spec for {base_path}")
        base_module = importlib.util.module_from_spec(base_spec)
        base_module.__package__ = searchers_pkg_name
        sys.modules[base_module_name] = base_module
        base_spec.loader.exec_module(base_module)

    target_module_name = f"searcher.searchers.{target_path.stem}"
    target_spec = importlib.util.spec_from_file_location(target_module_name, target_path)
    if target_spec is None or target_spec.loader is None:
        raise RuntimeError(f"Failed to create spec for {target_path}")
    target_module = importlib.util.module_from_spec(target_spec)
    target_module.__package__ = searchers_pkg_name
    sys.modules[target_module_name] = target_module
    target_spec.loader.exec_module(target_module)

    return getattr(target_module, class_name)


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
        description="Run VALOR on BrowseComp-Plus with local retriever tools."
    )
    parser.add_argument("--browsecomp-root", required=True, help="Path to BrowseComp-Plus repository root.")
    parser.add_argument(
        "--queries",
        default=None,
        help="Path to queries TSV file. Defaults to <browsecomp-root>/topics-qrels/queries.tsv",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write run files.")

    parser.add_argument("--model-path", required=True, help="VALOR policy checkpoint directory or HF model id.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-memory", default=None, help="Per-GPU memory limit (e.g. 40GiB) or JSON dict.")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")

    parser.add_argument(
        "--vllm-url",
        "--sglang-url",
        dest="vllm_url",
        default="",
        help="vLLM OpenAI-compatible base URL. If set, rollouts use vLLM for generation.",
    )
    parser.add_argument(
        "--vllm-model",
        "--sglang-model",
        dest="vllm_model",
        default="",
        help="Model name sent to vLLM. Defaults to --model-path when omitted.",
    )
    parser.add_argument(
        "--vllm-api-key",
        "--sglang-api-key",
        dest="vllm_api_key",
        default=os.getenv("VLLM_API_KEY", os.getenv("SGLANG_API_KEY", "")),
        help="API key for vLLM server (optional).",
    )
    parser.add_argument(
        "--vllm-timeout",
        "--sglang-timeout",
        dest="vllm_timeout",
        type=int,
        default=120,
        help="Timeout (seconds) for vLLM generation requests.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--format-retries",
        type=int,
        default=1,
        help="Retry generation this many times when output format/tool_call JSON is invalid.",
    )
    parser.add_argument("--date", default=None, help="Date injected into prompt (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agent-prompt-template",
        choices=AGENT_PROMPT_TEMPLATE_CHOICES,
        default="browsecomp",
        help="System prompt template used by VALOR agent for this benchmark.",
    )

    parser.add_argument("--searcher-type", choices=SEARCHER_CHOICES, required=True)
    parser.add_argument("--k", type=int, default=5, help="Top-k results returned by search tool.")
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Truncate snippets to this many tokens (0 disables truncation).",
    )
    parser.add_argument(
        "--snippet-tokenizer",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used for snippet truncation.",
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
        help="BrowseComp-Plus query template.",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for number of queries.")
    parser.add_argument(
        "--query-id-file",
        default=None,
        help="Optional file containing query ids (one per line) to run.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run queries even if run files already exist.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save checkpoint_state.json every N processed queries.",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save detailed per-query reasoning/tool traces.",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="HF token forwarded to searcher process.")
    parser.add_argument("--hf-home", type=str, default=None, help="HF home forwarded to searcher process.")

    searcher_class.parse_args(parser)
    args = parser.parse_args(argv)
    args._searcher_class = searcher_class
    return args, format_query


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_experiment(args: argparse.Namespace, format_query: Callable[[str, str | None], str]) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(output_dir)

    set_seed(args.seed)

    if args.hf_token:
        import os

        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        import os

        os.environ["HF_HOME"] = args.hf_home

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

    use_vllm = bool(args.vllm_url.strip())
    vllm_model_name = args.vllm_model.strip() or args.model_path

    model: PolicyModel | None = None
    tokenizer: AutoTokenizer | None = None
    device: torch.device | None = None

    if use_vllm:
        logger.info(
            "Using vLLM backend | url=%s | model=%s",
            args.vllm_url,
            vllm_model_name,
        )
    else:
        torch_dtype = _resolve_dtype(args.dtype, args.device)
        gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
        max_memory = _parse_max_memory(args.max_memory, gpu_count)
        if max_memory is None and args.device_map is not None and gpu_count > 0:
            max_memory = {idx: int(torch.cuda.mem_get_info(idx)[0] * 0.9) for idx in range(gpu_count)}

        logger.info("Loading model from %s", args.model_path)
        model = PolicyModel(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
            offload_state_dict=args.offload_state_dict,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if args.device_map is None:
            if not getattr(model.backbone, "is_loaded_in_4bit", False):
                model.to(device)
        model.eval()

    snippet_tokenizer = None
    if args.snippet_max_tokens > 0 and args.snippet_tokenizer:
        try:
            snippet_tokenizer = AutoTokenizer.from_pretrained(args.snippet_tokenizer, trust_remote_code=True)
        except Exception as exc:
            logger.warning(
                "Could not load snippet tokenizer '%s' (%s). Using full snippets.",
                args.snippet_tokenizer,
                exc,
            )

    searcher = args._searcher_class(args)
    logger.info("Initialized searcher type: %s", args.searcher_type)

    if args.query_template == "QUERY_TEMPLATE" and not args.get_document:
        logger.warning(
            "QUERY_TEMPLATE asks for get_document usage, but --get-document is disabled."
        )
    if args.query_template != "QUERY_TEMPLATE" and args.get_document:
        logger.info(
            "get_document tool enabled while using %s query template.",
            args.query_template,
        )

    if args.agent_prompt_template == "browsecomp":
        initial_prompt_template = browsecomp_initial_instruction_prompt
        instruction_prompt_template = browsecomp_instruction_prompt
    else:
        initial_prompt_template = initial_instruction_prompt
        instruction_prompt_template = instruction_prompt
    logger.info("Using agent prompt template: %s", args.agent_prompt_template)

    trace_dir = output_dir / "traces"
    if args.save_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    run_args_snapshot = {
        k: v
        for (k, v) in vars(args).items()
        if not k.startswith("_") and k != "_searcher_class"
    }
    checkpoint_state = (
        existing_state
        if existing_state is not None and not args.overwrite
        else CheckpointState(
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
            total_queries=len(query_rows),
            completed_query_ids=[],
            failed_query_ids=[],
            run_args=run_args_snapshot,
        )
    )

    processed_since_save = 0
    for query_id, raw_query in tqdm(pending_rows, desc="BrowseComp-Plus", unit="query"):
        runtime = BrowseCompToolRuntime(
            searcher=searcher,
            k=args.k,
            include_get_document=args.get_document,
            snippet_max_tokens=args.snippet_max_tokens,
            snippet_tokenizer=snippet_tokenizer,
        )
        tools_prompt = runtime.build_tools_prompt()
        formatted_query = format_query(raw_query, args.query_template)
        # For VALOR prompts, use raw question text to avoid nested prompt instructions
        # embedded by BrowseComp query templates.
        agent_question = raw_query if args.agent_prompt_template == "browsecomp" else formatted_query
        query_date = args.date or datetime.now().date().isoformat()

        logger.info("Query %s | start", query_id)
        steps: list[dict[str, Any]] = []
        result_items: list[dict[str, Any]] = []
        last_report = ""
        last_action = ""
        last_observation = ""
        status = "incomplete"

        for step_idx in range(1, args.max_steps + 1):
            if step_idx == 1:
                base_prompt = initial_prompt_template.format(
                    date_to_use=query_date,
                    question=agent_question,
                    tools=tools_prompt,
                )
            else:
                base_prompt = instruction_prompt_template.format(
                    date_to_use=query_date,
                    question=agent_question,
                    tools=tools_prompt,
                    report=last_report,
                    action=last_action,
                    observation=last_observation,
                )

            completion = ""
            report_text = ""
            answer_text = ""
            tool_call_text = ""
            format_error = ""
            previous_report = last_report
            for attempt in range(max(0, int(args.format_retries)) + 1):
                prompt = (
                    base_prompt
                    if attempt == 0
                    else base_prompt + _format_retry_suffix(format_error)
                )
                if use_vllm:
                    completion = _vllm_chat(
                        args.vllm_url,
                        vllm_model_name,
                        prompt,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        api_key=args.vllm_api_key,
                        timeout=args.vllm_timeout,
                    )
                else:
                    assert model is not None and tokenizer is not None and device is not None
                    completion = _generate_completion(
                        model,
                        tokenizer,
                        prompt,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                report_text, answer_text, tool_call_text = _extract_sections(completion)
                format_error = _step_format_error(step_idx, answer_text, tool_call_text)
                if not format_error:
                    break

            trace_item = {
                "step": step_idx,
                "completion": completion,
                "report": report_text,
                "answer": answer_text,
                "tool_call": tool_call_text,
            }

            if format_error:
                # Recovery path: keep the old report and set tool_call to "Invalid tool call."
                # with empty tool_output so the trajectory can still progress.
                error_payload = {
                    "tool": "error",
                    "parameters": {"message": "Invalid tool call."}
                }
                tool_call_text = json.dumps(error_payload, ensure_ascii=False)
                # Keep the old report unchanged.
                if previous_report:
                    trace_item["report"] = previous_report
                else:
                    # If this is step 1 and there's no previous report, create a basic one
                    last_report = "Starting analysis."
                    trace_item["report"] = last_report
                    result_items.append(
                        {
                            "type": "reasoning",
                            "tool_name": None,
                            "arguments": None,
                            "output": last_report,
                        }
                    )
                if previous_report:
                    last_report = previous_report
                trace_item["tool_call"] = tool_call_text
                trace_item["format_error"] = format_error
                trace_item["format_recovered"] = True
                # Never treat malformed generations as final answers.
                answer_text = ""
                trace_item["answer"] = ""
            elif report_text:
                result_items.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": report_text,
                    }
                )
                last_report = report_text

            if answer_text:
                result_items.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": answer_text.strip(),
                    }
                )
                status = "completed"
                steps.append(trace_item)
                break

            if not tool_call_text:
                last_action = ""
                last_observation = "[Tool Error] Missing <tool_call> tag."
                trace_item["tool_error"] = last_observation
                steps.append(trace_item)
                continue

            # Check if this is an "Invalid tool call." from format error recovery
            if tool_call_text == '{"tool": "error", "parameters": {"message": "Invalid tool call."}}':
                # Handle invalid tool call from format error
                last_action = tool_call_text
                last_observation = ""  # Empty tool_output as specified
                trace_item["tool_error"] = "Invalid tool call from format error recovery"
                result_items.append(
                    {
                        "type": "tool_call",
                        "tool_name": "error",
                        "arguments": {"message": "Invalid tool call."},
                        "output": "",
                    }
                )
                steps.append(trace_item)
                continue

            tool_payload = _safe_json_loads(tool_call_text)
            if tool_payload is None:
                last_action = tool_call_text
                last_observation = "[Tool Error] Could not parse tool_call JSON."
                trace_item["tool_error"] = last_observation
                result_items.append(
                    {
                        "type": "tool_call",
                        "tool_name": "invalid_tool_call",
                        "arguments": tool_call_text,
                        "output": last_observation,
                    }
                )
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
                result_items.append(
                    {
                        "type": "tool_call",
                        "tool_name": canonical_tool,
                        "arguments": tool_params,
                        "output": tool_output,
                    }
                )
                trace_item["tool_name"] = canonical_tool
                trace_item["tool_params"] = tool_params
                trace_item["tool_output"] = tool_output
            except Exception as exc:
                last_action = tool_call_text
                last_observation = f"[Tool Error] {exc}"
                trace_item["tool_error"] = last_observation
                result_items.append(
                    {
                        "type": "tool_call",
                        "tool_name": "tool_error",
                        "arguments": tool_payload,
                        "output": last_observation,
                    }
                )

            steps.append(trace_item)

        run_record = {
            "metadata": {
                "model": args.model_path,
                "searcher_type": args.searcher_type,
                "query_template": args.query_template,
                "k": args.k,
                "max_steps": args.max_steps,
                "max_new_tokens": args.max_new_tokens,
                "agent_prompt_template": args.agent_prompt_template,
                "parser_mode": "strict_v2",
                "generation_backend": "vllm" if use_vllm else "local",
                "generation_model": vllm_model_name if use_vllm else args.model_path,
            },
            "query_id": query_id,
            "tool_call_counts": runtime.tool_call_counts,
            "status": status,
            "retrieved_docids": sorted(runtime.retrieved_docids),
            "result": result_items,
        }

        run_path = output_dir / f"run_{_safe_query_id(query_id)}.json"
        _write_json(run_path, run_record)

        if args.save_traces:
            trace_path = trace_dir / f"trace_{_safe_query_id(query_id)}.json"
            _write_json(
                trace_path,
                {
                    "query_id": query_id,
                    "query": raw_query,
                    "formatted_query": formatted_query,
                    "agent_question": agent_question,
                    "parser_mode": "strict_v2",
                    "steps": steps,
                    "status": status,
                },
            )

        if status == "completed":
            checkpoint_state.completed_query_ids.append(query_id)
            logger.info(
                "Query %s | completed | tool_counts=%s | retrieved=%d",
                query_id,
                runtime.tool_call_counts,
                len(runtime.retrieved_docids),
            )
        else:
            checkpoint_state.failed_query_ids.append(query_id)
            logger.warning(
                "Query %s | incomplete | tool_counts=%s",
                query_id,
                runtime.tool_call_counts,
            )

        processed_since_save += 1
        if processed_since_save >= max(1, args.checkpoint_every):
            checkpoint_state.updated_at = _utc_now_iso()
            checkpoint_state.total_queries = len(query_rows)
            checkpoint_state.run_args = run_args_snapshot
            checkpoint_store.save(checkpoint_state)
            processed_since_save = 0

    checkpoint_state.updated_at = _utc_now_iso()
    checkpoint_state.total_queries = len(query_rows)
    checkpoint_state.run_args = run_args_snapshot
    checkpoint_store.save(checkpoint_state)

    completed = len(set(checkpoint_state.completed_query_ids))
    failed = len(set(checkpoint_state.failed_query_ids))
    logger.info("Finished. Completed=%d Failed/Incomplete=%d", completed, failed)


def main(argv: list[str] | None = None) -> None:
    args, format_query = _build_arg_parser(argv)
    run_experiment(args, format_query)


if __name__ == "__main__":
    main()



