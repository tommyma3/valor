"""Shared utilities for RL training scripts."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from valor.io_utils import append_jsonl, load_json, read_jsonl, save_json, write_jsonl


def utc_now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.strip().lower().split())


def safe_query_id(query_id: str) -> str:
    """Sanitize query ID for file systems."""
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in query_id)


def configure_logger(output_root: Path, name: str) -> logging.Logger:
    """Configure logger with file and stream handlers."""
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{name}_{stamp}.log"

    logger = logging.getLogger(f"valor.{name}")
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

    logger.info("Logs: %s", log_path)
    return logger

def load_query_ids(path: Path) -> list[str]:
    """Load query IDs from text file."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid = line.strip()
            if qid:
                ids.append(qid)
    return ids


def write_query_ids(path: Path, query_ids: list[str]) -> None:
    """Write query IDs to text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(qid)
            f.write("\n")


def write_queries_tsv(path: Path, query_ids: list[str], queries: dict[str, str]) -> None:
    """Write queries to TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for qid in query_ids:
            query = queries.get(qid)
            if query is None:
                continue
            writer.writerow([qid, query])


def extract_final_answer(run_record: dict[str, Any]) -> str:
    """Extract final answer from run record."""
    result = run_record.get("result", [])
    if not isinstance(result, list):
        return ""
    for item in reversed(result):
        if isinstance(item, dict) and item.get("type") == "output_text":
            return str(item.get("output", "")).strip()
    return ""
