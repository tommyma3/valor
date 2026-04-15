from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Optional

from prompts import browsecomp_initial_instruction_prompt, browsecomp_instruction_prompt
from valor.generation import STRICT_FORMAT_SYSTEM_PROMPT, build_chat_messages


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


@dataclass
class BrowseCompPromptState:
    question: str
    last_report: str = ""
    last_tool_call: str = ""
    last_tool_response: str = ""


def build_browsecomp_tools_prompt(
    *,
    k: int = 5,
    include_get_document: bool = False,
) -> str:
    entries = [
        {
            "name": "search",
            "description": (
                f"Perform a search on a knowledge source. Returns top-{k} hits with "
                "docid, score, and snippet. The snippet contains the document's "
                "contents (may be truncated based on token limits)."
            ),
            "parameters": SEARCH_TOOL_PARAMETERS,
        }
    ]
    if include_get_document:
        entries.append(
            {
                "name": "get_document",
                "description": "Retrieve a full document by its docid.",
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


def append_advantage_label(prompt: str, advantage_label: int | None) -> str:
    if advantage_label is None:
        return prompt
    indicator = "positive" if advantage_label > 0 else "negative"
    return prompt.rstrip() + f"\n\n### Advantage\nAdvantage: {indicator}\n"


def _resolve_date(date_to_use: Optional[str]) -> str:
    if date_to_use is not None and date_to_use.strip():
        return date_to_use.strip()
    return date.today().isoformat()


def is_initial_browsecomp_state(state: BrowseCompPromptState) -> bool:
    return not any(
        [
            state.last_report.strip(),
            state.last_tool_call.strip(),
            state.last_tool_response.strip(),
        ]
    )


def build_browsecomp_prompt(
    state: BrowseCompPromptState,
    *,
    tools_prompt: str,
    date_to_use: Optional[str] = None,
    advantage_label: int | None = None,
) -> str:
    resolved_date = _resolve_date(date_to_use)
    if is_initial_browsecomp_state(state):
        base_prompt = browsecomp_initial_instruction_prompt.format(
            date_to_use=resolved_date,
            question=state.question,
            tools=tools_prompt,
        )
    else:
        base_prompt = browsecomp_instruction_prompt.format(
            date_to_use=resolved_date,
            question=state.question,
            tools=tools_prompt,
            report=state.last_report,
            action=state.last_tool_call,
            observation=state.last_tool_response,
        )
    return append_advantage_label(base_prompt, advantage_label)


def build_browsecomp_messages(prompt: str) -> list[dict[str, str]]:
    return build_chat_messages(prompt, system_prompt=STRICT_FORMAT_SYSTEM_PROMPT)


def format_browsecomp_target(
    *,
    report: str,
    tool_call: str | None = None,
    answer: str | None = None,
) -> str:
    normalized_report = report.strip()
    if not normalized_report:
        normalized_report = "No report."

    if bool(tool_call and tool_call.strip()) == bool(answer and answer.strip()):
        raise ValueError("BrowseComp target must include exactly one of tool_call or answer.")

    if tool_call and tool_call.strip():
        body = tool_call.strip()
        second_block = f"<tool_call>\n{body}\n</tool_call>"
    else:
        body = (answer or "").strip()
        second_block = f"<answer>\n{body}\n</answer>"

    return f"<report>\n{normalized_report}\n</report>\n{second_block}"
