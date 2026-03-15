import re
from dataclasses import dataclass
from typing import Optional

from valor.system_prompts import SYSTEM_PROMPT


@dataclass
class State:
    question: str
    memory: str
    prev_tool_query: str
    prev_tool_result: str


@dataclass
class Action:
    think: str
    memory_update: str
    tool_query: str


ACTION_PATTERN = re.compile(
    r"<THINK>\n(?P<think>.*?)\n</THINK>\n"
    r"<MEMORY>\n(?P<memory>.*?)\n</MEMORY>\n"
    r"<TOOL>\n(?P<tool>.*?)\n</TOOL>",
    re.DOTALL,
)


def format_state_prompt(
    state: State,
    include_advantage: bool = False,
    advantage_label: Optional[int] = None,
    include_system_prompt: bool = True,
) -> str:
    parts = []
    if include_system_prompt:
        parts.extend([SYSTEM_PROMPT.strip(), ""])

    parts.extend(
        [
            "### Question",
            state.question.strip(),
            "",
            "### Memory",
            state.memory.strip() if state.memory.strip() else "<empty>",
            "",
            "### Previous Tool Query",
            state.prev_tool_query.strip() if state.prev_tool_query.strip() else "<empty>",
            "",
            "### Previous Tool Result",
            state.prev_tool_result.strip() if state.prev_tool_result.strip() else "<empty>",
            "",
        ]
    )

    if include_advantage and advantage_label is not None:
        indicator = "positive" if advantage_label > 0 else "negative"
        parts.extend(["### Advantage", f"Advantage: {indicator}", ""])

    parts.append("### Action")
    return "\n".join(parts) + "\n"


def format_action(action: Action) -> str:
    return (
        "<THINK>\n"
        f"{action.think.strip()}\n"
        "</THINK>\n"
        "<MEMORY>\n"
        f"{action.memory_update.strip()}\n"
        "</MEMORY>\n"
        "<TOOL>\n"
        f"{action.tool_query.strip()}\n"
        "</TOOL>"
    )


def parse_action(text: str) -> Action:
    match = ACTION_PATTERN.search(text)
    if not match:
        raise ValueError("Could not parse action from model output.")
    return Action(
        think=match.group("think").strip(),
        memory_update=match.group("memory").strip(),
        tool_query=match.group("tool").strip(),
    )
