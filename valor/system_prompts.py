from __future__ import annotations

import importlib
import json
import pkgutil
from datetime import date
from typing import List, Type

try:
    from prompts import initial_instruction_prompt as _SYSTEM_TEMPLATE
except Exception:
    _SYSTEM_TEMPLATE = None

DEFAULT_SYSTEM_PROMPT = """You are a VALOR agent. Follow the input format and output a structured action.

Rules:
- Only output the action in the required XML-like tags.
- Keep the tool query concise and executable by the environment.
- Update memory with only the information needed for future steps.
"""


def _extract_tool_classes(module) -> List[Type]:
    tool_classes: List[Type] = []
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "parameters"):
            tool_classes.append(obj)
    return tool_classes


def _load_tool_classes() -> List[Type]:
    try:
        import tools as tools_pkg
    except Exception:
        return []

    tool_classes: List[Type] = []

    # Include classes exposed in tools/__init__.py
    tool_classes.extend(_extract_tool_classes(tools_pkg))

    # Include classes from any module in tools/
    if hasattr(tools_pkg, "__path__"):
        for mod_info in pkgutil.iter_modules(tools_pkg.__path__):
            try:
                module = importlib.import_module(f"{tools_pkg.__name__}.{mod_info.name}")
            except Exception:
                continue
            tool_classes.extend(_extract_tool_classes(module))

    unique = {}
    for cls in tool_classes:
        name = getattr(cls, "name", cls.__name__)
        unique[name] = cls

    tool_classes = list(unique.values())
    tool_classes.sort(key=lambda cls: getattr(cls, "name", cls.__name__))
    return tool_classes


def _format_tool_entry(tool_cls: Type) -> str:
    name = getattr(tool_cls, "name", tool_cls.__name__)
    description = getattr(tool_cls, "description", "").strip()
    parameters = getattr(tool_cls, "parameters", None)

    lines = [f"{name}: {description}" if description else name]
    if parameters is not None:
        lines.append(f"parameters: {json.dumps(parameters, ensure_ascii=False)}")
    return "\n".join(lines)


def build_tools_prompt() -> str:
    tool_classes = _load_tool_classes()
    if not tool_classes:
        return ""
    entries = [_format_tool_entry(cls) for cls in tool_classes]
    return "\n\n".join(entries)


def render_system_prompt(
    question: str = "",
    tools: str = "",
    date_to_use: str | None = None,
) -> str:
    template = _SYSTEM_TEMPLATE or DEFAULT_SYSTEM_PROMPT
    if date_to_use is None:
        date_to_use = date.today().isoformat()

    tools_text = tools.strip() if tools.strip() else build_tools_prompt()

    try:
        return template.format(
            date_to_use=date_to_use,
            question=question,
            tools=tools_text,
        )
    except KeyError:
        return template


SYSTEM_PROMPT = render_system_prompt()
