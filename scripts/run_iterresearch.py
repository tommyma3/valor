import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts import initial_instruction_prompt, instruction_prompt
from valor.model import PolicyModel
from valor.system_prompts import build_tools_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IterResearch prompt with a local model.")
    parser.add_argument("--model-path", default="model/Qwen3.5-9B")
    parser.add_argument("--question", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Device map for multi-GPU inference (e.g., auto, balanced, balanced_low_0).",
    )
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"], help="Model dtype.")
    parser.add_argument(
        "--max-memory",
        default=None,
        help="Per-GPU max memory (e.g., 20GiB) or JSON dict for max_memory.",
    )
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6)
    return parser.parse_args()


def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None




def _resolve_dtype(dtype: Optional[str], device: str) -> Optional[torch.dtype]:
    if dtype is None:
        return torch.bfloat16 if device == "cuda" else None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _parse_max_memory(value: Optional[str], gpu_count: int) -> Optional[dict]:
    if not value:
        return None
    raw = value.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    return {idx: raw for idx in range(gpu_count)}


def _normalize_tool_call(payload: dict) -> Tuple[str, Any]:
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
        params = {k: v for k, v in payload.items() if k not in {"tool", "name", "tool_name"}}

    return tool_name, params


def _load_tool_classes():
    try:
        import tools as tools_pkg
    except Exception:
        return []

    tool_classes = []
    for attr in dir(tools_pkg):
        obj = getattr(tools_pkg, attr)
        if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "parameters"):
            tool_classes.append(obj)

    if hasattr(tools_pkg, "__path__"):
        import importlib
        import pkgutil

        for mod_info in pkgutil.iter_modules(tools_pkg.__path__):
            try:
                module = importlib.import_module(f"{tools_pkg.__name__}.{mod_info.name}")
            except Exception:
                continue
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and hasattr(obj, "name") and hasattr(obj, "parameters"):
                    tool_classes.append(obj)

    unique = {}
    for cls in tool_classes:
        name = getattr(cls, "name", cls.__name__)
        unique[name] = cls

    tool_classes = list(unique.values())
    tool_classes.sort(key=lambda cls: getattr(cls, "name", cls.__name__))
    return tool_classes


def _build_tool_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for cls in _load_tool_classes():
        name = getattr(cls, "name", cls.__name__)
        registry[name] = cls
    return registry


def _generate_completion(
    model: PolicyModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.backbone.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text[len(prompt) :].strip()


def main() -> None:
    args = parse_args()

    progress_idx = 0

    def report(message: str) -> None:
        nonlocal progress_idx
        progress_idx += 1
        print(f"[Progress {progress_idx}] {message}")

    report("Building tools prompt.")
    tools_text = build_tools_prompt()

    report("Initializing tool registry.")
    tool_registry = _build_tool_registry()
    tool_instances: Dict[str, Any] = {}
    tool_init_errors: Dict[str, str] = {}

    report("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report("Resolving device map.")
    gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
    device_map = args.device_map
    if device_map is None and gpu_count > 1:
        device_map = "auto"
    if device_map is not None and args.device != "cuda":
        device_map = None
    report(f"Resolved device map: {device_map or 'none'} (GPU count: {gpu_count}).")

    report("Resolving dtype and memory limits.")
    torch_dtype = _resolve_dtype(args.dtype, args.device)
    try:
        max_memory = _parse_max_memory(args.max_memory, gpu_count)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid --max-memory JSON.") from exc

    if max_memory is None and device_map is not None and gpu_count > 0:
        max_memory = {
            idx: int(torch.cuda.mem_get_info(idx)[0] * 0.9) for idx in range(gpu_count)
        }
        report("Inferred max_memory from free GPU memory.")

    report("Loading model.")
    model = PolicyModel(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        max_memory=max_memory,
        offload_folder=args.offload_folder,
        offload_state_dict=args.offload_state_dict,
    )

    report("Preparing device.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device_map is None:
        if not getattr(model.backbone, "is_loaded_in_4bit", False):
            model.to(device)
    model.eval()

    report("Starting iterative tool loop.")
    report_text = ""
    last_action = ""
    last_observation = ""
    date_to_use = args.date or date.today().isoformat()

    for step in range(1, args.max_steps + 1):
        if step == 1:
            prompt = initial_instruction_prompt.format(
                date_to_use=date_to_use,
                question=args.question,
                tools=tools_text,
            )
        else:
            prompt = instruction_prompt.format(
                date_to_use=date_to_use,
                question=args.question,
                tools=tools_text,
                report=report_text,
                action=last_action,
                observation=last_observation,
            )

        report(f"Generating model output (step {step}/{args.max_steps}).")
        completion = _generate_completion(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )

        report(f"Parsing model output (step {step}/{args.max_steps}).")
        report_text = _extract_tag(completion, "report")
        answer_text = _extract_tag(completion, "answer")
        tool_call_text = _extract_tag(completion, "tool_call")

        print(f"=== Iteration {step} Report ===")
        print(report_text or "[No report tag found]")

        if answer_text:
            print("=== Answer ===")
            print(answer_text.strip())
            return

        if not tool_call_text:
            print("=== Tool Call ===")
            print("[No tool_call tag found in model output]")
            print("=== Raw Output ===")
            print(completion)
            return

        tool_payload = _safe_json_loads(tool_call_text)
        if tool_payload is None:
            print("=== Tool Call ===")
            print(tool_call_text)
            last_action = tool_call_text
            last_observation = "[Tool Error] Could not parse tool_call JSON."
            print("=== Tool Response ===")
            print(last_observation)
            continue

        try:
            tool_name, tool_params = _normalize_tool_call(tool_payload)
        except ValueError as exc:
            last_action = tool_call_text
            last_observation = f"[Tool Error] {exc}"
            print("=== Tool Call ===")
            print(tool_call_text)
            print("=== Tool Response ===")
            print(last_observation)
            continue

        last_action = json.dumps({"tool": tool_name, "parameters": tool_params}, ensure_ascii=False)
        print("=== Tool Call ===")
        print(last_action)

        if tool_name not in tool_registry:
            last_observation = (
                f"[Tool Error] Unknown tool '{tool_name}'. Available: {', '.join(sorted(tool_registry))}"
            )
        else:
            if tool_name not in tool_instances and tool_name not in tool_init_errors:
                try:
                    tool_instances[tool_name] = tool_registry[tool_name]()
                except Exception as exc:
                    tool_init_errors[tool_name] = str(exc)

            if tool_name in tool_init_errors:
                last_observation = f"[Tool Error] Failed to initialize '{tool_name}': {tool_init_errors[tool_name]}"
            else:
                report(f"Executing tool '{tool_name}' (step {step}/{args.max_steps}).")
                try:
                    last_observation = tool_instances[tool_name].call(tool_params)
                except Exception as exc:
                    last_observation = f"[Tool Error] {tool_name} failed: {exc}"

        print("=== Tool Response ===")
        print(last_observation)

    print("=== Final Output ===")
    print("Reached max steps without an answer.")


if __name__ == "__main__":
    main()
