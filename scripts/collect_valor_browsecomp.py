from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from valor.browsecomp_prompting import BrowseCompPromptState, build_browsecomp_prompt
from valor.generation import STRICT_FORMAT_SYSTEM_PROMPT, generate_local_completion
from valor.io_utils import write_jsonl
from valor.model import PolicyModel
from valor.rollout_data import load_browsecomp_qa_pairs
from valor.utils import set_seed


def _load_browsecomp_helpers():
    helper_path = REPO_ROOT / "scripts" / "run_browsecomp_plus.py"
    spec = importlib.util.spec_from_file_location("valor_browsecomp_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = _load_browsecomp_helpers()


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def _normalize_valor_tool_call(tool_query: str) -> tuple[str, dict[str, Any], str]:
    raw_tool_query = tool_query.strip()
    tool_payload = HELPERS._safe_json_loads(raw_tool_query)
    if tool_payload is None:
        return "search", {"query": raw_tool_query}, raw_tool_query

    tool_name, tool_params = HELPERS._normalize_tool_call(tool_payload)
    serialized = json.dumps(
        {"tool": tool_name, "parameters": tool_params},
        ensure_ascii=False,
    )
    return tool_name, tool_params, serialized


def _generate_completion(
    *,
    args: argparse.Namespace,
    prompt: str,
    model: PolicyModel | None,
    tokenizer: AutoTokenizer | None,
    device: torch.device | None,
) -> str:
    if args.vllm_url:
        return HELPERS._vllm_chat(
            args.vllm_url,
            args.vllm_model.strip() or args.model_path,
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            api_key=args.vllm_api_key,
            timeout=args.vllm_timeout,
        )

    assert model is not None and tokenizer is not None and device is not None
    return generate_local_completion(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        system_prompt=STRICT_FORMAT_SYSTEM_PROMPT,
    ).completion


def _build_parser(argv: list[str] | None = None) -> tuple[argparse.Namespace, Callable[[str, str | None], str]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--browsecomp-root", required=True)
    bootstrap.add_argument("--searcher-type", required=True, choices=HELPERS.SEARCHER_CHOICES)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)

    browsecomp_root = Path(bootstrap_args.browsecomp_root).expanduser().resolve()
    searcher_type = str(bootstrap_args.searcher_type)
    format_query = HELPERS._prepare_browsecomp_imports(browsecomp_root)
    searcher_class = HELPERS._load_searcher_class(searcher_type, browsecomp_root)

    parser = argparse.ArgumentParser(
        description="Collect BrowseComp-Plus trajectories with BrowseComp-native prompts and a positive advantage indicator."
    )
    parser.add_argument("--browsecomp-root", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--answers-jsonl", required=True, help="BrowseComp answers jsonl for gold answers.")
    parser.add_argument("--output-dir", required=True, help="Directory to write trajectories, runs, and traces.")
    parser.add_argument("--model-path", required=True, help="VALOR policy checkpoint directory or HF model id.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--offload-state-dict", action="store_true")
    parser.add_argument(
        "--vllm-url",
        "--sglang-url",
        dest="vllm_url",
        default="",
        help="Optional vLLM OpenAI-compatible base URL.",
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
    )
    parser.add_argument(
        "--vllm-timeout",
        "--sglang-timeout",
        dest="vllm_timeout",
        type=int,
        default=120,
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--date", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--searcher-type", choices=HELPERS.SEARCHER_CHOICES, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--snippet-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--get-document", action="store_true")
    parser.add_argument(
        "--query-template",
        choices=HELPERS.QUERY_TEMPLATE_CHOICES,
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-id-file", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)

    searcher_class.parse_args(parser)
    args = parser.parse_args(argv)
    args._searcher_class = searcher_class
    return args, format_query


def run_collection(args: argparse.Namespace, format_query: Callable[[str, str | None], str]) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = HELPERS._configure_logging(output_dir)
    set_seed(args.seed)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    browsecomp_root = Path(args.browsecomp_root).expanduser().resolve()
    queries_path = (
        Path(args.queries).expanduser().resolve()
        if args.queries is not None
        else browsecomp_root / "topics-qrels" / "queries.tsv"
    )
    answers_path = Path(args.answers_jsonl).expanduser().resolve()
    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries TSV not found: {queries_path}")
    if not answers_path.is_file():
        raise FileNotFoundError(f"Answers JSONL not found: {answers_path}")

    qa_pairs = load_browsecomp_qa_pairs(queries_path, answers_path)
    query_rows = HELPERS._read_queries_tsv(queries_path)
    query_filter = HELPERS._read_query_filter(args.query_id_file)
    if query_filter is not None:
        query_rows = [(qid, q) for (qid, q) in query_rows if qid in query_filter]
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]

    existing_query_ids = HELPERS._load_existing_query_ids(output_dir)
    pending_rows = [(qid, q) for (qid, q) in query_rows if qid not in existing_query_ids]
    if not pending_rows:
        logger.info("Nothing to run.")
        return

    logger.info("Loaded %d queries from %s", len(query_rows), queries_path)
    logger.info("Pending queries: %d (skipping %d already processed).", len(pending_rows), len(query_rows) - len(pending_rows))

    use_vllm = bool(args.vllm_url.strip())
    model: PolicyModel | None = None
    tokenizer: AutoTokenizer | None = None
    device: torch.device | None = None

    if use_vllm:
        logger.info("Using vLLM backend | url=%s | model=%s", args.vllm_url, args.vllm_model.strip() or args.model_path)
    else:
        torch_dtype = HELPERS._resolve_dtype(args.dtype, args.device)
        gpu_count = torch.cuda.device_count() if args.device == "cuda" else 0
        max_memory = HELPERS._parse_max_memory(args.max_memory, gpu_count)
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
        if args.device_map is None and not getattr(model.backbone, "is_loaded_in_4bit", False):
            model.to(device)
        model.eval()

    snippet_tokenizer = None
    if args.snippet_max_tokens > 0 and args.snippet_tokenizer:
        try:
            snippet_tokenizer = AutoTokenizer.from_pretrained(args.snippet_tokenizer, trust_remote_code=True)
        except Exception as exc:
            logger.warning("Could not load snippet tokenizer '%s' (%s). Using full snippets.", args.snippet_tokenizer, exc)

    searcher = args._searcher_class(args)
    trace_dir = output_dir / "traces"
    if args.save_traces:
        trace_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_store = HELPERS.CheckpointStore(output_dir / "checkpoint_state.json")
    checkpoint_state = HELPERS.CheckpointState(
        created_at=HELPERS._utc_now_iso(),
        updated_at=HELPERS._utc_now_iso(),
        total_queries=len(query_rows),
        completed_query_ids=list(existing_query_ids),
        failed_query_ids=[],
        run_args={k: v for (k, v) in vars(args).items() if not k.startswith("_")},
    )

    trajectories_path = output_dir / "trajectories.jsonl"
    processed_since_save = 0

    for query_id, raw_query in tqdm(pending_rows, desc="VALOR BrowseComp", unit="query"):
        qa = qa_pairs.get(query_id)
        if qa is None:
            logger.warning("Skipping query %s because no gold answer was found.", query_id)
            continue

        runtime = HELPERS.BrowseCompToolRuntime(
            searcher=searcher,
            k=args.k,
            include_get_document=args.get_document,
            snippet_max_tokens=args.snippet_max_tokens,
            snippet_tokenizer=snippet_tokenizer,
        )
        tools_prompt = runtime.build_tools_prompt()
        formatted_query = format_query(raw_query, args.query_template)
        agent_question = raw_query
        query_date = args.date or datetime.now().date().isoformat()

        memory = ""
        prev_tool_query = ""
        prev_tool_result = ""
        last_report = ""
        last_action = ""
        last_observation = ""
        status = "incomplete"
        final_answer = ""
        transitions_for_query: list[dict[str, Any]] = []
        result_items: list[dict[str, Any]] = []
        trace_steps: list[dict[str, Any]] = []

        for step_idx in range(1, args.max_steps + 1):
            prompt = build_browsecomp_prompt(
                BrowseCompPromptState(
                    question=agent_question,
                    last_report=last_report,
                    last_tool_call=last_action,
                    last_tool_response=last_observation,
                ),
                tools_prompt=tools_prompt,
                date_to_use=query_date,
                advantage_label=1,
            )

            trace_item: dict[str, Any] = {
                "step": step_idx,
                "state": {
                    "question": agent_question,
                    "memory": memory,
                    "prev_tool_query": prev_tool_query,
                    "prev_tool_result": prev_tool_result,
                },
                "prompt": prompt,
            }
            transition = {
                "trajectory_id": query_id,
                "t": step_idx - 1,
                "question": agent_question,
                "memory": memory,
                "prev_tool_query": prev_tool_query,
                "prev_tool_result": prev_tool_result,
            }

            completion = _generate_completion(
                args=args,
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            trace_item["completion"] = completion

            report_text, answer_text, tool_call_text = HELPERS._extract_sections(completion)
            format_error = HELPERS._step_format_error(step_idx, answer_text, tool_call_text)
            trace_item["report"] = report_text
            trace_item["answer"] = answer_text
            trace_item["tool_call"] = tool_call_text

            if format_error:
                trace_item["format_error"] = format_error
                trace_steps.append(trace_item)
                break

            action_think = report_text if report_text else "No report."
            action_memory_update = report_text if report_text else (memory if memory else "<empty>")
            transition["action_think"] = action_think
            transition["action_memory_update"] = action_memory_update
            result_items.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": action_think,
                }
            )

            if answer_text:
                final_answer = answer_text.strip()
                transition["action_tool_query"] = "<NO_TOOL_CALL>"
                transition["final_answer"] = final_answer
                transition["gold_answer"] = qa.answer
                transitions_for_query.append(transition)
                result_items.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": final_answer,
                    }
                )
                trace_item["final_answer"] = final_answer
                trace_steps.append(trace_item)
                status = "completed"
                break

            if not tool_call_text:
                trace_item["tool_error"] = "[Tool Error] Missing <tool_call> tag."
                trace_steps.append(trace_item)
                break

            try:
                tool_name, tool_params, serialized_query = _normalize_valor_tool_call(tool_call_text)
                canonical_tool, tool_output = runtime.execute(tool_name, tool_params)
                action_tool_query = json.dumps(
                    {"tool": canonical_tool, "parameters": tool_params},
                    ensure_ascii=False,
                )
                transition["action_tool_query"] = action_tool_query
                prev_tool_query = action_tool_query
                prev_tool_result = tool_output
                last_action = action_tool_query
                last_observation = tool_output
                if report_text:
                    memory = report_text
                    last_report = report_text
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
                trace_item["tool_query"] = serialized_query
                trace_item["tool_output"] = tool_output
            except Exception as exc:
                transition["action_tool_query"] = tool_call_text
                prev_tool_query = tool_call_text
                prev_tool_result = f"[Tool Error] {exc}"
                last_action = tool_call_text
                last_observation = prev_tool_result
                if report_text:
                    memory = report_text
                    last_report = report_text
                result_items.append(
                    {
                        "type": "tool_call",
                        "tool_name": "tool_error",
                        "arguments": tool_call_text,
                        "output": prev_tool_result,
                    }
                )
                trace_item["tool_error"] = prev_tool_result

            transitions_for_query.append(transition)
            trace_steps.append(trace_item)

        run_record = {
            "metadata": {
                "generator": "collect_valor_browsecomp",
                "generated_at": HELPERS._utc_now_iso(),
                "model": args.vllm_model.strip() or args.model_path,
                "prompt_style": "browsecomp_positive_advantage",
            },
            "query_id": query_id,
            "status": status,
            "retrieved_docids": sorted(runtime.retrieved_docids),
            "tool_call_counts": runtime.tool_call_counts,
            "result": result_items,
        }
        HELPERS._write_json(output_dir / f"run_{HELPERS._safe_query_id(query_id)}.json", run_record)
        _append_jsonl(trajectories_path, transitions_for_query)

        if args.save_traces:
            HELPERS._write_json(
                trace_dir / f"trace_{HELPERS._safe_query_id(query_id)}.json",
                {
                    "query_id": query_id,
                    "query": raw_query,
                    "formatted_query": formatted_query,
                    "status": status,
                    "final_answer": final_answer,
                    "steps": trace_steps,
                },
            )

        checkpoint_state.completed_query_ids.append(query_id)
        processed_since_save += 1
        if processed_since_save >= max(1, args.checkpoint_every):
            checkpoint_state.updated_at = HELPERS._utc_now_iso()
            checkpoint_store.save(checkpoint_state)
            processed_since_save = 0

    checkpoint_state.updated_at = HELPERS._utc_now_iso()
    checkpoint_store.save(checkpoint_state)
    logger.info("Finished. Wrote trajectories to %s", trajectories_path)


def main(argv: list[str] | None = None) -> None:
    args, format_query = _build_parser(argv)
    run_collection(args, format_query)


if __name__ == "__main__":
    main()
