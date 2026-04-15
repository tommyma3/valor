# BrowseComp-Native Policy and Value Training

This guide explains how to train policy and value models on the **BrowseComp prompt format** while still consuming the existing trajectory JSONL files used elsewhere in the repository.

The new training scripts are:

- `scripts/train_browsecomp_policy.py`
- `scripts/train_browsecomp_value.py`

These scripts are intended for checkpoints that will be used with BrowseComp-style inference, including `scripts/collect_valor_browsecomp.py`.

## What Changed

The original trainers use the generic VALOR prompt and action schema:

- prompt: `### Question / ### Memory / ### Previous Tool Query / ### Previous Tool Result`
- target: `<THINK>...</THINK><MEMORY>...</MEMORY><TOOL>...</TOOL>`

The new BrowseComp trainers use the BrowseComp schema end to end:

- prompt: BrowseComp initial/follow-up prompt templates from `prompts/prompts.py`
- output: `<report>...</report>` plus exactly one of:
  - `<tool_call>...</tool_call>`
  - `<answer>...</answer>`

Training and inference now share:

- the same BrowseComp prompt templates
- the same appended advantage label format
- the same chat system prompt (`STRICT_FORMAT_SYSTEM_PROMPT`)

## Input Data

The new scripts still consume the existing trajectory JSONL format. No preprocessing export step is required.

Expected fields per record:

```json
{
  "trajectory_id": "123",
  "t": 0,
  "question": "...",
  "memory": "...",
  "prev_tool_query": "...",
  "prev_tool_result": "...",
  "action_think": "...",
  "action_memory_update": "...",
  "action_tool_query": "...",
  "advantage_label": 1,
  "reward": 0,
  "return": 1,
  "value_label": 1,
  "final_answer": "..."
}
```

Internal conversion used by the BrowseComp trainers:

- `memory` -> previous `<report>` in the next-step prompt
- `prev_tool_query` -> previous `<tool_call>` in the next-step prompt
- `prev_tool_result` -> previous `<tool_response>` in the next-step prompt
- `action_memory_update` -> supervised `<report>`
- `action_tool_query` -> supervised `<tool_call>` unless it is `<NO_TOOL_CALL>`
- `final_answer` -> supervised `<answer>` on terminal no-tool steps

## Tool Prompt Configuration

Because the trajectory JSONL does not store tool descriptions, the training scripts rebuild a BrowseComp tool block from CLI flags:

- `--search-top-k`
- `--include-get-document`

Use the same settings during training and inference whenever possible.

## Train the Value Model

Example:

```bash
uv run python scripts/train_browsecomp_value.py \
  --data runs/iter_002_data/trajectories.jsonl \
  --output checkpoints/browsecomp_value \
  --backbone Qwen/Qwen3.5-9B \
  --batch-size 2 \
  --epochs 1 \
  --lr 2e-5 \
  --max-length 2048 \
  --search-top-k 5 \
  --include-get-document
```

Useful flags:

- `--date 2026-04-15`: fix the prompt date during training
- `--freeze-backbone`: train only the value head
- `--device-map auto`: shard the backbone across GPUs

## Train the Policy Model

Example:

```bash
uv run python scripts/train_browsecomp_policy.py \
  --data runs/iter_002_data/trajectories_adv.jsonl \
  --output checkpoints/browsecomp_policy \
  --backbone Qwen/Qwen3.5-9B \
  --batch-size 1 \
  --epochs 1 \
  --lr 2e-4 \
  --max-length 2048 \
  --alpha 1.0 \
  --indicator-drop-prob 0.1 \
  --gradient-accumulation-steps 4 \
  --search-top-k 5 \
  --include-get-document
```

Important behavior:

- The base loss trains on BrowseComp targets without an advantage label.
- If `--alpha > 0`, an additional loss term is added using the same BrowseComp prompt plus the appended advantage label.
- Unlike the original VALOR policy trainer, the BrowseComp policy trainer does **not** mask the `<report>` span. The report is part of the public supervised output schema.

## Inference With the New Checkpoints

Use `scripts/collect_valor_browsecomp.py` for BrowseComp-style inference. It now shares the same prompt builder used by the new trainers.

Example:

```bash
PYTHONPATH=. uv run scripts/collect_valor_browsecomp.py \
  --browsecomp-root external/BrowseComp-Plus \
  --answers-jsonl external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \
  --output-dir runs/browsecomp_eval \
  --model-path checkpoints/browsecomp_policy \
  --searcher-type faiss \
  --index-path "external/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name Qwen/Qwen3-Embedding-8B \
  --normalize \
  --get-document \
  --query-template QUERY_TEMPLATE \
  --max-new-tokens 2048 \
  --max-steps 20
```

For vLLM:

```bash
vllm serve checkpoints/browsecomp_policy \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --language-model-only
```

Then point the collector to the OpenAI-compatible endpoint:

```bash
PYTHONPATH=. uv run scripts/collect_valor_browsecomp.py \
  ... \
  --vllm-url http://127.0.0.1:8000 \
  --vllm-model checkpoints/browsecomp_policy
```

## Compatibility Notes

- `scripts/train_policy.py` and `scripts/train_value.py` remain the correct trainers for the original VALOR `<THINK>/<MEMORY>/<TOOL>` setup.
- `scripts/train_browsecomp_policy.py` and `scripts/train_browsecomp_value.py` are for the BrowseComp `<report>/<tool_call>/<answer>` setup.
- Do not expect checkpoints trained with one schema to behave well under the other schema.

## Recommended Workflow

1. Collect or prepare trajectory JSONL data in the existing format.
2. Run `scripts/compute_rewards.py` if rewards are missing.
3. Run `scripts/compute_advantages.py` if `advantage_label` is missing for policy training.
4. Train the BrowseComp value model with `scripts/train_browsecomp_value.py`.
5. Train the BrowseComp policy model with `scripts/train_browsecomp_policy.py`.
6. Run BrowseComp inference with `scripts/collect_valor_browsecomp.py`.
