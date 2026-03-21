# BrowseComp-Plus Evaluation for VALOR

This note explains how to run VALOR on [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) using `scripts/run_browsecomp_plus.py`.

## 1) Prerequisites

1. Clone BrowseComp-Plus:

```bash
git clone https://github.com/texttron/BrowseComp-Plus external/BrowseComp-Plus
```

2. Prepare BrowseComp-Plus data and indexes (run inside `external/BrowseComp-Plus`):

```bash
python scripts_build_index/decrypt_dataset.py --output data/browsecomp_plus_decrypted.jsonl --generate-tsv topics-qrels/queries.tsv
bash scripts_build_index/download_indexes.sh
```

3. If you use BM25, install Java 21 and Pyserini dependencies as required by BrowseComp-Plus.

4. Make sure your VALOR checkpoint/model path is available (local path or HF model id).

## 2) Tool Setup (Retriever for Benchmark)

`run_browsecomp_plus.py` does not use live web tools. It loads BrowseComp-Plus retrievers directly and exposes them to VALOR as tools:

- `search(query)`
- `get_document(docid)` (optional, enable with `--get-document`)

Choose retriever via `--searcher-type`:

- `bm25`: requires `--index-path indexes/bm25`
- `faiss`: requires FAISS index args from BrowseComp-Plus (for example `--index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" --model-name "Qwen/Qwen3-Embedding-8B" --normalize`)
- `reasonir` / `custom`: pass their required searcher args

The script auto-imports the searcher class from `--browsecomp-root` and forwards searcher-specific CLI arguments.

Prompting note:

- The runner defaults to `--agent-prompt-template browsecomp`, a benchmark-specific system prompt tuned for evidence-first retrieval and strict tool-call formatting.
- You can switch to legacy prompts with `--agent-prompt-template default`.

## 3) Run Experiments

Example (BM25):

```bash
uv run python scripts/run_browsecomp_plus.py \
  --browsecomp-root external/BrowseComp-Plus \
  --queries external/BrowseComp-Plus/topics-qrels/queries.tsv \
  --output-dir runs/browsecomp_plus/valor_bm25 \
  --model-path checkpoints/policy_adv \
  --searcher-type bm25 \
  --index-path external/BrowseComp-Plus/indexes/bm25 \
  --max-steps 24 \
  --max-new-tokens 768 \
  --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT \
  --checkpoint-every 20 \
  --save-traces
```

Optional useful flags:

- `--max-queries 100`: quick smoke test
- `--query-id-file path/to/query_ids.txt`: run a subset
- `--overwrite`: rerun queries even if run files already exist
- `--get-document`: register `get_document` tool
- `--device-map auto --max-memory 40GiB`: multi-GPU/offload style runs

## 4) Outputs, Checkpoints, Logs

Inside `--output-dir`, the script writes:

- `run_<query_id>.json`: per-query run result (BrowseComp-Plus evaluation format)
- `checkpoint_state.json`: resumable checkpoint state (`completed_query_ids`, `failed_query_ids`, config snapshot)
- `logs/run_<timestamp>.log`: experiment logs
- `traces/trace_<query_id>.json` (if `--save-traces`): full step trace and tool I/O

You can stop/restart safely; already completed query ids are skipped unless `--overwrite` is used.

## 5) Evaluate with BrowseComp-Plus Judge

Run inside `external/BrowseComp-Plus`:

```bash
python scripts_evaluation/evaluate_run.py \
  --input_dir ../runs/browsecomp_plus/valor_bm25 \
  --ground_truth data/browsecomp_plus_decrypted.jsonl \
  --tensor_parallel_size 1
```

The summary will be generated under `evals/.../evaluation_summary.json`.
