# Train VALOR With vLLM Rollouts (BrowseComp-Plus)

This note explains how to set up **vLLM** and run the new RL loop script:

- `scripts/train_browsecomp_plus_rl.py`
- rollout backend: `scripts/run_browsecomp_plus.py --vllm-*`

## 1) Prerequisites

- BrowseComp-Plus is prepared for evaluation (queries + decrypted answers + retriever index).
- WebShaper training data is accessible on Hugging Face (`Alibaba-NLP/WebShaper`).
- WebShaper loader defaults: `--webshaper-split main`, question/answer/id fields = `question`/`answer`/`id`.
- Your Python env has the project dependencies installed.
- You can run a retriever (`--searcher-type faiss` or `bm25`).

Useful sources:

- [vLLM docs](https://docs.vllm.ai/)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

## 2) Install and Start vLLM

Install vLLM in the same environment where you run VALOR:

```bash
uv pip install vllm
```

Start an OpenAI-compatible vLLM server (example on port `8000`):

```bash
vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000
```

Then sanity-check the endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 3) Run RL Training Loop (vLLM Rollouts)

From repo root:

```bash
uv run python scripts/train_browsecomp_plus_rl.py \
  --browsecomp-root external/BrowseComp-Plus \
  --output-root runs/browsecomp_plus/rl_qwen9b_sglang \
  --searcher-type faiss \
  --index-path "external/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --retriever-model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --policy-init-model "Qwen/Qwen3.5-9B" \
  --value-init-model "Qwen/Qwen3.5-9B" \
  --num-iters 1 \
  --rollout-vllm-url "http://127.0.0.1:8000" \
  --rollout-vllm-model "Qwen/Qwen3.5-9B" \
  --rollout-max-steps 24 \
  --rollout-max-new-tokens 768 \
  --agent-prompt-template browsecomp \
  --train-qa-source webshaper \
  --webshaper-dataset "Alibaba-NLP/WebShaper" \
  --webshaper-split main
```

What this does per iteration:

1. rollout training trajectories on WebShaper QA pairs (via vLLM)
2. build transition dataset from traces
3. compute rewards (`final_answer` vs `gold_answer`)
4. train value model (`Qwen/Qwen3.5-9B` backbone)
5. compute advantages
6. train policy model (`Qwen/Qwen3.5-9B` backbone, full fine-tuning)
7. evaluate on BrowseComp-Plus and log score

Note: the revised VALOR plan uses separate policy and value models. The RL loop keeps that split configurable, and the current defaults use `Qwen/Qwen3.5-9B` for policy and `Qwen/Qwen3.5-9B` for value.

## 4) Multi-Iteration With Updated Policy Checkpoints

If your vLLM server is serving one fixed model, you should run **one iteration at a time** and restart vLLM with the latest policy checkpoint.

After iteration 1, the policy checkpoint is:

- `runs/browsecomp_plus/rl_qwen9b_sglang/iter_001/checkpoints/policy`

Restart vLLM with that checkpoint, then resume training:

```bash
uv run python scripts/train_browsecomp_plus_rl.py \
  --browsecomp-root external/BrowseComp-Plus \
  --output-root runs/browsecomp_plus/rl_qwen9b_sglang \
  --searcher-type faiss \
  --index-path "external/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --retriever-model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --policy-init-model "Qwen/Qwen3.5-9B" \
  --value-init-model "Qwen/Qwen3.5-9B" \
  --num-iters 3 \
  --resume \
  --rollout-vllm-url "http://127.0.0.1:8000" \
  --rollout-vllm-model "runs/browsecomp_plus/rl_qwen9b_sglang/iter_001/checkpoints/policy" \
  --train-qa-source webshaper \
  --webshaper-dataset "Alibaba-NLP/WebShaper" \
  --webshaper-split main
```

Repeat this pattern between iterations (restart server with newest `iter_XXX/checkpoints/policy`).

## 5) Logs, Scores, and Checkpoints

Under `--output-root`:

- `logs/train_loop_*.log`: full training loop logs
- `metrics_history.jsonl`: per-iteration scores and metadata
- `training_state.json`: resumable state
- `iter_XXX/checkpoints/policy`: policy checkpoint (default: Qwen3.5-9B)
- `iter_XXX/checkpoints/value`: value checkpoint (Qwen3.5-9B)

EM benchmark status is logged each iteration as:

- `score_em.accuracy_em`
- `score_em.completion_rate`

Optionally enable official BrowseComp evaluation each iteration:

```bash
--official-eval --official-eval-model Qwen/Qwen3-32B --official-eval-tensor-parallel-size 1
```

## 6) Common Pitfalls

- If rollouts fail with unknown model on vLLM, check `--rollout-vllm-model` exactly matches what vLLM serves.
- If FAISS retriever fails, verify `tevatron`, `qwen-omni-utils`, and index paths.
- If model loading fails, use absolute paths for local checkpoints.
