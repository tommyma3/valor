# Train VALOR With SGLang Rollouts (BrowseComp-Plus)

This note explains how to set up **SGLang** and run the new RL loop script:

- `scripts/train_browsecomp_plus_rl.py`
- rollout backend: `scripts/run_browsecomp_plus.py --sglang-*`

## 1) Prerequisites

- BrowseComp-Plus is prepared for evaluation (queries + decrypted answers + retriever index).
- WebShaper training data is accessible on Hugging Face (`Alibaba-NLP/WebShaper`).
- WebShaper loader defaults: `--webshaper-split main`, question/answer/id fields = `question`/`answer`/`id`.
- Your Python env has the project dependencies installed.
- You can run a retriever (`--searcher-type faiss` or `bm25`).

Useful sources:

- [SGLang install docs](https://docs.sglang.io/get_started/install.html)
- [SGLang OpenAI-compatible API docs](https://docs.sglang.io/basic_usage/openai_api_completions.html)

## 2) Install and Start SGLang

Install SGLang in the same environment where you run VALOR:

```bash
uv pip install sglang
```

Start an OpenAI-compatible SGLang server (example on port `8000`):

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000
```

Then sanity-check the endpoint:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 3) Run RL Training Loop (SGLang Rollouts)

From repo root:

```bash
uv run python scripts/train_browsecomp_plus_rl.py \
  --browsecomp-root external/BrowseComp-Plus \
  --output-root runs/browsecomp_plus/rl_qwen35b_sglang \
  --searcher-type faiss \
  --index-path "external/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --retriever-model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --policy-init-model "Qwen/Qwen3.5-35B-A3B" \
  --value-init-model "Qwen/Qwen3.5-9B" \
  --num-iters 1 \
  --rollout-sglang-url "http://127.0.0.1:8000" \
  --rollout-sglang-model "Qwen/Qwen3.5-35B-A3B" \
  --rollout-max-steps 24 \
  --rollout-max-new-tokens 768 \
  --agent-prompt-template browsecomp \
  --train-qa-source webshaper \
  --webshaper-dataset "Alibaba-NLP/WebShaper" \
  --webshaper-split main
```

What this does per iteration:

1. rollout training trajectories on WebShaper QA pairs (via SGLang)
2. build transition dataset from traces
3. compute rewards (`final_answer` vs `gold_answer`)
4. train value model (`Qwen/Qwen3.5-9B` backbone)
5. compute advantages
6. train policy model (`Qwen/Qwen3.5-35B-A3B` backbone, QLoRA adapters over attention + MLP/expert projections)
7. evaluate on BrowseComp-Plus and log score

## 4) Multi-Iteration With Updated Policy Checkpoints

If your SGLang server is serving one fixed model, you should run **one iteration at a time** and restart SGLang with the latest policy checkpoint.

After iteration 1, the policy checkpoint is:

- `runs/browsecomp_plus/rl_qwen35b_sglang/iter_001/checkpoints/policy`

Restart SGLang with that checkpoint, then resume training:

```bash
uv run python scripts/train_browsecomp_plus_rl.py \
  --browsecomp-root external/BrowseComp-Plus \
  --output-root runs/browsecomp_plus/rl_qwen35b_sglang \
  --searcher-type faiss \
  --index-path "external/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --retriever-model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize \
  --policy-init-model "Qwen/Qwen3.5-35B-A3B" \
  --value-init-model "Qwen/Qwen3.5-9B" \
  --num-iters 3 \
  --resume \
  --rollout-sglang-url "http://127.0.0.1:8000" \
  --rollout-sglang-model "runs/browsecomp_plus/rl_qwen35b_sglang/iter_001/checkpoints/policy" \
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
- `iter_XXX/checkpoints/policy`: policy checkpoint (Qwen3.5-35B-A3B)
- `iter_XXX/checkpoints/value`: value checkpoint (Qwen3.5-9B)

EM benchmark status is logged each iteration as:

- `score_em.accuracy_em`
- `score_em.completion_rate`

Optionally enable official BrowseComp evaluation each iteration:

```bash
--official-eval --official-eval-model Qwen/Qwen3-32B --official-eval-tensor-parallel-size 1
```

## 6) Common Pitfalls

- If rollouts fail with unknown model on SGLang, check `--rollout-sglang-model` exactly matches what SGLang serves.
- If FAISS retriever fails, verify `tevatron`, `qwen-omni-utils`, and index paths.
- If model loading fails, use absolute paths for local checkpoints.
