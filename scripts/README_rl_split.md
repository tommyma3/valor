# RL Training Script Usage Guide

This directory contains the split RL training scripts for VALOR. The original monolithic training script has been separated into three focused scripts for better modularity and flexibility.

## Scripts Overview

### 1. `collect_trajectories_rl.py`
Collects trajectories using SGLang for RL training. This script:
- Loads QA pairs (from BrowseComp or WebShaper)
- Runs rollouts using the specified model (via SGLang or local)
- Converts rollouts to RL transition format
- Computes binary rewards based on exact match

**Key outputs:**
- `trajectories.jsonl` - RL transitions from rollouts
- `trajectories_rewarded.jsonl` - Transitions with rewards computed
- `rollouts/` - Directory containing rollout traces and logs
- `state.json` - Completion status and metadata

### 2. `train_value_policy.py`
Trains value and policy models using the collected trajectories. This script:
- Trains value model on rewarded trajectories
- Computes advantages using the trained value model
- Trains policy model with PPO-like loss

**Key outputs:**
- `checkpoints/value/` - Trained value model
- `checkpoints/policy/` - Trained policy model
- `trajectories_adv.jsonl` - Transitions with advantages
- `metrics.json` - Training metadata and configuration

### 3. `evaluate_baseline.py`
Evaluates a model on BrowseComp-Plus to establish baseline performance. This script:
- Runs rollouts with the specified model
- Computes EM (exact match) accuracy
- Optionally runs official BrowseComp evaluation

**Key outputs:**
- `rollouts/` - Evaluation rollouts
- `results.json` - EM score and optional official evaluation results

### 4. `valor/rl_utils.py`
Shared utilities used by the training scripts. Contains helper functions for:
- I/O operations (JSONL, TSV, etc.)
- Text normalization
- Logging configuration
- QA pair loading
- Command execution

## Usage Examples

### 1. Collect Training Trajectories

```bash
# Using BrowseComp data
python scripts/collect_trajectories_rl.py \\
    --browsecomp-root /path/to/BrowseComp \\
    --queries-tsv /path/to/BrowseComp/topics-qrels/queries.tsv \\
    --answers-jsonl /path/to/BrowseComp/data/browsecomp_plus_decrypted.jsonl \\
    --query-id-file /path/to/train_ids.txt \\
    --output-dir runs/iter_001_data \\
    --model-path Qwen/Qwen3.5-9B \\
    --searcher-type bm25 \\
    --index-path /path/to/index \\
    --sglang-url http://localhost:30000 \\
    --sglang-model Qwen/Qwen3.5-9B

# Using WebShaper data
python scripts/collect_trajectories_rl.py \\
    --browsecomp-root /path/to/BrowseComp \\
    --queries-tsv /path/to/output/queries.tsv \\
    --answers-jsonl /path/to/output/answers.jsonl \\
    --query-id-file /path/to/train_ids.txt \\
    --output-dir runs/iter_001_data \\
    --model-path Qwen/Qwen3.5-9B \\
    --searcher-type bm25 \\
    --index-path /path/to/index \\
    --sglang-url http://localhost:30000
```

### 2. Train Value and Policy Models

```bash
python scripts/train_value_policy.py \\
    --trajectories runs/iter_001_data/trajectories_rewarded.jsonl \\
    --value-model Qwen/Qwen3.5-9B \\
    --policy-model Qwen/Qwen3.5-9B \\
    --output-dir runs/iter_001_checkpoints \\
    --value-batch-size 2 \\
    --policy-batch-size 1 \\
    --value-lr 2e-5 \\
    --policy-lr 2e-5
```

### 3. Evaluate Baseline Performance

```bash
# Before RL training
python scripts/evaluate_baseline.py \\
    --browsecomp-root /path/to/BrowseComp \\
    --queries-tsv /path/to/BrowseComp/topics-qrels/queries.tsv \\
    --answers-jsonl /path/to/BrowseComp/data/browsecomp_plus_decrypted.jsonl \\
    --query-id-file /path/to/eval_ids.txt \\
    --model-path Qwen/Qwen3.5-9B \\
    --searcher-type bm25 \\
    --index-path /path/to/index \\
    --sglang-url http://localhost:30000 \\
    --output-dir runs/baseline_eval \\
    --official-eval

# After RL training
python scripts/evaluate_baseline.py \\
    --browsecomp-root /path/to/BrowseComp \\
    --queries-tsv /path/to/BrowseComp/topics-qrels/queries.tsv \\
    --answers-jsonl /path/to/BrowseComp/data/browsecomp_plus_decrypted.jsonl \\
    --query-id-file /path/to/eval_ids.txt \\
    --model-path runs/iter_001_checkpoints/checkpoints/policy \\
    --searcher-type bm25 \\
    --index-path /path/to/index \\
    --sglang-url http://localhost:30000 \\
    --output-dir runs/iter_001_eval \\
    --official-eval
```

## Full RL Training Loop

A complete RL training iteration consists of:

1. **Collect trajectories** (from training queries)
   ```bash
   python scripts/collect_trajectories_rl.py [args] --output-dir runs/iter_001_data
   ```

2. **Train models** (value + policy)
   ```bash
   python scripts/train_value_policy.py \\
       --trajectories runs/iter_001_data/trajectories_rewarded.jsonl \\
       --output-dir runs/iter_001_checkpoints
   ```

3. **Evaluate** (on evaluation queries)
   ```bash
   python scripts/evaluate_baseline.py \\
       --model-path runs/iter_001_checkpoints/checkpoints/policy \\
       --output-dir runs/iter_001_eval
   ```

4. **Repeat** for subsequent iterations, using the trained policy as the starting point

## Key Parameters

### Trajectory Collection
- `--model-path`: Model for rollouts (local path or HuggingFace ID)
- `--sglang-url`: SGLang server URL (optional, uses local model if not provided)
- `--max-steps`: Maximum steps per trajectory (default: 24)
- `--temperature`: Generation temperature (default: 0.0)

### Training
- `--value-batch-size`: Batch size for value model (default: 2)
- `--policy-batch-size`: Batch size for policy model (default: 1)
- `--value-lr`: Learning rate for value model (default: 2e-5)
- `--policy-lr`: Learning rate for policy model (default: 2e-5)
- `--policy-alpha`: KL penalty coefficient (default: 1.0)

### Evaluation
- `--official-eval`: Enable official BrowseComp evaluation
- `--official-eval-model`: Model for official evaluation (default: Qwen/Qwen3-32B)

## Resume Support

The trajectory collection script supports resuming:
```bash
python scripts/collect_trajectories_rl.py [args] --resume
```

If a `state.json` with `"completed": true` exists, the script will skip execution. Otherwise, it will continue from where it left off.
