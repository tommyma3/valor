"""Train value and policy models for RL."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from valor.rl_utils import configure_logger, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], logger: logging.Logger, cwd: Path | None = None) -> None:
    """Run a command and log its output."""
    logger.info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[cmd] %s", line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train value and policy models for RL."
    )
    parser.add_argument("--trajectories", type=Path, required=True, help="Path to rewarded trajectories JSONL")
    parser.add_argument("--value-model", required=True, help="Value model checkpoint or HF ID")
    parser.add_argument("--policy-model", required=True, help="Policy model checkpoint or HF ID")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints")

    parser.add_argument("--value-batch-size", type=int, default=2)
    parser.add_argument("--value-epochs", type=int, default=1)
    parser.add_argument("--value-lr", type=float, default=2e-5)
    parser.add_argument("--value-max-length", type=int, default=2048)
    parser.add_argument("--value-device-map", default=None)

    parser.add_argument("--policy-batch-size", type=int, default=1)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=2e-5)
    parser.add_argument("--policy-max-length", type=int, default=2048)
    parser.add_argument("--policy-alpha", type=float, default=1.0)
    parser.add_argument("--policy-indicator-drop-prob", type=float, default=0.1)
    parser.add_argument("--policy-device-map", default=None)

    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.trajectories = args.trajectories.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.trajectories.is_file():
        raise FileNotFoundError(f"Trajectories file not found: {args.trajectories}")

    return args


def train_value_model(
    args: argparse.Namespace,
    trajectories_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Train the value model."""
    logger.info("Training value model: %s", args.value_model)

    value_ckpt = output_dir / "value"
    value_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_value.py"),
        "--data",
        str(trajectories_path),
        "--output",
        str(value_ckpt),
        "--backbone",
        args.value_model,
        "--batch-size",
        str(args.value_batch_size),
        "--epochs",
        str(args.value_epochs),
        "--lr",
        str(args.value_lr),
        "--max-length",
        str(args.value_max_length),
        "--device",
        args.train_device,
        "--seed",
        str(args.seed),
    ]
    if args.value_device_map is not None:
        value_cmd.extend(["--device-map", args.value_device_map])

    run_command(value_cmd, logger, cwd=REPO_ROOT)
    logger.info("Value model training completed: %s", value_ckpt)
    return value_ckpt


def compute_advantages(
    args: argparse.Namespace,
    trajectories_path: Path,
    value_model_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Compute advantages using the trained value model."""
    logger.info("Computing advantages using value model: %s", value_model_path)

    adv_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compute_advantages.py"),
        "--data",
        str(trajectories_path),
        "--value-model",
        str(value_model_path),
        "--output",
        str(output_path),
        "--batch-size",
        str(args.value_batch_size),
        "--max-length",
        str(args.value_max_length),
        "--device",
        args.train_device,
    ]

    run_command(adv_cmd, logger, cwd=REPO_ROOT)
    logger.info("Advantages computed: %s", output_path)


def train_policy_model(
    args: argparse.Namespace,
    advantages_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Train the policy model."""
    logger.info("Training policy model: %s", args.policy_model)

    policy_ckpt = output_dir / "policy"
    policy_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_policy.py"),
        "--data",
        str(advantages_path),
        "--output",
        str(policy_ckpt),
        "--backbone",
        args.policy_model,
        "--batch-size",
        str(args.policy_batch_size),
        "--epochs",
        str(args.policy_epochs),
        "--lr",
        str(args.policy_lr),
        "--max-length",
        str(args.policy_max_length),
        "--device",
        args.train_device,
        "--alpha",
        str(args.policy_alpha),
        "--indicator-drop-prob",
        str(args.policy_indicator_drop_prob),
        "--seed",
        str(args.seed),
    ]
    if args.policy_device_map is not None:
        policy_cmd.extend(["--device-map", args.policy_device_map])

    run_command(policy_cmd, logger, cwd=REPO_ROOT)
    logger.info("Policy model training completed: %s", policy_ckpt)
    return policy_ckpt


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Save training metrics."""
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(args.output_dir, "train_value_policy")

    logger.info("Starting value and policy training")
    logger.info("Trajectories: %s", args.trajectories)
    logger.info("Value model: %s", args.value_model)
    logger.info("Policy model: %s", args.policy_model)

    # Train value model
    value_ckpt = train_value_model(args, args.trajectories, args.output_dir, logger)

    # Compute advantages
    advantages_path = args.output_dir / "trajectories_adv.jsonl"
    compute_advantages(args, args.trajectories, value_ckpt, advantages_path, logger)

    # Train policy model
    policy_ckpt = train_policy_model(args, advantages_path, args.output_dir, logger)

    # Save metrics
    metrics = {
        "timestamp": utc_now_iso(),
        "trajectories": str(args.trajectories),
        "value_model_input": args.value_model,
        "value_model_output": str(value_ckpt),
        "policy_model_input": args.policy_model,
        "policy_model_output": str(policy_ckpt),
        "config": {
            "value_batch_size": args.value_batch_size,
            "value_epochs": args.value_epochs,
            "value_lr": args.value_lr,
            "value_max_length": args.value_max_length,
            "policy_batch_size": args.policy_batch_size,
            "policy_epochs": args.policy_epochs,
            "policy_lr": args.policy_lr,
            "policy_max_length": args.policy_max_length,
            "policy_alpha": args.policy_alpha,
            "policy_indicator_drop_prob": args.policy_indicator_drop_prob,
            "seed": args.seed,
        }
    }
    save_metrics(args.output_dir, metrics)

    logger.info("Training completed successfully")
    logger.info("Value checkpoint: %s", value_ckpt)
    logger.info("Policy checkpoint: %s", policy_ckpt)


if __name__ == "__main__":
    main()
