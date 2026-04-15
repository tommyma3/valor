import argparse
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3.5 text checkpoint into a vLLM-friendly layout in a new directory."
    )
    parser.add_argument("--src", required=True, help="Source model directory.")
    parser.add_argument(
        "--dst",
        default=None,
        help="Destination directory. Defaults to <src>-vllm.",
    )
    return parser.parse_args()


def rename_tensor_key(key: str) -> str:
    if key.startswith("model."):
        return f"language_model.{key}"
    if key.startswith("lm_head."):
        return f"language_model.{key}"
    return key


def validate_paths(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source model directory does not exist: {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(f"Destination directory already exists: {dst_dir}")
    if not (src_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in source directory: {src_dir}")
    if not list(src_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"No .safetensors files found in source directory: {src_dir}")


def convert_safetensor_file(src_path: Path, dst_path: Path) -> None:
    print(f"[CONVERT] {src_path.name}")
    tensors = load_file(str(src_path), device="cpu")
    metadata = None
    with safe_open(str(src_path), framework="pt", device="cpu") as source_file:
        metadata = source_file.metadata()

    new_tensors = {}
    rename_count = 0
    for key, value in tensors.items():
        new_key = rename_tensor_key(key)
        new_tensors[new_key] = value
        if new_key != key:
            rename_count += 1
            print(f"  {key} -> {new_key}")

    save_file(new_tensors, str(dst_path), metadata=metadata)
    print(f"  saved: {dst_path} ({rename_count} renamed keys)")


def convert_config(src_dir: Path, dst_dir: Path) -> None:
    print("[CONVERT] config.json")
    config_path = src_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    old_model_type = config.get("model_type")
    old_architectures = config.get("architectures")
    config["model_type"] = "qwen3_5"
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]

    with (dst_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"  model_type: {old_model_type!r} -> 'qwen3_5'")
    print(f"  architectures: {old_architectures!r} -> ['Qwen3_5ForConditionalGeneration']")


def convert_index_files(src_dir: Path, dst_dir: Path) -> None:
    index_files = sorted(src_dir.glob("*.safetensors.index.json"))
    for index_path in index_files:
        print(f"[CONVERT] {index_path.name}")
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)

        weight_map = index_data.get("weight_map")
        if isinstance(weight_map, dict):
            index_data["weight_map"] = {
                rename_tensor_key(key): value for key, value in weight_map.items()
            }

        with (dst_dir / index_path.name).open("w", encoding="utf-8") as handle:
            json.dump(index_data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"  saved: {dst_dir / index_path.name}")


def copy_remaining_files(src_dir: Path, dst_dir: Path) -> None:
    excluded_names = {
        "config.json",
    }
    excluded_suffixes = {
        ".bak",
        ".safetensors",
    }
    for item in src_dir.iterdir():
        if item.name in excluded_names:
            continue
        if any(item.name.endswith(suffix) for suffix in excluded_suffixes):
            continue
        if item.name.endswith(".safetensors.index.json"):
            continue

        destination = dst_dir / item.name
        print(f"[COPY] {item.name}")
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def convert_model(src_dir: Path, dst_dir: Path) -> None:
    validate_paths(src_dir, dst_dir)
    dst_dir.mkdir(parents=True)

    print(f"[INFO] Source directory: {src_dir}")
    print(f"[INFO] Destination directory: {dst_dir}")

    safetensor_files = sorted(
        file_path for file_path in src_dir.glob("*.safetensors") if not file_path.name.endswith(".bak")
    )
    for safetensor_path in safetensor_files:
        convert_safetensor_file(safetensor_path, dst_dir / safetensor_path.name)

    convert_config(src_dir, dst_dir)
    convert_index_files(src_dir, dst_dir)
    copy_remaining_files(src_dir, dst_dir)

    print("\n[DONE] Conversion finished.")
    print(f"  Output directory: {dst_dir}")
    print("\n  Example launch command:")
    print(f"  vllm serve {dst_dir} --language-model-only --trust-remote-code")


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve() if args.dst else src_dir.with_name(f"{src_dir.name}-vllm")

    try:
        convert_model(src_dir, dst_dir)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
