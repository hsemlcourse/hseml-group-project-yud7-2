"""Dataset download and preparation CLI."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from src.config import RANDOM_SEED, ZENODO_FILES, ZENODO_SOURCE_URL
from src.preprocessing import build_feature_matrix, find_split_members, random_train_val_test_members


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def describe_remote_files() -> None:
    print(f"Zenodo source: {ZENODO_SOURCE_URL}")
    for name, remote_file in ZENODO_FILES.items():
        print(f"{name}: {remote_file.size_mb:.1f} MB ({remote_file.size_gib:.2f} GiB)")


def download_file(name: str, output_dir: Path = RAW_DIR) -> Path:
    if name not in ZENODO_FILES:
        known = ", ".join(sorted(ZENODO_FILES))
        raise ValueError(f"Unknown file '{name}'. Expected one of: {known}")

    output_dir.mkdir(parents=True, exist_ok=True)
    remote_file = ZENODO_FILES[name]
    output_path = output_dir / remote_file.filename
    if output_path.exists() and output_path.stat().st_size == remote_file.size_bytes:
        print(f"Already downloaded: {output_path}")
        return output_path

    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    print(f"Downloading {name} ({remote_file.size_mb:.1f} MB) to {output_path}")
    with urllib.request.urlopen(remote_file.url) as response, temp_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    temp_path.replace(output_path)
    return output_path


def prepare_baseline_dataset(
    preprocess_source: Path,
    split_source: Path | None,
    output_path: Path,
    max_samples_per_split: int | None,
    max_timesteps: int,
    seed: int,
) -> Path:
    if split_source is not None and split_source.exists():
        train_members = find_split_members(split_source, include=("train",), exclude=("val", "test"))
        val_members = find_split_members(split_source, include=("val",), exclude=("train", "test"))
        test_members = find_split_members(split_source, include=("test",), exclude=("train", "val"))
        if not train_members or not val_members or not test_members:
            print("Official split was found, but matching JSON files were not detected. Using random fallback split.")
            train_members, val_members, test_members = random_train_val_test_members(preprocess_source, seed=seed)
    else:
        print("Official split.zip was not found. Using random fallback split.")
        train_members, val_members, test_members = random_train_val_test_members(preprocess_source, seed=seed)

    max_samples = None if max_samples_per_split is None or max_samples_per_split <= 0 else max_samples_per_split
    print("Building train matrix")
    x_train, y_train, train_files = build_feature_matrix(
        preprocess_source,
        members=train_members,
        max_samples=max_samples,
        max_timesteps=max_timesteps,
        seed=seed,
    )
    print("Building validation matrix")
    x_val, y_val, val_files = build_feature_matrix(
        preprocess_source,
        members=val_members,
        max_samples=max_samples,
        max_timesteps=max_timesteps,
        seed=seed,
    )
    print("Building test matrix")
    x_test, y_test, test_files = build_feature_matrix(
        preprocess_source,
        members=test_members,
        max_samples=max_samples,
        max_timesteps=max_timesteps,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=x_train,
        y_train=y_train,
        train_files=train_files,
        X_val=x_val,
        y_val=y_val,
        val_files=val_files,
        X_test=x_test,
        y_test=y_test,
        test_files=test_files,
    )
    print(f"Saved prepared dataset to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Print Zenodo file sizes and URLs.")

    download_parser = subparsers.add_parser("download", help="Download selected Zenodo files.")
    download_parser.add_argument(
        "--files",
        nargs="+",
        default=("split.zip", "preprocess.zip"),
        choices=sorted(ZENODO_FILES),
    )
    download_parser.add_argument("--output-dir", type=Path, default=RAW_DIR)

    prepare_parser = subparsers.add_parser("prepare", help="Build train/val/test arrays for the baseline.")
    prepare_parser.add_argument("--preprocess-source", type=Path, default=RAW_DIR / "preprocess.zip")
    prepare_parser.add_argument("--split-source", type=Path, default=RAW_DIR / "split.zip")
    prepare_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "baseline_dataset.npz")
    prepare_parser.add_argument("--max-samples-per-split", type=int, default=5_000)
    prepare_parser.add_argument("--max-timesteps", type=int, default=366)
    prepare_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "info":
        describe_remote_files()
    elif args.command == "download":
        for file_name in args.files:
            download_file(file_name, output_dir=args.output_dir)
    elif args.command == "prepare":
        prepare_baseline_dataset(
            preprocess_source=args.preprocess_source,
            split_source=args.split_source,
            output_path=args.output,
            max_samples_per_split=args.max_samples_per_split,
            max_timesteps=args.max_timesteps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
