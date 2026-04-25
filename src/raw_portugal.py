"""Prepare the Portugal subset from EuroCropsML raw data."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TARGET_CANDIDATES = ("EC_hcat_c", "hcat_c", "crop", "crop_type", "label", "class")
DATE_CANDIDATES = ("date", "datetime", "timestamp", "time")
ID_CANDIDATES = ("parcel_id", "parcelID", "field_id", "id", "ID", "fid", "FID")
META_COLUMN_PARTS = ("country", "nuts", "geometry", "centroid", "lon", "lat", "area")


def extract_portugal_raw(raw_zip: Path = RAW_DIR / "raw_data.zip", output_dir: Path = RAW_DIR / "portugal") -> Path:
    """Extract only Portugal files from the monolithic raw_data.zip archive."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as archive:
        portugal_members = [member for member in archive.namelist() if Path(member).name.startswith("Portugal")]
        if not portugal_members:
            raise FileNotFoundError("No files with names starting with 'Portugal' were found in raw_data.zip")
        for member in portugal_members:
            archive.extract(member, output_dir)
            print(f"Extracted {member}")
    return output_dir


def _find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[0]


def _read_geojson_properties(path: Path) -> pd.DataFrame:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = [feature.get("properties", {}) for feature in document.get("features", [])]
    return pd.DataFrame(rows)


def _read_table_or_geojson(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".geojson":
        return _read_geojson_properties(path)
    raise ValueError(f"Unsupported label file format: {path}")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    for column in columns:
        column_lower = column.lower()
        if any(candidate.lower() in column_lower for candidate in candidates):
            return column
    return None


def _pick_target_column(columns: list[str]) -> str:
    target = _pick_column(columns, TARGET_CANDIDATES)
    if target is None:
        raise ValueError(f"Could not infer target column. Available columns: {columns}")
    return target


def _pick_id_column(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, str]:
    for left_col in left.columns:
        for right_col in right.columns:
            if left_col == right_col and any(candidate.lower() in left_col.lower() for candidate in ID_CANDIDATES):
                return left_col, right_col

    best_pair: tuple[str, str] | None = None
    best_overlap = 0
    left_candidates = [
        column
        for column in left.columns
        if any(candidate.lower() in column.lower() for candidate in ID_CANDIDATES)
    ]
    right_candidates = [
        column
        for column in right.columns
        if any(candidate.lower() in column.lower() for candidate in ID_CANDIDATES)
    ]
    for left_col in left_candidates:
        left_values = set(left[left_col].dropna().astype(str).head(20_000))
        for right_col in right_candidates:
            right_values = set(right[right_col].dropna().astype(str).head(20_000))
            overlap = len(left_values & right_values)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (left_col, right_col)

    if best_pair is None or best_overlap == 0:
        raise ValueError(
            "Could not infer parcel id columns for joining raw features with labels. "
            f"Raw columns: {list(left.columns)}; label columns: {list(right.columns)}"
        )
    return best_pair


def _is_sequence_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.iloc[0]
    return isinstance(sample, list | tuple | np.ndarray)


def _expand_sequence_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    expanded_parts: list[pd.DataFrame] = []
    for column in columns:
        values = frame[column].apply(lambda item: item if isinstance(item, list | tuple | np.ndarray) else [])
        max_length = int(values.apply(len).max())
        if max_length == 0:
            continue
        expanded = pd.DataFrame(values.tolist(), index=frame.index).iloc[:, :max_length]
        expanded.columns = [f"{column}_{idx}" for idx in range(max_length)]
        expanded_parts.append(expanded)
    if not expanded_parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(expanded_parts, axis=1)


def _prepare_wide_features(frame: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = frame[target_col].astype(str)
    ignored_columns = {target_col}
    for column in frame.columns:
        column_lower = column.lower()
        is_id = any(candidate.lower() in column_lower for candidate in ID_CANDIDATES)
        is_label_name = column_lower in {"ec_hcat_n", "hcat_n", "crop_name", "class_name"}
        if is_id or is_label_name or any(part in column_lower for part in META_COLUMN_PARTS):
            ignored_columns.add(column)

    sequence_columns = [
        column for column in frame.columns if column not in ignored_columns and _is_sequence_like(frame[column])
    ]
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in ignored_columns and column not in sequence_columns
    ]

    numeric_features = frame[numeric_columns].copy()
    sequence_features = _expand_sequence_columns(frame, sequence_columns)
    x = pd.concat([numeric_features, sequence_features], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, y


def _prepare_long_features(frame: pd.DataFrame, id_col: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    date_col = _pick_column(list(frame.columns), DATE_CANDIDATES)
    ignored_columns = {id_col, target_col}
    if date_col is not None:
        ignored_columns.add(date_col)

    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in ignored_columns
        and not any(part in column.lower() for part in META_COLUMN_PARTS)
    ]
    if not numeric_columns:
        raise ValueError("Could not find numeric spectral columns in Portugal raw data.")

    work = frame[[id_col, target_col, *numeric_columns]].copy()
    if date_col is not None:
        work["_step"] = pd.to_datetime(frame[date_col], errors="coerce").rank(method="dense").astype("Int64")
    else:
        work["_step"] = work.groupby(id_col).cumcount()

    work = work.sort_values([id_col, "_step"])
    work["_step"] = work.groupby(id_col).cumcount()
    long = work.pivot_table(index=id_col, columns="_step", values=numeric_columns, aggfunc="mean", fill_value=0.0)
    long.columns = [f"{band}_t{step}" for band, step in long.columns]
    labels = work.groupby(id_col)[target_col].first().astype(str).reindex(long.index)
    return long.replace([np.inf, -np.inf], np.nan).fillna(0.0), labels


def load_portugal_raw(portugal_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    parquet_path = _find_file(portugal_dir, "Portugal.parquet")
    raw = pd.read_parquet(parquet_path)

    raw_target_col = _pick_column(list(raw.columns), TARGET_CANDIDATES)
    if raw_target_col is not None:
        return _prepare_wide_features(raw, target_col=raw_target_col)

    labels_path = _find_file(portugal_dir, "Portugal_labels.parquet")
    labels = _read_table_or_geojson(labels_path)

    target_col = _pick_target_column(list(labels.columns))
    raw_id_col, labels_id_col = _pick_id_column(raw, labels)
    labels = labels[[labels_id_col, target_col]].drop_duplicates(subset=[labels_id_col])
    merged = raw.merge(labels, left_on=raw_id_col, right_on=labels_id_col, how="inner")

    if merged[raw_id_col].duplicated().any():
        return _prepare_long_features(merged, id_col=raw_id_col, target_col=target_col)
    return _prepare_wide_features(merged, target_col=target_col)


def _stratify_or_none(y: pd.Series, test_size: float) -> pd.Series | None:
    counts = y.value_counts()
    test_count = int(np.ceil(len(y) * test_size))
    train_count = len(y) - test_count
    enough_class_space = test_count >= len(counts) and train_count >= len(counts)
    return y if counts.min() >= 2 and len(counts) > 1 and enough_class_space else None


def prepare_portugal_dataset(
    portugal_dir: Path = RAW_DIR / "portugal",
    output_path: Path = PROCESSED_DIR / "portugal_baseline_dataset.npz",
    test_size: float = 0.2,
    val_size: float = 0.2,
    min_class_count: int = 2,
    seed: int = RANDOM_SEED,
) -> Path:
    x, y = load_portugal_raw(portugal_dir)
    class_counts = y.value_counts()
    kept_classes = class_counts[class_counts >= min_class_count].index
    mask = y.isin(kept_classes)
    x = x.loc[mask]
    y = y.loc[mask]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=_stratify_or_none(y, test_size),
    )
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=_stratify_or_none(y_train_val, relative_val_size),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=x_train.to_numpy(dtype=np.float32),
        y_train=y_train.to_numpy(dtype=str),
        X_val=x_val.to_numpy(dtype=np.float32),
        y_val=y_val.to_numpy(dtype=str),
        X_test=x_test.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=str),
        feature_names=x_train.columns.to_numpy(dtype=str),
    )
    print(f"Portugal raw dataset: {len(x):,} samples, {x.shape[1]:,} features, {y.nunique():,} classes")
    print(f"Saved prepared dataset to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract Portugal files from raw_data.zip.")
    extract_parser.add_argument("--raw-zip", type=Path, default=RAW_DIR / "raw_data.zip")
    extract_parser.add_argument("--output-dir", type=Path, default=RAW_DIR / "portugal")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare Portugal train/val/test arrays.")
    prepare_parser.add_argument("--portugal-dir", type=Path, default=RAW_DIR / "portugal")
    prepare_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "portugal_baseline_dataset.npz")
    prepare_parser.add_argument("--test-size", type=float, default=0.2)
    prepare_parser.add_argument("--val-size", type=float, default=0.2)
    prepare_parser.add_argument("--min-class-count", type=int, default=2)
    prepare_parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract_portugal_raw(raw_zip=args.raw_zip, output_dir=args.output_dir)
    elif args.command == "prepare":
        prepare_portugal_dataset(
            portugal_dir=args.portugal_dir,
            output_path=args.output,
            test_size=args.test_size,
            val_size=args.val_size,
            min_class_count=args.min_class_count,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
