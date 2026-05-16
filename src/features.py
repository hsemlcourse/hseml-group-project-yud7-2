"""Feature engineering for the Portugal EuroCropsML subset."""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED


RAW_PORTUGAL_PATH = Path("data/raw/portugal/raw_data/Portugal.parquet")
PROCESSED_DIR = Path("data/processed")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BAND_NAMES = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


@dataclass(frozen=True)
class PreparedFeatures:
    x: np.ndarray
    y: np.ndarray
    feature_names: np.ndarray
    parcel_ids: np.ndarray


def get_date_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if DATE_PATTERN.match(str(column))]


def load_portugal_frame(path: Path = RAW_PORTUGAL_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.raw_portugal extract")
    return pd.read_parquet(path)


def build_timeseries_cube(frame: pd.DataFrame, date_columns: list[str]) -> np.ndarray:
    """Convert object date columns to a dense ``fields x dates x bands`` cube."""

    cube = np.full((len(frame), len(date_columns), len(BAND_NAMES)), np.nan, dtype=np.float32)
    for date_idx, column in enumerate(date_columns):
        values = frame[column]
        mask = values.notna().to_numpy()
        if mask.any():
            cube[mask, date_idx, :] = np.asarray(values.loc[mask].tolist(), dtype=np.float32)
    return cube


def _append_features(
    parts: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    feature_names: list[str],
) -> None:
    parts.append(values.astype(np.float32, copy=False))
    names.extend(feature_names)


def _safe_normalized_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = left + right
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denominator) > 1e-6, (left - right) / denominator, np.nan)


def engineer_features(cube: np.ndarray, date_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build compact statistical and seasonal features from the raw time series."""

    parts: list[np.ndarray] = []
    names: list[str] = []
    valid_mask = ~np.isnan(cube).all(axis=2)
    valid_count = valid_mask.sum(axis=1).astype(np.float32)
    missing_rate = 1.0 - valid_count / len(date_columns)
    zero_share = np.nanmean(cube == 0, axis=(1, 2))
    _append_features(parts, names, valid_count[:, None], ["valid_observation_count"])
    _append_features(parts, names, missing_rate[:, None], ["missing_rate"])
    _append_features(parts, names, zero_share[:, None], ["zero_share"])

    stat_functions = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
    }
    for stat_name, function in stat_functions.items():
        values = function(cube, axis=1)
        feature_names = [f"{band}_{stat_name}" for band in BAND_NAMES]
        _append_features(parts, names, values, feature_names)

    q25 = np.nanpercentile(cube, 25, axis=1)
    q75 = np.nanpercentile(cube, 75, axis=1)
    _append_features(parts, names, q25, [f"{band}_q25" for band in BAND_NAMES])
    _append_features(parts, names, q75, [f"{band}_q75" for band in BAND_NAMES])
    _append_features(parts, names, q75 - q25, [f"{band}_iqr" for band in BAND_NAMES])
    band_range = np.nanmax(cube, axis=1) - np.nanmin(cube, axis=1)
    _append_features(parts, names, band_range, [f"{band}_range" for band in BAND_NAMES])

    dates = pd.to_datetime(pd.Index(date_columns))
    for month in range(1, 13):
        month_mask = dates.month == month
        if month_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                month_mean = np.nanmean(cube[:, month_mask, :], axis=1)
            feature_names = [f"{band}_month_{month:02d}_mean" for band in BAND_NAMES]
            _append_features(parts, names, month_mean, feature_names)

    # Standard Sentinel-2 order: B04 is red, B03 is green, B08 is NIR, B12 is SWIR2.
    red = cube[:, :, 3]
    green = cube[:, :, 2]
    nir = cube[:, :, 7]
    swir2 = cube[:, :, 12]
    indices = {
        "ndvi": _safe_normalized_difference(nir, red),
        "ndwi": _safe_normalized_difference(green, nir),
        "nbr": _safe_normalized_difference(nir, swir2),
    }
    for index_name, index_values in indices.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            index_stats = np.column_stack(
                [
                    np.nanmean(index_values, axis=1),
                    np.nanstd(index_values, axis=1),
                    np.nanmin(index_values, axis=1),
                    np.nanmax(index_values, axis=1),
                ]
            )
        feature_names = [
            f"{index_name}_mean",
            f"{index_name}_std",
            f"{index_name}_min",
            f"{index_name}_max",
        ]
        _append_features(parts, names, index_stats, feature_names)

    x = np.concatenate(parts, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return x, np.asarray(names, dtype=str)


def build_engineered_features(
    raw_path: Path = RAW_PORTUGAL_PATH,
    min_class_count: int = 3,
) -> PreparedFeatures:
    frame = load_portugal_frame(raw_path)
    class_counts = frame["EC_hcat_c"].value_counts()
    kept_classes = class_counts[class_counts >= min_class_count].index
    frame = frame[frame["EC_hcat_c"].isin(kept_classes)].reset_index(drop=True)
    date_columns = get_date_columns(frame)
    cube = build_timeseries_cube(frame, date_columns)
    x, feature_names = engineer_features(cube, date_columns)
    return PreparedFeatures(
        x=x,
        y=frame["EC_hcat_c"].astype(str).to_numpy(dtype=str),
        feature_names=feature_names,
        parcel_ids=frame["parcel_id"].astype(str).to_numpy(dtype=str),
    )


def split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=seed,
        stratify=y[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def save_feature_dataset(
    output_path: Path = PROCESSED_DIR / "portugal_engineered_features.npz",
    raw_path: Path = RAW_PORTUGAL_PATH,
    seed: int = RANDOM_SEED,
) -> Path:
    prepared = build_engineered_features(raw_path)
    train_idx, val_idx, test_idx = split_indices(prepared.y, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=prepared.x,
        y=prepared.y,
        feature_names=prepared.feature_names,
        parcel_ids=prepared.parcel_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"Saved engineered feature dataset to {output_path}")
    print(f"Shape: {prepared.x.shape[0]:,} rows x {prepared.x.shape[1]:,} features")
    return output_path


def compute_feature_scores(
    dataset_path: Path = PROCESSED_DIR / "portugal_engineered_features.npz",
    output_path: Path = Path("report/feature_scores.csv"),
) -> Path:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        x = dataset["X"]
        y = dataset["y"]
        train_idx = dataset["train_idx"]
        feature_names = dataset["feature_names"]
    f_scores, p_values = f_classif(x[train_idx], y[train_idx])
    scores = pd.DataFrame(
        {
            "feature": feature_names,
            "f_score": np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0),
            "p_value": np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0),
        }
    ).sort_values("f_score", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Saved feature scores to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_PORTUGAL_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "portugal_engineered_features.npz")
    parser.add_argument("--scores-output", type=Path, default=Path("report/feature_scores.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = save_feature_dataset(args.output, raw_path=args.raw_path, seed=args.seed)
    compute_feature_scores(dataset_path, output_path=args.scores_output)


if __name__ == "__main__":
    main()
