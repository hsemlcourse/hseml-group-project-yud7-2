"""Utilities for preparing EuroCropsML time-series data."""

from __future__ import annotations

import json
import random
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from io import BytesIO
from pathlib import Path

import numpy as np

from src.config import RANDOM_SEED


IGNORED_KEY_PARTS = (
    "date",
    "day",
    "id",
    "label",
    "lat",
    "lon",
    "meta",
    "target",
    "time",
    "utm",
    "x",
    "y",
)


def label_from_npz_name(path: str | Path) -> str:
    """Extract crop label from an EuroCropsML file name."""

    stem = Path(path).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", maxsplit=1)[-1]


def _numeric_candidate_score(key: str, array: np.ndarray) -> tuple[int, int]:
    key_lower = key.lower()
    if not np.issubdtype(array.dtype, np.number):
        return (-1, -1)
    if any(part == key_lower or part in key_lower for part in IGNORED_KEY_PARTS):
        return (-1, -1)
    ndim_score = {2: 4, 3: 3, 1: 2}.get(array.ndim, 1)
    return (ndim_score, int(array.size))


def select_timeseries_array(npz_file: np.lib.npyio.NpzFile) -> np.ndarray:
    """Pick the most likely multispectral time-series array from an NPZ file."""

    scored_arrays: list[tuple[tuple[int, int], str, np.ndarray]] = []
    for key in npz_file.files:
        array = np.asarray(npz_file[key])
        score = _numeric_candidate_score(key, array)
        if score[0] >= 0:
            scored_arrays.append((score, key, array))

    if not scored_arrays:
        keys = ", ".join(npz_file.files)
        raise ValueError(f"Could not find a numeric feature array in NPZ file. Available keys: {keys}")

    scored_arrays.sort(reverse=True, key=lambda item: item[0])
    return scored_arrays[0][2]


def vectorize_timeseries(array: np.ndarray, max_timesteps: int = 366) -> np.ndarray:
    """Convert a time-series array to a fixed-length flat vector.

    This is intentionally plain preprocessing for the CP1 baseline: no derived
    vegetation indices or aggregations are added.
    """

    values = np.asarray(array, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] > max_timesteps:
        values = values[:max_timesteps]
    elif values.shape[0] < max_timesteps:
        pad_rows = max_timesteps - values.shape[0]
        values = np.pad(values, ((0, pad_rows), (0, 0)), mode="constant")

    return values.ravel()


def _iter_npz_members(source: Path) -> Iterator[str]:
    if source.is_dir():
        for path in sorted(source.rglob("*.npz")):
            yield str(path)
        return

    with zipfile.ZipFile(source) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith(".npz"):
                yield member


def _load_npz(source: Path, member: str) -> np.lib.npyio.NpzFile:
    if source.is_dir():
        return np.load(member, allow_pickle=False)

    with zipfile.ZipFile(source) as archive:
        with archive.open(member) as file:
            return np.load(BytesIO(file.read()), allow_pickle=False)


def _build_member_index(source: Path) -> dict[str, str]:
    members = list(_iter_npz_members(source))
    index = {Path(member).name: member for member in members}
    index.update({member: member for member in members})
    return index


def _fit_vector_length(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors_list = list(vectors)
    if not vectors_list:
        return np.empty((0, 0), dtype=np.float32)

    feature_count = max(vector.size for vector in vectors_list)
    output = np.zeros((len(vectors_list), feature_count), dtype=np.float32)
    for row, vector in enumerate(vectors_list):
        length = min(feature_count, vector.size)
        output[row, :length] = vector[:length]
    return output


def build_feature_matrix(
    source: str | Path,
    members: Sequence[str] | None = None,
    max_samples: int | None = None,
    max_timesteps: int = 366,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``X, y, file_names`` from a directory or zip archive of NPZ files."""

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file or directory does not exist: {source_path}")

    if members is None:
        selected_members = list(_iter_npz_members(source_path))
    else:
        index = _build_member_index(source_path)
        selected_members = [index[item] for item in members if item in index]
        missing_count = len(members) - len(selected_members)
        if missing_count:
            print(f"Skipped {missing_count} split entries that were not found in {source_path}")

    if max_samples is not None and max_samples > 0 and len(selected_members) > max_samples:
        rng = random.Random(seed)
        selected_members = rng.sample(selected_members, max_samples)

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for member in selected_members:
        with _load_npz(source_path, member) as npz_file:
            vector = vectorize_timeseries(select_timeseries_array(npz_file), max_timesteps=max_timesteps)
        vectors.append(vector)
        labels.append(label_from_npz_name(member))

    return _fit_vector_length(vectors), np.asarray(labels), np.asarray(selected_members)


def _extract_npz_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [Path(value).name] if value.endswith(".npz") else []
    if isinstance(value, list | tuple):
        names: list[str] = []
        for item in value:
            names.extend(_extract_npz_strings(item))
        return names
    if isinstance(value, dict):
        names = []
        for item in value.values():
            names.extend(_extract_npz_strings(item))
        return names
    return []


def _read_json_documents(source: Path) -> Iterator[tuple[str, object]]:
    if source.is_dir():
        for path in sorted(source.rglob("*.json")):
            yield str(path), json.loads(path.read_text(encoding="utf-8"))
        return

    with zipfile.ZipFile(source) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith(".json"):
                with archive.open(member) as file:
                    yield member, json.loads(file.read().decode("utf-8"))


def find_split_members(
    split_source: str | Path,
    include: Sequence[str],
    exclude: Sequence[str] = (),
) -> list[str]:
    """Find NPZ file names inside split JSON files whose paths match filters."""

    split_path = Path(split_source)
    include_lower = tuple(part.lower() for part in include if part)
    exclude_lower = tuple(part.lower() for part in exclude if part)
    file_names: list[str] = []

    for document_name, document in _read_json_documents(split_path):
        normalized = document_name.lower()
        if any(part not in normalized for part in include_lower):
            continue
        if any(part in normalized for part in exclude_lower):
            continue
        file_names.extend(_extract_npz_strings(document))

    return sorted(set(file_names))


def random_train_val_test_members(
    source: str | Path,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Fallback split when official split files are not available."""

    members = [Path(member).name for member in _iter_npz_members(Path(source))]
    rng = random.Random(seed)
    rng.shuffle(members)
    train_end = int(len(members) * train_size)
    val_end = train_end + int(len(members) * val_size)
    return members[:train_end], members[train_end:val_end], members[val_end:]
