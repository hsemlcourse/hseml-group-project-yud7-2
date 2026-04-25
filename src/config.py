"""Shared project configuration."""

from __future__ import annotations

from dataclasses import dataclass


RANDOM_SEED = 42


@dataclass(frozen=True)
class RemoteFile:
    """Metadata for a file hosted on Zenodo."""

    filename: str
    size_bytes: int
    url: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000

    @property
    def size_gib(self) -> float:
        return self.size_bytes / (1024**3)


ZENODO_RECORD_ID = "15095445"
ZENODO_SOURCE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

ZENODO_FILES = {
    "split.zip": RemoteFile(
        filename="split.zip",
        size_bytes=20_699_453,
        url=f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/split.zip/content",
    ),
    "preprocess.zip": RemoteFile(
        filename="preprocess.zip",
        size_bytes=1_468_255_227,
        url=f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/preprocess.zip/content",
    ),
    "raw_data.zip": RemoteFile(
        filename="raw_data.zip",
        size_bytes=3_279_983_432,
        url=f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/raw_data.zip/content",
    ),
}
