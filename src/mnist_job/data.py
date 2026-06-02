from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import TensorDataset

IMAGE_FILES = {
    "train": "train-images-idx3-ubyte.gz",
    "test": "t10k-images-idx3-ubyte.gz",
}
LABEL_FILES = {
    "train": "train-labels-idx1-ubyte.gz",
    "test": "t10k-labels-idx1-ubyte.gz",
}


def find_mnist_root(data_dir: str | Path) -> Path:
    """Find the directory containing the four canonical MNIST gzip files."""
    root = Path(data_dir)
    candidates: Iterable[Path] = [root, *root.rglob("*")] if root.exists() else []
    required = set(IMAGE_FILES.values()) | set(LABEL_FILES.values())
    for candidate in candidates:
        if candidate.is_dir() and required.issubset({p.name for p in candidate.iterdir()}):
            return candidate
    raise FileNotFoundError(
        f"Could not find canonical MNIST gzip files under {root}. "
        "This demo intentionally does not download data ad-hoc; mount the Zenodo "
        "data connector for DOI 10.5281/zenodo.10058130 and pass its mount path."
    )


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    return data


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        if len(labels) != count:
            raise ValueError(f"Expected {count} labels in {path}, got {len(labels)}")
    return labels


def load_numpy(data_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    mnist_root = find_mnist_root(data_dir)
    images = _read_images(mnist_root / IMAGE_FILES[split])
    labels = _read_labels(mnist_root / LABEL_FILES[split])
    return images, labels


def load_dataset(data_dir: str | Path, split: str) -> TensorDataset:
    images, labels = load_numpy(data_dir, split)
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    # MNIST global normalization constants.
    x = (x - 0.1307) / 0.3081
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)
