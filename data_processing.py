"""Utilities for preparing gearbox fault diagnosis datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

WINDOW_SIZE = 500
WINDOW_STEP = 500


def read_excel_signals(file_path: Path) -> np.ndarray:
    """Read a vibration Excel file into a numpy array.

    Parameters
    ----------
    file_path: Path
        The path of the Excel file. Files are expected to contain three columns
        (speed pulse, vertical vibration, horizontal vibration).

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, num_channels=3).
    """

    data_frame = pd.read_excel(file_path, header=None)
    return data_frame.to_numpy(dtype=np.float32)


def sliding_window(samples: np.ndarray, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP) -> np.ndarray:
    """Convert a long time-series into a stack of windows without overlap."""

    num_points, num_channels = samples.shape
    windows: List[np.ndarray] = []
    for start in range(0, num_points - window_size + 1, step):
        end = start + window_size
        window = samples[start:end]
        if window.shape[0] == window_size:
            windows.append(window)
    if not windows:
        return np.empty((0, window_size, num_channels), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def build_dataset_from_directory(
    directory: Path,
    label_index: int,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load every Excel file in the directory and generate windowed samples."""

    all_samples: List[np.ndarray] = []
    for excel_path in sorted(directory.glob("*.xlsx")):
        raw_signals = read_excel_signals(excel_path)
        windows = sliding_window(raw_signals, window_size, step)
        if windows.size == 0:
            continue
        all_samples.append(windows)
    if not all_samples:
        return np.empty((0, window_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    samples = np.concatenate(all_samples, axis=0)
    labels = np.full((samples.shape[0],), label_index, dtype=np.int64)
    return samples, labels


def assemble_dataset(class_dirs: Dict[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Generate samples and labels for all class directories."""

    samples_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    label_mapping: Dict[str, int] = {}

    for label_idx, (class_name, class_path) in enumerate(sorted(class_dirs.items())):
        label_mapping[class_name] = label_idx
        class_samples, class_labels = build_dataset_from_directory(class_path, label_idx)
        if class_samples.size == 0:
            continue
        samples_list.append(class_samples)
        labels_list.append(class_labels)

    samples = np.concatenate(samples_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return samples, labels, label_mapping


def load_inference_windows(
    directory: Path,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> Dict[str, np.ndarray]:
    """Generate windowed samples for every Excel file in ``directory``.

    The returned dictionary maps the stem of each file (e.g. ``test0001``) to
    an array of shape ``(num_windows, window_size, num_channels)``. Files that
    do not yield at least one full window are skipped with a warning message.
    """

    window_mapping: Dict[str, np.ndarray] = {}
    for excel_path in sorted(directory.glob("*.xlsx")):
        raw_signals = read_excel_signals(excel_path)
        windows = sliding_window(raw_signals, window_size, step)
        if windows.size == 0:
            print(
                f"[WARN] 文件 {excel_path.name} 不足以切分出一个长度为 {window_size} 的窗口，已跳过。"
            )
            continue
        window_mapping[excel_path.stem] = windows
    return window_mapping


def create_dataloaders(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split samples into training/validation/testing sets and wrap in DataLoaders.

    Parameters
    ----------
    samples, labels:
        Pre-processed dataset arrays.
    batch_size:
        Mini-batch size used by the returned DataLoaders.
    test_size:
        Fraction of the total dataset reserved for the final test split.
    val_size:
        Fraction of the total dataset reserved for validation. The validation
        portion is taken from the remaining data after the test split.
    random_state:
        Seed controlling the deterministic behaviour of the shuffling.
    """

    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be < 1.0")

    all_indices = np.arange(samples.shape[0])
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    remaining_val_fraction = val_size / (1.0 - test_size)
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=remaining_val_fraction,
        random_state=random_state,
        stratify=labels[train_indices],
    )

    def _to_tensor(index_array: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(samples[index_array], dtype=torch.float32),
            torch.tensor(labels[index_array], dtype=torch.long),
        )

    train_samples, train_labels = _to_tensor(train_indices)
    val_samples, val_labels = _to_tensor(val_indices)
    test_samples, test_labels = _to_tensor(test_indices)

    train_dataset = TensorDataset(train_samples, train_labels)
    val_dataset = TensorDataset(val_samples, val_labels)
    test_dataset = TensorDataset(test_samples, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_samples,
        train_labels,
        val_samples,
        val_labels,
        test_samples,
        test_labels,
    )


def save_numpy_dataset(
    output_path: Path,
    samples: np.ndarray,
    labels: np.ndarray,
    label_mapping: Dict[str, int],
) -> None:
    """Persist processed dataset to an ``.npz`` file with meta information."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"label_mapping": label_mapping, "window_size": WINDOW_SIZE, "window_step": WINDOW_STEP}
    np.savez_compressed(output_path, samples=samples, labels=labels, metadata=json.dumps(metadata))


def load_numpy_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load dataset produced by :func:`save_numpy_dataset`."""

    npz_data = np.load(dataset_path, allow_pickle=False)
    metadata = json.loads(str(npz_data["metadata"]))
    return npz_data["samples"], npz_data["labels"], metadata["label_mapping"]
