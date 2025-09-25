"""Train and evaluate an InceptionTime model on the competition dataset.

This script reads the vibration Excel files from the training split, builds a
windowed dataset, trains an InceptionTime classifier and finally performs
inference on the official test split. The predictions for the test split are
saved to ``inference_results.txt``.

The implementation only relies on the packages that ship with the repository
(torch, numpy and scikit-learn). Reading ``.xlsx`` files is handled with Python's
standard library by parsing the underlying XML files.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from inception_time_pytorch.model import InceptionTime


# Namespace used inside XLSX worksheet XML files.
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def column_index(cell_reference: str) -> int:
    """Return the zero-based column index for an Excel cell reference.

    Examples
    --------
    >>> column_index("A1")
    0
    >>> column_index("C532")
    2
    """

    col_letters = "".join(ch for ch in cell_reference if ch.isalpha())
    index = 0
    for char in col_letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def iter_excel_rows(
    file_path: Path,
    *,
    num_channels: int,
    max_rows: Optional[int] = None,
) -> Iterator[List[float]]:
    """Yield rows from an Excel file as lists of floats.

    The reader uses ``xml.etree.ElementTree.iterparse`` to stream the XML
    worksheet. Only numeric values are expected in the dataset and any empty
    cell is represented as ``math.nan``.
    """

    with zipfile.ZipFile(file_path) as zf:
        with zf.open("xl/worksheets/sheet1.xml") as worksheet:
            context = ET.iterparse(worksheet, events=("end",))
            rows_read = 0
            for _, element in context:
                if element.tag != f"{XML_NS}row":
                    continue

                row: List[float] = [math.nan] * num_channels
                for cell in element.findall(f"{XML_NS}c"):
                    ref = cell.attrib.get("r")
                    if not ref:
                        continue
                    col_idx = column_index(ref)
                    if col_idx >= num_channels:
                        # Ignore trailing metadata columns if they exist.
                        continue
                    value_node = cell.find(f"{XML_NS}v")
                    if value_node is None or value_node.text is None:
                        continue
                    try:
                        row[col_idx] = float(value_node.text)
                    except ValueError:
                        row[col_idx] = math.nan

                rows_read += 1
                yield row

                # Clear the element to keep memory usage bounded.
                element.clear()

                if max_rows is not None and rows_read >= max_rows:
                    break


def read_excel_signal(
    file_path: Path,
    *,
    num_channels: int,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    """Read the requested number of rows from an Excel file.

    Parameters
    ----------
    file_path:
        Path to the ``.xlsx`` file.
    num_channels:
        Number of channels (columns) to keep.
    max_rows:
        If provided, limit the number of rows that are parsed from the file.

    Returns
    -------
    np.ndarray
        Array with shape ``(num_rows, num_channels)`` containing ``float32``
        values. Rows that contained only missing values are removed.
    """

    rows = [row for row in iter_excel_rows(file_path, num_channels=num_channels, max_rows=max_rows)]
    if not rows:
        raise ValueError(f"No data found inside {file_path}")

    array = np.asarray(rows, dtype=np.float32)
    # Drop rows that are completely NaN, which can happen if the spreadsheet
    # stored trailing empty lines.
    mask = ~np.isnan(array).all(axis=1)
    array = array[mask]
    return array


def window_signal(
    signal: np.ndarray,
    *,
    window_size: int,
    step_size: int,
) -> np.ndarray:
    """Split a multivariate signal into non-overlapping windows.

    The input ``signal`` is expected to have shape ``(num_samples, num_channels)``
    with samples along the first axis. The returned array has shape
    ``(num_windows, num_channels, window_size)`` so that it can be passed
    directly to ``InceptionTime``.
    """

    if signal.shape[0] < window_size:
        return np.empty((0, signal.shape[1], window_size), dtype=np.float32)

    windows: List[np.ndarray] = []
    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        end = start + window_size
        window = signal[start:end]
        windows.append(window.T.astype(np.float32))

    if not windows:
        return np.empty((0, signal.shape[1], window_size), dtype=np.float32)

    return np.stack(windows)


def build_dataset(
    class_folders: Dict[str, Path],
    *,
    window_size: int,
    step_size: int,
    num_channels: int,
    max_windows_per_file: Optional[int] = None,
    max_files_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load every class folder and return a dataset compatible with the model."""

    all_samples: List[np.ndarray] = []
    all_labels: List[int] = []
    label_names: List[str] = []

    for label_index, (label_name, folder) in enumerate(sorted(class_folders.items())):
        label_names.append(label_name)
        files = sorted(folder.glob("*.xlsx"))
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        for file_path in files:
            max_rows = None
            if max_windows_per_file is not None:
                max_rows = window_size * max_windows_per_file
            signal = read_excel_signal(file_path, num_channels=num_channels, max_rows=max_rows)
            windows = window_signal(signal, window_size=window_size, step_size=step_size)
            if windows.size == 0:
                continue
            all_samples.append(windows)
            all_labels.extend([label_index] * windows.shape[0])

    if not all_samples:
        raise RuntimeError("No samples were generated from the provided folders.")

    x = np.concatenate(all_samples, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return x, y, label_names


def infer_on_file(
    model: InceptionTime,
    file_path: Path,
    *,
    window_size: int,
    step_size: int,
    num_channels: int,
    max_windows: Optional[int],
) -> Optional[int]:
    """Return the predicted class index for a single ``.xlsx`` file."""

    max_rows = None if max_windows is None else window_size * max_windows
    signal = read_excel_signal(file_path, num_channels=num_channels, max_rows=max_rows)
    windows = window_signal(signal, window_size=window_size, step_size=step_size)
    if windows.size == 0:
        return None
    predictions = model.predict(windows)
    most_common_label, _ = Counter(predictions).most_common(1)[0]
    return int(most_common_label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InceptionTime on the vibration dataset")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("初赛数据集A") / "A" / "初赛数据集(6种)" / "初赛训练集",
        help="Path to the directory that stores the class-wise training sub-folders.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("初赛数据集A") / "A" / "初赛数据集(6种)" / "初赛测试集",
        help="Path to the directory that stores the official test files.",
    )
    parser.add_argument("--window-size", type=int, default=1024, help="Sliding window length in samples.")
    parser.add_argument("--step-size", type=int, default=1024, help="Sliding window step. Use the same value as window-size for non-overlapping windows.")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of signal channels to use from every Excel file.")
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=10,
        help="Maximum number of consecutive windows to extract from a single training file (limits memory usage).",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Optional limit on the number of Excel files processed per class during training.",
    )
    parser.add_argument(
        "--test-windows",
        type=int,
        default=10,
        help="Maximum number of windows used when aggregating predictions for each test file.",
    )
    parser.add_argument("--filters", type=int, default=32, help="Number of convolutional filters inside InceptionTime blocks.")
    parser.add_argument("--depth", type=int, default=3, help="Number of InceptionTime blocks.")
    parser.add_argument("--models", type=int, default=1, help="Number of ensemble models trained inside InceptionTime.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_results.txt"),
        help="Where to save the predictions for the official test set.",
    )

    args = parser.parse_args()

    class_folders = {
        "inner_broken": args.train_dir / "inner_broken_train150",
        "inner_wear": args.train_dir / "inner_wear_train120",
        "normal": args.train_dir / "normal_train160",
        "outer_missing": args.train_dir / "outer_missing_train180",
        "roller_broken": args.train_dir / "roller_broken_train150",
        "roller_wear": args.train_dir / "roller_wear_train100",
    }

    # Load the training dataset and split it into train/validation subsets.
    x, y, label_names = build_dataset(
        class_folders,
        window_size=args.window_size,
        step_size=args.step_size,
        num_channels=args.num_channels,
        max_windows_per_file=args.max_windows_per_file,
        max_files_per_class=args.max_files_per_class,
    )

    print(f"Generated {x.shape[0]} samples with shape (channels={x.shape[1]}, length={x.shape[2]}).")

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train the InceptionTime model.
    model = InceptionTime(
        x=x_train,
        y=y_train,
        filters=args.filters,
        depth=args.depth,
        models=args.models,
    )

    model.fit(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=True,
    )

    # Evaluate on both the training and validation splits.
    train_predictions = model.predict(x_train)
    valid_predictions = model.predict(x_valid)
    train_accuracy = accuracy_score(y_train, train_predictions)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    valid_f1 = f1_score(y_valid, valid_predictions, average="macro")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {valid_accuracy:.4f}")
    print(f"Validation macro F1: {valid_f1:.4f}")

    # Generate predictions for the official test files.
    test_files = sorted(args.test_dir.glob("*.xlsx"))
    predictions = []
    for file_path in test_files:
        label_index = infer_on_file(
            model,
            file_path,
            window_size=args.window_size,
            step_size=args.step_size,
            num_channels=args.num_channels,
            max_windows=args.test_windows,
        )
        if label_index is None:
            continue
        predictions.append((file_path.stem, label_names[label_index]))

    # Save the predictions in the requested text format.
    with args.output.open("w", encoding="utf-8") as fh:
        fh.write("测试集名称\t故障类型\n")
        for file_name, label_name in predictions:
            fh.write(f"{file_name}\t{label_name}\n")

    print(f"Saved {len(predictions)} predictions to {args.output}.")


if __name__ == "__main__":
    main()
