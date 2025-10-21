"""Train and evaluate an InceptionTime model on the competition dataset.

Reads vibration Excel files, builds a windowed dataset, trains an InceptionTime
classifier, evaluates, and runs inference on the official test split.

Only depends on torch/numpy/sklearn. XLSX is parsed via stdlib (zipfile + xml.etree).
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Set
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

from inception_time_pytorch.model import InceptionTime

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# ------------------------------ utils ------------------------------ #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def column_index(cell_reference: str) -> int:
    col_letters = "".join(ch for ch in cell_reference if ch.isalpha())
    index = 0
    for char in col_letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def excel_detect_num_channels(file_path: Path) -> int:
    with zipfile.ZipFile(file_path) as zf:
        with zf.open("xl/worksheets/sheet1.xml") as ws:
            for event, elem in ET.iterparse(ws, events=("start",)):
                if elem.tag == f"{XML_NS}dimension":
                    ref = elem.attrib.get("ref", "")
                    last = ref.split(":")[-1] if ":" in ref else ref
                    return column_index(last) + 1
    with zipfile.ZipFile(file_path) as zf:
        with zf.open("xl/worksheets/sheet1.xml") as ws:
            for _, row in ET.iterparse(ws, events=("end",)):
                if row.tag == f"{XML_NS}row":
                    max_col = 0
                    for cell in row.findall(f"{XML_NS}c"):
                        ref = cell.attrib.get("r")
                        if ref:
                            max_col = max(max_col, column_index(ref) + 1)
                    if max_col > 0:
                        return max_col
                    break
    raise ValueError(f"Cannot detect number of columns for {file_path}")


# ------------------------------ IO (raw) ------------------------------ #
def iter_excel_rows(
    file_path: Path,
    *,
    num_channels: int,
    max_rows: Optional[int] = None,
) -> Iterator[List[float]]:
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
                element.clear()

                if max_rows is not None and rows_read >= max_rows:
                    break


def read_excel_signal(
    file_path: Path,
    *,
    num_channels: int,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    effective_nc = num_channels if num_channels and num_channels > 0 else excel_detect_num_channels(file_path)
    rows = [row for row in iter_excel_rows(file_path, num_channels=effective_nc, max_rows=max_rows)]
    if not rows:
        raise ValueError(f"No data found inside {file_path}")

    array = np.asarray(rows, dtype=np.float32)
    mask = ~np.isnan(array).all(axis=1)
    array = array[mask]

    # per-file column median imputation（不做逐文件z-score）
    med = np.nanmedian(array, axis=0)
    inds = np.where(np.isnan(array))
    if inds[0].size > 0:
        array[inds] = np.take(med, inds[1])
    return array


def window_signal(
    signal: np.ndarray,
    *,
    window_size: int,
    step_size: int,
) -> np.ndarray:
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


# ------------------------------ features / normalization ------------------------------ #
def add_derived_channels(x: np.ndarray, names: List[str]) -> np.ndarray:
    """Add derived channels (using first two physical channels)."""
    if not names:
        return x
    out = [x]
    C = x.shape[1]
    if C >= 2:
        ch0 = x[:, 0:1, :]
        ch1 = x[:, 1:2, :]
        for nm in names:
            if nm == "diff":
                out.append(ch0 - ch1)
            elif nm == "absdiff":
                out.append(np.abs(ch0 - ch1))
            elif nm == "sum":
                out.append(ch0 + ch1)
    return np.concatenate(out, axis=1)


def compute_global_stats(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2))
    std = x_train.std(axis=(0, 2))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(
    x: np.ndarray,
    norm: str,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    std_floor_factor: float = 0.0,
    groups: Optional[List[str]] = None,
) -> np.ndarray:
    """norm: none | global | global_scale | file"""
    if norm == "none":
        return x

    if norm in ("global", "global_scale"):
        assert mean is not None and std is not None
        eff_std = std.copy()
        if std_floor_factor > 0:
            # 防止极小std被放大：以中位数为基准设置下限
            floor = np.median(std) * float(std_floor_factor)
            eff_std = np.maximum(eff_std, floor)
        if norm == "global":
            x -= mean[None, :, None]
        x /= eff_std[None, :, None]
        return x

    if norm == "file":
        assert groups is not None
        x_out = x
        groups_arr = np.asarray(groups)
        uniq = np.unique(groups_arr)
        for g in uniq:
            idx = np.where(groups_arr == g)[0]
            mu = x_out[idx].mean(axis=(0, 2))
            sd = x_out[idx].std(axis=(0, 2))
            sd[sd == 0] = 1.0
            x_out[idx] = (x_out[idx] - mu[None, :, None]) / sd[None, :, None]
        return x_out

    raise ValueError(f"Unknown norm: {norm}")


# ------------------------------ dataset building ------------------------------ #
def build_dataset(
    class_folders: Dict[str, Path],
    *,
    window_size: int,
    step_size: int,
    num_channels: int,
    max_windows_per_file: Optional[int] = None,
    max_files_per_class: Optional[int] = None,
    dense_classes: Optional[Set[str]] = None,
    dense_step: Optional[int] = None,
    dense_max_windows_per_file: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if dense_classes is None:
        dense_classes = set()

    all_samples: List[np.ndarray] = []
    all_labels: List[int] = []
    label_names: List[str] = []
    groups: List[str] = []

    if num_channels <= 0:
        first_file: Optional[Path] = None
        for _, folder in sorted(class_folders.items()):
            files = sorted(folder.glob("*.xlsx"))
            if files:
                first_file = files[0]
                break
        if first_file is None:
            raise RuntimeError("No Excel files found to detect num_channels.")
        detected = excel_detect_num_channels(first_file)
        print(f"[info] Detected num_channels = {detected} from {first_file.name}")
        num_channels = detected

    for label_index, (label_name, folder) in enumerate(sorted(class_folders.items())):
        label_names.append(label_name)
        files = sorted(folder.glob("*.xlsx"))
        if max_files_per_class is not None:
            files = files[:max_files_per_class]

        for file_path in files:
            use_step = dense_step if (label_name in dense_classes and dense_step is not None) else step_size
            cap_windows = (dense_max_windows_per_file if (label_name in dense_classes and dense_max_windows_per_file is not None)
                           else max_windows_per_file)

            signal = read_excel_signal(file_path, num_channels=num_channels, max_rows=None)
            windows = window_signal(signal, window_size=window_size, step_size=use_step)
            if windows.size == 0:
                continue

            if cap_windows is not None and windows.shape[0] > cap_windows:
                idx = np.linspace(0, windows.shape[0] - 1, num=cap_windows, dtype=int)
                windows = windows[idx]

            all_samples.append(windows)
            all_labels.extend([label_index] * windows.shape[0])
            groups.extend([str(file_path)] * windows.shape[0])

    if not all_samples:
        raise RuntimeError("No samples were generated from the provided folders.")

    x = np.concatenate(all_samples, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    return x, y, label_names, groups


# ------------------------------ sampling & augmentation ------------------------------ #
def oversample_minority(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    for c in classes:
        idx = np.where(y == c)[0]
        rep = int(np.ceil(target / len(idx)))
        take = np.tile(idx, rep)[:target]
        xs.append(x[take])
        ys.append(y[take])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def augment_windows(
    x: np.ndarray,
    noise_std: float = 0.02,
    scale_std: float = 0.10,
    max_shift: int = 16,
) -> np.ndarray:
    x_aug = x.copy()
    if scale_std > 0:
        scale = 1.0 + np.random.randn(x.shape[0], x.shape[1], 1).astype(np.float32) * scale_std
        x_aug *= scale
    if noise_std > 0:
        x_aug += np.random.randn(*x.shape).astype(np.float32) * noise_std
    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=(x.shape[0],))
        for i, s in enumerate(shifts):
            if s != 0:
                x_aug[i] = np.roll(x_aug[i], shift=s, axis=1)
    return x_aug


def _tta_windows(windows: np.ndarray, noise_std: float, max_shift: int) -> np.ndarray:
    w = windows.copy()
    if noise_std > 0:
        w += np.random.randn(*w.shape).astype(np.float32) * noise_std
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift != 0:
            w = np.roll(w, shift=shift, axis=2)
    return w


# ------------------------------ inference per file ------------------------------ #
def infer_on_file(
    model: InceptionTime,
    file_path: Path,
    *,
    window_size: int,
    step_size: int,
    num_channels: int,
    max_windows: Optional[int],
    tta: int = 1,
    tta_noise_std: float = 0.0,
    tta_max_shift: int = 0,
    norm: str = "global",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    std_floor_factor: float = 0.0,
    extra_channels: Optional[List[str]] = None,
) -> Optional[int]:
    max_rows = None if max_windows is None else window_size * max_windows
    signal = read_excel_signal(file_path, num_channels=num_channels, max_rows=max_rows)
    windows = window_signal(signal, window_size=window_size, step_size=step_size)
    if windows.size == 0:
        return None

    # 派生通道（和训练一致）
    windows = add_derived_channels(windows, extra_channels or [])

    # 归一化（与训练一致）
    if norm in ("global", "global_scale"):
        windows = apply_norm(windows, norm, mean=global_mean, std=global_std, std_floor_factor=std_floor_factor)
    elif norm == "file":
        mu = windows.mean(axis=(0, 2))
        sd = windows.std(axis=(0, 2))
        sd[sd == 0] = 1.0
        windows = (windows - mu[None, :, None]) / sd[None, :, None]
    elif norm == "none":
        pass
    else:
        raise ValueError(f"Unknown norm: {norm}")

    if hasattr(model, "predict_proba"):
        rounds = max(1, tta)
        agg = None
        for _ in range(rounds):
            w = windows if rounds == 1 else _tta_windows(windows, tta_noise_std, tta_max_shift)
            p = model.predict_proba(w)           # (num_windows, num_classes)
            mp = p.mean(axis=0, keepdims=True)
            agg = mp if agg is None else agg + mp
        mean_prob = (agg / rounds).squeeze(0)
        return int(mean_prob.argmax())
    else:
        predictions = model.predict(windows)
        most_common_label, _ = Counter(predictions).most_common(1)[0]
        return int(most_common_label)


# ------------------------------ main ------------------------------ #
def main() -> None:
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train InceptionTime on the vibration dataset")
    parser.add_argument("--train-dir", type=Path, default=Path("初赛数据集A") / "A" / "初赛数据集(6种)" / "初赛训练集")
    parser.add_argument("--test-dir",  type=Path, default=Path("初赛数据集A") / "A" / "初赛数据集(6种)" / "初赛测试集")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--num-channels", type=int, default=2, help="Use 0 to auto-detect.")
    parser.add_argument("--max-windows-per-file", type=int, default=12)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--test-windows", type=int, default=30)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # Normalization
    parser.add_argument("--norm", type=str, default="global", choices=["none", "global", "global_scale", "file"])
    parser.add_argument("--std-floor-factor", type=float, default=0.6,
                        help="clip per-channel std to at least median(std)*factor to avoid tiny-std explosion")

    # Derived channels
    parser.add_argument("--extra-channels", type=str, default="",
                        help="comma-separated among: diff,absdiff,sum (built from first two channels)")

    # Train balancing & augmentation
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--scale-std", type=float, default=0.0)
    parser.add_argument("--max-shift", type=int, default=16)

    # Dense windowing
    parser.add_argument("--dense-classes", type=str, default="")
    parser.add_argument("--dense-step", type=int, default=256)
    parser.add_argument("--dense-max-windows", type=int, default=24)

    # Validation & inference TTA
    parser.add_argument("--eval-tta", type=int, default=1)
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--tta-noise-std", type=float, default=0.01)
    parser.add_argument("--tta-max-shift", type=int, default=8)

    parser.add_argument("--output", type=Path, default=Path("inference_results.txt"))
    args = parser.parse_args()

    class_folders = {
        "inner_broken": args.train_dir / "inner_broken_train150",
        "inner_wear": args.train_dir / "inner_wear_train120",
        "normal": args.train_dir / "normal_train160",
        "outer_missing": args.train_dir / "outer_missing_train180",
        "roller_broken": args.train_dir / "roller_broken_train150",
        "roller_wear": args.train_dir / "roller_wear_train100",
    }

    dense_names: Set[str] = set([s.strip() for s in args.dense_classes.split(",") if s.strip()])
    extra_names: List[str] = [s.strip() for s in args.extra_channels.split(",") if s.strip()]

    # 1) build raw windows
    if dense_names:
        print(f"[dense] classes={sorted(list(dense_names))}, dense_step={args.dense_step}, dense_cap={args.dense_max_windows}")
    x, y, label_names, groups = build_dataset(
        class_folders,
        window_size=args.window_size,
        step_size=args.step_size,
        num_channels=args.num_channels,
        max_windows_per_file=args.max_windows_per_file,
        max_files_per_class=args.max_files_per_class,
        dense_classes=dense_names,
        dense_step=args.dense_step,
        dense_max_windows_per_file=args.dense_max_windows,
    )
    print(f"Generated {x.shape[0]} samples with shape (channels={x.shape[1]}, length={x.shape[2]}).")

    # 2) add derived channels
    if extra_names:
        x = add_derived_channels(x, extra_names)
        print(f"[features] extra channels {extra_names} -> new shape: (N={x.shape[0]}, C={x.shape[1]}, L={x.shape[2]})")

    # 3) group split by file
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(x, y, groups=groups))
    x_train, x_valid = x[train_idx].copy(), x[valid_idx].copy()
    y_train, y_valid = y[train_idx], y[valid_idx]
    groups_train = [groups[i] for i in train_idx]
    groups_valid = [groups[i] for i in valid_idx]

    # 4) normalization (global with std floor by default)
    global_mean: Optional[np.ndarray] = None
    global_std: Optional[np.ndarray] = None
    if args.norm in ("global", "global_scale"):
        global_mean, global_std = compute_global_stats(x_train)
        x_train = apply_norm(x_train, args.norm, mean=global_mean, std=global_std,
                             std_floor_factor=args.std_floor_factor)
        x_valid = apply_norm(x_valid, args.norm, mean=global_mean, std=global_std,
                             std_floor_factor=args.std_floor_factor)
        print(f"[norm:{args.norm}] mean={global_mean}, std(before floor)={global_std}, floor_factor={args.std_floor_factor}")
    elif args.norm == "file":
        x_train = apply_norm(x_train, "file", groups=groups_train)
        x_valid = apply_norm(x_valid, "file", groups=groups_valid)
        print("[norm:file] applied per-file z-score on train/valid")
    else:
        print("[norm:none] no normalization")

    # 5) oversample & augmentation
    if args.oversample:
        before = np.bincount(y_train, minlength=len(label_names))
        x_train, y_train = oversample_minority(x_train, y_train)
        after = np.bincount(y_train, minlength=len(label_names))
        print(f"[oversample] class counts before: {before.tolist()} -> after: {after.tolist()}")

    if args.augment:
        x_aug = augment_windows(x_train, noise_std=args.noise_std, scale_std=args.scale_std, max_shift=args.max_shift)
        x_train = np.concatenate([x_train, x_aug], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
        print(f"[augment] doubled train windows: {x_train.shape[0]}")

    # 6) train
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

    # 7) eval (supports eval-TTA)
    def _predict_labels(m: InceptionTime, X: np.ndarray) -> np.ndarray:
        if hasattr(m, "predict_proba"):
            if args.eval_tta <= 1:
                return m.predict_proba(X).argmax(axis=1)
            agg = None
            for _ in range(args.eval_tta):
                Xt = _tta_windows(X, args.tta_noise_std, args.tta_max_shift)
                p = m.predict_proba(Xt)
                agg = p if agg is None else agg + p
            return agg.argmax(axis=1)
        return m.predict(X)

    train_predictions = _predict_labels(model, x_train)
    valid_predictions = _predict_labels(model, x_valid)
    train_accuracy = accuracy_score(y_train, train_predictions)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    valid_f1 = f1_score(y_valid, valid_predictions, average="macro")
    print("-" * 41)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {valid_accuracy:.4f}")
    print(f"Validation macro F1: {valid_f1:.4f}")
    print(classification_report(y_valid, valid_predictions, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_valid, valid_predictions))

    # 8) inference on official test
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
            tta=args.tta,
            tta_noise_std=args.tta_noise_std,
            tta_max_shift=args.tta_max_shift,
            norm=args.norm,
            global_mean=global_mean,
            global_std=global_std,
            std_floor_factor=args.std_floor_factor,
            extra_channels=extra_names,
        )
        if label_index is None:
            continue
        predictions.append((file_path.stem, label_names[label_index]))

    with args.output.open("w", encoding="utf-8") as fh:
        fh.write("测试集名称\t故障类型\n")
        for file_name, label_name in predictions:
            fh.write(f"{file_name}\t{label_name}\n")
    print(f"Saved {len(predictions)} predictions to {args.output}.")


if __name__ == "__main__":
    main()
    print(111)
