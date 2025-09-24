from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    random_split = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "初赛数据集A" / "A" / "初赛数据集(6种)"
TRAIN_DIR = DATA_ROOT / "初赛训练集"
TEST_DIR = DATA_ROOT / "初赛测试集"

CLASS_NAME_MAP: Dict[str, str] = {
    "roller_broken_train150": "roller_broken",
    "inner_broken_train150": "inner_broken",
    "normal_train160": "normal",
    "roller_wear_train100": "roller_wear",
    "outer_missing_train180": "outer_missing",
    "inner_wear_train120": "inner_wear",
}

CLASS_LABELS: List[str] = sorted(CLASS_NAME_MAP.values())
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_LABELS)}
INDEX_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_INDEX.items()}

SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on environment
            torch.cuda.manual_seed_all(seed)


def excel_column_to_index(cell_ref: str) -> int:
    col_letters = ""
    for ch in cell_ref:
        if ch.isalpha():
            col_letters += ch.upper()
        else:
            break
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def load_shared_strings(zip_file: ZipFile) -> List[str]:
    try:
        with zip_file.open("xl/sharedStrings.xml") as shared_file:
            values: List[str] = []
            for _, elem in iterparse(shared_file, events=("end",)):
                if elem.tag == f"{SPREADSHEET_NS}t":
                    values.append(elem.text or "")
                elem.clear()
            return values
    except KeyError:
        return []


def read_xlsx_columns(file_path: Path, num_columns: int = 3) -> List[List[float]]:
    with ZipFile(file_path) as archive:
        shared_strings = load_shared_strings(archive)
        columns: List[List[float]] = [[] for _ in range(num_columns)]
        with archive.open("xl/worksheets/sheet1.xml") as sheet_file:
            row_values: List[float] = [0.0] * num_columns
            for event, elem in iterparse(sheet_file, events=("start", "end")):
                tag = elem.tag
                if event == "start" and tag == f"{SPREADSHEET_NS}row":
                    row_values = [0.0] * num_columns
                elif event == "end" and tag == f"{SPREADSHEET_NS}c":
                    cell_ref = elem.attrib.get("r", "A1")
                    col_idx = excel_column_to_index(cell_ref)
                    if 0 <= col_idx < num_columns:
                        cell_type = elem.attrib.get("t")
                        value_element = elem.find(f"{SPREADSHEET_NS}v")
                        text = value_element.text if value_element is not None else ""
                        if cell_type == "s" and text:
                            try:
                                text = shared_strings[int(text)]
                            except (IndexError, ValueError):
                                text = ""
                        try:
                            value = float(text) if text else 0.0
                        except ValueError:
                            value = 0.0
                        row_values[col_idx] = value
                    elem.clear()
                elif event == "end" and tag == f"{SPREADSHEET_NS}row":
                    for idx, value in enumerate(row_values):
                        columns[idx].append(value)
                    elem.clear()
            return columns


def compute_confusion_matrix(true_labels: Sequence[int], pred_labels: Sequence[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(true_labels, pred_labels):
        matrix[true][pred] += 1
    return matrix


def compute_f1_macro(true_labels: Sequence[int], pred_labels: Sequence[int], num_classes: int) -> float:
    f1_values: List[float] = []
    for class_idx in range(num_classes):
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == class_idx and p == class_idx)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != class_idx and p == class_idx)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == class_idx and p != class_idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_values.append(f1)
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


@dataclass
class ModelConfig:
    in_channels: int = 3
    num_classes: int = len(CLASS_LABELS)
    num_blocks: int = 3
    num_filters: int = 32
    bottleneck_channels: int = 32
    kernel_sizes: Tuple[int, int, int] = (9, 19, 39)


@dataclass
class TrainingConfig:
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    output_dir: Path = BASE_DIR / "outputs"
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.8
    patience: int = 10
    limit_per_class: Optional[int] = None
    limit_test_files: Optional[int] = None
    fast_dev_run: bool = False
    num_workers: int = 0
    device: Optional[str] = None
    max_steps_per_epoch: Optional[int] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42


def determine_device(device_str: Optional[str] = None) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch 未安装，无法构建 InceptionTime 模型。")
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gather_training_samples(train_dir: Path, limit_per_class: Optional[int] = None) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for folder_name, class_name in CLASS_NAME_MAP.items():
        class_dir = train_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"训练目录不存在: {class_dir}")
        files = sorted(class_dir.glob("*.xlsx"))
        if limit_per_class is not None:
            files = files[:limit_per_class]
        label = CLASS_TO_INDEX[class_name]
        samples.extend((file_path, label) for file_path in files)
    return samples


def gather_test_samples(test_dir: Path, limit: Optional[int] = None) -> List[Tuple[Path, Optional[int]]]:
    files = sorted(test_dir.glob("*.xlsx"))
    if limit is not None:
        files = files[:limit]
    return [(file_path, None) for file_path in files]


def ensure_sequence_length(samples: Sequence[Tuple[Path, Optional[int]]]) -> int:
    if not samples:
        raise ValueError("数据集中没有任何样本。")
    first_path = samples[0][0]
    columns = read_xlsx_columns(first_path, num_columns=3)
    if not columns:
        raise ValueError(f"文件 {first_path} 中未找到有效数据。")
    return len(columns[0])


if torch is not None:

    class SignalDataset(Dataset):
        def __init__(
            self,
            samples: Sequence[Tuple[Path, Optional[int]]],
            sequence_length: int,
            normalize: bool = True,
            return_file_name: bool = False,
        ) -> None:
            self.samples = list(samples)
            self.sequence_length = sequence_length
            self.normalize = normalize
            self.return_file_name = return_file_name

        def __len__(self) -> int:
            return len(self.samples)

        def _load_signal(self, file_path: Path) -> "torch.Tensor":
            columns = read_xlsx_columns(file_path, num_columns=3)
            processed: List[List[float]] = []
            for channel in columns:
                if len(channel) >= self.sequence_length:
                    cropped = channel[: self.sequence_length]
                else:
                    cropped = channel + [0.0] * (self.sequence_length - len(channel))
                processed.append(cropped)
            tensor = torch.tensor(processed, dtype=torch.float32)
            if self.normalize:
                mean = tensor.mean(dim=1, keepdim=True)
                std = tensor.std(dim=1, keepdim=True)
                tensor = (tensor - mean) / (std + 1e-6)
            return tensor

        def __getitem__(self, index: int) -> Tuple["torch.Tensor", object]:
            file_path, label = self.samples[index]
            tensor = self._load_signal(file_path)
            if self.return_file_name:
                return tensor, file_path.name
            if label is None:
                raise ValueError("训练/验证数据必须包含标签。")
            return tensor, label


    class InceptionModule(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bottleneck_channels: int,
            kernel_sizes: Sequence[int],
        ) -> None:
            super().__init__()
            if bottleneck_channels > 0 and in_channels > bottleneck_channels:
                self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
                reduced_channels = bottleneck_channels
            else:
                self.bottleneck = nn.Identity()
                reduced_channels = in_channels
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        reduced_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
            self.maxpool = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            )
            self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            reduced = self.bottleneck(x)
            outputs = [conv(reduced) for conv in self.convs]
            outputs.append(self.maxpool(x))
            concatenated = torch.cat(outputs, dim=1)
            return self.relu(self.batch_norm(concatenated))


    class InceptionBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bottleneck_channels: int,
            kernel_sizes: Sequence[int],
        ) -> None:
            super().__init__()
            self.module1 = InceptionModule(in_channels, out_channels, bottleneck_channels, kernel_sizes)
            self.module2 = InceptionModule(out_channels * (len(kernel_sizes) + 1), out_channels, bottleneck_channels, kernel_sizes)
            self.module3 = InceptionModule(out_channels * (len(kernel_sizes) + 1), out_channels, bottleneck_channels, kernel_sizes)
            total_out_channels = out_channels * (len(kernel_sizes) + 1)
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, total_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(total_out_channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            x = self.module1(x)
            x = self.module2(x)
            x = self.module3(x)
            if residual.shape[1] != x.shape[1]:
                residual = self.shortcut(residual)
            return self.relu(x + residual)


    class InceptionTime(nn.Module):
        def __init__(
            self,
            in_channels: int,
            num_classes: int,
            num_blocks: int = 3,
            num_filters: int = 32,
            bottleneck_channels: int = 32,
            kernel_sizes: Sequence[int] = (9, 19, 39),
        ) -> None:
            super().__init__()
            blocks: List[nn.Module] = []
            current_channels = in_channels
            for _ in range(num_blocks):
                block = InceptionBlock(
                    in_channels=current_channels,
                    out_channels=num_filters,
                    bottleneck_channels=bottleneck_channels,
                    kernel_sizes=kernel_sizes,
                )
                blocks.append(block)
                current_channels = num_filters * (len(kernel_sizes) + 1)
            self.feature_extractor = nn.Sequential(*blocks)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(current_channels, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.feature_extractor(x)
            pooled = self.global_pool(features).squeeze(-1)
            return self.classifier(pooled)

else:

    class SignalDataset:  # pragma: no cover - only used when dependency is missing
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch 未安装，无法构建数据集。")


    class InceptionTime:  # pragma: no cover - only used when dependency is missing
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch 未安装，无法构建 InceptionTime 模型。")


def train_one_epoch(
    model: "InceptionTime",
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    criterion: nn.Module,
    device: "torch.device",
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    average_loss = total_loss / max(1, total_samples)
    accuracy = (total_correct / max(1, total_samples)) * 100.0
    return {"loss": average_loss, "accuracy": accuracy}


def evaluate(
    model: "InceptionTime",
    loader: "DataLoader",
    criterion: nn.Module,
    device: "torch.device",
    max_steps: Optional[int] = None,
    return_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    steps = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

            all_preds.extend(int(p) for p in preds.cpu().tolist())
            all_targets.extend(int(t) for t in batch_y.cpu().tolist())

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    average_loss = total_loss / max(1, total_samples)
    accuracy = (total_correct / max(1, total_samples)) * 100.0
    f1_macro = compute_f1_macro(all_targets, all_preds, len(CLASS_LABELS)) if all_targets else 0.0
    result: Dict[str, object] = {"loss": average_loss, "accuracy": accuracy, "f1_macro": f1_macro}
    if return_predictions:
        result["predictions"] = all_preds
        result["targets"] = all_targets
    return result


def predict_test_set(
    model: "InceptionTime",
    loader: "DataLoader",
    device: "torch.device",
    max_steps: Optional[int] = None,
) -> List[Tuple[str, str]]:
    model.eval()
    predictions: List[Tuple[str, str]] = []
    steps = 0
    with torch.no_grad():
        for batch_x, file_names in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().tolist()
            for file_name, pred_idx in zip(file_names, preds):
                predictions.append((file_name, INDEX_TO_CLASS[int(pred_idx)]))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    return predictions


def save_predictions(predictions: Sequence[Tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("测试集名称\t故障类型\n")
        for file_name, label in predictions:
            f.write(f"{file_name}\t{label}\n")
    print(f"预测结果已保存至 {output_path}")


def run_training_pipeline(config: TrainingConfig) -> None:
    if torch is None:
        print(
            "未检测到 PyTorch，无法运行 InceptionTime 训练。请在目标环境中安装 PyTorch "
            "(例如 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124)。"
        )
        return

    set_random_seed(config.seed)
    device = determine_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    config_dict["train_dir"] = str(config.train_dir)
    config_dict["test_dir"] = str(config.test_dir)
    config_dict["output_dir"] = str(config.output_dir)
    config_path = config.output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存至 {config_path}")

    train_samples = gather_training_samples(config.train_dir, config.limit_per_class)
    if config.fast_dev_run:
        train_samples = train_samples[: max(1, len(CLASS_LABELS) * 2)]
    sequence_length = ensure_sequence_length(train_samples)

    dataset = SignalDataset(train_samples, sequence_length=sequence_length, normalize=True, return_file_name=False)
    val_size = max(1, int(len(dataset) * (1 - config.train_split)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("训练样本数量不足，无法划分训练/验证集。请增大数据量或调整 train_split。")
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    print(
        "数据集概览: 总样本 {total}, 训练 {train}, 验证 {val}, 每条序列长度 {length}".format(
            total=len(dataset), train=train_size, val=val_size, length=sequence_length
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = InceptionTime(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        num_blocks=config.model.num_blocks,
        num_filters=config.model.num_filters,
        bottleneck_channels=config.model.bottleneck_channels,
        kernel_sizes=config.model.kernel_sizes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    print("开始 InceptionTime 训练...")
    for epoch in range(1, config.epochs + 1):
        train_max_steps = (
            1
            if config.fast_dev_run and config.max_steps_per_epoch is None
            else config.max_steps_per_epoch
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_steps=train_max_steps,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_steps=1 if config.fast_dev_run else None,
        )
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics.get("f1_macro", 0.0))
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | Train Loss {record['train_loss']:.4f} | Train Acc {record['train_accuracy']:.2f}% "
            f"| Val Loss {record['val_loss']:.4f} | Val Acc {record['val_accuracy']:.2f}% | Val F1 {record['val_f1_macro']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if config.fast_dev_run and epoch >= 1:
            print("fast_dev_run 模式下提前结束训练。")
            break
        if epochs_without_improvement >= config.patience:
            print("早停触发，停止训练。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = config.output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练过程指标已保存至 {history_path}")

    model_path = config.output_dir / "inception_time_state.pt"
    torch.save(model.state_dict(), model_path)
    print(f"最优模型参数已保存至 {model_path} (最佳 Epoch: {best_epoch})")

    final_val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        return_predictions=True,
    )
    targets = [int(x) for x in final_val_metrics.get("targets", [])]
    predictions = [int(x) for x in final_val_metrics.get("predictions", [])]
    confusion_matrix = compute_confusion_matrix(targets, predictions, len(CLASS_LABELS))
    metrics_summary = {
        "val_loss": float(final_val_metrics["loss"]),
        "val_accuracy": float(final_val_metrics["accuracy"]),
        "val_f1_macro": float(final_val_metrics.get("f1_macro", 0.0)),
        "confusion_matrix": confusion_matrix,
    }
    metrics_path = config.output_dir / "validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"验证集指标已保存至 {metrics_path}")

    test_samples = gather_test_samples(config.test_dir, config.limit_test_files)
    if test_samples:
        test_dataset = SignalDataset(
            test_samples,
            sequence_length=sequence_length,
            normalize=True,
            return_file_name=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        predictions = predict_test_set(
            model,
            test_loader,
            device,
            max_steps=1 if config.fast_dev_run else None,
        )
        predictions_path = config.output_dir / "predictions.txt"
        save_predictions(predictions, predictions_path)

    label_map_path = config.output_dir / "label_mapping.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(INDEX_TO_CLASS, f, indent=2, ensure_ascii=False)
    print(f"类别映射已保存至 {label_map_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 InceptionTime 进行轴承故障诊断的训练脚本")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批量大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减系数")
    parser.add_argument("--train-split", type=float, default=0.8, help="训练集划分比例")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心轮数")
    parser.add_argument("--limit-per-class", type=int, default=None, help="每个类别最多使用的训练样本数")
    parser.add_argument("--limit-test-files", type=int, default=None, help="测试集最多使用的文件数")
    parser.add_argument("--device", type=str, default=None, help="训练设备 (例如 cuda 或 cpu)")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "outputs"), help="输出目录")
    parser.add_argument("--num-blocks", type=int, default=3, help="InceptionTime 模块堆叠数量")
    parser.add_argument("--num-filters", type=int, default=32, help="每条支路的卷积通道数")
    parser.add_argument("--bottleneck-channels", type=int, default=32, help="瓶颈层通道数")
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=(9, 19, 39),
        help="InceptionTime 分支卷积核大小",
    )
    parser.add_argument("--fast-dev-run", action="store_true", help="快速调试模式，仅运行极少步数")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="每个 epoch 最多训练的 batch 数")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 的工作进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        patience=args.patience,
        limit_per_class=args.limit_per_class,
        limit_test_files=args.limit_test_files,
        fast_dev_run=args.fast_dev_run,
        num_workers=args.num_workers,
        device=args.device,
        max_steps_per_epoch=args.max_steps_per_epoch,
        seed=args.seed,
        model=ModelConfig(
            in_channels=3,
            num_classes=len(CLASS_LABELS),
            num_blocks=args.num_blocks,
            num_filters=args.num_filters,
            bottleneck_channels=args.bottleneck_channels,
            kernel_sizes=tuple(args.kernel_sizes),
        ),
    )
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
