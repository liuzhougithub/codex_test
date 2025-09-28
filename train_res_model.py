"""Training script for gearbox fault diagnosis using a simple residual network."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.manifold import TSNE

from data_processing import (
    assemble_dataset,
    create_dataloaders,
    load_inference_windows,
    load_numpy_dataset,
    save_numpy_dataset,
)
from models.batchtst import BatchTSTConfig, BatchTSTNet
from models.configurable_convnet import ConfigurableConvNet, ConfigurableConvNetConfig
from models.inception_time import InceptionTimeConfig, InceptionTimeNet
from models.simple_resnet import ResNetConfig, SimpleResNet


plt.switch_backend("Agg")


@dataclass
class TrainingHyperParameters:
    """Central place to tweak training behaviour.

    Attributes
    ----------
    batch_size:
        每个 mini-batch 包含的样本数；增大能提高吞吐但需要更多显存/内存。
    epochs:
        针对一次训练运行的完整数据遍历次数。
    learning_rate:
        Adam 优化器的基础学习率，决定参数更新幅度。
    weight_decay:
        L2 正则化强度，用于抑制过拟合（Adam 的 ``weight_decay`` 参数）。
    test_size:
        划分为最终测试集的数据比例。
    val_size:
        划分为验证集的数据比例（从训练剩余部分中切分，用于调参与早停）。
    random_state:
        控制 ``train_test_split`` 随机性的随机种子，便于结果复现。
    """

    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


HYPERPARAMETER_DESCRIPTIONS: Dict[str, str] = {
    "batch_size": "每个迭代喂入网络的样本数量，影响训练稳定性和显存占用。",
    "epochs": "一次 run 内训练集被完整遍历的次数。",
    "learning_rate": "优化器更新步长，过大易发散，过小收敛慢。",
    "weight_decay": "对网络权重的L2惩罚系数，用于正则化。",
    "test_size": "整体数据中用于最终评估的比例。",
    "val_size": "整体数据中用于验证调参的比例。",
    "random_state": "划分训练/验证/测试集时的随机种子。",
}


def load_model_configuration(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load optional JSON model configuration from disk."""

    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件 {config_path} 不存在。")
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("模型配置文件需要是一个 JSON 对象。")
    return data


def create_model(
    model_name: str,
    model_config: Dict[str, Any],
    input_channels: int,
    num_classes: int,
    seq_len: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate the requested model architecture."""

    if model_name == "resnet":
        config = ResNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            base_filters=int(model_config.get("base_filters", 32)),
            num_blocks=int(model_config.get("num_blocks", 3)),
            kernel_size=int(model_config.get("kernel_size", 7)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        model = SimpleResNet(config, seq_len)
        return model, asdict(config)

    if model_name == "inceptiontime":
        kernel_sizes = tuple(int(k) for k in model_config.get("kernel_sizes", (9, 19, 39)))
        config = InceptionTimeConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            num_blocks=int(model_config.get("num_blocks", 6)),
            in_channels=int(model_config.get("in_channels", 32)),
            bottleneck_channels=int(model_config.get("bottleneck_channels", 32)),
            kernel_sizes=kernel_sizes,
            use_residual=bool(model_config.get("use_residual", True)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        model = InceptionTimeNet(config, seq_len)
        return model, asdict(config)

    if model_name == "batchtst":
        config = BatchTSTConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            patch_len=int(model_config.get("patch_len", 32)),
            stride=int(model_config.get("stride", 16)),
            d_model=int(model_config.get("d_model", 128)),
            num_heads=int(model_config.get("num_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        model = BatchTSTNet(config, seq_len)
        return model, asdict(config)

    if model_name == "custom":
        config = ConfigurableConvNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            layers=model_config.get("layers", []),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        model = ConfigurableConvNet(config, seq_len)
        config_dict = asdict(config)
        # layers may contain non-serialisable objects; ensure pure python types
        config_dict["layers"] = model_config.get("layers", config.layers)
        return model, config_dict

    raise ValueError(f"未知的模型类型: {model_name}")


def describe_model(name: str, config: Dict[str, Any]) -> None:
    print(f"\n==== 当前模型: {name} ====")
    for key, value in config.items():
        print(f"{key:>16}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("初赛数据集A/A/初赛数据集(6种)/初赛训练集"), help="Root directory containing class folders.")
    parser.add_argument("--cache", type=Path, default=Path("processed/dataset.npz"), help="Optional cache path for processed dataset.")
    parser.add_argument("--model", choices=["resnet", "inceptiontime", "batchtst", "custom"], default="resnet", help="选择训练使用的网络结构。")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="自定义模型或覆盖默认结构的 JSON 配置文件路径。",
    )
    parser.add_argument("--batch-size", type=int, default=TrainingHyperParameters.batch_size, help="Override default batch size.")
    parser.add_argument("--epochs", type=int, default=TrainingHyperParameters.epochs, help="Training epochs per run.")
    parser.add_argument("--learning-rate", type=float, default=TrainingHyperParameters.learning_rate, help="Learning rate for Adam optimizer.")
    parser.add_argument("--weight-decay", type=float, default=TrainingHyperParameters.weight_decay, help="Weight decay (L2 regularisation strength).")
    parser.add_argument("--test-size", type=float, default=TrainingHyperParameters.test_size, help="Fraction reserved for test set.")
    parser.add_argument("--val-size", type=float, default=TrainingHyperParameters.val_size, help="Fraction reserved for validation set.")
    parser.add_argument("--random-state", type=int, default=TrainingHyperParameters.random_state, help="Seed for deterministic dataset splits.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated experiments.")
    parser.add_argument("--tsne", action="store_true", help="Whether to produce a TSNE visualization from the test set features.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached dataset even if it exists.")
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=Path("初赛数据集A/A/初赛测试集"),
        help="真实测试集所在的文件夹路径。",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("outputs/test_predictions.txt"),
        help="推理结果保存的 txt 文件路径。",
    )
    parser.add_argument("--skip-inference", action="store_true", help="只训练评估，不对真实测试集进行推理。")
    return parser.parse_args()


def build_hyperparameters(args: argparse.Namespace) -> TrainingHyperParameters:
    return TrainingHyperParameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )


def describe_hyperparameters(config: TrainingHyperParameters) -> None:
    print("\n==== 当前可调训练超参数 ====")
    for key, value in asdict(config).items():
        description = HYPERPARAMETER_DESCRIPTIONS.get(key, "")
        print(f"{key:>12}: {value:<10} -> {description}")


def discover_class_directories(data_root: Path) -> Dict[str, Path]:
    class_dirs: Dict[str, Path] = {}
    for item in sorted(data_root.iterdir()):
        if item.is_dir():
            class_dirs[item.name] = item
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {data_root}.")
    return class_dirs


def prepare_dataset(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if args.cache.exists() and not args.no_cache:
        print(f"Loading cached dataset from {args.cache}...")
        return load_numpy_dataset(args.cache)

    class_dirs = discover_class_directories(args.data_root)
    samples, labels, label_mapping = assemble_dataset(class_dirs)
    save_numpy_dataset(args.cache, samples, labels, label_mapping)
    return samples, labels, label_mapping


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train the model while tracking the best validation performance.

    Returns
    -------
    best_metrics: Dict[str, float]
        记录最佳验证集表现的指标。
    history: Dict[str, float]
        包含最终一个 epoch 的训练损失等信息，用于日志打印。
    """

    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {"val_accuracy": 0.0, "val_f1": 0.0, "epoch": 0, "train_accuracy": 0.0}
    history = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)
        avg_loss = epoch_loss / max(total_batches, 1)
        train_accuracy = (train_correct / max(train_total, 1)) * 100

        val_accuracy, val_f1, *_ = evaluate_model(model, val_loader, device, return_features=False)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}, TrainAcc={train_accuracy:.2f}%, "
            f"ValAcc={val_accuracy:.2f}%, ValF1={val_f1:.4f}"
        )

        if val_f1 > best_metrics["val_f1"]:
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "epoch": epoch + 1,
                "train_accuracy": train_accuracy,
            }

        history = {
            "epoch_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "train_accuracy": train_accuracy,
        }

    model.load_state_dict(best_state)
    return best_metrics, history


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_features: bool = True,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluate accuracy and macro F1 score on the provided loader."""

    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    feature_storage: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(targets.cpu().numpy().tolist())
            all_preds.extend(predictions.cpu().numpy().tolist())
            if return_features:
                feature_storage.append(logits.cpu().numpy())

    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    accuracy = (all_labels_array == all_preds_array).mean() * 100
    f1_macro = f1_score(all_labels_array, all_preds_array, average="macro")
    features: Optional[np.ndarray]
    if return_features and feature_storage:
        features = np.concatenate(feature_storage, axis=0)
    else:
        features = None
    return accuracy, f1_macro, all_labels_array, all_preds_array, features


def plot_confusion(all_labels: np.ndarray, all_preds: np.ndarray, class_names: List[str], output_path: Path) -> None:
    matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_tsne(features: np.ndarray, labels: np.ndarray, class_names: List[str], output_path: Path) -> None:
    tsne = TSNE(n_components=2, init="random", random_state=42, perplexity=30)
    embedding = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        plt.scatter(embedding[mask, 0], embedding[mask, 1], label=class_name, alpha=0.6)
    plt.legend()
    plt.title("t-SNE of Test Set Features")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def run_inference(
    model: nn.Module,
    device: torch.device,
    inference_data: Dict[str, np.ndarray],
    class_names: List[str],
    batch_size: int,
    output_path: Path,
) -> None:
    """Generate predictions for unseen test files and save them to ``output_path``."""

    model.eval()
    predictions: Dict[str, str] = {}
    with torch.no_grad():
        for file_stem, windows in inference_data.items():
            tensor = torch.tensor(windows, dtype=torch.float32)
            loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, shuffle=False)
            probs: List[torch.Tensor] = []
            for batch in loader:
                inputs = batch.permute(0, 2, 1).to(device)
                logits = model(inputs)
                probs.append(torch.softmax(logits, dim=1).cpu())
            if not probs:
                continue
            mean_prob = torch.cat(probs, dim=0).mean(dim=0)
            predicted_index = int(torch.argmax(mean_prob).item())
            predictions[file_stem] = class_names[predicted_index]
            print(f"推理结果 -> {file_stem}: {predictions[file_stem]}")

    with output_path.open("w", encoding="utf-8") as writer:
        for file_name in sorted(predictions.keys()):
            writer.write(f"{file_name}\t{predictions[file_name]}\n")
    print(f"真实测试集推理结果已保存至 {output_path}")


def main() -> None:
    args = parse_args()

    samples, labels, label_mapping = prepare_dataset(args)
    class_names = [name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    seq_len = samples.shape[1]
    input_channels = samples.shape[2]

    hyperparams = build_hyperparameters(args)
    describe_hyperparameters(hyperparams)

    model_config = load_model_configuration(args.model_config)
    preview_model, model_details = create_model(
        args.model,
        model_config,
        input_channels,
        len(class_names),
        seq_len,
    )
    describe_model(args.model, model_details)
    print(f"模型参数量(初始化预览): {preview_model.count_parameters():,}")
    del preview_model

    (
        train_loader,
        val_loader,
        test_loader,
        *_
    ) = create_dataloaders(
        samples,
        labels,
        batch_size=hyperparams.batch_size,
        test_size=hyperparams.test_size,
        val_size=hyperparams.val_size,
        random_state=hyperparams.random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用的设备:", device)

    acc_list: List[float] = []
    f1_list: List[float] = []
    best_val_runs: List[Dict[str, float]] = []
    features_runs: List[np.ndarray] = []
    labels_runs: List[np.ndarray] = []

    best_overall_state: Optional[Dict[str, torch.Tensor]] = None
    best_overall_metrics: Optional[Dict[str, float]] = None

    for run in range(1, args.num_runs + 1):
        print(f"\n==== 第 {run} 次训练 ====")
        model, _ = create_model(
            args.model,
            model_config,
            input_channels,
            len(class_names),
            seq_len,
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams.learning_rate,
            weight_decay=hyperparams.weight_decay,
        )

        print(f"模型参数量: {model.count_parameters():,}")
        best_metrics, _ = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=hyperparams.epochs,
        )
        print(
            "最佳验证表现 -> "
            f"Epoch={best_metrics['epoch']}, TrainAcc={best_metrics['train_accuracy']:.2f}%, "
            f"ValAcc={best_metrics['val_accuracy']:.2f}%, ValF1={best_metrics['val_f1']:.4f}"
        )
        best_val_runs.append(best_metrics.copy())

        if (best_overall_metrics is None) or (best_metrics["val_f1"] > best_overall_metrics["val_f1"]):
            best_overall_metrics = best_metrics.copy()
            best_overall_state = copy.deepcopy(model.state_dict())

        accuracy, f1_macro, all_labels, all_preds, features = evaluate_model(model, test_loader, device)
        acc_list.append(accuracy)
        f1_list.append(f1_macro)
        if features is not None:
            features_runs.append(features)
        labels_runs.append(all_labels)
        print(f"第{run}次评估结果: Accuracy={accuracy:.2f}%, F1-macro={f1_macro:.4f}")

        cm_path = Path("outputs") / f"confusion_matrix_run_{run}.png"
        plot_confusion(all_labels, all_preds, class_names, cm_path)
        print(f"混淆矩阵已保存至 {cm_path}")

        if args.tsne and features is not None:
            tsne_path = Path("outputs") / f"tsne_run_{run}.png"
            plot_tsne(features, all_labels, class_names, tsne_path)
            print(f"t-SNE 图已保存至 {tsne_path}")

    print("\n==== 每次训练的最佳验证表现 ====")
    for idx, metrics in enumerate(best_val_runs, start=1):
        print(
            f"Run {idx}: BestEpoch={metrics['epoch']}, TrainAcc={metrics['train_accuracy']:.2f}%, "
            f"ValAcc={metrics['val_accuracy']:.2f}%, ValF1={metrics['val_f1']:.4f}"
        )

    print("\n==== 测试集指标平均值 ====")
    for idx, (acc, f1_macro) in enumerate(zip(acc_list, f1_list), start=1):
        print(f"Run {idx}: Accuracy={acc:.2f}%, F1-macro={f1_macro:.4f}")
    print(f"平均 Accuracy = {np.mean(acc_list):.2f}%")
    print(f"平均 F1-macro = {np.mean(f1_list):.4f}")

    if args.tsne and features_runs:
        avg_features = np.concatenate(features_runs, axis=0)
        avg_labels = np.concatenate(labels_runs, axis=0)
        tsne_path = Path("outputs") / "tsne_overall.png"
        plot_tsne(avg_features, avg_labels, class_names, tsne_path)
        print(f"平均 t-SNE 图已保存至 {tsne_path}")

    if not args.skip_inference:
        if best_overall_state is None:
            print("没有可用于推理的训练权重，已跳过真实测试集预测。")
        else:
            inference_data = load_inference_windows(args.inference_root)
            if not inference_data:
                print(f"在 {args.inference_root} 下未找到可用的测试文件，推理已跳过。")
            else:
                prediction_path = args.prediction_output
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                inference_model, _ = create_model(
                    args.model,
                    model_config,
                    input_channels,
                    len(class_names),
                    seq_len,
                )
                inference_model.load_state_dict(best_overall_state)
                inference_model = inference_model.to(device)
                run_inference(
                    inference_model,
                    device,
                    inference_data,
                    class_names,
                    args.batch_size,
                    prediction_path,
                )


if __name__ == "__main__":
    main()
