"""Training script for gearbox fault diagnosis using a simple residual network."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
    load_numpy_dataset,
    save_numpy_dataset,
)
from models.simple_resnet import ResNetConfig, SimpleResNet


plt.switch_backend("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("初赛数据集A/A/初赛数据集(6种)/初赛训练集"), help="Root directory containing class folders.")
    parser.add_argument("--cache", type=Path, default=Path("processed/dataset.npz"), help="Optional cache path for processed dataset.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated experiments.")
    parser.add_argument("--tsne", action="store_true", help="Whether to produce a TSNE visualization from the test set features.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached dataset even if it exists.")
    return parser.parse_args()


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
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    """Train the model while printing epoch loss."""

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = 0
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
        avg_loss = epoch_loss / max(total_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}")


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
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
            feature_storage.append(logits.cpu().numpy())

    all_labels_array = np.array(all_labels)
    all_preds_array = np.array(all_preds)
    accuracy = (all_labels_array == all_preds_array).mean() * 100
    f1_macro = f1_score(all_labels_array, all_preds_array, average="macro")
    features = np.concatenate(feature_storage, axis=0)
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


def main() -> None:
    args = parse_args()

    samples, labels, label_mapping = prepare_dataset(args)
    class_names = [name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    seq_len = samples.shape[1]
    input_channels = samples.shape[2]

    train_loader, test_loader, *_ = create_dataloaders(samples, labels, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前使用的设备:", device)

    acc_list: List[float] = []
    f1_list: List[float] = []
    features_runs: List[np.ndarray] = []
    labels_runs: List[np.ndarray] = []

    for run in range(1, args.num_runs + 1):
        print(f"\n==== 第 {run} 次训练 ====")
        config = ResNetConfig(input_channels=input_channels, num_classes=len(class_names))
        model = SimpleResNet(config, seq_len=seq_len).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        print(f"模型参数量: {model.count_parameters():,}")
        train_model(model, train_loader, criterion, optimizer, device, epochs=args.epochs)

        accuracy, f1_macro, all_labels, all_preds, features = evaluate_model(model, test_loader, device)
        acc_list.append(accuracy)
        f1_list.append(f1_macro)
        features_runs.append(features)
        labels_runs.append(all_labels)
        print(f"第{run}次评估结果: Accuracy={accuracy:.2f}%, F1-macro={f1_macro:.4f}")

        cm_path = Path("outputs") / f"confusion_matrix_run_{run}.png"
        plot_confusion(all_labels, all_preds, class_names, cm_path)
        print(f"混淆矩阵已保存至 {cm_path}")

        if args.tsne:
            tsne_path = Path("outputs") / f"tsne_run_{run}.png"
            plot_tsne(features, all_labels, class_names, tsne_path)
            print(f"t-SNE 图已保存至 {tsne_path}")

    print("\n==== 三次训练结果平均值 ====")
    for idx, (acc, f1_macro) in enumerate(zip(acc_list, f1_list), start=1):
        print(f"Run {idx}: Accuracy={acc:.2f}%, F1-macro={f1_macro:.4f}")
    print(f"\n平均 Accuracy = {np.mean(acc_list):.2f}%")
    print(f"平均 F1-macro = {np.mean(f1_list):.4f}")

    if args.tsne and features_runs:
        avg_features = np.concatenate(features_runs, axis=0)
        avg_labels = np.concatenate(labels_runs, axis=0)
        tsne_path = Path("outputs") / "tsne_overall.png"
        plot_tsne(avg_features, avg_labels, class_names, tsne_path)
        print(f"平均 t-SNE 图已保存至 {tsne_path}")


if __name__ == "__main__":
    main()
