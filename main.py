import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

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
                        if cell_type == "s":
                            if text:
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


def compute_moment_features(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return [0.0] * 12

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    sabs = 0.0
    min_v = float("inf")
    max_v = float("-inf")
    max_abs = 0.0

    for x in values:
        s1 += x
        s2 += x * x
        s3 += x * x * x
        s4 += x * x * x * x
        sabs += abs(x)
        if x < min_v:
            min_v = x
        if x > max_v:
            max_v = x
        ax = abs(x)
        if ax > max_abs:
            max_abs = ax

    mean = s1 / n
    mean_sq = s2 / n
    variance = max(mean_sq - mean * mean, 0.0)
    std = math.sqrt(variance)
    rms = math.sqrt(mean_sq)
    abs_mean = sabs / n

    mu3 = s3 / n
    mu4 = s4 / n
    central_m3 = mu3 - 3 * mean * mean_sq + 2 * mean ** 3
    central_m4 = mu4 - 4 * mean * mu3 + 6 * (mean ** 2) * mean_sq - 3 * mean ** 4

    skew = central_m3 / (std ** 3) if std > 1e-12 else 0.0
    kurt = central_m4 / (std ** 4) if std > 1e-12 else 0.0

    crest_factor = max_abs / rms if rms > 1e-12 else 0.0
    impulse_factor = max_abs / abs_mean if abs_mean > 1e-12 else 0.0
    shape_factor = rms / abs_mean if abs_mean > 1e-12 else 0.0

    features = [
        mean,
        std,
        min_v,
        max_v,
        abs_mean,
        rms,
        mean_sq,
        max_v - min_v,
        skew,
        kurt,
        crest_factor,
        impulse_factor,
        shape_factor,
    ]
    return features


def extract_features_from_file(file_path: Path) -> List[float]:
    columns = read_xlsx_columns(file_path, num_columns=3)
    features: List[float] = []
    for column in columns:
        features.extend(compute_moment_features(column))
    return features


def load_training_data(train_dir: Path) -> Tuple[List[List[float]], List[int]]:
    feature_rows: List[List[float]] = []
    label_rows: List[int] = []
    for folder_name, class_name in CLASS_NAME_MAP.items():
        class_dir = train_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"训练目录不存在: {class_dir}")
        for file_path in sorted(class_dir.glob("*.xlsx")):
            features = extract_features_from_file(file_path)
            feature_rows.append(features)
            label_rows.append(CLASS_TO_INDEX[class_name])
    return feature_rows, label_rows


def load_test_data(test_dir: Path) -> Tuple[List[List[float]], List[str]]:
    feature_rows: List[List[float]] = []
    file_names: List[str] = []
    for file_path in sorted(test_dir.glob("*.xlsx")):
        features = extract_features_from_file(file_path)
        feature_rows.append(features)
        file_names.append(file_path.name)
    return feature_rows, file_names


@dataclass
class GaussianClassStats:
    prior_log_prob: float
    mean: List[float]
    variance: List[float]


class GaussianNaiveBayes:
    def __init__(self, num_classes: int, num_features: int, epsilon: float = 1e-6) -> None:
        self.num_classes = num_classes
        self.num_features = num_features
        self.epsilon = epsilon
        self.class_stats: List[GaussianClassStats] = []

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        counts = [0] * self.num_classes
        sums = [[0.0 for _ in range(self.num_features)] for _ in range(self.num_classes)]
        sumsq = [[0.0 for _ in range(self.num_features)] for _ in range(self.num_classes)]

        for row, label in zip(features, labels):
            counts[label] += 1
            for i, value in enumerate(row):
                sums[label][i] += value
                sumsq[label][i] += value * value

        total = sum(counts)
        stats: List[GaussianClassStats] = []
        for class_idx in range(self.num_classes):
            count = counts[class_idx]
            if count == 0:
                prior_log_prob = math.log(1.0 / self.num_classes)
                mean = [0.0] * self.num_features
                variance = [1.0] * self.num_features
            else:
                prior_log_prob = math.log(count / total)
                mean = [s / count for s in sums[class_idx]]
                variance = []
                for i in range(self.num_features):
                    mean_value = mean[i]
                    avg_square = sumsq[class_idx][i] / count
                    var = avg_square - mean_value * mean_value
                    if var < self.epsilon:
                        var = self.epsilon
                    variance.append(var)
            stats.append(GaussianClassStats(prior_log_prob, mean, variance))
        self.class_stats = stats

    def predict_row(self, row: Sequence[float]) -> int:
        best_class = 0
        best_log_prob = float("-inf")
        for class_idx, stat in enumerate(self.class_stats):
            log_prob = stat.prior_log_prob
            for feature_value, mean, variance in zip(row, stat.mean, stat.variance):
                diff = feature_value - mean
                log_prob -= 0.5 * math.log(2 * math.pi * variance)
                log_prob -= (diff * diff) / (2 * variance)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = class_idx
        return best_class

    def predict(self, features: Sequence[Sequence[float]]) -> List[int]:
        return [self.predict_row(row) for row in features]


@dataclass
class EvaluationResult:
    accuracy: float
    f1_macro: float
    confusion_matrix: List[List[int]]


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


def evaluate_predictions(true_labels: Sequence[int], pred_labels: Sequence[int], num_classes: int) -> EvaluationResult:
    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels) * 100.0
    f1_macro = compute_f1_macro(true_labels, pred_labels, num_classes)
    matrix = compute_confusion_matrix(true_labels, pred_labels, num_classes)
    return EvaluationResult(accuracy=accuracy, f1_macro=f1_macro, confusion_matrix=matrix)


def run_cross_validation(features: List[List[float]], labels: List[int], runs: int = 3, train_ratio: float = 0.8) -> None:
    num_samples = len(features)
    indices = list(range(num_samples))
    for run in range(1, runs + 1):
        seed = 42 + run
        random.Random(seed).shuffle(indices)
        split = int(num_samples * train_ratio)
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_features = [features[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]

        model = GaussianNaiveBayes(num_classes=len(CLASS_LABELS), num_features=len(features[0]))
        model.fit(train_features, train_labels)
        preds = model.predict(val_features)
        result = evaluate_predictions(val_labels, preds, len(CLASS_LABELS))

        print(f"\n==== 第 {run} 次训练 ====")
        print(f"验证集 Accuracy = {result.accuracy:.2f}%")
        print(f"验证集 F1-macro = {result.f1_macro:.4f}")
        print("混淆矩阵:")
        for row in result.confusion_matrix:
            print("\t" + " ".join(f"{value:4d}" for value in row))

    print("\n==== 三次训练结果平均值 ====")
    # 重新计算平均值
    acc_values: List[float] = []
    f1_values: List[float] = []
    for run in range(1, runs + 1):
        seed = 42 + run
        random.Random(seed).shuffle(indices)
        split = int(num_samples * train_ratio)
        train_indices = indices[:split]
        val_indices = indices[split:]
        train_features = [features[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_features = [features[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        model = GaussianNaiveBayes(num_classes=len(CLASS_LABELS), num_features=len(features[0]))
        model.fit(train_features, train_labels)
        preds = model.predict(val_features)
        result = evaluate_predictions(val_labels, preds, len(CLASS_LABELS))
        acc_values.append(result.accuracy)
        f1_values.append(result.f1_macro)
        print(f"Run {run}: Accuracy={result.accuracy:.2f}%, F1-macro={result.f1_macro:.4f}")
    mean_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0
    mean_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    print(f"\n平均 Accuracy = {mean_acc:.2f}%")
    print(f"平均 F1-macro = {mean_f1:.4f}")


def train_full_model(features: List[List[float]], labels: List[int]) -> GaussianNaiveBayes:
    model = GaussianNaiveBayes(num_classes=len(CLASS_LABELS), num_features=len(features[0]))
    model.fit(features, labels)
    return model


def predict_test_set(model: GaussianNaiveBayes, features: List[List[float]], file_names: Sequence[str]) -> List[Tuple[str, str]]:
    predictions = model.predict(features)
    return [(file_name, INDEX_TO_CLASS[pred]) for file_name, pred in zip(file_names, predictions)]


def save_predictions(predictions: Sequence[Tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("测试集名称\t故障类型\n")
        for file_name, label in predictions:
            f.write(f"{file_name}\t{label}\n")
    print(f"预测结果已保存至 {output_path}")


def main() -> None:
    set_random_seed(42)
    print("开始加载训练数据...")
    train_features, train_labels = load_training_data(TRAIN_DIR)
    print(f"训练样本数: {len(train_features)}, 特征维度: {len(train_features[0])}")

    run_cross_validation(train_features, train_labels)

    print("\n开始加载测试数据...")
    test_features, test_file_names = load_test_data(TEST_DIR)

    model = train_full_model(train_features, train_labels)
    predictions = predict_test_set(model, test_features, test_file_names)
    save_predictions(predictions, BASE_DIR / "predictions.txt")


if __name__ == "__main__":
    main()
