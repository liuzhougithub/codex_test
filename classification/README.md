# 通用分类模型训练框架

本目录提供了一个以 PyTorch 为核心、组件可插拔的通用深度学习分类训练脚手架。通过配置文件即可切换不同的数据加载器、模型和优化器，方便快速实验。

## 目录结构

- `config.py`：定义训练、数据和模型相关的配置数据类，以及 JSON 配置文件的加载逻辑。
- `data.py`：包含数据模块协议定义和一个基于 `sklearn.make_classification` 的示例数据模块 `ToyClassificationDataModule`。
- `models.py`：给出简单的 MLP 分类器示例实现，可替换为任意自定义模型。
- `trainer.py`：训练循环的核心实现，负责模型训练、验证、测试及模型保存。
- `__init__.py`：便于外部直接导入关键组件。

## 快速开始

1. 安装依赖（若尚未安装）：

   ```bash
   pip install -r requirements.txt
   ```

2. 查看或修改示例配置文件 `configs/toy_classification.json`，配置项包括：
   - `dataset.target`：数据模块的导入路径，默认为 `classification.data.ToyClassificationDataModule`。
   - `model.target`：模型类的导入路径，默认为 `classification.models.MLPClassifier`。
   - `optimizer.target`：优化器的导入路径（例如 `torch.optim.Adam`）。
   - `params`：对应模块初始化时需要的参数，采用 JSON 格式传入。

3. 运行训练：

   ```bash
   python train_classifier.py --config configs/toy_classification.json
   ```

   训练完成后会在 `output_dir` 指定的目录下生成最优模型的 `best_model.pt`。

## 替换数据模块

要接入自定义数据集，只需编写满足 `ClassificationDataModule` 协议的数据模块。例如：

```python
from torch.utils.data import DataLoader

class MyDataModule:
    num_classes = 10
    input_shape = (3, 32, 32)

    def __init__(self, data_dir: str, batch_size: int = 64) -> None:
        self._train_loader = DataLoader(...)
        self._val_loader = DataLoader(...)
        self._test_loader = DataLoader(...)

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def test_dataloader(self) -> DataLoader:
        return self._test_loader
```

将类保存为 `my_project/data.py` 后，在配置文件中设置：

```json
"dataset": {
  "target": "my_project.data.MyDataModule",
  "params": {"data_dir": "./data", "batch_size": 64}
}
```

## 替换模型

自定义模型只需继承 `torch.nn.Module`，接收配置文件中的参数，输出形状为 `[batch_size, num_classes]` 的张量即可。例如：

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
```

配置文件中设置：

```json
"model": {
  "target": "my_project.models.MyNet",
  "params": {"in_features": 32, "num_classes": 4}
}
```

## 学习率调度器

可选地配置学习率调度器：

```json
"scheduler": {
  "target": "torch.optim.lr_scheduler.StepLR",
  "params": {"step_size": 5, "gamma": 0.5}
}
```

只要对应的类在运行环境中可导入，训练器会自动实例化并在每个 epoch 末调用 `step()`。

## 输出

- `best_model.pt`：保存验证集上性能最优的模型及优化器状态，便于继续训练或推理。
- 终端日志：包含每个 epoch 的训练/验证损失与准确率，以及测试集评估结果（若提供测试集）。

通过这种模块化的设计，可以快速尝试不同的数据加载方式、模型结构和优化策略，构建适用于多种任务的分类训练流程。
