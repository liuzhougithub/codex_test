"""Dataset utilities for interchangeable classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import torch
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ClassificationDataModule(Protocol):
    """Protocol describing the expected behaviour of data modules."""

    num_classes: int
    input_shape: Sequence[int]

    def train_dataloader(self) -> DataLoader:
        ...

    def val_dataloader(self) -> DataLoader:
        ...

    def test_dataloader(self) -> Optional[DataLoader]:
        ...


@dataclass
class ToyDatasetConfig:
    """Configuration for :class:`ToyClassificationDataModule`."""

    num_samples: int = 2000
    num_features: int = 20
    num_classes: int = 3
    class_sep: float = 1.0
    test_split: float = 0.2
    val_split: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42


class ToyClassificationDataModule:
    """Simple data module that generates synthetic tabular data."""

    def __init__(self, **kwargs: object) -> None:
        cfg = ToyDatasetConfig(**kwargs)
        self.config = cfg

        generator = torch.Generator().manual_seed(cfg.seed)
        data, target = make_classification(
            n_samples=cfg.num_samples,
            n_features=cfg.num_features,
            n_informative=cfg.num_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=cfg.num_classes,
            class_sep=cfg.class_sep,
            random_state=cfg.seed,
        )

        tensor_x = torch.tensor(data, dtype=torch.float32)
        tensor_y = torch.tensor(target, dtype=torch.long)

        test_size = int(cfg.num_samples * cfg.test_split)
        val_size = int(cfg.num_samples * cfg.val_split)
        train_size = cfg.num_samples - test_size - val_size
        if train_size <= 0:
            raise ValueError("Train split must contain at least one sample.")

        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            TensorDataset(tensor_x, tensor_y),
            lengths=[train_size, val_size, test_size],
            generator=generator,
        )

        self._train_loader = self._create_loader(self.train_set, shuffle=True)
        self._val_loader = self._create_loader(self.val_set, shuffle=False)
        self._test_loader = self._create_loader(self.test_set, shuffle=False)

        self.num_classes = cfg.num_classes
        self.input_shape = (cfg.num_features,)

    def _create_loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def test_dataloader(self) -> Optional[DataLoader]:
        return self._test_loader


def to_device(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move batch tensors to the given device."""

    inputs, targets = batch
    return inputs.to(device), targets.to(device)
