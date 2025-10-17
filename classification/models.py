"""Model definitions for interchangeable classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class MLPConfig:
    """Configuration for :class:`MLPClassifier`."""

    in_features: int
    hidden_dims: Sequence[int] = (128, 64)
    num_classes: int = 2
    dropout: float = 0.0


class MLPClassifier(nn.Module):
    """Simple fully connected classifier."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        cfg = MLPConfig(**kwargs)  # type: ignore[arg-type]

        layers: list[nn.Module] = []
        input_dim = cfg.in_features
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, cfg.num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
