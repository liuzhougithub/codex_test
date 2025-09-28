"""Simple residual network for time series classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ResNetConfig:
    """Configuration container for :class:`SimpleResNet`."""

    input_channels: int
    num_classes: int
    base_filters: int = 32
    num_blocks: int = 3
    kernel_size: int = 7
    dropout: float = 0.1


class ResidualBlock(nn.Module):
    """A standard residual block with two 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class SimpleResNet(nn.Module):
    """A lightweight residual network for vibration signal classification."""

    def __init__(self, config: ResNetConfig, seq_len: int):
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.input_channels
        out_channels = config.base_filters
        for block_idx in range(config.num_blocks):
            stride = 2 if block_idx > 0 else 1
            layers.append(ResidualBlock(in_channels, out_channels, config.kernel_size, stride=stride))
            in_channels = out_channels
            out_channels *= 2
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, config.input_channels, seq_len)
            feature_tensor = self.global_pool(self.features(dummy))
            self.feature_length = feature_tensor.shape[1]

        self.classifier = nn.Linear(self.feature_length, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["ResNetConfig", "SimpleResNet"]
