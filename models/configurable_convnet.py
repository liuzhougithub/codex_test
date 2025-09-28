"""Configurable 1D CNN that can be described via a simple JSON file."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn


@dataclass
class ConfigurableConvNetConfig:
    """Configuration for :class:`ConfigurableConvNet`."""

    input_channels: int
    num_classes: int
    layers: List[Dict[str, Any]] = field(default_factory=list)
    dropout: float = 0.1

    @staticmethod
    def default_layers() -> List[Dict[str, Any]]:
        return [
            {"type": "conv", "out_channels": 32, "kernel_size": 7, "stride": 1, "padding": "same"},
            {"type": "batchnorm"},
            {"type": "relu"},
            {"type": "maxpool", "kernel_size": 2},
            {"type": "conv", "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": "same"},
            {"type": "batchnorm"},
            {"type": "relu"},
            {"type": "maxpool", "kernel_size": 2},
            {"type": "conv", "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": "same"},
            {"type": "batchnorm"},
            {"type": "relu"},
            {"type": "globalavgpool"},
        ]


class ConfigurableConvNet(nn.Module):
    """Sequential CNN assembled from a list of layer specifications."""

    def __init__(self, config: ConfigurableConvNetConfig, seq_len: int) -> None:
        super().__init__()
        self.config = config

        layers_cfg = config.layers or ConfigurableConvNetConfig.default_layers()
        layers: List[nn.Module] = []
        in_channels = config.input_channels
        current_length = seq_len

        for layer in layers_cfg:
            layer_type = layer.get("type", "").lower()
            if layer_type == "conv":
                out_channels = int(layer["out_channels"])
                kernel_size = int(layer.get("kernel_size", 3))
                stride = int(layer.get("stride", 1))
                padding_value = layer.get("padding", 0)
                if padding_value == "same":
                    padding = kernel_size // 2
                else:
                    padding = int(padding_value)
                conv = nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=layer.get("bias", False),
                )
                layers.append(conv)
                in_channels = out_channels
                current_length = (current_length + 2 * padding - kernel_size) // stride + 1
            elif layer_type == "batchnorm":
                layers.append(nn.BatchNorm1d(in_channels))
            elif layer_type == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif layer_type == "gelu":
                layers.append(nn.GELU())
            elif layer_type == "dropout":
                layers.append(nn.Dropout(float(layer.get("p", config.dropout))))
            elif layer_type == "maxpool":
                kernel_size = int(layer.get("kernel_size", 2))
                stride = int(layer.get("stride", kernel_size))
                layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
                current_length = (current_length - kernel_size) // stride + 1
            elif layer_type == "avgpool":
                kernel_size = int(layer.get("kernel_size", 2))
                stride = int(layer.get("stride", kernel_size))
                layers.append(nn.AvgPool1d(kernel_size=kernel_size, stride=stride))
                current_length = (current_length - kernel_size) // stride + 1
            elif layer_type == "globalavgpool":
                layers.append(nn.AdaptiveAvgPool1d(1))
                current_length = 1
            elif layer_type == "flatten":
                layers.append(nn.Flatten())
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, config.input_channels, seq_len)
            features = self.features(dummy)
            if features.ndim == 3:
                features = torch.mean(features, dim=-1)
            self.feature_dim = features.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.feature_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        if out.ndim == 3:
            out = out.mean(dim=-1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["ConfigurableConvNetConfig", "ConfigurableConvNet"]
