"""InceptionTime-style network for multivariate time-series classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn


@dataclass
class InceptionTimeConfig:
    """Configuration for :class:`InceptionTimeNet`."""

    input_channels: int
    num_classes: int
    num_blocks: int = 6
    in_channels: int = 32
    bottleneck_channels: int = 32
    kernel_sizes: Tuple[int, int, int] = (9, 19, 39)
    use_residual: bool = True
    dropout: float = 0.1


class InceptionBlock(nn.Module):
    """Single inception block with optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: Iterable[int],
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = in_channels

        conv_layers = []
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(
                    bottleneck_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
        self.branches = nn.ModuleList(conv_layers)
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

        if self.use_residual:
            if in_channels == out_channels * (len(kernel_sizes) + 1):
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels * (len(kernel_sizes) + 1),
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1)),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        branch_outputs = [conv(bottleneck) for conv in self.branches]
        pooled = self.avg_pool(x)
        branch_outputs.append(self.pool_conv(pooled))

        out = torch.cat(branch_outputs, dim=1)
        out = self.batch_norm(out)
        out = self.relu(out)

        if self.use_residual:
            out = out + self.residual(x)
            out = self.relu(out)
        return out


class InceptionTimeNet(nn.Module):
    """Stacked inception blocks followed by global average pooling."""

    def __init__(self, config: InceptionTimeConfig, seq_len: int) -> None:
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.input_channels
        out_channels = config.in_channels
        for block_idx in range(config.num_blocks):
            block = InceptionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bottleneck_channels=config.bottleneck_channels,
                kernel_sizes=config.kernel_sizes,
                use_residual=config.use_residual,
            )
            layers.append(block)
            in_channels = out_channels * (len(config.kernel_sizes) + 1)
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(in_channels, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["InceptionTimeConfig", "InceptionTimeNet"]
