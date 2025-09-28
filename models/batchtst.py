"""A lightweight BatchTST-inspired transformer for time-series classification."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BatchTSTConfig:
    """Configuration for :class:`BatchTSTNet`."""

    input_channels: int
    num_classes: int
    patch_len: int = 32
    stride: int = 16
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class BatchTSTNet(nn.Module):
    """Temporal set transformer with learnable class token."""

    def __init__(self, config: BatchTSTConfig, seq_len: int) -> None:
        super().__init__()
        self.config = config

        if config.patch_len > seq_len:
            raise ValueError("patch_len must be <= sequence length")

        self.patch_embed = nn.Conv1d(
            config.input_channels,
            config.d_model,
            kernel_size=config.patch_len,
            stride=config.stride,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, config.input_channels, seq_len)
            patches = self.patch_embed(dummy)
            self.num_patches = patches.shape[-1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        patches = self.patch_embed(x)  # (batch, d_model, num_patches)
        patches = patches.transpose(1, 2)  # (batch, num_patches, d_model)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.size(1)]
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)
        cls = encoded[:, 0, :]
        cls = self.norm(cls)
        logits = self.classifier(cls)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BatchTSTConfig", "BatchTSTNet"]
