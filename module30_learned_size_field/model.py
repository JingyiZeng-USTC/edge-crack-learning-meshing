"""Pointwise neural regressor for a continuous mesh-size residual."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PointSizeFieldMLP(nn.Module):
    """Predict a bounded log-size correction at any point."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
        log_limit: float = 0.5108256237659907,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.log_limit = float(log_limit)
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, dropout) for _ in range(max(layers - 1, 0))
        )
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input(features)
        for block in self.blocks:
            value = block(value)
        return self.log_limit * torch.tanh(self.output(value).squeeze(-1))

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "log_limit": self.log_limit,
        }


class AbsolutePointSizeFieldMLP(nn.Module):
    """Predict normalized log(h) directly, without an analytic base field."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, dropout) for _ in range(max(layers - 1, 0))
        )
        self.output = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input(features)
        for block in self.blocks:
            value = block(value)
        return torch.sigmoid(self.output(value).squeeze(-1))

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
        }


class DirectNormalizedLogSizeMLP(nn.Module):
    """Predict normalized log(h) with a linear head.

    Clipping is deliberately postponed until physical h values are recovered.
    This avoids the zero-gradient collapse that a sigmoid head can exhibit when
    the training target is dominated by far-field points at h_max.
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.input = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU())
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, dropout) for _ in range(max(layers - 1, 0))
        )
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input(features)
        for block in self.blocks:
            value = block(value)
        return self.output(value).squeeze(-1)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
        }


__all__ = [
    "PointSizeFieldMLP",
    "AbsolutePointSizeFieldMLP",
    "DirectNormalizedLogSizeMLP",
    "ResidualBlock",
]
