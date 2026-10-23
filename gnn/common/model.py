from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class EdgeGraphSAGE(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 96, dropout: float = 0.10, layers: int = 4):
        super().__init__()
        self.input = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(layers)])
        self.classifier = nn.Linear(hidden_channels, 2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = self.dropout(self.activation(norm(x))) + residual
        return self.classifier(x)
