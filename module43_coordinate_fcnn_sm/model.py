"""Node-coordinate network with a mesh-independent input/output width."""

from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    options = {
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }
    if name not in options:
        raise ValueError(f"Unsupported activation: {name}")
    return options[name]()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = _activation(activation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.linear2(self.activation(self.linear1(values)))
        return self.activation((values + update) / 1.4142135623730951)


class CoordinateFCNNSM(nn.Module):
    """Predict nodal displacement corrections from reusable neural basis functions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        residual_blocks: int = 4,
        activation: str = "silu",
        readout_features: str = "final",
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 4 or residual_blocks < 1:
            raise ValueError("Invalid coordinate FCNN-SM dimensions")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_blocks = int(residual_blocks)
        self.activation_name = str(activation)
        if readout_features not in {"final", "all", "all_with_input"}:
            raise ValueError(
                "readout_features must be 'final', 'all' or 'all_with_input'"
            )
        self.readout_features = str(readout_features)
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_activation = _activation(activation)
        self.blocks = nn.ModuleList(
            ResidualBlock(self.hidden_dim, activation)
            for _ in range(self.residual_blocks)
        )
        readout_dim = self.hidden_dim
        if self.readout_features in {"all", "all_with_input"}:
            readout_dim *= self.residual_blocks + 1
        if self.readout_features == "all_with_input":
            readout_dim += self.input_dim
        self.readout_dim = int(readout_dim)
        self.output_layer = nn.Linear(self.readout_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # The initial displacement is close to the exact prescribed lifting.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def basis_features(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return the neural basis evaluated at every mesh node."""
        values = self.input_activation(self.input_layer(node_features))
        levels = [values]
        for block in self.blocks:
            values = block(values)
            levels.append(values)
        if self.readout_features in {"all", "all_with_input"}:
            combined = torch.cat(levels, dim=-1)
            if self.readout_features == "all_with_input":
                combined = torch.cat((node_features, combined), dim=-1)
            return combined
        return values

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.basis_features(node_features))

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "residual_blocks": self.residual_blocks,
            "activation": self.activation_name,
            "readout_features": self.readout_features,
        }
