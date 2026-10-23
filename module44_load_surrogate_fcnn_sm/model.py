from __future__ import annotations

import torch
from torch import nn


class LoadToDisplacementFCNNSM(nn.Module):
    """Five-layer fixed-mesh surrogate from known loads to all free DOFs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width_multiplier: int = 1,
        activation: str = "silu",
        linear_skip: bool = True,
        low_mode_basis: torch.Tensor | None = None,
        low_mode_count: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or width_multiplier < 1:
            raise ValueError("Invalid load-surrogate network dimensions")
        activations = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        scale = int(width_multiplier)
        widths = (
            max(2 * input_dim * scale, 32),
            max(4 * input_dim * scale, 64),
            max(8 * input_dim * scale, 128),
        )
        activation_type = activations[activation]
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.width_multiplier = scale
        self.activation_name = str(activation)
        self.linear_skip_enabled = bool(linear_skip)
        if low_mode_basis is None and int(low_mode_count) > 0:
            low_mode_basis = torch.zeros(
                (output_dim, int(low_mode_count)), dtype=torch.get_default_dtype(),
            )
        if low_mode_basis is not None:
            if low_mode_basis.ndim != 2 or low_mode_basis.shape[0] != output_dim:
                raise ValueError("low_mode_basis must have shape (output_dim, n_modes)")
            self.register_buffer("low_mode_basis", low_mode_basis.detach().clone())
            self.spectral_head = nn.Linear(
                self.input_dim, int(low_mode_basis.shape[1]), bias=False,
            )
        else:
            self.register_buffer("low_mode_basis", None)
            self.spectral_head = None
        self.hidden_widths = tuple(int(value) for value in widths)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, widths[0]),
            activation_type(),
            nn.Linear(widths[0], widths[1]),
            activation_type(),
            nn.Linear(widths[1], widths[2]),
            activation_type(),
            nn.Linear(widths[2], self.output_dim),
        )
        self.linear_skip = (
            nn.Linear(self.input_dim, self.output_dim, bias=False)
            if self.linear_skip_enabled
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start from the zero free-displacement field.  This prevents a wide
        # random output layer from injecting very large initial strain energy.
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        if self.linear_skip is not None:
            nn.init.zeros_(self.linear_skip.weight)
        if self.spectral_head is not None:
            nn.init.zeros_(self.spectral_head.weight)

    def forward(self, known_loads: torch.Tensor) -> torch.Tensor:
        values = self.network(known_loads)
        if self.linear_skip is not None:
            values = values + self.linear_skip(known_loads)
        if self.spectral_head is not None:
            values = values + self.spectral_head(known_loads) @ self.low_mode_basis.T
        return values

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "width_multiplier": self.width_multiplier,
            "activation": self.activation_name,
            "linear_skip": self.linear_skip_enabled,
            "low_mode_count": (
                int(self.low_mode_basis.shape[1])
                if self.low_mode_basis is not None
                else 0
            ),
        }
