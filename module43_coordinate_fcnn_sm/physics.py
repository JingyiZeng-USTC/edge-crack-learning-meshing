"""Differentiable finite-element equilibrium residual for one current mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warnings
from scipy.sparse import csr_matrix, diags


def scipy_csr_to_torch(matrix: csr_matrix, device, dtype) -> torch.Tensor:
    values = torch.as_tensor(matrix.data, dtype=dtype, device=device)
    indices = torch.as_tensor(matrix.indices, dtype=torch.int64, device=device)
    indptr = torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")
        return torch.sparse_csr_tensor(
            indptr,
            indices,
            values,
            size=matrix.shape,
            dtype=dtype,
            device=device,
            check_invariants=False,
        )


@dataclass(frozen=True)
class DiscreteEquilibrium:
    stiffness_ff: torch.Tensor
    rhs: torch.Tensor
    inverse_diagonal: torch.Tensor
    weighted_rhs_norm: torch.Tensor
    rhs_norm: torch.Tensor

    @classmethod
    def from_scipy(cls, stiffness_ff, rhs, *, device, dtype):
        matrix = stiffness_ff.tocsr()
        diagonal = np.maximum(np.abs(matrix.diagonal()), 1.0e-30)
        rhs_array = np.asarray(rhs, dtype=float)
        K = scipy_csr_to_torch(matrix, device, dtype)
        b = torch.as_tensor(rhs_array, dtype=dtype, device=device)
        inverse = torch.as_tensor(1.0 / diagonal, dtype=dtype, device=device)
        weighted = torch.sum(b * b * inverse).clamp_min(torch.finfo(dtype).tiny)
        norm = torch.linalg.vector_norm(b).clamp_min(torch.finfo(dtype).tiny)
        return cls(K, b, inverse, weighted, norm)

    def residual(self, free_displacement: torch.Tensor) -> torch.Tensor:
        internal = torch.sparse.mm(
            self.stiffness_ff,
            free_displacement.reshape(-1, 1),
        ).reshape(-1)
        return internal - self.rhs

    def loss_and_metrics(self, free_displacement: torch.Tensor):
        residual = self.residual(free_displacement)
        loss = torch.sum(residual * residual * self.inverse_diagonal) / self.weighted_rhs_norm
        relative_l2 = torch.linalg.vector_norm(residual) / self.rhs_norm
        return loss, relative_l2, residual


@dataclass(frozen=True)
class ScaledRitzEquilibrium:
    """Jacobi-scaled Ritz system with a dimensionless network unknown."""

    stiffness_scaled: torch.Tensor
    rhs_scaled: torch.Tensor
    sqrt_diagonal: torch.Tensor
    inverse_sqrt_diagonal: torch.Tensor
    physical_rhs_norm: torch.Tensor
    scaled_rhs_norm_squared: torch.Tensor
    scaled_variable_scale: torch.Tensor
    ritz_scale: torch.Tensor
    base_free_displacement: torch.Tensor

    @classmethod
    def from_scipy(
        cls,
        stiffness_ff,
        rhs,
        *,
        displacement_scale: float,
        base_free_displacement=None,
        device,
        dtype,
    ):
        matrix = stiffness_ff.tocsr()
        diagonal = np.maximum(np.abs(matrix.diagonal()), 1.0e-30)
        sqrt_diagonal = np.sqrt(diagonal)
        inverse_sqrt = 1.0 / sqrt_diagonal
        scaling = diags(inverse_sqrt)
        scaled_matrix = (scaling @ matrix @ scaling).tocsr()
        rhs_array = np.asarray(rhs, dtype=float).reshape(-1)
        if base_free_displacement is None:
            base_array = np.zeros_like(rhs_array)
        else:
            base_array = np.asarray(base_free_displacement, dtype=float).reshape(-1)
            if base_array.shape != rhs_array.shape:
                raise ValueError("base_free_displacement must have the same shape as rhs")
        correction_rhs = rhs_array - np.asarray(matrix @ base_array).reshape(-1)
        rhs_scaled = inverse_sqrt * correction_rhs
        variable_scale = max(
            float(np.median(sqrt_diagonal)) * float(displacement_scale),
            1.0e-30,
        )
        scaled_rhs_norm = max(float(np.linalg.norm(rhs_scaled)), 1.0e-30)
        ritz_scale = max(
            variable_scale * scaled_rhs_norm * np.sqrt(len(rhs_scaled)),
            1.0e-30,
        )
        return cls(
            stiffness_scaled=scipy_csr_to_torch(scaled_matrix, device, dtype),
            rhs_scaled=torch.as_tensor(rhs_scaled, dtype=dtype, device=device),
            sqrt_diagonal=torch.as_tensor(sqrt_diagonal, dtype=dtype, device=device),
            inverse_sqrt_diagonal=torch.as_tensor(
                inverse_sqrt, dtype=dtype, device=device,
            ),
            physical_rhs_norm=torch.as_tensor(
                max(float(np.linalg.norm(rhs_array)), 1.0e-30),
                dtype=dtype,
                device=device,
            ),
            scaled_rhs_norm_squared=torch.as_tensor(
                scaled_rhs_norm * scaled_rhs_norm, dtype=dtype, device=device,
            ),
            scaled_variable_scale=torch.as_tensor(
                variable_scale, dtype=dtype, device=device,
            ),
            ritz_scale=torch.as_tensor(ritz_scale, dtype=dtype, device=device),
            base_free_displacement=torch.as_tensor(
                base_array, dtype=dtype, device=device,
            ),
        )

    def scaled_variable(self, dimensionless: torch.Tensor) -> torch.Tensor:
        return dimensionless * self.scaled_variable_scale

    def free_displacement(self, dimensionless: torch.Tensor) -> torch.Tensor:
        return self.base_free_displacement + (
            self.scaled_variable(dimensionless) * self.inverse_sqrt_diagonal
        )

    def losses_and_metrics(self, dimensionless: torch.Tensor):
        z = self.scaled_variable(dimensionless)
        Az = torch.sparse.mm(self.stiffness_scaled, z.reshape(-1, 1)).reshape(-1)
        scaled_residual = Az - self.rhs_scaled
        physical_residual = self.sqrt_diagonal * scaled_residual
        potential = 0.5 * torch.dot(z, Az) - torch.dot(z, self.rhs_scaled)
        ritz = potential / self.ritz_scale
        residual_loss = (
            torch.sum(scaled_residual * scaled_residual)
            / self.scaled_rhs_norm_squared
        )
        physical_relative = (
            torch.linalg.vector_norm(physical_residual) / self.physical_rhs_norm
        )
        scaled_relative = torch.sqrt(residual_loss)
        return ritz, residual_loss, physical_relative, scaled_relative, physical_residual


__all__ = [
    "DiscreteEquilibrium",
    "ScaledRitzEquilibrium",
    "scipy_csr_to_torch",
]
