"""Linear static solution by sparse partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from .assembly import FEMSystem
from .boundary import BoundaryConditions


@dataclass(frozen=True)
class Solution:
    displacement: np.ndarray
    displacement_xy: np.ndarray
    reactions: np.ndarray
    free_dofs: np.ndarray
    prescribed_dofs: np.ndarray
    rhs_free: np.ndarray
    residual_free: np.ndarray
    strain_energy: float


def solve_linear_system(system: FEMSystem, bc: BoundaryConditions) -> Solution:
    """Solve K_ff U_f = F_f - K_fc U_c without forming K inverse."""
    K = system.stiffness
    ndof = system.n_dofs
    c = np.asarray(bc.prescribed_dofs, dtype=np.int64)
    uc = np.asarray(bc.prescribed_values, dtype=float)
    if len(c) == 0:
        raise ValueError("At least one prescribed displacement is required.")
    if c.min() < 0 or c.max() >= ndof:
        raise ValueError("A prescribed DOF lies outside the global system.")
    if len(np.unique(c)) != len(c):
        raise ValueError("prescribed_dofs must be unique.")

    f = bc.free_dofs(ndof)
    rhs = bc.force[f] - K[f][:, c] @ uc
    uf = np.asarray(spsolve(K[f][:, f], rhs), dtype=float)
    if not np.all(np.isfinite(uf)):
        raise np.linalg.LinAlgError("The constrained stiffness system is singular or ill-conditioned.")

    U = np.zeros(ndof, dtype=float)
    U[c] = uc
    U[f] = uf
    internal = np.asarray(K @ U).ravel()
    residual = internal - bc.force
    return Solution(
        displacement=U,
        displacement_xy=U.reshape(-1, 2),
        reactions=residual[c],
        free_dofs=f,
        prescribed_dofs=c,
        rhs_free=np.asarray(rhs).ravel(),
        residual_free=residual[f],
        strain_energy=0.5 * float(U @ internal),
    )
