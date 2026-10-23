"""Element and area-weighted nodal strain/stress recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assembly import FEMSystem


def von_mises_plane_stress(stress: np.ndarray) -> np.ndarray:
    s = np.asarray(stress, dtype=float)
    sx, sy, txy = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))


@dataclass(frozen=True)
class FieldResults:
    element_strain: np.ndarray
    element_stress: np.ndarray
    element_von_mises: np.ndarray
    nodal_strain: np.ndarray
    nodal_stress: np.ndarray
    nodal_von_mises: np.ndarray


def recover_fields(system: FEMSystem, displacement: np.ndarray) -> FieldResults:
    """Recover [eps_x, eps_y, gamma_xy] and [sig_x, sig_y, tau_xy]."""
    U = np.asarray(displacement, dtype=float)
    if U.shape == (system.n_nodes, 2):
        U = U.reshape(-1)
    if U.shape != (system.n_dofs,):
        raise ValueError(f"displacement must have shape ({system.n_dofs},) or ({system.n_nodes}, 2).")

    ue = U[system.element_dof_map]
    strain_e = np.einsum("eij,ej->ei", system.element_B, ue)
    stress_e = strain_e @ system.constitutive.T
    vm_e = von_mises_plane_stress(stress_e)

    strain_n = np.zeros((system.n_nodes, 3), dtype=float)
    stress_n = np.zeros((system.n_nodes, 3), dtype=float)
    weights = np.zeros(system.n_nodes, dtype=float)
    for local in range(3):
        nodes = system.triangles[:, local]
        np.add.at(strain_n, nodes, strain_e * system.element_areas[:, None])
        np.add.at(stress_n, nodes, stress_e * system.element_areas[:, None])
        np.add.at(weights, nodes, system.element_areas)
    used = weights > 0.0
    strain_n[used] /= weights[used, None]
    stress_n[used] /= weights[used, None]
    vm_n = von_mises_plane_stress(stress_n)
    return FieldResults(strain_e, stress_e, vm_e, strain_n, stress_n, vm_n)
