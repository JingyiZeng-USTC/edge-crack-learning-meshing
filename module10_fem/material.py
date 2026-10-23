"""Isotropic constitutive matrices using engineering shear strain gamma_xy."""

from __future__ import annotations

import numpy as np


def _validate(young_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    E = float(young_modulus)
    nu = float(poisson_ratio)
    if E <= 0.0:
        raise ValueError("Young's modulus must be positive.")
    if not -1.0 < nu < 0.5:
        raise ValueError("Poisson's ratio must lie in (-1, 0.5).")
    return E, nu


def plane_stress_matrix(young_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Return D for [eps_x, eps_y, gamma_xy] under plane stress."""
    E, nu = _validate(young_modulus, poisson_ratio)
    return E / (1.0 - nu * nu) * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )


def plane_strain_matrix(young_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Return D for [eps_x, eps_y, gamma_xy] under plane strain."""
    E, nu = _validate(young_modulus, poisson_ratio)
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return factor * np.array(
        [[1.0 - nu, nu, 0.0],
         [nu, 1.0 - nu, 0.0],
         [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]],
        dtype=float,
    )
