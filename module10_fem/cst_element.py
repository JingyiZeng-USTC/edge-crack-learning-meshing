"""Three-node constant-strain triangle (CST) element."""

from __future__ import annotations

import numpy as np


def cst_kinematics(coordinates: np.ndarray, area_tolerance: float = 1e-14):
    """Return positive area and the 3x6 strain-displacement matrix B."""
    xy = np.asarray(coordinates, dtype=float)
    if xy.shape != (3, 2):
        raise ValueError(f"CST coordinates must have shape (3, 2), got {xy.shape}.")

    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    twice_signed_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(twice_signed_area)
    if area <= area_tolerance:
        raise ValueError(f"Degenerate CST element with area {area:.6e}.")

    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    B = np.array(
        [[b1, 0.0, b2, 0.0, b3, 0.0],
         [0.0, c1, 0.0, c2, 0.0, c3],
         [c1, b1, c2, b2, c3, b3]],
        dtype=float,
    ) / twice_signed_area
    return area, B


def cst_stiffness(
    coordinates: np.ndarray,
    constitutive_matrix: np.ndarray,
    thickness: float,
):
    """Return (Ke, area, B) for a CST element."""
    t = float(thickness)
    if t <= 0.0:
        raise ValueError("Thickness must be positive.")
    D = np.asarray(constitutive_matrix, dtype=float)
    if D.shape != (3, 3):
        raise ValueError(f"Constitutive matrix must have shape (3, 3), got {D.shape}.")
    area, B = cst_kinematics(coordinates)
    Ke = t * area * (B.T @ D @ B)
    return Ke, area, B


def element_dofs(nodes: np.ndarray) -> np.ndarray:
    """Map three zero-based node IDs to six global displacement DOFs."""
    n = np.asarray(nodes, dtype=np.int64)
    if n.shape != (3,):
        raise ValueError("A CST element must contain exactly three node IDs.")
    return np.column_stack((2 * n, 2 * n + 1)).reshape(-1)
