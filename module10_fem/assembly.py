"""Sparse global stiffness assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .cst_element import cst_stiffness, element_dofs


@dataclass(frozen=True)
class FEMSystem:
    points: np.ndarray
    triangles: np.ndarray
    stiffness: csr_matrix
    constitutive: np.ndarray
    thickness: float
    element_areas: np.ndarray
    element_B: np.ndarray
    element_dof_map: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes


def assemble_stiffness(
    points: np.ndarray,
    triangles: np.ndarray,
    constitutive_matrix: np.ndarray,
    thickness: float,
) -> FEMSystem:
    """Assemble the global 2N x 2N CSR stiffness matrix."""
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    D = np.asarray(constitutive_matrix, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N, 2) or (N, 3).")
    pts = pts[:, :2].copy()
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3).")
    if len(tri) == 0:
        raise ValueError("Cannot assemble an empty triangle mesh.")
    if tri.min() < 0 or tri.max() >= len(pts):
        raise ValueError("Triangle connectivity contains an invalid node index.")

    n_elements = len(tri)
    rows = np.empty(n_elements * 36, dtype=np.int64)
    cols = np.empty(n_elements * 36, dtype=np.int64)
    values = np.empty(n_elements * 36, dtype=float)
    areas = np.empty(n_elements, dtype=float)
    B_all = np.empty((n_elements, 3, 6), dtype=float)
    dof_map = np.empty((n_elements, 6), dtype=np.int64)

    cursor = 0
    for e, nodes in enumerate(tri):
        Ke, area, B = cst_stiffness(pts[nodes], D, thickness)
        dofs = element_dofs(nodes)
        sl = slice(cursor, cursor + 36)
        rows[sl] = np.repeat(dofs, 6)
        cols[sl] = np.tile(dofs, 6)
        values[sl] = Ke.ravel()
        areas[e] = area
        B_all[e] = B
        dof_map[e] = dofs
        cursor += 36

    ndof = 2 * len(pts)
    K = coo_matrix((values, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return FEMSystem(
        points=pts,
        triangles=tri.copy(),
        stiffness=K,
        constitutive=D.copy(),
        thickness=float(thickness),
        element_areas=areas,
        element_B=B_all,
        element_dof_map=dof_map,
    )
