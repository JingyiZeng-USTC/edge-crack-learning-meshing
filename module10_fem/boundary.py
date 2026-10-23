"""Boundary-condition construction and DOF partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryConditions:
    prescribed_dofs: np.ndarray
    prescribed_values: np.ndarray
    force: np.ndarray
    named_nodes: dict[str, np.ndarray]

    def free_dofs(self, n_dofs: int) -> np.ndarray:
        mask = np.ones(int(n_dofs), dtype=bool)
        mask[self.prescribed_dofs] = False
        return np.flatnonzero(mask)


def boundary_nodes(points: np.ndarray, tolerance: float | None = None):
    """Identify left, right, bottom and top nodes from mesh extents."""
    pts = np.asarray(points, dtype=float)[:, :2]
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    tol = float(tolerance) if tolerance is not None else 1e-8 * span
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    return {
        "LEFT": np.flatnonzero(np.abs(pts[:, 0] - xmin) <= tol),
        "RIGHT": np.flatnonzero(np.abs(pts[:, 0] - xmax) <= tol),
        "BOTTOM": np.flatnonzero(np.abs(pts[:, 1] - ymin) <= tol),
        "TOP": np.flatnonzero(np.abs(pts[:, 1] - ymax) <= tol),
    }


def vertical_tension_boundary(
    points: np.ndarray,
    top_displacement: float,
    bottom_displacement: float = 0.0,
    pin_node: int | None = None,
    force: np.ndarray | None = None,
    tolerance: float | None = None,
) -> BoundaryConditions:
    """Create BOTTOM-UY, TOP-UY and one PIN-UX constraints."""
    pts = np.asarray(points, dtype=float)[:, :2]
    n_nodes = len(pts)
    named = boundary_nodes(pts, tolerance)
    bottom = named["BOTTOM"]
    top = named["TOP"]
    if len(bottom) == 0 or len(top) == 0:
        raise ValueError("Top or bottom boundary contains no nodes.")

    if pin_node is None:
        pin = int(bottom[np.argmin(pts[bottom, 0])])
    else:
        pin = int(pin_node)
        if pin < 0 or pin >= n_nodes:
            raise ValueError(f"pin_node {pin} is outside [0, {n_nodes}).")
    named["PIN"] = np.asarray([pin], dtype=np.int64)

    dofs = np.concatenate((2 * bottom + 1, 2 * top + 1, np.asarray([2 * pin])))
    values = np.concatenate((
        np.full(len(bottom), float(bottom_displacement)),
        np.full(len(top), float(top_displacement)),
        np.asarray([0.0]),
    ))
    order = np.argsort(dofs)
    dofs, values = dofs[order].astype(np.int64), values[order]
    unique, first = np.unique(dofs, return_index=True)
    if len(unique) != len(dofs):
        for d in unique:
            v = values[dofs == d]
            if not np.allclose(v, v[0]):
                raise ValueError(f"Conflicting prescribed values on DOF {d}.")
        dofs, values = unique, values[first]

    F = np.zeros(2 * n_nodes, dtype=float) if force is None else np.asarray(force, dtype=float).copy()
    if F.shape != (2 * n_nodes,):
        raise ValueError(f"force must have shape ({2 * n_nodes},).")
    return BoundaryConditions(dofs, values, F, named)


def top_uniform_line_load_boundary(
    points: np.ndarray,
    line_load_y: float,
    bottom_displacement: float = 0.0,
    pin_node: int | None = None,
    tolerance: float | None = None,
) -> BoundaryConditions:
    """Apply a uniform vertical line load to the top boundary.

    ``line_load_y`` is the signed physical line load in N/m.  Every ordered
    two-node top-boundary segment of length ``L`` contributes the consistent
    nodal loads ``line_load_y * L / 2`` to both endpoints.  Therefore the
    assembled vertical resultant is exactly ``line_load_y * loaded_length``
    even when the top-boundary node spacing is nonuniform.

    The bottom vertical DOFs are prescribed and one bottom node is pinned in
    the horizontal direction.  All top displacement DOFs remain free.
    """
    pts = np.asarray(points, dtype=float)[:, :2]
    n_nodes = len(pts)
    named = boundary_nodes(pts, tolerance)
    bottom = np.asarray(named["BOTTOM"], dtype=np.int64)
    top = np.asarray(named["TOP"], dtype=np.int64)
    if len(bottom) == 0 or len(top) < 2:
        raise ValueError("Bottom boundary is empty or top boundary has fewer than two nodes.")

    if pin_node is None:
        pin = int(bottom[np.argmin(pts[bottom, 0])])
    else:
        pin = int(pin_node)
        if pin < 0 or pin >= n_nodes:
            raise ValueError(f"pin_node {pin} is outside [0, {n_nodes}).")
    named["PIN"] = np.asarray([pin], dtype=np.int64)

    dofs = np.concatenate((2 * bottom + 1, np.asarray([2 * pin], dtype=np.int64)))
    values = np.concatenate((
        np.full(len(bottom), float(bottom_displacement)),
        np.asarray([0.0]),
    ))
    order = np.argsort(dofs)
    dofs, values = dofs[order].astype(np.int64), values[order]
    unique, first = np.unique(dofs, return_index=True)
    if len(unique) != len(dofs):
        for dof in unique:
            repeated = values[dofs == dof]
            if not np.allclose(repeated, repeated[0]):
                raise ValueError(f"Conflicting prescribed values on DOF {dof}.")
        dofs, values = unique, values[first]

    # Sort along the physical top edge.  Coincident nodes can occur where a
    # top-edge crack is duplicated; their zero-length interval contributes no
    # load while the two adjacent real intervals retain the exact resultant.
    top = top[np.lexsort((top, pts[top, 0]))]
    named["TOP"] = top
    segment_lengths = np.linalg.norm(np.diff(pts[top], axis=0), axis=1)
    loaded_length = float(np.sum(segment_lengths))
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    if loaded_length <= np.finfo(float).eps * span:
        raise ValueError("Top boundary has zero loaded length.")
    weights = np.zeros(len(top), dtype=float)
    weights[:-1] += 0.5 * segment_lengths
    weights[1:] += 0.5 * segment_lengths

    force = np.zeros(2 * n_nodes, dtype=float)
    force[2 * top + 1] += float(line_load_y) * weights
    return BoundaryConditions(dofs, values, force, named)
