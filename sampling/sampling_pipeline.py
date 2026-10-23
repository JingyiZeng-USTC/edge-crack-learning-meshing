"""Sampling pipeline orchestrator.

Runs all sub-modules in order:
  1. boundary
  2. crack
  3. crack layers
  4. tip ring
  5. interior Poisson
  6. merge

Usage
-----
>>> from sampling.sampling_pipeline import sample_all
>>> points = sample_all(geometry, size_field, seed=42)
"""

import numpy as np

from .boundary_sampling import sample_boundary
from .crack_sampling import sample_crack
from .crack_layer_sampling import sample_crack_layers
from .tip_ring_sampling import sample_tip_ring
from .poisson_sampling import sample_interior
from .merge_points import merge_all


def _clear_exterior_boundary(
    points: np.ndarray,
    geometry,
    size_field,
    factor: float,
) -> np.ndarray:
    """Remove non-constraint points that would form exterior-boundary slivers."""
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(values):
        return values
    distance = np.minimum.reduce(
        (values[:, 0], geometry.W - values[:, 0], values[:, 1], geometry.H - values[:, 1])
    )
    local_h = np.maximum(size_field.evaluate_bulk(values), 0.5 * size_field.h_min)
    return values[distance >= factor * local_h]


def _priority_auxiliary_points(
    constraints: np.ndarray,
    ring: np.ndarray,
    layers: np.ndarray,
    size_field,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ring/layer points only when separated at the local size scale.

    Boundary and crack points are immutable constraints.  Tip-ring points are
    considered before crack-layer points because the ring supplies the most
    regular radial pattern around a crack tip.  No constraint point is ever
    removed.
    """
    accepted = [point.copy() for point in np.asarray(constraints, dtype=float).reshape(-1, 2)]
    accepted_h = list(np.maximum(size_field.evaluate_bulk(np.asarray(accepted)), 0.5 * size_field.h_min))
    kept_ring: list[np.ndarray] = []
    kept_layers: list[np.ndarray] = []

    def append_group(values: np.ndarray, destination: list[np.ndarray]) -> None:
        for point in np.asarray(values, dtype=float).reshape(-1, 2):
            h_point = max(float(size_field.evaluate(float(point[0]), float(point[1]))), 0.5 * size_field.h_min)
            stored = np.asarray(accepted, dtype=float)
            stored_h = np.asarray(accepted_h, dtype=float)
            distance = np.linalg.norm(stored - point, axis=1)
            required = factor * np.minimum(stored_h, h_point)
            if np.any(distance < np.maximum(required, 1.0e-10)):
                continue
            accepted.append(point.copy())
            accepted_h.append(h_point)
            destination.append(point.copy())

    append_group(ring, kept_ring)
    append_group(layers, kept_layers)
    return (
        np.asarray(kept_ring, dtype=float).reshape(-1, 2),
        np.asarray(kept_layers, dtype=float).reshape(-1, 2),
    )


def sample_all(
    geometry,
    size_field,
    seed: int | None = None,
    n_crack_segments: int = 30,
    crack_spacing_factor: float = 0.90,
    n_layers: int = 3,
    n_tip_ring: int = 8,
    k_candidates: int = 30,
    boundary_clearance_factor: float = 0.5,
    auxiliary_separation_factor: float = 0.5,
) -> np.ndarray:
    """Run the complete sampling pipeline.

    Returns
    -------
    points : (N, 2) np.ndarray
        Merged, deduplicated point cloud.
    """
    # 1. boundary
    bnd = sample_boundary(geometry, size_field)

    # 2. crack
    crk = sample_crack(
        geometry,
        size_field=size_field,
        n_segments=n_crack_segments,
        spacing_factor=crack_spacing_factor,
    )

    # 3. crack layers (structured band along both sides)
    lay = sample_crack_layers(geometry, size_field, n_layers=n_layers)
    lay = _clear_exterior_boundary(
        lay, geometry, size_field, boundary_clearance_factor,
    )

    # 4. tip ring
    ring = sample_tip_ring(geometry, size_field, n_ring=n_tip_ring)
    ring = _clear_exterior_boundary(
        ring, geometry, size_field, boundary_clearance_factor,
    )

    # Constraints have absolute priority.  Ring points then take priority over
    # crack-layer points so independently generated structured sets cannot
    # create sub-h_min edges near a crack tip.
    constraints = merge_all(bnd, crk)
    ring, lay = _priority_auxiliary_points(
        constraints,
        ring,
        lay,
        size_field,
        auxiliary_separation_factor,
    )

    # 5. interior Poisson (avoids existing points)
    existing = merge_all(constraints, ring, lay)
    interior = sample_interior(
        geometry,
        size_field,
        existing,
        seed=seed,
        k_candidates=k_candidates,
        boundary_clearance_factor=boundary_clearance_factor,
    )

    # 6. final merge
    all_pts = merge_all(constraints, ring, lay, interior)
    return all_pts
