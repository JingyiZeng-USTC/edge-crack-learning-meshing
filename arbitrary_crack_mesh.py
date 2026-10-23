"""Deterministic constrained mesh construction for arbitrary crack polylines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_sampling import AdaptivePoissonSampler
from module8_inference import _boundary_edges, _ek, _tri_min_angle, quality_optimize
from module9_crack_graph_nodes import _chain_node_ids, duplicate_crack_graph_nodes


@dataclass
class ArbitraryCrackMesh:
    points: np.ndarray
    triangles: np.ndarray
    crack_info: dict
    constraints: set[tuple[int, int]]
    quality: dict


def constraint_edges(points: np.ndarray, geometry, tolerance: float = 1.0e-8):
    """Map outer-boundary and every ordered crack-chain segment to node IDs."""
    constraints = set(_boundary_edges(points, geometry.W, geometry.H))
    chain_ids = _chain_node_ids(points, geometry.crack_chains, tolerance)
    values = np.asarray(points, dtype=float)
    for chain in chain_ids:
        for first, second in zip(chain[:-1], chain[1:]):
            start = values[first]
            vector = values[second] - start
            length2 = float(vector @ vector)
            parameter = ((values - start) @ vector) / length2
            closest = start + np.clip(parameter, 0.0, 1.0)[:, None] * vector
            on_segment = (
                (np.linalg.norm(values - closest, axis=1) <= tolerance)
                & (parameter >= -tolerance)
                & (parameter <= 1.0 + tolerance)
            )
            atomic = np.flatnonzero(on_segment)
            atomic = atomic[np.argsort(parameter[atomic])]
            constraints.update(
                _ek(a, b) for a, b in zip(atomic[:-1], atomic[1:]) if int(a) != int(b)
            )
    return constraints


def constrained_triangulation(
    points: np.ndarray,
    constraints: set[tuple[int, int]],
    *,
    minimum_angle: float = 20.0,
    max_steiner_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    """Run quality-constrained PSLG triangulation and return atomic constraints.

    Triangle may append Steiner points in element interiors or on a constrained
    segment.  Its returned segment list is therefore authoritative: every
    split child segment remains locked during subsequent quality optimization
    and crack-face duplication.
    """
    import triangle

    values = np.asarray(points, dtype=float)
    if not 0.0 <= float(minimum_angle) < 34.0:
        raise ValueError("minimum_angle must lie in [0, 34) degrees")
    if int(max_steiner_points) < 0:
        raise ValueError("max_steiner_points cannot be negative")
    quality_option = "" if minimum_angle <= 0.0 else f"q{float(minimum_angle):.8g}"
    steiner_option = f"S{int(max_steiner_points)}"
    result = triangle.triangulate(
        {
            "vertices": values,
            "segments": np.asarray(sorted(constraints), dtype=np.int32),
        },
        f"p{quality_option}{steiner_option}Q",
    )
    triangles = np.asarray(result.get("triangles", ()), dtype=np.int32)
    output_points = np.asarray(result.get("vertices", values), dtype=float)
    if len(output_points) < len(values):
        raise ValueError("Constrained triangulation removed input vertices")
    if len(output_points) - len(values) > int(max_steiner_points):
        raise ValueError("Constrained triangulation exceeded the Steiner-node limit")
    output_segments = np.asarray(result.get("segments", ()), dtype=np.int64)
    if output_segments.ndim != 2 or output_segments.shape[1] != 2:
        raise ValueError("Constrained triangulation did not return its atomic segments")
    atomic_constraints = {
        _ek(first, second) for first, second in output_segments
    }
    mesh_edges = {
        _ek(triangle_nodes[first], triangle_nodes[second])
        for triangle_nodes in triangles
        for first, second in ((0, 1), (1, 2), (2, 0))
    }
    missing = atomic_constraints - mesh_edges
    if missing:
        raise ValueError(
            f"Constrained triangulation omitted {len(missing)} hard segments; "
            f"first={sorted(missing)[:10]}"
        )
    return output_points, triangles, atomic_constraints


def _quality(points: np.ndarray, triangles: np.ndarray) -> dict:
    if len(triangles) == 0:
        return {"minimum_angle_deg": 0.0, "mean_minimum_angle_deg": 0.0}
    angles = np.asarray(
        [_tri_min_angle(points, int(a), int(b), int(c)) for a, b, c in triangles],
        dtype=float,
    )
    cells = points[triangles]
    twice_area = (
        (cells[:, 1, 0] - cells[:, 0, 0]) * (cells[:, 2, 1] - cells[:, 0, 1])
        - (cells[:, 2, 0] - cells[:, 0, 0]) * (cells[:, 1, 1] - cells[:, 0, 1])
    )
    return {
        "minimum_angle_deg": float(np.degrees(np.min(angles))),
        "angle_p05_deg": float(np.degrees(np.quantile(angles, 0.05))),
        "mean_minimum_angle_deg": float(np.degrees(np.mean(angles))),
        "degenerate_elements": int(np.count_nonzero(np.abs(twice_area) <= 1.0e-14)),
    }


def generate_arbitrary_crack_mesh(
    geometry,
    size_field,
    *,
    seed: int = 0,
    n_crack_segments: int = 30,
    n_layers: int = 3,
    n_tip_ring: int = 8,
    k_candidates: int = 30,
    quality_iterations: int = 2,
    triangulation_minimum_angle: float = 20.0,
    max_steiner_points: int = 20000,
    open_crack_faces: bool = True,
) -> ArbitraryCrackMesh:
    """Sample points, enforce every PSLG segment, then open crack faces."""
    sampler = AdaptivePoissonSampler(geometry, size_field)
    points = sampler.sample(
        seed=seed,
        n_crack_segments=n_crack_segments,
        n_layers=n_layers,
        n_tip_ring=n_tip_ring,
        k_candidates=k_candidates,
    )
    constraints = constraint_edges(points, geometry)
    original_point_count = len(points)
    points, triangles, constraints = constrained_triangulation(
        points,
        constraints,
        minimum_angle=triangulation_minimum_angle,
        max_steiner_points=max_steiner_points,
    )
    steiner_point_count = len(points) - original_point_count
    if quality_iterations > 0:
        triangles = quality_optimize(
            triangles, points, constraints, max_iters=int(quality_iterations),
        )
    if open_crack_faces:
        triangles, points, crack_info = duplicate_crack_graph_nodes(
            triangles, points, geometry,
        )
    else:
        crack_info = {
            "has_branches": bool(geometry.has_branches),
            "shared_crack_nodes": True,
        }
    quality = _quality(points, triangles)
    # Crack-face opening duplicates existing constraint nodes.  Those copies
    # are topology nodes, not Triangle refinement (Steiner) nodes.
    quality["steiner_points"] = int(steiner_point_count)
    quality["requested_minimum_angle_deg"] = float(triangulation_minimum_angle)
    quality["quality_target_met"] = bool(
        triangulation_minimum_angle <= 0.0
        or quality["minimum_angle_deg"] + 0.05 >= triangulation_minimum_angle
    )
    if quality["degenerate_elements"]:
        raise ValueError("The arbitrary-crack mesh contains degenerate triangles")
    return ArbitraryCrackMesh(
        points=np.asarray(points, dtype=float),
        triangles=np.asarray(triangles, dtype=np.int32),
        crack_info=crack_info,
        constraints=constraints,
        quality=quality,
    )


__all__ = [
    "ArbitraryCrackMesh",
    "constraint_edges",
    "constrained_triangulation",
    "generate_arbitrary_crack_mesh",
]
