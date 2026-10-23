"""Geometry-only supervision extracted from a reference triangle mesh."""

from __future__ import annotations

import numpy as np


def _validated_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(triangles, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise ValueError("triangles must have shape (M, 3) with M > 0")
    if tri.min() < 0 or tri.max() >= len(pts):
        raise ValueError("Triangle connectivity contains an invalid node index")
    xy = pts[:, :2]
    cell = xy[tri]
    twice_signed_area = (
        (cell[:, 1, 0] - cell[:, 0, 0])
        * (cell[:, 2, 1] - cell[:, 0, 1])
        - (cell[:, 2, 0] - cell[:, 0, 0])
        * (cell[:, 1, 1] - cell[:, 0, 1])
    )
    areas = 0.5 * np.abs(twice_signed_area)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    if np.any(areas <= np.finfo(float).eps * span * span):
        raise ValueError("The reference mesh contains a degenerate triangle")
    return xy, tri, areas


def mesh_size_targets(
    points: np.ndarray,
    triangles: np.ndarray,
    analytic_h: np.ndarray,
    h_min: float,
    h_max: float,
    min_ratio: float = 0.50,
    max_ratio: float = 2.00,
) -> dict[str, np.ndarray]:
    """Build nodal size labels using only reference-mesh geometry.

    Each element is assigned the side length of an equal-area equilateral
    triangle.  A node receives the median size of its incident elements.  The
    loss weight is the node's barycentric dual area, normalized to mean one;
    this prevents already-refined regions from dominating merely because they
    contain more nodes.
    """
    pts, tri, areas = _validated_mesh(points, triangles)
    h0 = np.asarray(analytic_h, dtype=float).reshape(-1)
    if len(h0) != len(pts):
        raise ValueError("analytic_h must contain one value per mesh node")
    if not (0.0 < float(h_min) < float(h_max)):
        raise ValueError("Require 0 < h_min < h_max")
    if not (0.0 < float(min_ratio) <= 1.0 <= float(max_ratio)):
        raise ValueError("Require 0 < min_ratio <= 1 <= max_ratio")

    element_size = np.sqrt(4.0 * areas / np.sqrt(3.0))
    incident_sizes: list[list[float]] = [[] for _ in range(len(pts))]
    for local in range(3):
        for node, size in zip(tri[:, local], element_size, strict=True):
            incident_sizes[int(node)].append(float(size))
    if any(not values for values in incident_sizes):
        raise ValueError("The reference mesh contains an unconnected node")
    nodal_size = np.asarray(
        [np.median(values) for values in incident_sizes], dtype=float,
    )

    lower = np.maximum(float(h_min), h0 * float(min_ratio))
    upper = np.minimum(float(h_max), h0 * float(max_ratio))
    target_h = np.clip(nodal_size, lower, upper)
    target_log_ratio = np.log(
        np.maximum(target_h, 1e-30) / np.maximum(h0, 1e-30),
    )

    dual_area = np.zeros(len(pts), dtype=float)
    for local in range(3):
        np.add.at(dual_area, tri[:, local], areas / 3.0)
    positive = dual_area > 0.0
    scale = float(np.mean(dual_area[positive]))
    sample_weight = dual_area / max(scale, np.finfo(float).tiny)
    # Extremely irregular boundary cells should not determine the optimizer.
    sample_weight = np.clip(sample_weight, 0.10, 10.0)

    return {
        "target_h": target_h.astype(np.float32),
        "target_log_ratio": target_log_ratio.astype(np.float32),
        "sample_weight": sample_weight.astype(np.float32),
        "element_size": element_size.astype(np.float32),
        "nodal_size_raw": nodal_size.astype(np.float32),
        "nodal_dual_area": dual_area.astype(np.float32),
    }


__all__ = ["mesh_size_targets"]
