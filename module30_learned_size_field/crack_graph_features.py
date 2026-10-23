"""Fixed-width local features for an arbitrary crack constraint graph."""

from __future__ import annotations

import math

import numpy as np


CRACK_GRAPH_FEATURE_NAMES = [
    "x_over_W",
    "y_over_H",
    "nearest_segment_local_t",
    "nearest_segment_signed_normal",
    "distance_to_nearest_segment",
    "nearest_segment_tangent_x",
    "nearest_segment_tangent_y",
    "nearest_segment_normal_x",
    "nearest_segment_normal_y",
    "nearest_segment_turning",
    "distance_to_nearest_tip",
    "nearest_tip_direction_x",
    "nearest_tip_direction_y",
    "distance_to_nearest_branch",
    "nearest_branch_direction_x",
    "nearest_branch_direction_y",
    "distance_to_nearest_entry",
    "distance_to_outer_boundary",
    "nearby_segment_count",
    "analytic_h_over_hmax",
    "W_over_diagonal",
    "H_over_diagonal",
]

# Absolute v2 deliberately excludes the analytic size-field value.  Every
# remaining item is computed from the query point, plate and crack graph only.
ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES = [
    name for name in CRACK_GRAPH_FEATURE_NAMES
    if name != "analytic_h_over_hmax"
]


def build_absolute_crack_graph_features(
    points: np.ndarray,
    geometry,
    *,
    neighborhood_radius: float | None = None,
) -> np.ndarray:
    """Build the 21 geometry-only features used by the absolute v2 model."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    diagonal = max(float(math.hypot(geometry.W, geometry.H)), 1.0e-30)
    local = geometry.local_geometry_query(
        values, neighborhood_radius=neighborhood_radius,
    )
    segment_index = np.asarray(local["segment_index"], dtype=np.int64)
    segment_lengths = np.linalg.norm(
        geometry.segment_coordinates[:, 1] - geometry.segment_coordinates[:, 0],
        axis=1,
    )
    local_t = (
        np.asarray(local["projection"], dtype=float)
        * segment_lengths[segment_index]
        / diagonal
    )
    signed_normal = np.asarray(local["signed_distance"], dtype=float) / diagonal
    tip_distance = np.asarray(local["tip_distance"], dtype=float)
    branch_distance = np.asarray(local["branch_distance"], dtype=float)
    entry_distance = np.asarray(local["entry_distance"], dtype=float)
    tip_distance = np.where(np.isfinite(tip_distance), tip_distance / diagonal, 2.0)
    branch_distance = np.where(
        np.isfinite(branch_distance), branch_distance / diagonal, 2.0,
    )
    entry_distance = np.where(
        np.isfinite(entry_distance), entry_distance / diagonal, 2.0,
    )
    nearby = np.log1p(np.asarray(local["nearby_segment_count"], dtype=float))
    tangent = np.asarray(local["tangent"], dtype=float)
    normal = np.asarray(local["normal"], dtype=float)
    tip_direction = np.asarray(local["tip_direction"], dtype=float)
    branch_direction = np.asarray(local["branch_direction"], dtype=float)
    features = np.column_stack(
        (
            values[:, 0] / max(float(geometry.W), 1.0e-30),
            values[:, 1] / max(float(geometry.H), 1.0e-30),
            local_t,
            signed_normal,
            np.asarray(local["distance"], dtype=float) / diagonal,
            tangent[:, 0],
            tangent[:, 1],
            normal[:, 0],
            normal[:, 1],
            np.asarray(local["turning"], dtype=float),
            tip_distance,
            tip_direction[:, 0],
            tip_direction[:, 1],
            branch_distance,
            branch_direction[:, 0],
            branch_direction[:, 1],
            entry_distance,
            np.asarray(local["boundary_distance"], dtype=float) / diagonal,
            nearby,
            np.full(len(values), float(geometry.W) / diagonal),
            np.full(len(values), float(geometry.H) / diagonal),
        )
    ).astype(np.float32)
    if features.shape[1] != len(ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES):
        raise RuntimeError("Absolute crack-graph feature schema width mismatch")
    if not np.all(np.isfinite(features)):
        raise ValueError("Absolute crack-graph features contain NaN or infinity")
    return features


def build_crack_graph_features(
    points: np.ndarray,
    geometry,
    analytic_h: np.ndarray,
    h_max: float,
    *,
    neighborhood_radius: float | None = None,
) -> np.ndarray:
    """Build features whose width is independent of crack point/segment count."""
    values = np.asarray(points, dtype=float)
    h0 = np.asarray(analytic_h, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(values) != len(h0):
        raise ValueError("analytic_h must contain one value per query point")
    pure = build_absolute_crack_graph_features(
        values, geometry, neighborhood_radius=neighborhood_radius,
    )
    # Preserve the exact legacy 22-column order for residual-v1 checkpoints.
    analytic_column = (
        h0 / max(float(h_max), 1.0e-30)
    ).astype(np.float32)[:, None]
    features = np.concatenate(
        (pure[:, :19], analytic_column, pure[:, 19:]), axis=1,
    ).astype(np.float32, copy=False)
    if features.shape[1] != len(CRACK_GRAPH_FEATURE_NAMES):
        raise RuntimeError("Crack-graph feature schema width mismatch")
    if not np.all(np.isfinite(features)):
        raise ValueError("Crack-graph features contain NaN or infinity")
    return features


__all__ = [
    "CRACK_GRAPH_FEATURE_NAMES",
    "ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES",
    "build_crack_graph_features",
    "build_absolute_crack_graph_features",
]
