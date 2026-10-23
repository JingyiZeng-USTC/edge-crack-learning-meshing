"""Geometry-aware point features for the continuous size-field regressor."""

from __future__ import annotations

import numpy as np

from size_field import point_to_segment_distance_bulk


FEATURE_NAMES = [
    "x_over_W",
    "y_over_H",
    "tip_tangent_coordinate",
    "tip_normal_coordinate",
    "distance_to_tip",
    "distance_to_crack_segment",
    "distance_to_crack_entry",
    "distance_to_plate_boundary",
    "crack_length",
    "sin_theta",
    "cos_theta",
    "edge_left",
    "edge_right",
    "edge_top",
    "edge_bottom",
    "analytic_h_over_hmax",
    "W_over_diagonal",
    "H_over_diagonal",
]


def build_point_features(
    points: np.ndarray,
    geometry,
    analytic_h: np.ndarray,
    h_max: float,
) -> np.ndarray:
    """Build dimensionless features at arbitrary spatial coordinates."""
    pts = np.asarray(points, dtype=float)
    h0 = np.asarray(analytic_h, dtype=float).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(h0) != len(pts):
        raise ValueError("analytic_h must contain one value per point")

    W = float(geometry.W)
    H = float(geometry.H)
    diagonal = max(float(np.hypot(W, H)), 1e-30)
    tip = np.asarray(geometry.crack_tip, dtype=float)
    start = np.asarray(geometry.crack_start, dtype=float)
    tangent = np.asarray(geometry.crack_tangent, dtype=float)
    normal = np.asarray(geometry.crack_normal, dtype=float)
    relative = pts - tip[None, :]

    local_t = relative @ tangent / diagonal
    local_n = relative @ normal / diagonal
    d_tip = np.linalg.norm(relative, axis=1) / diagonal
    d_crack = point_to_segment_distance_bulk(
        pts, start[0], start[1], tip[0], tip[1],
    ) / diagonal
    d_entry = np.linalg.norm(pts - start[None, :], axis=1) / diagonal
    d_boundary = np.minimum.reduce(
        [pts[:, 0], W - pts[:, 0], pts[:, 1], H - pts[:, 1]],
    ) / diagonal

    one_hot = np.zeros((len(pts), 4), dtype=float)
    edge_index = {"left": 0, "right": 1, "top": 2, "bottom": 3}
    one_hot[:, edge_index[str(geometry.edge_side)]] = 1.0
    constant_W = np.full(len(pts), W / diagonal)
    constant_H = np.full(len(pts), H / diagonal)

    return np.column_stack(
        [
            pts[:, 0] / max(W, 1e-30),
            pts[:, 1] / max(H, 1e-30),
            local_t,
            local_n,
            d_tip,
            d_crack,
            d_entry,
            np.maximum(d_boundary, 0.0),
            np.full(len(pts), float(geometry.crack_length) / diagonal),
            np.full(len(pts), np.sin(float(geometry.theta))),
            np.full(len(pts), np.cos(float(geometry.theta))),
            one_hot,
            h0 / max(float(h_max), 1e-30),
            constant_W,
            constant_H,
        ],
    ).astype(np.float32)
