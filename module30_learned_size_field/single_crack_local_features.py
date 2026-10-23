"""Segment-count-invariant local features for one edge crack.

This schema is intentionally restricted to a single, non-branched crack.  It
is designed for the strict experiment in which only straight cracks are used
for training and gently curved or mildly kinked cracks are held out for OOD
testing.  Every feature has a fixed physical normalization; no train-set
standard deviation is used at runtime.
"""

from __future__ import annotations

import math

import numpy as np


SINGLE_CRACK_LOCAL_MODEL_FAMILY = "single-crack-local-size-field-v4"

SINGLE_CRACK_LOCAL_FEATURE_NAMES = [
    "log_distance_to_crack",
    "signed_log_distance_to_crack",
    "log_distance_to_tip",
    "log_distance_to_entry",
    "crack_arc_fraction",
    "local_tangent_x",
    "local_tangent_y",
    "distance_to_left_edge",
    "distance_to_right_edge",
    "distance_to_bottom_edge",
    "distance_to_top_edge",
    "crack_length_over_diagonal",
    "width_over_diagonal",
    "height_over_diagonal",
]


def _oriented_polyline(geometry) -> np.ndarray:
    if len(geometry.crack_polylines) != 1:
        raise ValueError("single-crack-local-v4 requires exactly one crack polyline")
    if bool(geometry.has_branches):
        raise ValueError("single-crack-local-v4 does not support branched cracks")
    if len(geometry.crack_entries) != 1 or len(geometry.crack_tips) != 1:
        raise ValueError(
            "single-crack-local-v4 requires one boundary entry and one crack tip",
        )
    polyline = np.asarray(geometry.crack_polylines[0], dtype=float).copy()
    entry = np.asarray(geometry.crack_entries[0], dtype=float)
    if np.linalg.norm(polyline[-1] - entry) < np.linalg.norm(polyline[0] - entry):
        polyline = polyline[::-1].copy()
    return polyline


def _log_distance(distance: np.ndarray, h_min: float, diagonal: float) -> np.ndarray:
    values = np.maximum(np.asarray(distance, dtype=float), 0.0)
    denominator = math.log1p(diagonal / h_min)
    return np.clip(np.log1p(values / h_min) / denominator, 0.0, 1.0)


def build_single_crack_local_features(
    points: np.ndarray,
    geometry,
    *,
    h_min: float,
) -> np.ndarray:
    """Return 14 bounded local features independent of polyline segmentation."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError("points contain NaN or infinity")
    if h_min <= 0.0:
        raise ValueError("h_min must be positive")

    polyline = _oriented_polyline(geometry)
    starts = polyline[:-1]
    vectors = polyline[1:] - polyline[:-1]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("crack polyline contains a zero-length segment")
    tangents = vectors / lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])

    # Query all segments.  A single-crack polyline normally has only a handful
    # of segments, so this vectorized form is both simple and fast.
    relative = values[:, None, :] - starts[None, :, :]
    parameter = np.einsum("nsi,si->ns", relative, vectors)
    parameter /= (lengths * lengths)[None, :]
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = starts[None, :, :] + parameter[:, :, None] * vectors[None, :, :]
    offsets = values[:, None, :] - closest
    distance2 = np.einsum("nsi,nsi->ns", offsets, offsets)
    nearest = np.argmin(distance2, axis=1)
    row = np.arange(len(values))

    local_tangent = tangents[nearest]
    local_normal = np.column_stack((-local_tangent[:, 1], local_tangent[:, 0]))
    nearest_offset = offsets[row, nearest]
    signed_distance = np.einsum("ni,ni->n", nearest_offset, local_normal)
    crack_distance = np.sqrt(np.maximum(distance2[row, nearest], 0.0))
    arc_length = cumulative[nearest] + parameter[row, nearest] * lengths[nearest]
    arc_fraction = np.clip(arc_length / total_length, 0.0, 1.0)

    diagonal = max(float(math.hypot(geometry.W, geometry.H)), 1.0e-30)
    tip = np.asarray(geometry.crack_tips[0], dtype=float)
    entry = np.asarray(geometry.crack_entries[0], dtype=float)
    tip_distance = np.linalg.norm(values - tip, axis=1)
    entry_distance = np.linalg.norm(values - entry, axis=1)
    log_crack_distance = _log_distance(crack_distance, h_min, diagonal)
    signed_log_distance = np.sign(signed_distance) * log_crack_distance

    features = np.column_stack(
        (
            log_crack_distance,
            signed_log_distance,
            _log_distance(tip_distance, h_min, diagonal),
            _log_distance(entry_distance, h_min, diagonal),
            arc_fraction,
            local_tangent[:, 0],
            local_tangent[:, 1],
            np.clip(values[:, 0] / float(geometry.W), 0.0, 1.0),
            np.clip((float(geometry.W) - values[:, 0]) / float(geometry.W), 0.0, 1.0),
            np.clip(values[:, 1] / float(geometry.H), 0.0, 1.0),
            np.clip((float(geometry.H) - values[:, 1]) / float(geometry.H), 0.0, 1.0),
            np.full(len(values), total_length / diagonal),
            np.full(len(values), float(geometry.W) / diagonal),
            np.full(len(values), float(geometry.H) / diagonal),
        )
    ).astype(np.float32)
    if features.shape[1] != len(SINGLE_CRACK_LOCAL_FEATURE_NAMES):
        raise RuntimeError("single-crack-local-v4 feature width mismatch")
    if not np.isfinite(features).all():
        raise ValueError("single-crack-local-v4 features contain NaN or infinity")
    if np.max(np.abs(features)) > 1.0 + 1.0e-6:
        raise ValueError("single-crack-local-v4 feature escaped its fixed range")
    return features


__all__ = [
    "SINGLE_CRACK_LOCAL_FEATURE_NAMES",
    "SINGLE_CRACK_LOCAL_MODEL_FAMILY",
    "build_single_crack_local_features",
]
