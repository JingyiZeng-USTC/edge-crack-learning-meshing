"""Local-invariant and multiscale node features for one crack."""

from __future__ import annotations

import math

import numpy as np

from crack_graph_geometry import CrackGraphGeometry
from module10_fem.boundary import BoundaryConditions
from module30_learned_size_field.single_crack_local_features import (
    SINGLE_CRACK_LOCAL_FEATURE_NAMES,
    build_single_crack_local_features,
)


FOURIER_SOURCE_NAMES = (
    "distance_to_left_edge",
    "distance_to_bottom_edge",
    "log_distance_to_crack",
    "log_distance_to_tip",
)
FOURIER_FREQUENCIES = (1, 2, 4, 8)
FOURIER_FEATURE_NAMES = [
    f"{function}_{source}_f{frequency}"
    for source in FOURIER_SOURCE_NAMES
    for frequency in FOURIER_FREQUENCIES
    for function in ("sin", "cos")
]

CRACK_TIP_BASIS_NAMES = [
    "sqrt_r_cos_half_theta",
    "sqrt_r_sin_half_theta",
    "sqrt_r_cos_half_theta_cos_theta",
    "sqrt_r_sin_half_theta_cos_theta",
    "sqrt_r_cos_half_theta_sin_theta",
    "sqrt_r_sin_half_theta_sin_theta",
    "r_three_halves_cos_three_half_theta",
    "r_three_halves_sin_three_half_theta",
]

ROLE_FEATURE_NAMES = [
    "is_left_boundary",
    "is_right_boundary",
    "is_bottom_boundary",
    "is_top_boundary",
    "is_prescribed_ux",
    "is_prescribed_uy",
    "prescribed_ux_scaled",
    "prescribed_uy_scaled",
    "is_crack_positive_face",
    "is_crack_negative_face",
    "is_crack_tip_node",
    "is_generic_crack_face_node",
    "log10_E_minus_11",
    "poisson_ratio",
    "log10_thickness_plus_2",
]

FEATURE_NAMES = (
    list(SINGLE_CRACK_LOCAL_FEATURE_NAMES)
    + FOURIER_FEATURE_NAMES
    + CRACK_TIP_BASIS_NAMES
    + ROLE_FEATURE_NAMES
)


def _scalar(metadata: dict, key: str) -> float:
    if key not in metadata:
        raise ValueError(f"Pure mesh metadata is missing {key!r}")
    return float(np.asarray(metadata[key]).reshape(()))


def geometry_from_mesh_metadata(metadata: dict) -> CrackGraphGeometry:
    required = {"W", "H", "crack_polyline_points", "crack_polyline_offsets"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(
            "Coordinate FCNN-SM requires an NPZ pure mesh with crack geometry; "
            f"missing {sorted(missing)}"
        )
    points = np.asarray(metadata["crack_polyline_points"], dtype=float)
    offsets = np.asarray(metadata["crack_polyline_offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(points):
        raise ValueError("Invalid crack_polyline_offsets in pure mesh")
    polylines = [points[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return CrackGraphGeometry(_scalar(metadata, "W"), _scalar(metadata, "H"), polylines)


def _node_marker(metadata: dict, key: str, n_nodes: int) -> np.ndarray:
    marker = np.zeros(n_nodes, dtype=np.float64)
    if key not in metadata:
        return marker
    indices = np.asarray(metadata[key], dtype=np.int64).reshape(-1)
    if len(indices):
        if indices.min() < 0 or indices.max() >= n_nodes:
            raise ValueError(f"{key} contains a node outside the mesh")
        marker[indices] = 1.0
    return marker


def _fourier_features(local_features: np.ndarray) -> np.ndarray:
    lookup = {
        name: local_features[:, SINGLE_CRACK_LOCAL_FEATURE_NAMES.index(name)]
        for name in FOURIER_SOURCE_NAMES
    }
    columns = []
    for source in FOURIER_SOURCE_NAMES:
        values = lookup[source]
        for frequency in FOURIER_FREQUENCIES:
            phase = math.pi * float(frequency) * values
            columns.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(columns)


def _crack_tip_basis(
    points: np.ndarray,
    geometry: CrackGraphGeometry,
    positive_face: np.ndarray,
    negative_face: np.ndarray,
) -> np.ndarray:
    """Geometry-only Williams-type functions used by the neural trial space."""
    if len(geometry.crack_polylines) != 1 or len(geometry.crack_tips) != 1:
        raise ValueError("Coordinate FCNN-SM currently expects one non-branched crack")
    polyline = np.asarray(geometry.crack_polylines[0], dtype=float)
    tip = np.asarray(geometry.crack_tips[0], dtype=float)
    if np.linalg.norm(polyline[-1] - tip) <= np.linalg.norm(polyline[0] - tip):
        tangent = polyline[-1] - polyline[-2]
    else:
        tangent = polyline[0] - polyline[1]
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-30)
    normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    delta = np.asarray(points, dtype=float) - tip[None, :]
    axial = delta @ tangent
    transverse = delta @ normal
    scale = max(float(np.hypot(geometry.W, geometry.H)), 1.0e-30)
    face_tolerance = 1.0e-12 * scale
    transverse = transverse.copy()
    transverse[(positive_face > 0.5) & (np.abs(transverse) <= face_tolerance)] = face_tolerance
    transverse[(negative_face > 0.5) & (np.abs(transverse) <= face_tolerance)] = -face_tolerance
    radius = np.hypot(axial, transverse) / scale
    theta = np.arctan2(transverse, axial)
    sqrt_radius = np.sqrt(np.maximum(radius, 0.0))
    half = 0.5 * theta
    cos_half = np.cos(half)
    sin_half = np.sin(half)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    radius_three_halves = radius * sqrt_radius
    return np.column_stack((
        sqrt_radius * cos_half,
        sqrt_radius * sin_half,
        sqrt_radius * cos_half * cos_theta,
        sqrt_radius * sin_half * cos_theta,
        sqrt_radius * cos_half * sin_theta,
        sqrt_radius * sin_half * sin_theta,
        radius_three_halves * np.cos(1.5 * theta),
        radius_three_halves * np.sin(1.5 * theta),
    ))


def build_node_features(
    points: np.ndarray,
    metadata: dict,
    boundary: BoundaryConditions,
    *,
    young_modulus: float,
    poisson_ratio: float,
    thickness: float,
    displacement_scale: float,
    local_feature_h_min: float = 0.004,
) -> tuple[np.ndarray, CrackGraphGeometry]:
    """Create fixed-width features for a single straight/curved/kinked crack."""
    pts = np.asarray(points, dtype=float)[:, :2]
    n_nodes = len(pts)
    geometry = geometry_from_mesh_metadata(metadata)
    local = build_single_crack_local_features(
        pts, geometry, h_min=float(local_feature_h_min),
    ).astype(np.float64)
    fourier = _fourier_features(local).astype(np.float64)

    named = boundary.named_nodes
    boundary_columns = []
    for name in ("LEFT", "RIGHT", "BOTTOM", "TOP"):
        marker = np.zeros(n_nodes, dtype=np.float64)
        marker[np.asarray(named.get(name, ()), dtype=np.int64)] = 1.0
        boundary_columns.append(marker)

    prescribed_mask = np.zeros(2 * n_nodes, dtype=np.float64)
    prescribed_values = np.zeros(2 * n_nodes, dtype=np.float64)
    prescribed_mask[boundary.prescribed_dofs] = 1.0
    prescribed_values[boundary.prescribed_dofs] = boundary.prescribed_values
    prescribed_mask = prescribed_mask.reshape(n_nodes, 2)
    prescribed_values = prescribed_values.reshape(n_nodes, 2) / float(displacement_scale)

    positive = _node_marker(metadata, "crack_upper_nodes", n_nodes)
    negative = _node_marker(metadata, "crack_lower_nodes", n_nodes)
    tips = _node_marker(metadata, "crack_tip_nodes", n_nodes)
    generic_face = _node_marker(metadata, "crack_face_nodes", n_nodes)
    crack_tip_basis = _crack_tip_basis(pts, geometry, positive, negative)
    material = np.column_stack(
        (
            np.full(n_nodes, math.log10(float(young_modulus)) - 11.0),
            np.full(n_nodes, float(poisson_ratio)),
            np.full(n_nodes, math.log10(float(thickness)) + 2.0),
        )
    )
    roles = np.column_stack(
        (
            *boundary_columns,
            prescribed_mask[:, 0],
            prescribed_mask[:, 1],
            prescribed_values[:, 0],
            prescribed_values[:, 1],
            positive,
            negative,
            tips,
            generic_face,
            material,
        )
    )
    features = np.concatenate((local, fourier, crack_tip_basis, roles), axis=1)
    if features.shape != (n_nodes, len(FEATURE_NAMES)):
        raise RuntimeError(
            f"Feature shape {features.shape} does not match schema width {len(FEATURE_NAMES)}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Coordinate FCNN-SM node features contain NaN or infinity")
    return features.astype(np.float64, copy=False), geometry


__all__ = [
    "FEATURE_NAMES",
    "CRACK_TIP_BASIS_NAMES",
    "FOURIER_FEATURE_NAMES",
    "build_node_features",
    "geometry_from_mesh_metadata",
]
