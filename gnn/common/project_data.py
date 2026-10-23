from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .meshing import edge_features, line_graph_edge_index, triangles_to_edges


def crack_segments_from_sample(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """Return physical crack segments as coordinates, supporting all crack families."""
    points = np.asarray(data["crack_polyline_points"], dtype=np.float64)
    offsets = np.asarray(data["crack_polyline_offsets"], dtype=np.int64)
    segments: list[np.ndarray] = []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        polyline = points[int(begin):int(end)]
        if len(polyline) >= 2:
            segments.extend(np.stack([polyline[:-1], polyline[1:]], axis=1))
    if not segments and "crack_graph_vertices" in data and "crack_graph_segments" in data:
        vertices = np.asarray(data["crack_graph_vertices"], dtype=np.float64)
        graph_edges = np.asarray(data["crack_graph_segments"], dtype=np.int64)
        segments = list(vertices[graph_edges])
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)


def _consecutive_collinear_edges(
    points: np.ndarray, start: np.ndarray, end: np.ndarray, tol: float
) -> list[tuple[int, int]]:
    direction = end - start
    length2 = float(direction @ direction)
    if length2 <= tol * tol:
        return []
    relative = points - start
    projection = (relative @ direction) / length2
    distance = np.abs(direction[0] * relative[:, 1] - direction[1] * relative[:, 0]) / np.sqrt(length2)
    ids = np.flatnonzero((distance <= tol) & (projection >= -tol) & (projection <= 1.0 + tol))
    ordered = ids[np.argsort(projection[ids])]
    return [(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(ordered[:-1], ordered[1:])]


def mandatory_project_edges(
    points: np.ndarray, width: float, height: float, crack_segments: np.ndarray, tol: float = 2e-6
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    boundary_specs = [
        (np.flatnonzero(np.abs(points[:, 0]) <= tol), points[:, 1]),
        (np.flatnonzero(np.abs(points[:, 0] - width) <= tol), points[:, 1]),
        (np.flatnonzero(np.abs(points[:, 1]) <= tol), points[:, 0]),
        (np.flatnonzero(np.abs(points[:, 1] - height) <= tol), points[:, 0]),
    ]
    for ids, coordinate in boundary_specs:
        ordered = ids[np.argsort(coordinate[ids])]
        edges.update((min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(ordered[:-1], ordered[1:]))
    for segment in crack_segments:
        edges.update(_consecutive_collinear_edges(points, segment[0], segment[1], tol))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def _crossing_mask(points: np.ndarray, candidates: np.ndarray, segment: np.ndarray, tol: float = 2e-6) -> np.ndarray:
    a = points[candidates[:, 0]].astype(np.float64)
    b = points[candidates[:, 1]].astype(np.float64)
    c, d = segment
    direction = d - c
    length2 = float(direction @ direction)
    if length2 <= tol * tol:
        return np.zeros(len(candidates), dtype=bool)

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    o1 = cross(b - a, c[None, :] - a)
    o2 = cross(b - a, d[None, :] - a)
    o3 = cross(np.broadcast_to(direction, a.shape), a - c)
    o4 = cross(np.broadcast_to(direction, b.shape), b - c)
    proper = (o1 * o2 < -1e-12) & (o3 * o4 < -1e-12)

    # Touching a crack at a candidate endpoint is valid and must not be
    # misclassified as crossing because of float32/float64 round-off.
    for endpoint in (a, b):
        relative = endpoint - c
        projection = (relative @ direction) / length2
        distance = np.abs(direction[0] * relative[:, 1] - direction[1] * relative[:, 0]) / np.sqrt(length2)
        proper &= ~((distance <= tol) & (projection >= -tol) & (projection <= 1.0 + tol))
    return proper


def project_candidate_edges(
    points: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    """Teacher-free kNN candidates for arbitrary crack graphs."""
    _, neighbors = cKDTree(points).query(points, k=min(k + 1, len(points)))
    row = np.repeat(np.arange(len(points), dtype=np.int64), neighbors.shape[1] - 1)
    col = neighbors[:, 1:].reshape(-1).astype(np.int64)
    raw = np.sort(np.column_stack([row, col]), axis=1)
    candidates = np.unique(raw[raw[:, 0] != raw[:, 1]], axis=0)
    keep = np.ones(len(candidates), dtype=bool)
    for segment in crack_segments:
        keep &= ~_crossing_mask(points, candidates, segment)
    mandatory = mandatory_project_edges(points, width, height, crack_segments)
    return np.unique(np.vstack([candidates[keep], mandatory]), axis=0).astype(np.int64)


def convert_project_sample(source: Path, k: int = 16) -> dict[str, np.ndarray | float | str]:
    data = np.load(source, allow_pickle=False)
    points = np.asarray(data["points"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int64)
    width, height = float(data["W"]), float(data["H"])
    crack_segments = crack_segments_from_sample(data)
    gt_edges = triangles_to_edges(triangles)
    candidates = project_candidate_edges(points, width, height, crack_segments, k=k)
    gt_set = {tuple(map(int, edge)) for edge in gt_edges}
    labels = np.asarray([int(tuple(map(int, edge)) in gt_set) for edge in candidates], dtype=np.int64)
    features = edge_features(points, candidates, width, height)
    return {
        "nodes": points,
        "triangles": triangles,
        "gt_edges": gt_edges,
        "candidates": candidates,
        "labels": labels,
        "features": features,
        "line_edge_index": line_graph_edge_index(candidates),
        "width": width,
        "height": height,
        "crack_segments": crack_segments.astype(np.float32),
        "crack_family": str(data["crack_family"]),
        "edge_side": str(data["edge_side"]),
        "coverage": float(labels.sum() / len(gt_edges)),
    }

