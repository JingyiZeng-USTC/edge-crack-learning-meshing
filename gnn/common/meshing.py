from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (
    EdgeCrack, collinear_interior_overlap, edge_crosses_crack,
    point_in_plate, proper_segment_intersection,
)


def _unique_rows(points: np.ndarray, decimals: int = 10) -> np.ndarray:
    rounded = np.round(points, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]


def sample_nodes(crack: EdgeCrack, spacing: float, seed: int) -> np.ndarray:
    """Jittered interior grid plus exact plate/crack boundary points."""
    rng = np.random.default_rng(seed)
    xs = np.arange(spacing, crack.width, spacing)
    ys = np.arange(spacing, crack.height, spacing)
    xx, yy = np.meshgrid(xs, ys)
    interior = np.column_stack([xx.ravel(), yy.ravel()])
    interior += rng.uniform(-0.22 * spacing, 0.22 * spacing, interior.shape)
    interior[:, 0] = np.clip(interior[:, 0], 0.02 * spacing, crack.width - 0.02 * spacing)
    interior[:, 1] = np.clip(interior[:, 1], 0.02 * spacing, crack.height - 0.02 * spacing)

    bx = np.linspace(0.0, crack.width, max(3, int(round(crack.width / spacing)) + 1))
    by = np.linspace(0.0, crack.height, max(3, int(round(crack.height / spacing)) + 1))
    boundary = np.vstack(
        [
            np.column_stack([bx, np.zeros_like(bx)]),
            np.column_stack([bx, np.full_like(bx, crack.height)]),
            np.column_stack([np.zeros_like(by), by]),
            np.column_stack([np.full_like(by, crack.width), by]),
        ]
    )
    n_crack = max(3, int(np.ceil(crack.length / spacing)) + 1)
    t = np.linspace(0.0, 1.0, n_crack)[:, None]
    crack_nodes = crack.start[None, :] * (1.0 - t) + crack.tip[None, :] * t
    return _unique_rows(np.vstack([boundary, interior, crack_nodes])).astype(np.float32)


def triangles_to_edges(triangles: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tri in triangles:
        for i, j in combinations(map(int, tri), 2):
            edges.add((min(i, j), max(i, j)))
    return np.asarray(sorted(edges), dtype=np.int64)


def reference_mesh(nodes: np.ndarray, crack: EdgeCrack) -> tuple[np.ndarray, np.ndarray]:
    """True constrained Delaunay teacher; the crack is an internal segment, not a hole."""
    try:
        import triangle as tr
    except ImportError as exc:
        raise RuntimeError(
            "The CDT teacher requires the 'triangle' package. Run: pip install triangle"
        ) from exc
    segments = mandatory_constraint_edges(nodes, crack)
    result = tr.triangulate({"vertices": np.asarray(nodes), "segments": segments}, "pQ")
    if "triangles" not in result:
        raise RuntimeError("Triangle failed to produce a constrained triangulation")
    output_nodes = np.asarray(result["vertices"], dtype=np.float32)
    if output_nodes.shape != nodes.shape or not np.allclose(output_nodes, nodes, atol=1e-7):
        raise RuntimeError(
            "CDT unexpectedly inserted or reordered nodes. Disable refinement flags; "
            "the GNN pipeline requires a fixed input node set."
        )
    triangles = np.asarray(result["triangles"], dtype=np.int64)
    # Defensive validation: the mesher must preserve every supplied segment.
    mesh_edges = triangles_to_edges(triangles)
    mesh_edge_set = {tuple(map(int, edge)) for edge in mesh_edges}
    missing = [tuple(map(int, edge)) for edge in segments if tuple(map(int, edge)) not in mesh_edge_set]
    if missing:
        raise RuntimeError(f"CDT omitted {len(missing)} required boundary/crack segments")
    return triangles, mesh_edges


def mandatory_constraint_edges(nodes: np.ndarray, crack: EdgeCrack, tol: float = 1e-6) -> np.ndarray:
    """Known consecutive plate-boundary and crack-polyline edges (no teacher data)."""
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    groups.append((np.flatnonzero(np.abs(nodes[:, 0]) < tol), nodes[:, 1]))
    groups.append((np.flatnonzero(np.abs(nodes[:, 0] - crack.width) < tol), nodes[:, 1]))
    groups.append((np.flatnonzero(np.abs(nodes[:, 1]) < tol), nodes[:, 0]))
    groups.append((np.flatnonzero(np.abs(nodes[:, 1] - crack.height) < tol), nodes[:, 0]))
    direction = crack.tip - crack.start
    length2 = float(direction @ direction)
    rel = nodes - crack.start
    projection = (rel @ direction) / length2
    distance = np.abs(np.cross(direction, rel)) / np.sqrt(length2)
    crack_ids = np.flatnonzero((distance < tol) & (projection >= -tol) & (projection <= 1.0 + tol))
    groups.append((crack_ids, projection))
    edges: set[tuple[int, int]] = set()
    for ids, coordinate in groups:
        ordered = ids[np.argsort(coordinate[ids])]
        for u, v in zip(ordered[:-1], ordered[1:]):
            if u != v:
                edges.add((min(int(u), int(v)), max(int(u), int(v))))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def candidate_edges(nodes: np.ndarray, crack: EdgeCrack, k: int = 12) -> np.ndarray:
    """Teacher-free kNN candidates filtered only by the known crack geometry."""
    tree = cKDTree(nodes)
    _, nbrs = tree.query(nodes, k=min(k + 1, len(nodes)))
    candidates = {tuple(map(int, e)) for e in mandatory_constraint_edges(nodes, crack)}
    for i, row in enumerate(np.atleast_2d(nbrs)):
        for j in row[1:]:
            if i != int(j):
                edge = (min(i, int(j)), max(i, int(j)))
                if not edge_crosses_crack(nodes[edge[0]], nodes[edge[1]], crack):
                    candidates.add(edge)
    return np.asarray(sorted(candidates), dtype=np.int64)


def edge_labels(candidates: np.ndarray, gt_edges: np.ndarray) -> np.ndarray:
    positives = {tuple(map(int, e)) for e in gt_edges}
    return np.asarray([int(tuple(map(int, e)) in positives) for e in candidates], dtype=np.int64)


def edge_features(nodes: np.ndarray, candidates: np.ndarray, width: float, height: float) -> np.ndarray:
    p0, p1 = nodes[candidates[:, 0]].copy(), nodes[candidates[:, 1]].copy()
    scale = np.array([width, height], dtype=np.float32)
    p0, p1 = p0 / scale, p1 / scale
    # A mesh edge is unordered. Lexicographic coordinate ordering removes the
    # accidental dependence on node numbering present in the first version.
    swap = (p0[:, 0] > p1[:, 0]) | ((p0[:, 0] == p1[:, 0]) & (p0[:, 1] > p1[:, 1]))
    p0[swap], p1[swap] = p1[swap].copy(), p0[swap].copy()
    delta = p1 - p0
    length = np.linalg.norm(p1 - p0, axis=1, keepdims=True)
    midpoint = 0.5 * (p0 + p1)
    incident_lengths: dict[int, list[float]] = {i: [] for i in range(len(nodes))}
    for edge, value in zip(candidates, length[:, 0]):
        incident_lengths[int(edge[0])].append(float(value))
        incident_lengths[int(edge[1])].append(float(value))
    local_scale = np.asarray(
        [0.5 * (np.mean(incident_lengths[int(u)]) + np.mean(incident_lengths[int(v)])) for u, v in candidates],
        dtype=np.float32,
    )[:, None]
    relative_length = length / np.maximum(local_scale, 1e-8)
    return np.hstack([p0, p1, length, np.abs(delta), midpoint, relative_length]).astype(np.float32)


def line_graph_edge_index(candidates: np.ndarray) -> np.ndarray:
    incident: dict[int, list[int]] = {}
    for edge_id, (u, v) in enumerate(candidates):
        incident.setdefault(int(u), []).append(edge_id)
        incident.setdefault(int(v), []).append(edge_id)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for ids in incident.values():
        array = np.asarray(ids, dtype=np.int64)
        if len(array) < 2:
            continue
        source = np.repeat(array, len(array))
        target = np.tile(array, len(array))
        mask = source != target
        sources.append(source[mask])
        targets.append(target[mask])
    if not sources:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([np.concatenate(sources), np.concatenate(targets)])


def recover_triangles(nodes: np.ndarray, edges: np.ndarray, crack: EdgeCrack) -> np.ndarray:
    edge_set = {tuple(map(int, e)) for e in edges}
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for u, v in edge_set:
        adjacency[u].add(v)
        adjacency[v].add(u)
    triangles: list[tuple[int, int, int]] = []
    for u in range(len(nodes)):
        for v, w in combinations(sorted(x for x in adjacency[u] if x > u), 2):
            if (min(v, w), max(v, w)) not in edge_set:
                continue
            pts = nodes[[u, v, w]]
            area2 = abs(float(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
            if area2 <= 1e-10 or not point_in_plate(pts.mean(axis=0), crack):
                continue
            if any(edge_crosses_crack(pts[i], pts[j], crack) for i, j in ((0, 1), (1, 2), (2, 0))):
                continue
            triangles.append((u, v, w))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def greedy_planar_edges(
    nodes: np.ndarray,
    candidates: np.ndarray,
    probability: np.ndarray,
    crack: EdgeCrack,
    fill_to_maximal: bool = True,
) -> np.ndarray:
    """Confidence-ordered straight-line planar graph with known constraints first."""
    mandatory = {tuple(map(int, e)) for e in mandatory_constraint_edges(nodes, crack)}
    score = {tuple(map(int, e)): float(p) for e, p in zip(candidates, probability)}
    ordered = sorted((tuple(map(int, e)) for e in candidates), key=lambda e: (e not in mandatory, -score[e]))
    selected: list[tuple[int, int]] = []
    for edge in ordered:
        u, v = edge
        if edge_crosses_crack(nodes[u], nodes[v], crack):
            continue
        if any(
            collinear_interior_overlap(nodes[u], nodes[v], nodes[a], nodes[b])
            or (len({u, v, a, b}) == 4 and proper_segment_intersection(nodes[u], nodes[v], nodes[a], nodes[b]))
            for a, b in selected
        ):
            continue
        if fill_to_maximal or edge in mandatory or score[edge] >= 0.5:
            selected.append(edge)
    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def _planar_inner_faces(nodes: np.ndarray, edges: np.ndarray) -> list[list[int]]:
    """Walk the bounded faces of a planar straight-line graph."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))
    for u, neighbors in adjacency.items():
        neighbors.sort(key=lambda v: np.arctan2(nodes[v, 1] - nodes[u, 1], nodes[v, 0] - nodes[u, 0]))
    visited: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    directed_edges: list[tuple[int, int]] = []
    for edge in edges:
        u, v = map(int, edge)
        directed_edges.extend([(u, v), (v, u)])
    for u, v in directed_edges:
        if (u, v) in visited:
            continue
        face: list[int] = []
        a, b = u, v
        for _ in range(2 * len(edges) + 1):
            if (a, b) in visited:
                break
            visited.add((a, b))
            face.append(a)
            neighbors = adjacency[b]
            if not neighbors:
                break
            idx = neighbors.index(a)
            a, b = b, neighbors[(idx - 1) % len(neighbors)]
            if (a, b) == (u, v):
                pts = nodes[face]
                signed_area2 = float(
                    np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))
                )
                if len(face) >= 3 and signed_area2 > 1e-10:
                    faces.append(face)
                break
    return faces


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-10) -> bool:
    o1 = np.cross(b - a, p - a)
    o2 = np.cross(c - b, p - b)
    o3 = np.cross(a - c, p - c)
    return bool(o1 >= -eps and o2 >= -eps and o3 >= -eps)


def _ear_clip_face(nodes: np.ndarray, face: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate one CCW simple face, preferring the shortest valid ear."""
    polygon = list(face)
    result: list[tuple[int, int, int]] = []
    while len(polygon) > 3:
        ears: list[tuple[float, int, tuple[int, int, int]]] = []
        for i, current in enumerate(polygon):
            previous = polygon[(i - 1) % len(polygon)]
            following = polygon[(i + 1) % len(polygon)]
            a, b, c = nodes[[previous, current, following]]
            if np.cross(b - a, c - b) <= 1e-10:
                continue
            if any(
                _point_in_triangle(nodes[other], a, b, c)
                for other in polygon
                if other not in (previous, current, following)
            ):
                continue
            diagonal = float(np.linalg.norm(c - a))
            ears.append((diagonal, i, (previous, current, following)))
        if not ears:
            break
        _, index, triangle = min(ears, key=lambda item: item[0])
        result.append(tuple(sorted(triangle)))
        polygon.pop(index)
    if len(polygon) == 3:
        result.append(tuple(sorted(polygon)))
    return result


def triangular_faces(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Extract planar faces and triangulate any residual polygonal faces."""
    triangles: set[tuple[int, int, int]] = set()
    for face in _planar_inner_faces(nodes, edges):
        triangles.update(_ear_clip_face(nodes, face))
    return np.asarray(sorted(triangles), dtype=np.int64).reshape(-1, 3)


def strict_triangular_faces(
    nodes: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract empty three-edge cycles from the GNN-selected planar graph.

    Every returned side must already be present in ``edges``.  No diagonal,
    node, or face is created.  Empty-cycle extraction is more robust than a
    half-edge angle walk for extremely short boundary segments: a valid thin
    triangle remains a graph 3-clique even when two incident directions are
    numerically almost collinear.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_set = {tuple(sorted(map(int, edge))) for edge in edges}
    adjacency: list[set[int]] = [set() for _ in range(len(nodes))]
    for u, v in edge_set:
        adjacency[u].add(v)
        adjacency[v].add(u)

    scale = max(float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])), 1.0)
    area_tolerance = 1.0e-12 * scale * scale
    point_tree = cKDTree(nodes)
    triangles: set[tuple[int, int, int]] = set()
    rejected_nonempty_cycles = 0
    rejected_degenerate_cycles = 0

    for u in range(len(nodes)):
        higher = sorted(v for v in adjacency[u] if v > u)
        for v, w in combinations(higher, 2):
            if (min(v, w), max(v, w)) not in edge_set:
                continue
            a, b, c = nodes[[u, v, w]]
            area2 = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if abs(area2) <= area_tolerance:
                rejected_degenerate_cycles += 1
                continue
            centroid = (a + b + c) / 3.0
            radius = max(
                float(np.linalg.norm(a - centroid)),
                float(np.linalg.norm(b - centroid)),
                float(np.linalg.norm(c - centroid)),
            )
            nearby = point_tree.query_ball_point(centroid, radius + 1.0e-12 * scale)
            nonempty = False
            for point_id in nearby:
                if point_id in (u, v, w):
                    continue
                p = nodes[point_id]
                first, second = b - a, p - a
                o1 = float(first[0] * second[1] - first[1] * second[0])
                first, second = c - b, p - b
                o2 = float(first[0] * second[1] - first[1] * second[0])
                first, second = a - c, p - c
                o3 = float(first[0] * second[1] - first[1] * second[0])
                strictly_positive = min(o1, o2, o3) > area_tolerance
                strictly_negative = max(o1, o2, o3) < -area_tolerance
                if strictly_positive or strictly_negative:
                    nonempty = True
                    break
            if nonempty:
                rejected_nonempty_cycles += 1
                continue
            triangles.add((u, v, w))

    stats = {
        "bounded_faces": len(triangles),
        "triangular_faces": len(triangles),
        "nontriangular_faces": 0,
        "max_face_vertices": 3 if triangles else 0,
        "rejected_nonempty_three_cycles": rejected_nonempty_cycles,
        "rejected_degenerate_three_cycles": rejected_degenerate_cycles,
        "face_extraction_empty_three_cycles": 1,
    }
    return np.asarray(sorted(triangles), dtype=np.int64).reshape(-1, 3), stats
