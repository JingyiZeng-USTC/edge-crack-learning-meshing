"""Module 9: split a finite crack into two coincident, disconnected faces."""

from __future__ import annotations

import math
from typing import Dict, Set, Tuple

import numpy as np

_EPS = 1e-12


def _pt_seg(px, py, sx, sy, tx, ty):
    dx, dy = tx - sx, ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-30:
        return math.hypot(px - sx, py - sy)
    u = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / L2))
    return math.hypot(px - (sx + u * dx), py - (sy + u * dy))


def _ek(i: int, j: int) -> Tuple[int, int]:
    return (min(int(i), int(j)), max(int(i), int(j)))


def duplicate_crack_nodes(triangles, points, geometry, eps=1e-8):
    """Create two coincident crack faces by duplicating crack-line nodes.

    The crack tip remains shared. For an edge crack, the mouth node is
    duplicated so the crack is open at the external boundary.

    Returns
    -------
    triangles_new, points_new, crack_info
    """
    triangles = np.asarray(triangles, dtype=np.int32)
    points = np.asarray(points, dtype=float)

    if len(points) < 3 or len(triangles) < 1:
        return triangles, points, {}

    sx, sy = map(float, geometry.crack_start)
    tx, ty = map(float, geometry.crack_tip)
    nx, ny = map(float, geometry.crack_normal)

    dx, dy = tx - sx, ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-30:
        raise ValueError("Crack start and tip are coincident.")

    # Select only nodes on the finite crack segment and sort them from mouth to tip.
    dist = np.array([_pt_seg(x, y, sx, sy, tx, ty) for x, y in points])
    projection = ((points[:, 0] - sx) * dx + (points[:, 1] - sy) * dy) / L2
    on_segment = (dist <= eps) & (projection >= -eps) & (projection <= 1.0 + eps)
    all_crack_nodes = np.where(on_segment)[0]

    if len(all_crack_nodes) < 2:
        raise ValueError(
            f"Only {len(all_crack_nodes)} crack-line nodes were found. "
            "The sampler must place an ordered node chain on the crack segment."
        )

    all_crack_nodes = all_crack_nodes[np.argsort(projection[all_crack_nodes])]

    # The closest node to the geometric tip remains shared by both crack faces.
    tip_idx = int(all_crack_nodes[np.argmin(
        np.hypot(points[all_crack_nodes, 0] - tx,
                 points[all_crack_nodes, 1] - ty)
    )])
    if np.linalg.norm(points[tip_idx] - np.array([tx, ty])) > max(eps * 10.0, 1e-10):
        raise ValueError("No mesh node coincides with the geometric crack tip.")

    duplicate_nodes = [int(v) for v in all_crack_nodes if int(v) != tip_idx]
    if not duplicate_nodes:
        raise ValueError("No crack-face nodes are available for duplication.")

    # A triangle may straddle the crack's *supporting line* beyond the finite
    # tip.  Only a straddling triangle incident to a crack-face node is a real
    # bridge that would make node duplication ambiguous.
    signed = (points[:, 0] - sx) * nx + (points[:, 1] - sy) * ny
    duplicate_node_set = set(duplicate_nodes)
    bridge_triangles = []
    for ki, t in enumerate(triangles):
        ids = np.asarray(t, dtype=int)
        sv = signed[ids]
        if (np.any(sv > eps) and np.any(sv < -eps)
                and any(int(v) in duplicate_node_set for v in ids)):
            bridge_triangles.append(ki)
    if bridge_triangles:
        raise ValueError(
            f"Crack split aborted: {len(bridge_triangles)} triangles still bridge "
            f"the crack. First IDs: {bridge_triangles[:10]}"
        )

    # Append coincident duplicates.
    points_new = points.copy()
    old_to_dup: Dict[int, int] = {}
    for old in duplicate_nodes:
        new = len(points_new)
        points_new = np.vstack((points_new, points[old]))
        old_to_dup[old] = new

    # Positive side keeps original IDs; negative side uses duplicated IDs.
    triangles_new = triangles.copy()
    for ki, t in enumerate(triangles):
        ids = np.asarray(t, dtype=int)
        sv = signed[ids]
        has_negative_vertex = np.any(sv < -eps)
        has_positive_vertex = np.any(sv > eps)

        if (has_negative_vertex and has_positive_vertex
                and any(int(v) in duplicate_node_set for v in ids)):
            raise ValueError(f"Triangle {ki} still crosses the crack: {tuple(ids)}")

        if has_negative_vertex:
            new_t = ids.copy()
            for loc, old in enumerate(ids):
                if int(old) in old_to_dup:
                    new_t[loc] = old_to_dup[int(old)]
            triangles_new[ki] = new_t

    upper_chain = [int(v) for v in all_crack_nodes]
    lower_chain = [old_to_dup.get(int(v), int(v)) for v in all_crack_nodes]

    upper_edges: Set[Tuple[int, int]] = {
        _ek(a, b) for a, b in zip(upper_chain[:-1], upper_chain[1:])
    }
    lower_edges: Set[Tuple[int, int]] = {
        _ek(a, b) for a, b in zip(lower_chain[:-1], lower_chain[1:])
    }

    crack_info = {
        "crack_upper": upper_edges,
        "crack_lower": lower_edges,
        "crack_upper_nodes": set(upper_chain[:-1]),
        "crack_lower_nodes": set(lower_chain[:-1]),
        "crack_tip_nodes": {tip_idx},
        "tip_idx": tip_idx,
        "old_to_dup": old_to_dup,
        "upper_chain": upper_chain,
        "lower_chain": lower_chain,
    }

    validate_crack_topology(triangles_new, points_new, geometry, crack_info, eps=eps)
    return triangles_new, points_new, crack_info


def validate_crack_topology(triangles, points, geometry, crack_info, eps=1e-8):
    """Validate that the finite crack is disconnected and has two free faces."""
    triangles = np.asarray(triangles, dtype=np.int32)
    points = np.asarray(points, dtype=float)

    sx, sy = map(float, geometry.crack_start)
    nx, ny = map(float, geometry.crack_normal)
    signed = (points[:, 0] - sx) * nx + (points[:, 1] - sy) * ny

    # Straddling the infinite supporting line beyond the crack tip is valid.
    # A genuine topology error is detected below through crack-face edge
    # incidence and triangles mixing upper/lower face node IDs.

    # Count triangle incidences of every mesh edge.
    edge_count: Dict[Tuple[int, int], int] = {}
    for t in triangles:
        a, b, c = map(int, t)
        for e in (_ek(a, b), _ek(b, c), _ek(c, a)):
            edge_count[e] = edge_count.get(e, 0) + 1

    bad_upper = [e for e in crack_info.get("crack_upper", set()) if edge_count.get(e, 0) != 1]
    bad_lower = [e for e in crack_info.get("crack_lower", set()) if edge_count.get(e, 0) != 1]
    if bad_upper or bad_lower:
        raise ValueError(
            "Crack faces are not free boundaries. "
            f"Bad upper edges={bad_upper[:5]}, bad lower edges={bad_lower[:5]}"
        )

    upper = set(crack_info.get("crack_upper_nodes", set()))
    lower = set(crack_info.get("crack_lower_nodes", set()))
    mixed = []
    for ki, t in enumerate(triangles):
        ids = set(map(int, t))
        if ids & upper and ids & lower:
            mixed.append(ki)
    if mixed:
        raise ValueError(
            f"{len(mixed)} triangles contain nodes from both crack faces. "
            f"First IDs: {mixed[:10]}"
        )

    validate_domain_boundaries(triangles, points, geometry, eps=eps)

    print(
        "Crack topology OK | "
        f"upper nodes={len(upper)}, lower nodes={len(lower)}, "
        f"upper edges={len(crack_info.get('crack_upper', set()))}, "
        f"lower edges={len(crack_info.get('crack_lower', set()))}, "
        f"tip={crack_info.get('tip_idx')}"
    )
    return True


def unintended_internal_boundary_edges(triangles, points, geometry, eps=1e-8):
    """Return free edges that are neither plate boundary nor crack-face edges."""
    triangles = np.asarray(triangles, dtype=np.int32)
    points = np.asarray(points, dtype=float)[:, :2]
    W, H = float(geometry.W), float(geometry.H)
    start = np.asarray(geometry.crack_start, dtype=float)
    tip = np.asarray(geometry.crack_tip, dtype=float)
    direction = tip - start
    length2 = float(direction @ direction)
    if length2 <= 0.0:
        raise ValueError("Crack start and tip are coincident.")
    scale = max(W, H, math.sqrt(length2), 1.0)
    tol = max(float(eps), 1e-10 * scale)

    edge_count: Dict[Tuple[int, int], int] = {}
    for triangle in triangles:
        a, b, c = map(int, triangle)
        for edge in (_ek(a, b), _ek(b, c), _ek(c, a)):
            edge_count[edge] = edge_count.get(edge, 0) + 1

    def on_same_outer_edge(edge):
        p, q = points[list(edge)]
        return (
            (abs(p[0]) <= tol and abs(q[0]) <= tol)
            or (abs(p[0] - W) <= tol and abs(q[0] - W) <= tol)
            or (abs(p[1]) <= tol and abs(q[1]) <= tol)
            or (abs(p[1] - H) <= tol and abs(q[1] - H) <= tol)
        )

    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    normal /= math.sqrt(length2)

    def on_crack_segment(edge):
        coordinates = points[list(edge)]
        signed_distance = np.abs((coordinates - start) @ normal)
        projection = ((coordinates - start) @ direction) / length2
        return bool(
            np.all(signed_distance <= tol)
            and np.all(projection >= -tol)
            and np.all(projection <= 1.0 + tol)
        )

    free_edges = [edge for edge, count in edge_count.items() if count == 1]
    return [
        edge for edge in free_edges
        if not on_same_outer_edge(edge) and not on_crack_segment(edge)
    ]


def validate_domain_boundaries(triangles, points, geometry, eps=1e-8):
    """Reject holes and unintended internal free boundaries in the plate mesh."""
    bad = unintended_internal_boundary_edges(
        triangles, points, geometry, eps=eps,
    )
    if bad:
        raise ValueError(
            f"Mesh contains {len(bad)} unintended internal boundary edges "
            f"(a hole or missing triangles). First edges: {bad[:10]}"
        )
    return True
