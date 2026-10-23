"""Topology-preserving node smoothing for an already recovered GNN mesh."""

from __future__ import annotations

import numpy as np


def _triangle_angles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    q = points[triangles]
    a = np.linalg.norm(q[:, 1] - q[:, 2], axis=1)
    b = np.linalg.norm(q[:, 0] - q[:, 2], axis=1)
    c = np.linalg.norm(q[:, 0] - q[:, 1], axis=1)
    denominator = np.maximum(
        np.column_stack((2.0 * b * c, 2.0 * a * c, 2.0 * a * b)),
        1.0e-30,
    )
    cosine = np.column_stack((
        b * b + c * c - a * a,
        a * a + c * c - b * b,
        a * a + b * b - c * c,
    )) / denominator
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _signed_double_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    q = points[triangles]
    return (
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )


def _vertex_constraints(
    points: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = 2.0e-7 * max(width, height, 1.0)
    on_left = points[:, 0] <= tolerance
    on_right = points[:, 0] >= width - tolerance
    on_bottom = points[:, 1] <= tolerance
    on_top = points[:, 1] >= height - tolerance
    # -1: free interior; 0: slide vertically; 1: slide horizontally.
    boundary_axis = np.full(len(points), -1, dtype=np.int8)
    boundary_axis[on_left | on_right] = 0
    boundary_axis[on_bottom | on_top] = 1
    fixed = (on_left | on_right) & (on_bottom | on_top)
    for start, end in np.asarray(crack_segments, dtype=float):
        vector = end - start
        length2 = float(vector @ vector)
        parameter = np.clip((points - start) @ vector / max(length2, 1.0e-30), 0.0, 1.0)
        closest = start + parameter[:, None] * vector
        fixed |= np.linalg.norm(points - closest, axis=1) <= tolerance
    return fixed, boundary_axis


def _capped_target(current: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    delta = target - current
    length = float(np.linalg.norm(delta))
    if length > limit:
        delta *= limit / length
    return current + delta


def constrained_quality_smoothing(
    points: np.ndarray,
    triangles: np.ndarray,
    learned_h: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
    *,
    target_angle_deg: float = 20.0,
    max_passes: int = 10,
    move_factor: float = 0.30,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Improve angles without changing connectivity or constraint vertices.

    Each proposed interior-node move is accepted only if the worst angle in
    its incident triangle fan improves and every incident triangle preserves
    its orientation with a positive area margin.  Thus this stage introduces
    no node, edge, triangulation operation, or teacher information.
    """
    values = np.asarray(points, dtype=float).copy()
    triangles = np.asarray(triangles, dtype=np.int64)
    learned_h = np.asarray(learned_h, dtype=float)
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    if move_factor <= 0.0:
        raise ValueError("move_factor must be positive")
    if len(values) != len(learned_h):
        raise ValueError("learned_h must contain one value per node")

    incident: list[list[int]] = [[] for _ in range(len(values))]
    neighbors: list[set[int]] = [set() for _ in range(len(values))]
    for triangle_id, triangle in enumerate(triangles):
        a, b, c = map(int, triangle)
        for node in (a, b, c):
            incident[node].append(triangle_id)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    fixed, boundary_axis = _vertex_constraints(
        values, width, height, crack_segments,
    )
    initial_area = _signed_double_area(values, triangles)
    reference_sign = np.sign(initial_area)
    area_floor = max(float(np.abs(initial_area).mean()) * 1.0e-8, 1.0e-16)
    before = float(_triangle_angles(values, triangles).min())
    accepted = 0
    completed_passes = 0

    for pass_index in range(max_passes):
        completed_passes = pass_index + 1
        all_angles = _triangle_angles(values, triangles)
        local_quality = np.full(len(values), 180.0)
        for node, local_triangles in enumerate(incident):
            if local_triangles:
                local_quality[node] = float(all_angles[local_triangles].min())

        accepted_this_pass = 0
        for node in np.argsort(local_quality):
            if fixed[node] or local_quality[node] >= target_angle_deg:
                continue
            triangle_ids = np.asarray(incident[node], dtype=np.int64)
            old_quality = float(
                _triangle_angles(values, triangles[triangle_ids]).min()
            )
            current = values[node].copy()
            neighbor_ids = np.fromiter(neighbors[node], dtype=np.int64)
            if len(neighbor_ids) == 0:
                continue
            neighbor_points = values[neighbor_ids]
            distances = np.maximum(
                np.linalg.norm(neighbor_points - current, axis=1), 1.0e-12,
            )
            targets = [
                neighbor_points.mean(axis=0),
                np.average(neighbor_points, axis=0, weights=1.0 / distances),
            ]
            for triangle_id in triangle_ids:
                triangle = triangles[triangle_id]
                opposite = triangle[triangle != node]
                first, second = values[opposite]
                midpoint = 0.5 * (first + second)
                edge = second - first
                edge_length = float(np.linalg.norm(edge))
                if edge_length <= 1.0e-14:
                    continue
                normal = np.asarray((-edge[1], edge[0]), dtype=float) / edge_length
                side = 1.0 if float((current - midpoint) @ normal) >= 0.0 else -1.0
                targets.append(midpoint + side * np.sqrt(3.0) * 0.5 * edge_length * normal)

            best = current
            best_quality = old_quality
            move_limit = move_factor * max(float(learned_h[node]), 1.0e-9)
            for target in targets:
                endpoint = _capped_target(current, np.asarray(target), move_limit)
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    proposal = current + alpha * (endpoint - current)
                    if boundary_axis[node] == 0:
                        proposal[0] = current[0]
                    elif boundary_axis[node] == 1:
                        proposal[1] = current[1]
                    values[node] = proposal
                    signed_area = _signed_double_area(values, triangles[triangle_ids])
                    if np.any(signed_area * reference_sign[triangle_ids] <= area_floor):
                        continue
                    quality = float(
                        _triangle_angles(values, triangles[triangle_ids]).min()
                    )
                    if quality > best_quality + 1.0e-8:
                        best_quality = quality
                        best = proposal.copy()
            values[node] = best
            if best_quality > old_quality + 1.0e-8:
                accepted += 1
                accepted_this_pass += 1
            else:
                values[node] = current

        current_min = float(_triangle_angles(values, triangles).min())
        if accepted_this_pass == 0 or current_min >= target_angle_deg:
            break

    after_angles = _triangle_angles(values, triangles)
    after_area = _signed_double_area(values, triangles)
    return values, {
        "enabled": True,
        "target_angle_deg": float(target_angle_deg),
        "move_factor": float(move_factor),
        "passes": int(completed_passes),
        "accepted_moves": int(accepted),
        "fixed_constraint_nodes": int(np.count_nonzero(fixed)),
        "sliding_boundary_nodes": int(np.count_nonzero((boundary_axis >= 0) & ~fixed)),
        "min_angle_before_deg": before,
        "min_angle_after_deg": float(after_angles.min()),
        "fraction_below_target": float(
            np.mean(after_angles.min(axis=1) < target_angle_deg)
        ),
        "orientation_preserved": bool(np.all(after_area * reference_sign > area_floor)),
    }
