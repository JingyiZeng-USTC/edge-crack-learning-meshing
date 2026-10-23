from __future__ import annotations

from collections import Counter, defaultdict, deque
import numpy as np
from scipy.spatial import cKDTree

from .geometry import collinear_interior_overlap, proper_segment_intersection
from .metrics import set_recovery_metrics
from .meshing import strict_triangular_faces, triangles_to_edges
from .project_data import mandatory_project_edges


def _spatial_cells(a: np.ndarray, b: np.ndarray, cell_size: float) -> list[tuple[int, int]]:
    lo = np.floor(np.minimum(a, b) / cell_size).astype(int)
    hi = np.floor(np.maximum(a, b) / cell_size).astype(int)
    return [(ix, iy) for ix in range(lo[0], hi[0] + 1) for iy in range(lo[1], hi[1] + 1)]


def select_non_crossing_project_edges(
    points: np.ndarray,
    predicted_edges: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
    predicted_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep mandatory edges first, then accept predictions in descending confidence order."""
    points = np.asarray(points, dtype=np.float64)
    predicted_edges = np.asarray(predicted_edges, dtype=np.int64).reshape(-1, 2)
    if predicted_probability is None:
        predicted_probability = np.ones(len(predicted_edges), dtype=np.float64)
    predicted_probability = np.asarray(predicted_probability, dtype=np.float64).reshape(-1)
    if len(predicted_probability) != len(predicted_edges):
        raise ValueError("predicted_probability must align with predicted_edges")

    mandatory_array = mandatory_project_edges(points, width, height, crack_segments)
    mandatory = {tuple(map(int, edge)) for edge in mandatory_array}
    scores: dict[tuple[int, int], float] = {}
    for edge, probability in zip(predicted_edges, predicted_probability):
        u, v = sorted(map(int, edge))
        if u != v:
            scores[(u, v)] = max(scores.get((u, v), -np.inf), float(probability))

    all_lengths_edges = predicted_edges if len(predicted_edges) else mandatory_array
    lengths = np.linalg.norm(points[all_lengths_edges[:, 1]] - points[all_lengths_edges[:, 0]], axis=1)
    positive_lengths = lengths[lengths > 0.0]
    median_length = float(np.median(positive_lengths)) if len(positive_lengths) else min(width, height) / 100.0
    cell_size = max(median_length, min(width, height) / 250.0)
    point_tree = cKDTree(points)
    point_tolerance = max(width, height) * 2e-7
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    selected: list[tuple[int, int]] = []

    def register(edge: tuple[int, int]) -> None:
        edge_id = len(selected)
        selected.append(edge)
        a, b = points[edge[0]], points[edge[1]]
        for cell in _spatial_cells(a, b, cell_size):
            grid[cell].append(edge_id)

    # Boundary and crack edges are authoritative and already split at every
    # collinear project node, so they are inserted before learned predictions.
    for edge in sorted(mandatory):
        register(edge)

    stats = {
        "mandatory_edges": len(mandatory),
        "predicted_edges": len(scores),
        "kept_predicted_edges": 0,
        "rejected_crossing_edges": 0,
        "rejected_pass_through_edges": 0,
        "rejected_invalid_edges": 0,
    }
    for (u, v), _score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if (u, v) in mandatory:
            continue
        if not (0 <= u < len(points) and 0 <= v < len(points)):
            stats["rejected_invalid_edges"] += 1
            continue
        a, b = points[u], points[v]
        direction = b - a
        length2 = float(direction @ direction)
        if length2 <= point_tolerance * point_tolerance:
            stats["rejected_invalid_edges"] += 1
            continue

        midpoint = 0.5 * (a + b)
        nearby_points = point_tree.query_ball_point(midpoint, 0.5 * np.sqrt(length2) + point_tolerance)
        passes_through_node = False
        for node_id in nearby_points:
            if node_id in (u, v):
                continue
            relative = points[node_id] - a
            projection = float(relative @ direction / length2)
            distance = abs(float(direction[0] * relative[1] - direction[1] * relative[0])) / np.sqrt(length2)
            if point_tolerance < projection < 1.0 - point_tolerance and distance <= point_tolerance:
                passes_through_node = True
                break
        if passes_through_node:
            stats["rejected_pass_through_edges"] += 1
            continue

        cells = _spatial_cells(a, b, cell_size)
        nearby_edges: set[int] = set()
        for cell in cells:
            nearby_edges.update(grid.get(cell, ()))
        conflict = False
        for edge_id in nearby_edges:
            x, y = selected[edge_id]
            c, d = points[x], points[y]
            if collinear_interior_overlap(a, b, c, d):
                conflict = True
                break
            if len({u, v, x, y}) == 4 and proper_segment_intersection(a, b, c, d):
                conflict = True
                break
        if conflict:
            stats["rejected_crossing_edges"] += 1
            continue
        register((u, v))
        stats["kept_predicted_edges"] += 1

    return np.asarray(selected, dtype=np.int64).reshape(-1, 2), stats


def recover_planar_project_mesh(
    points: np.ndarray,
    candidates: np.ndarray,
    probability: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for strict, fully learned recovery."""
    _, triangles, planar_edges, _ = recover_strict_gnn_project_mesh(
        points, candidates, probability, width, height, crack_segments,
    )
    return planar_edges, triangles


def _minimum_triangle_angle(points: np.ndarray, triangle: tuple[int, int, int]) -> float:
    q = np.asarray(points, dtype=np.float64)[list(triangle)]
    lengths = np.asarray([
        np.linalg.norm(q[1] - q[2]),
        np.linalg.norm(q[0] - q[2]),
        np.linalg.norm(q[0] - q[1]),
    ])
    a, b, c = lengths
    if min(a, b, c) <= 1.0e-15:
        return 0.0
    cosines = np.asarray([
        (b * b + c * c - a * a) / (2.0 * b * c),
        (a * a + c * c - b * b) / (2.0 * a * c),
        (a * a + b * b - c * c) / (2.0 * a * b),
    ])
    return float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).min())


def _structured_quality_redecode(
    points: np.ndarray,
    triangles: np.ndarray,
    selected_edges: np.ndarray,
    candidates: np.ndarray,
    probability: np.ndarray,
    mandatory: set[tuple[int, int]],
    *,
    quality_target_deg: float,
    quality_weight: float,
    max_passes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Improve bad local connectivity using only GNN-scored candidate edges.

    The operation replaces a non-mandatory shared diagonal of two adjacent
    triangles by the other candidate diagonal of the same convex quadrilateral.
    It is accepted only when the minimum angle improves and the combined GNN
    log-probability/quality objective increases.  No edge outside the GNN
    candidate set is ever introduced.
    """
    points = np.asarray(points, dtype=np.float64)
    triangle_set = {tuple(sorted(map(int, tri))) for tri in triangles}
    selected_set = {tuple(sorted(map(int, edge))) for edge in selected_edges}
    candidate_score = {
        tuple(sorted(map(int, edge))): float(score)
        for edge, score in zip(candidates, probability)
    }
    before_min = min(
        (_minimum_triangle_angle(points, tri) for tri in triangle_set), default=0.0,
    )
    replacements = 0
    total_gnn_log_probability_delta = 0.0
    completed_passes = 0
    epsilon = 1.0e-9

    for pass_index in range(max_passes):
        edge_to_triangles: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for tri in triangle_set:
            u, v, w = tri
            for edge in ((u, v), (u, w), (v, w)):
                edge_to_triangles[tuple(sorted(edge))].append(tri)

        proposals: list[
            tuple[float, float, tuple[int, int], tuple[int, int],
                  tuple[int, int, int], tuple[int, int, int],
                  tuple[int, int, int], tuple[int, int, int], float]
        ] = []
        for old_edge, adjacent in edge_to_triangles.items():
            if old_edge in mandatory or len(adjacent) != 2:
                continue
            first, second = adjacent
            opposite_first = next(node for node in first if node not in old_edge)
            opposite_second = next(node for node in second if node not in old_edge)
            if opposite_first == opposite_second:
                continue
            new_edge = tuple(sorted((opposite_first, opposite_second)))
            if new_edge in selected_set or new_edge not in candidate_score:
                continue
            u, v = old_edge
            a, b = new_edge
            if not proper_segment_intersection(points[u], points[v], points[a], points[b]):
                continue
            new_first = tuple(sorted((a, b, u)))
            new_second = tuple(sorted((a, b, v)))
            old_quality = min(
                _minimum_triangle_angle(points, first),
                _minimum_triangle_angle(points, second),
            )
            if old_quality >= quality_target_deg:
                continue
            new_quality = min(
                _minimum_triangle_angle(points, new_first),
                _minimum_triangle_angle(points, new_second),
            )
            if new_quality <= old_quality + 1.0e-8:
                continue
            old_probability = candidate_score.get(old_edge, 1.0)
            new_probability = candidate_score[new_edge]
            learned_gain = np.log(new_probability + epsilon) - np.log(old_probability + epsilon)
            capped_quality_gain = (
                min(new_quality, quality_target_deg) - min(old_quality, quality_target_deg)
            ) / max(quality_target_deg, epsilon)
            objective_gain = float(learned_gain + quality_weight * capped_quality_gain)
            if objective_gain <= 0.0:
                continue
            proposals.append((
                old_quality, -objective_gain, old_edge, new_edge,
                first, second, new_first, new_second, float(learned_gain),
            ))

        if not proposals:
            break
        proposals.sort(key=lambda item: (item[0], item[1], item[2]))
        used_triangles: set[tuple[int, int, int]] = set()
        accepted_this_pass = 0
        for (
            _old_quality, _negative_gain, old_edge, new_edge,
            first, second, new_first, new_second, learned_gain,
        ) in proposals:
            if first in used_triangles or second in used_triangles:
                continue
            if first not in triangle_set or second not in triangle_set:
                continue
            if old_edge not in selected_set or new_edge in selected_set:
                continue
            triangle_set.remove(first)
            triangle_set.remove(second)
            triangle_set.add(new_first)
            triangle_set.add(new_second)
            selected_set.remove(old_edge)
            selected_set.add(new_edge)
            used_triangles.update((first, second, new_first, new_second))
            replacements += 1
            accepted_this_pass += 1
            total_gnn_log_probability_delta += learned_gain
        completed_passes = pass_index + 1
        if accepted_this_pass == 0:
            break

    output_edges = np.asarray(sorted(selected_set), dtype=np.int64).reshape(-1, 2)
    output_triangles, _ = strict_triangular_faces(points, output_edges)
    after_min = min(
        (_minimum_triangle_angle(points, tuple(map(int, tri))) for tri in output_triangles),
        default=0.0,
    )
    return output_triangles, output_edges, {
        "structured_quality_replacements": replacements,
        "structured_quality_passes": completed_passes,
        "structured_quality_target_deg": float(quality_target_deg),
        "structured_quality_weight": float(quality_weight),
        "structured_min_angle_before_deg": float(before_min),
        "structured_min_angle_after_deg": float(after_min),
        "structured_gnn_log_probability_delta": float(total_gnn_log_probability_delta),
    }


def recover_strict_gnn_project_mesh(
    points: np.ndarray,
    candidates: np.ndarray,
    probability: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
    threshold: float | None = None,
    quality_target_deg: float = 10.0,
    quality_weight: float = 6.0,
    quality_passes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | bool]]:
    """Recover a mesh entirely from GNN-ranked candidate connectivity.

    Known plate-boundary and crack-polyline segments are mandatory geometry.
    Every other accepted edge comes from the GNN candidate list, considered in
    descending class-1 probability.  Geometry checks only reject illegal edges;
    they never create a new edge, node, diagonal, or triangle.  If the selected
    graph does not already consist solely of triangular bounded faces, recovery
    fails explicitly.
    """
    points = np.asarray(points, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(candidates) != len(probability):
        raise ValueError("probability must align with candidates")
    if quality_target_deg <= 0.0 or quality_weight < 0.0 or quality_passes < 0:
        raise ValueError("structured quality settings are invalid")

    selected_edges, filter_stats = select_non_crossing_project_edges(
        points,
        candidates,
        width,
        height,
        crack_segments,
        probability,
    )
    triangles, face_stats = strict_triangular_faces(points, selected_edges)
    if len(triangles) == 0:
        raise RuntimeError("Strict GNN recovery produced no triangular faces.")

    mandatory = {
        tuple(sorted(map(int, edge)))
        for edge in mandatory_project_edges(points, width, height, crack_segments)
    }
    triangles, selected_edges, quality_decode_stats = _structured_quality_redecode(
        points,
        triangles,
        selected_edges,
        candidates,
        probability,
        mandatory,
        quality_target_deg=quality_target_deg,
        quality_weight=quality_weight,
        max_passes=quality_passes,
    )

    triangle_edges = triangles_to_edges(triangles)
    selected_set = {tuple(sorted(map(int, edge))) for edge in selected_edges}
    represented_set = {tuple(sorted(map(int, edge))) for edge in triangle_edges}
    unrepresented = selected_set - represented_set

    q = points[triangles].astype(np.float64)
    signed_area2 = (
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )
    covered_area = 0.5 * float(np.sum(np.abs(signed_area2)))
    domain_area = float(width * height)
    relative_area_error = abs(covered_area - domain_area) / max(domain_area, 1e-30)
    topology = topology_metrics(points, triangles, width, height, crack_segments)

    failures: list[str] = []
    if unrepresented:
        failures.append(f"{len(unrepresented)} selected edges are not triangle edges")
    if relative_area_error > 5e-6:
        failures.append(f"relative area error is {relative_area_error:.3e}")
    if int(topology["components"]) != 1:
        failures.append(f"components={topology['components']}")
    for key in (
        "isolated_nodes", "nonmanifold_edges", "missing_constraint_edges",
        "duplicate_triangles",
    ):
        if int(topology[key]) != 0:
            failures.append(f"{key}={topology[key]}")
    if failures:
        raise RuntimeError(
            "Strict GNN recovery failed legality/completeness checks without "
            "CDT fallback: " + "; ".join(failures)
        )

    candidate_scores = {
        tuple(sorted(map(int, edge))): float(score)
        for edge, score in zip(candidates, probability)
    }
    selected_learned = selected_set - mandatory
    selected_below_threshold = 0
    threshold_positive = len(candidates)
    if threshold is not None:
        threshold_positive = int(np.count_nonzero(probability >= threshold))
        selected_below_threshold = sum(
            candidate_scores.get(edge, -np.inf) < threshold for edge in selected_learned
        )

    stats: dict[str, int | float | bool] = {
        **filter_stats,
        **face_stats,
        **quality_decode_stats,
        "strict_gnn_recovery": True,
        "structured_gnn_decoder": True,
        "cdt_fallback_used": False,
        "ear_clipping_used": False,
        "selected_edges": len(selected_edges),
        "selected_learned_edges": len(selected_learned),
        "threshold_positive_edges": threshold_positive,
        "selected_below_threshold_edges": selected_below_threshold,
        "unrepresented_selected_edges": len(unrepresented),
        "covered_area": covered_area,
        "domain_area": domain_area,
        "relative_area_error": relative_area_error,
    }
    return points, triangles, selected_edges, stats


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    if len(triangles) == 0:
        return {"triangles": 0, "min_angle_deg": 0.0, "angle_p05_deg": 0.0, "angle_median_deg": 0.0,
                "fraction_below_10deg": 1.0, "fraction_below_20deg": 1.0, "max_angle_deg": 180.0,
                "fraction_above_150deg": 1.0, "quality_mean": 0.0, "quality_p05": 0.0,
                "quality_min": 0.0, "area_cv": float("inf"), "degenerate_triangles": 0}
    q = points[triangles].astype(np.float64)
    side_a = np.linalg.norm(q[:, 1] - q[:, 2], axis=1)
    side_b = np.linalg.norm(q[:, 0] - q[:, 2], axis=1)
    side_c = np.linalg.norm(q[:, 0] - q[:, 1], axis=1)
    area2_signed = (
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )
    area = 0.5 * np.abs(area2_signed)
    denominator_a = np.maximum(2.0 * side_b * side_c, 1e-30)
    denominator_b = np.maximum(2.0 * side_a * side_c, 1e-30)
    angle_a = np.degrees(np.arccos(np.clip((side_b**2 + side_c**2 - side_a**2) / denominator_a, -1.0, 1.0)))
    angle_b = np.degrees(np.arccos(np.clip((side_a**2 + side_c**2 - side_b**2) / denominator_b, -1.0, 1.0)))
    angle_c = np.clip(180.0 - angle_a - angle_b, 0.0, 180.0)
    min_angle = np.min(np.column_stack([angle_a, angle_b, angle_c]), axis=1)
    max_angle = np.max(np.column_stack([angle_a, angle_b, angle_c]), axis=1)
    normalized_quality = 4.0 * np.sqrt(3.0) * area / np.maximum(side_a**2 + side_b**2 + side_c**2, 1e-30)
    return {
        "triangles": int(len(triangles)),
        "min_angle_deg": float(np.min(min_angle)),
        "angle_p05_deg": float(np.quantile(min_angle, 0.05)),
        "angle_median_deg": float(np.median(min_angle)),
        "fraction_below_10deg": float(np.mean(min_angle < 10.0)),
        "fraction_below_20deg": float(np.mean(min_angle < 20.0)),
        "max_angle_deg": float(np.max(max_angle)),
        "fraction_above_150deg": float(np.mean(max_angle > 150.0)),
        "quality_mean": float(np.mean(normalized_quality)),
        "quality_p05": float(np.quantile(normalized_quality, 0.05)),
        "quality_min": float(np.min(normalized_quality)),
        "area_cv": float(np.std(area) / np.mean(area)),
        "degenerate_triangles": int(np.sum(area < 1e-12)),
    }


def _component_count(n_nodes: int, edges: np.ndarray) -> tuple[int, int]:
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    used = np.zeros(n_nodes, dtype=bool)
    for u, v in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))
        used[int(u)] = used[int(v)] = True
    components = 0
    visited = np.zeros(n_nodes, dtype=bool)
    for start in np.flatnonzero(used):
        if visited[start]:
            continue
        components += 1
        queue = deque([int(start)])
        visited[start] = True
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
    return components, int(np.sum(~used))


def topology_metrics(
    points: np.ndarray,
    triangles: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
) -> dict[str, float | int]:
    edges = triangles_to_edges(triangles) if len(triangles) else np.empty((0, 2), dtype=np.int64)
    components, isolated = _component_count(len(points), edges)
    counts: Counter[tuple[int, int]] = Counter()
    for tri in triangles:
        u, v, w = map(int, tri)
        for edge in ((u, v), (v, w), (u, w)):
            counts[tuple(sorted(edge))] += 1
    nonmanifold = sum(value > 2 for value in counts.values())
    required = {tuple(map(int, edge)) for edge in mandatory_project_edges(points, width, height, crack_segments)}
    present = set(counts)
    return {
        "components": components,
        "isolated_nodes": isolated,
        "nonmanifold_edges": int(nonmanifold),
        "constraint_edges": len(required),
        "missing_constraint_edges": len(required - present),
        "duplicate_triangles": int(len(triangles) - len({tuple(sorted(map(int, tri))) for tri in triangles})),
    }


def compare_meshes(reference: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    metrics = set_recovery_metrics(reference, recovered, "triangle")
    metrics.update(set_recovery_metrics(triangles_to_edges(reference), triangles_to_edges(recovered), "recovered_edge"))
    return metrics
