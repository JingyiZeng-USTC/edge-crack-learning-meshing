"""Adaptive sampling of the rectangular exterior boundary.

The former implementation used ``0.85*h_max`` everywhere.  Near a crack
entry this left long boundary segments next to the fine crack-layer points,
which created almost-collinear triples and sub-degree triangles.  Boundary
spacing now follows the learned size field in a neighbourhood of every crack
entry while remaining coarse in the far field.
"""

import math

import numpy as np


def _crack_entries(geometry) -> np.ndarray:
    if getattr(geometry, "is_crack_graph", False):
        return np.asarray(geometry.crack_entries, dtype=float).reshape(-1, 2)
    return np.asarray([geometry.crack_start], dtype=float)


def sample_boundary(
    geometry,
    size_field,
    entry_refinement_radius_factor: float = 4.0,
) -> np.ndarray:
    """Return adaptively spaced points on all four exterior edges."""
    W, H = float(geometry.W), float(geometry.H)
    h_min, h_max = float(size_field.h_min), float(size_field.h_max)
    far_spacing = 0.85 * h_max
    entries = _crack_entries(geometry)
    entry_radius = max(entry_refinement_radius_factor * h_max, 6.0 * h_min)

    def _spacing(x: float, y: float) -> float:
        if len(entries):
            distance = float(np.linalg.norm(entries - np.asarray([x, y]), axis=1).min())
        else:
            distance = float("inf")
        if distance <= entry_radius:
            local = 0.85 * float(size_field.evaluate(x, y))
            return min(far_spacing, max(local, 0.8 * h_min))
        return far_spacing

    def _march(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1.0e-15:
            return [(x0, y0)]
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        points: list[tuple[float, float]] = [(x0, y0)]
        travelled = 0.0
        while True:
            x, y = x0 + travelled * ux, y0 + travelled * uy
            spacing = _spacing(x, y)
            remaining = length - travelled
            # Balance the terminal interval instead of leaving an arbitrarily
            # short segment immediately before a crack entry or corner.
            if remaining <= 1.5 * spacing:
                if math.hypot(points[-1][0] - x1, points[-1][1] - y1) > 1.0e-12:
                    points.append((x1, y1))
                break
            travelled += spacing
            points.append((x0 + travelled * ux, y0 + travelled * uy))
        return points

    def _edge_with_entries(
        start: tuple[float, float], end: tuple[float, float]
    ) -> list[tuple[float, float]]:
        start_array, end_array = np.asarray(start), np.asarray(end)
        direction = end_array - start_array
        length2 = float(direction @ direction)
        tolerance = 1.0e-10 * max(W, H, 1.0)
        breakpoints: list[tuple[float, np.ndarray]] = [(0.0, start_array), (1.0, end_array)]
        for entry in entries:
            parameter = float((entry - start_array) @ direction / length2)
            projection = start_array + np.clip(parameter, 0.0, 1.0) * direction
            if -tolerance <= parameter <= 1.0 + tolerance and np.linalg.norm(entry - projection) <= tolerance:
                breakpoints.append((float(np.clip(parameter, 0.0, 1.0)), entry.copy()))
        ordered: list[np.ndarray] = []
        for parameter, point in sorted(breakpoints, key=lambda item: item[0]):
            if not ordered or np.linalg.norm(point - ordered[-1]) > tolerance:
                ordered.append(point)
        result: list[tuple[float, float]] = []
        for first, second in zip(ordered[:-1], ordered[1:]):
            segment = _march(float(first[0]), float(first[1]), float(second[0]), float(second[1]))
            if result and segment and np.allclose(result[-1], segment[0], atol=tolerance):
                result.extend(segment[1:])
            else:
                result.extend(segment)
        return result

    edges = (
        _edge_with_entries((0.0, 0.0), (W, 0.0)),
        _edge_with_entries((W, 0.0), (W, H)),
        _edge_with_entries((W, H), (0.0, H)),
        _edge_with_entries((0.0, H), (0.0, 0.0)),
    )
    result = list(edges[0])
    for next_edge in edges[1:]:
        if result and next_edge and np.allclose(result[-1], next_edge[0], atol=1.0e-10):
            result.extend(next_edge[1:])
        else:
            result.extend(next_edge)
    if result and np.allclose(result[-1], result[0], atol=1.0e-10):
        result.pop()
    return np.asarray(result, dtype=float)
