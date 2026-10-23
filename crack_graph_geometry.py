"""General 2-D crack polylines represented as a planar constraint graph.

The neural model never consumes a variable-length polyline directly.  This
module converts one or more crack polylines into fixed-size local queries while
also retaining the exact vertices and segments required by a deterministic
constrained triangulator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


_TOL = 1.0e-10


def _as_polyline(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
        raise ValueError(f"{name} must have shape (N, 2) with N >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite coordinate")
    return array


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _strict_segment_crossing(a, b, c, d, tolerance: float) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 * o2 < -(tolerance * tolerance) and o3 * o4 < -(tolerance * tolerance)


class CrackGraphGeometry:
    """Rectangular domain containing straight, curved, multiple or branched cracks.

    Curves are supplied as ordered polylines.  Separate polylines may share an
    endpoint to form a branch.  Crossing at a point that is not an explicit
    shared vertex is rejected because it is not a valid planar constraint graph.
    """

    is_crack_graph = True

    def __init__(
        self,
        W: float,
        H: float,
        crack_polylines: Sequence[Sequence[Sequence[float]] | np.ndarray],
        *,
        tolerance: float = _TOL,
    ) -> None:
        self.W = float(W)
        self.H = float(H)
        self.tolerance = float(tolerance)
        if self.W <= 0.0 or self.H <= 0.0:
            raise ValueError("W and H must be positive")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if not crack_polylines:
            raise ValueError("At least one crack polyline is required")
        self.crack_polylines = tuple(
            _as_polyline(values, f"crack_polylines[{index}]").copy()
            for index, values in enumerate(crack_polylines)
        )
        self._validate_coordinates()
        self._build_graph()
        self._validate_planarity()
        self._discretized_chains: tuple[np.ndarray, ...] | None = None
        self._crack_vertices: np.ndarray | None = None
        self._crack_segments: np.ndarray | None = None

    @property
    def boundary(self) -> list[tuple[float, float]]:
        return [(0.0, 0.0), (self.W, 0.0), (self.W, self.H), (0.0, self.H)]

    def inside(self, x: float, y: float) -> bool:
        tol = self.tolerance
        return -tol <= x <= self.W + tol and -tol <= y <= self.H + tol

    def inside_bulk(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        tol = self.tolerance
        return (
            (values[:, 0] >= -tol)
            & (values[:, 0] <= self.W + tol)
            & (values[:, 1] >= -tol)
            & (values[:, 1] <= self.H + tol)
        )

    def _validate_coordinates(self) -> None:
        scale = max(self.W, self.H, 1.0)
        minimum_length = self.tolerance * scale
        for index, polyline in enumerate(self.crack_polylines):
            if not np.all(self.inside_bulk(polyline)):
                raise ValueError(f"crack_polylines[{index}] leaves the rectangular domain")
            lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
            if np.any(lengths <= minimum_length):
                raise ValueError(f"crack_polylines[{index}] contains a zero-length segment")

    def _vertex_key(self, point: np.ndarray) -> tuple[int, int]:
        scale = max(self.W, self.H, 1.0)
        quantum = self.tolerance * scale
        return tuple(np.rint(point / quantum).astype(np.int64))

    def _build_graph(self) -> None:
        vertices: list[np.ndarray] = []
        vertex_lookup: dict[tuple[int, int], int] = {}
        graph_segments: list[tuple[int, int]] = []
        segment_coordinates: list[np.ndarray] = []
        segment_polyline: list[int] = []
        segment_local_index: list[int] = []
        polyline_vertex_ids: list[np.ndarray] = []

        for polyline_index, polyline in enumerate(self.crack_polylines):
            identifiers = []
            for point in polyline:
                key = self._vertex_key(point)
                identifier = vertex_lookup.get(key)
                if identifier is None:
                    identifier = len(vertices)
                    vertex_lookup[key] = identifier
                    vertices.append(point.copy())
                elif np.linalg.norm(vertices[identifier] - point) > self.tolerance * 10.0:
                    raise ValueError("Two graph vertices collapsed under the geometry tolerance")
                identifiers.append(identifier)
            ids = np.asarray(identifiers, dtype=np.int64)
            polyline_vertex_ids.append(ids)
            for local_index, (first, second) in enumerate(zip(ids[:-1], ids[1:])):
                graph_segments.append((int(first), int(second)))
                segment_coordinates.append(np.vstack((vertices[first], vertices[second])))
                segment_polyline.append(polyline_index)
                segment_local_index.append(local_index)

        self.graph_vertices = np.asarray(vertices, dtype=float)
        self.graph_segments = np.asarray(graph_segments, dtype=np.int64)
        self.polyline_vertex_ids = tuple(polyline_vertex_ids)
        self.segment_coordinates = np.asarray(segment_coordinates, dtype=float)
        self.segment_polyline = np.asarray(segment_polyline, dtype=np.int64)
        self.segment_local_index = np.asarray(segment_local_index, dtype=np.int64)

        degree = np.zeros(len(self.graph_vertices), dtype=np.int64)
        np.add.at(degree, self.graph_segments[:, 0], 1)
        np.add.at(degree, self.graph_segments[:, 1], 1)
        self.vertex_degree = degree
        endpoints = np.flatnonzero(degree == 1)
        on_boundary = self._points_on_outer_boundary(self.graph_vertices)
        self.entry_vertex_ids = endpoints[on_boundary[endpoints]]
        self.tip_vertex_ids = endpoints[~on_boundary[endpoints]]
        self.branch_vertex_ids = np.flatnonzero(degree >= 3)
        self.crack_entries = self.graph_vertices[self.entry_vertex_ids]
        self.crack_tips = self.graph_vertices[self.tip_vertex_ids]
        self.branch_points = self.graph_vertices[self.branch_vertex_ids]
        self.has_branches = bool(len(self.branch_vertex_ids))
        self.total_crack_length = float(
            np.linalg.norm(
                self.segment_coordinates[:, 1] - self.segment_coordinates[:, 0],
                axis=1,
            ).sum()
        )
        self.segment_turning = self._segment_turning_angles()

    def _points_on_outer_boundary(self, points: np.ndarray) -> np.ndarray:
        tolerance = self.tolerance * max(self.W, self.H, 1.0) * 10.0
        return (
            (np.abs(points[:, 0]) <= tolerance)
            | (np.abs(points[:, 0] - self.W) <= tolerance)
            | (np.abs(points[:, 1]) <= tolerance)
            | (np.abs(points[:, 1] - self.H) <= tolerance)
        )

    def _validate_planarity(self) -> None:
        tolerance = self.tolerance * max(self.W, self.H, 1.0) * 10.0
        for first in range(len(self.graph_segments)):
            a_id, b_id = map(int, self.graph_segments[first])
            a, b = self.graph_vertices[[a_id, b_id]]
            for second in range(first + 1, len(self.graph_segments)):
                c_id, d_id = map(int, self.graph_segments[second])
                if {a_id, b_id} & {c_id, d_id}:
                    continue
                c, d = self.graph_vertices[[c_id, d_id]]
                if _strict_segment_crossing(a, b, c, d, tolerance):
                    raise ValueError(
                        "Crack segments cross without an explicit shared graph vertex: "
                        f"segments {first} and {second}"
                    )

    def _segment_turning_angles(self) -> np.ndarray:
        turning = np.zeros(len(self.graph_segments), dtype=float)
        cursor = 0
        for polyline in self.crack_polylines:
            directions = np.diff(polyline, axis=0)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            local = np.zeros(len(directions), dtype=float)
            for index in range(1, len(directions)):
                cosine = float(np.clip(directions[index - 1] @ directions[index], -1.0, 1.0))
                angle = math.acos(cosine) / math.pi
                local[index - 1] = max(local[index - 1], angle)
                local[index] = max(local[index], angle)
            turning[cursor:cursor + len(local)] = local
            cursor += len(local)
        return turning

    def nearest_segment_query(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Return closest-segment distance, orientation and local curvature."""
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        starts = self.segment_coordinates[:, 0]
        vectors = self.segment_coordinates[:, 1] - starts
        length2 = np.einsum("si,si->s", vectors, vectors)
        lengths = np.sqrt(length2)
        tangents = vectors / lengths[:, None]
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        count = len(values)
        nearest = np.empty(count, dtype=np.int64)
        distance = np.empty(count, dtype=float)
        projection = np.empty(count, dtype=float)
        signed = np.empty(count, dtype=float)
        chunk_size = max(1, int(2_000_000 / max(len(starts), 1)))
        for begin in range(0, count, chunk_size):
            end = min(begin + chunk_size, count)
            relative = values[begin:end, None, :] - starts[None, :, :]
            parameter = np.einsum("nsi,si->ns", relative, vectors) / length2[None, :]
            parameter = np.clip(parameter, 0.0, 1.0)
            closest = starts[None, :, :] + parameter[:, :, None] * vectors[None, :, :]
            delta = values[begin:end, None, :] - closest
            distance2 = np.einsum("nsi,nsi->ns", delta, delta)
            selected = np.argmin(distance2, axis=1)
            rows = np.arange(end - begin)
            nearest[begin:end] = selected
            distance[begin:end] = np.sqrt(distance2[rows, selected])
            projection[begin:end] = parameter[rows, selected]
            signed[begin:end] = np.einsum(
                "ni,ni->n", delta[rows, selected], normals[selected],
            )
        return {
            "segment_index": nearest,
            "distance": distance,
            "signed_distance": signed,
            "projection": projection,
            "tangent": tangents[nearest],
            "normal": normals[nearest],
            "turning": self.segment_turning[nearest],
        }

    @staticmethod
    def _nearest_points(points: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(targets) == 0:
            return np.full(len(points), np.inf), np.zeros((len(points), 2), dtype=float)
        relative = targets[None, :, :] - points[:, None, :]
        distance2 = np.einsum("nti,nti->nt", relative, relative)
        selected = np.argmin(distance2, axis=1)
        rows = np.arange(len(points))
        vectors = relative[rows, selected]
        distances = np.sqrt(distance2[rows, selected])
        directions = vectors / np.maximum(distances[:, None], 1.0e-30)
        return distances, directions

    def distance_to_tip_bulk(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        return self._nearest_points(values, self.crack_tips)[0]

    def distance_to_tip(self, x: float, y: float) -> float:
        return float(self.distance_to_tip_bulk(np.asarray([[x, y]], dtype=float))[0])

    def distance_to_crack_bulk(self, points: np.ndarray) -> np.ndarray:
        return self.nearest_segment_query(points)["distance"]

    def distance_to_crack(self, x: float, y: float) -> float:
        return float(self.distance_to_crack_bulk(np.asarray([[x, y]], dtype=float))[0])

    def local_geometry_query(
        self,
        points: np.ndarray,
        *,
        neighborhood_radius: float | None = None,
    ) -> dict[str, np.ndarray]:
        values = np.asarray(points, dtype=float)
        segment = self.nearest_segment_query(values)
        tip_distance, tip_direction = self._nearest_points(values, self.crack_tips)
        branch_distance, branch_direction = self._nearest_points(values, self.branch_points)
        entry_distance, _ = self._nearest_points(values, self.crack_entries)
        boundary_distance = np.minimum.reduce(
            (values[:, 0], self.W - values[:, 0], values[:, 1], self.H - values[:, 1])
        )
        radius = float(
            neighborhood_radius
            if neighborhood_radius is not None
            else 0.05 * math.hypot(self.W, self.H)
        )
        starts = self.segment_coordinates[:, 0]
        ends = self.segment_coordinates[:, 1]
        nearby = np.zeros(len(values), dtype=float)
        for index, point in enumerate(values):
            vector = ends - starts
            length2 = np.einsum("si,si->s", vector, vector)
            parameter = np.clip(np.einsum("si,si->s", point - starts, vector) / length2, 0.0, 1.0)
            closest = starts + parameter[:, None] * vector
            nearby[index] = np.count_nonzero(np.linalg.norm(point - closest, axis=1) <= radius)
        return {
            **segment,
            "tip_distance": tip_distance,
            "tip_direction": tip_direction,
            "branch_distance": branch_distance,
            "branch_direction": branch_direction,
            "entry_distance": entry_distance,
            "boundary_distance": np.maximum(boundary_distance, 0.0),
            "nearby_segment_count": nearby,
        }

    def discretize_cracks(self, n_segments: int = 30) -> tuple[np.ndarray, ...]:
        """Densify every polyline while retaining every corner and branch vertex."""
        if n_segments < 1:
            raise ValueError("n_segments must be positive")
        chains: list[np.ndarray] = []
        for polyline in self.crack_polylines:
            lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
            total = float(lengths.sum())
            pieces = np.maximum(1, np.rint(n_segments * lengths / total).astype(np.int64))
            chain_parts = []
            for index, count in enumerate(pieces):
                part = np.linspace(polyline[index], polyline[index + 1], int(count) + 1)
                chain_parts.append(part if index == 0 else part[1:])
            chains.append(np.vstack(chain_parts))
        self._discretized_chains = tuple(chains)
        vertices: list[np.ndarray] = []
        lookup: dict[tuple[int, int], int] = {}
        segments: list[tuple[int, int]] = []
        for chain in chains:
            ids = []
            for point in chain:
                key = self._vertex_key(point)
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append(point.copy())
                ids.append(lookup[key])
            segments.extend(zip(ids[:-1], ids[1:]))
        self._crack_vertices = np.asarray(vertices, dtype=float)
        self._crack_segments = np.asarray(segments, dtype=np.int64)
        return self._discretized_chains

    def discretize_crack(self, n_segments: int = 30) -> None:
        """Compatibility alias used by the existing sampler."""
        self.discretize_cracks(n_segments)

    @property
    def crack_chains(self) -> tuple[np.ndarray, ...]:
        if self._discretized_chains is None:
            self.discretize_cracks(30)
        return self._discretized_chains or ()

    @property
    def crack_vertices(self) -> np.ndarray | None:
        return self._crack_vertices

    @property
    def crack_segments(self) -> np.ndarray | None:
        return self._crack_segments

    @property
    def crack_n_segments(self) -> int:
        return 0 if self._crack_segments is None else len(self._crack_segments)

    def to_dict(self) -> dict:
        return {
            "format": "crack-graph-geometry-v1",
            "W": self.W,
            "H": self.H,
            "crack_polylines": [polyline.tolist() for polyline in self.crack_polylines],
        }

    @classmethod
    def from_dict(cls, values: dict) -> "CrackGraphGeometry":
        return cls(values["W"], values["H"], values["crack_polylines"])

    @classmethod
    def from_json(cls, path: str | Path) -> "CrackGraphGeometry":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        return "\n".join(
            (
                "CrackGraphGeometry",
                f"  domain        = {self.W} x {self.H}",
                f"  polylines     = {len(self.crack_polylines)}",
                f"  graph vertices= {len(self.graph_vertices)}",
                f"  graph segments= {len(self.graph_segments)}",
                f"  crack tips    = {len(self.crack_tips)}",
                f"  entries       = {len(self.crack_entries)}",
                f"  branch points = {len(self.branch_points)}",
                f"  total length  = {self.total_crack_length:.6g}",
            )
        )


__all__ = ["CrackGraphGeometry"]
