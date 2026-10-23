"""Crack sampling driven by the learned absolute size field.

These points become hard crack-constraint edges in later modules.  Sampling
is deterministic and keeps every original polyline corner exactly.
"""

import numpy as np


def _size_values(size_field, points: np.ndarray) -> np.ndarray:
    if hasattr(size_field, "evaluate_bulk"):
        values = np.asarray(size_field.evaluate_bulk(points), dtype=float)
    else:
        values = np.asarray(
            [size_field.evaluate(float(x), float(y)) for x, y in points],
            dtype=float,
        )
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError("size field returned invalid values on the crack")
    return np.maximum(values, np.finfo(float).eps)


def _sample_segment(
    start: np.ndarray,
    end: np.ndarray,
    size_field,
    minimum_pieces: int,
    spacing_factor: float,
) -> np.ndarray:
    """Equidistribute one segment in the learned metric ds / h(x)."""
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= np.finfo(float).eps:
        return start[None, :]

    # Metric quadrature avoids the very short terminal interval produced by a
    # forward stepper.  It is cheap because this is only a one-dimensional path.
    probe_count = max(65, 8 * int(minimum_pieces) + 1)
    parameter = np.linspace(0.0, 1.0, probe_count)
    probes = start[None, :] + parameter[:, None] * vector[None, :]
    h = _size_values(size_field, probes)
    ds = length / float(probe_count - 1)
    increments = 0.5 * (1.0 / h[:-1] + 1.0 / h[1:]) * ds / spacing_factor
    metric = np.concatenate(([0.0], np.cumsum(increments)))
    pieces = max(int(minimum_pieces), int(np.ceil(metric[-1])))
    targets = np.linspace(0.0, metric[-1], pieces + 1)
    locations = np.interp(targets, metric, parameter)
    return start[None, :] + locations[:, None] * vector[None, :]


def sample_crack(
    geometry,
    size_field=None,
    n_segments: int = 30,
    spacing_factor: float = 0.90,
) -> np.ndarray:
    """Return crack vertices spaced at roughly ``spacing_factor * h``.

    ``n_segments`` is retained as a lower bound for each complete crack
    polyline, so coarse-field cases never receive fewer constraint vertices.
    Passing no size field preserves the former fixed-count behaviour.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be positive")
    if spacing_factor <= 0.0:
        raise ValueError("spacing_factor must be positive")

    if size_field is None:
        if getattr(geometry, "is_crack_graph", False):
            chains = geometry.discretize_cracks(n_segments)
            return (
                np.vstack(chains).astype(float, copy=False)
                if chains else np.empty((0, 2))
            )
        geometry.discretize_crack(n_segments)
        verts = geometry.crack_vertices
        return (
            np.asarray(verts, dtype=float)
            if verts is not None else np.empty((0, 2))
        )

    if getattr(geometry, "is_crack_graph", False):
        polylines = tuple(np.asarray(p, dtype=float) for p in geometry.crack_polylines)
    else:
        polylines = (
            np.asarray((geometry.crack_start, geometry.crack_tip), dtype=float),
        )

    chains = []
    for polyline in polylines:
        lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        total = float(lengths.sum())
        parts = []
        for index, length in enumerate(lengths):
            minimum = max(1, int(np.ceil(n_segments * float(length) / total)))
            part = _sample_segment(
                polyline[index], polyline[index + 1], size_field,
                minimum, spacing_factor,
            )
            parts.append(part if index == 0 else part[1:])
        chains.append(np.vstack(parts))
    return (
        np.vstack(chains).astype(float, copy=False)
        if chains else np.empty((0, 2))
    )
