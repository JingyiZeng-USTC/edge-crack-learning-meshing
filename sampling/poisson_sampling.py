"""Interior Poisson-disk sampling with exclusion zones.

Bridson's algorithm, adaptive radius from size field, subdomain-isolated.
Existing points (boundary, crack, layers, tip ring) are pre-loaded into
the spatial grid so new points respect their exclusion zones.
"""

import math
import random
from typing import List, Tuple

import numpy as np


# reuse the spatial grid from the original module
import sys
sys.path.insert(0, "/sessions/keen-brave-ritchie/mnt/3")
from adaptive_sampling import _SpatialGrid


def _signed_crack(x, y, sx, sy, nx, ny):
    return (x - sx) * nx + (y - sy) * ny


def _in_subdomain(x, y, sx, sy, nx, ny, side, eps=1e-12):
    d = _signed_crack(x, y, sx, sy, nx, ny)
    return d > eps if side == 0 else d < -eps


def sample_interior(
    geometry,
    size_field,
    existing_points: np.ndarray,
    seed: int | None = None,
    k_candidates: int = 30,
    seed_attempts: int = 300,
    boundary_clearance_factor: float = 0.5,
) -> np.ndarray:
    """Poisson-disk sample the interior, avoiding existing points.

    Parameters
    ----------
    existing_points : (P, 2) array
        Already-placed points (boundary, crack, layers, tip ring).
        These are pre-inserted into the spatial grid so new candidates
        respect their exclusion zones.
    """
    W, H = geometry.W, geometry.H
    h_min = size_field.h_min
    generic_graph = bool(getattr(geometry, "is_crack_graph", False))
    if not generic_graph:
        sx, sy = geometry.crack_start
        nx, ny = geometry.crack_normal

    rng = random.Random(seed)
    cell_size = h_min / math.sqrt(2)
    grid = _SpatialGrid(cell_size, W, H)

    # pre-load existing points into grid
    existing = np.asarray(existing_points, dtype=float)
    for i, (ex, ey) in enumerate(existing):
        grid.insert(-(i + 1), ex, ey)  # negative indices = fixed

    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def clears_exterior_boundary(x: float, y: float, local_h: float) -> bool:
        distance = min(x, W - x, y, H - y)
        return distance >= boundary_clearance_factor * local_h

    # seed point
    seed_pt = None
    for _ in range(seed_attempts):
        cx = rng.uniform(0, W)
        cy = rng.uniform(0, H)
        if not geometry.inside(cx, cy):
            continue
        admissible = generic_graph or any(
            _in_subdomain(cx, cy, sx, sy, nx, ny, side) for side in (0, 1)
        )
        if admissible:
            rc = max(size_field.evaluate(cx, cy), h_min * 0.5)
            if clears_exterior_boundary(cx, cy, rc) and not grid.query_radius(cx, cy, rc):
                seed_pt = (cx, cy)
        if seed_pt:
            break

    if seed_pt is None:
        return np.empty((0, 2))

    idx = len(points)
    points.append(seed_pt)
    active.append(idx)
    grid.insert(idx, seed_pt[0], seed_pt[1])

    # main Bridson loop
    while active:
        i = active[rng.randrange(len(active))]
        px, py = points[i]
        r_i = size_field.evaluate(px, py)
        r_i = max(r_i, h_min * 0.5)

        accepted = False
        for _ in range(k_candidates):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(r_i, 2 * r_i)
            cx = px + dist * math.cos(angle)
            cy = py + dist * math.sin(angle)

            if not geometry.inside(cx, cy):
                continue
            on_side = generic_graph or any(
                _in_subdomain(cx, cy, sx, sy, nx, ny, side) for side in (0, 1)
            )
            if not on_side:
                continue

            r_c = size_field.evaluate(cx, cy)
            r_c = max(r_c, h_min * 0.5)
            if not clears_exterior_boundary(cx, cy, r_c):
                continue

            # check against all nearby points (including existing)
            neighbours = grid.query_radius(cx, cy, r_c)
            too_close = False
            for _, nx_p, ny_p in neighbours:
                d2 = (cx - nx_p) ** 2 + (cy - ny_p) ** 2
                if d2 < r_c * r_c:
                    too_close = True
                    break

            if not too_close:
                nidx = len(points)
                points.append((cx, cy))
                active.append(nidx)
                grid.insert(nidx, cx, cy)
                accepted = True
                break

        if not accepted:
            last = active.pop()
            if last != i:
                active[active.index(i)] = last

    return np.array(points, dtype=float)
