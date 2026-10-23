"""
Module 3: Adaptive Point Distribution
======================================
Converts a geometry + continuous size field into a quality point cloud
using Bridson's Poisson-disk algorithm with adaptive radius.

Input:   Geometry (Module 1) + SizeField (Module 2)
Output:  points  (N x 2 numpy array)  -- no edges, no triangles.

Pipeline
--------
  1.  Boundary points   -- adaptive marching along the 4 plate edges
  2.  Crack points      -- direct from geometry.crack_vertices (Module 1)
  3.  Interior points   -- Bridson sampling, subdomain by subdomain
  4.  Merge + deduplicate

The crack acts as an internal barrier: each subdomain (above / below the
crack line) is sampled independently so points never cross the crack.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

import numpy as np


# ======================================================================
# Spatial grid for O(1) neighbour queries
# ======================================================================
class _SpatialGrid:
    """Uniform grid for fast radius-based nearest-neighbour checks."""

    def __init__(self, cell_size: float, W: float, H: float):
        self.cell_size = float(cell_size)
        self.W = W
        self.H = H
        self.nx = max(1, int(math.ceil(W / cell_size)))
        self.ny = max(1, int(math.ceil(H / cell_size)))
        # each cell holds a list of (point_index, x, y)
        self._cells: List[List[Tuple[int, float, float]]] = [
            [] for _ in range(self.nx * self.ny)
        ]

    def _cell_ix(self, cx: int, cy: int) -> int:
        return cy * self.nx + cx

    def _coords_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            max(0, min(self.nx - 1, int(x / self.cell_size))),
            max(0, min(self.ny - 1, int(y / self.cell_size))),
        )

    def insert(self, idx: int, x: float, y: float) -> None:
        cx, cy = self._coords_to_cell(x, y)
        self._cells[self._cell_ix(cx, cy)].append((idx, x, y))

    def query_radius(
        self, x: float, y: float, radius: float
    ) -> List[Tuple[int, float, float]]:
        """Return all stored points within *radius* of (x, y)."""
        r_cells = max(1, int(math.ceil(radius / self.cell_size)))
        cx, cy = self._coords_to_cell(x, y)
        r2 = radius * radius
        result = []
        for dx in range(-r_cells, r_cells + 1):
            nx_cell = cx + dx
            if nx_cell < 0 or nx_cell >= self.nx:
                continue
            for dy in range(-r_cells, r_cells + 1):
                ny_cell = cy + dy
                if ny_cell < 0 or ny_cell >= self.ny:
                    continue
                for idx, px, py in self._cells[self._cell_ix(nx_cell, ny_cell)]:
                    d2 = (x - px) * (x - px) + (y - py) * (y - py)
                    if d2 < r2:
                        result.append((idx, px, py))
        return result


# ======================================================================
# Subdomain helpers
# ======================================================================
def _signed_dist_to_crack(
    x: float, y: float,
    sx: float, sy: float,
    nx: float, ny: float,
) -> float:
    """Signed distance: positive on normal side, negative on anti-normal side."""
    return (x - sx) * nx + (y - sy) * ny


def _in_subdomain(
    x: float, y: float,
    sx: float, sy: float,
    nx: float, ny: float,
    side: int,
    eps: float = 1e-12,
) -> bool:
    """True if point is strictly on the requested side of the crack."""
    d = _signed_dist_to_crack(x, y, sx, sy, nx, ny)
    if side == 0:
        return d > eps   # positive side
    else:
        return d < -eps  # negative side


# ======================================================================
# Boundary point generation  (adaptive 1-D marching along each edge)
# ======================================================================
def _generate_boundary_points(
    W: float, H: float,
    size_field,
    crack_entry: Tuple[float, float],
    L_entry: float = 0.08,
) -> np.ndarray:
    """Walk the four edges placing points at uniform coarse spacing.

    Away from the crack entry, spacing = h_max.
    Near the crack entry (within ~2*L_entry), use the size-field spacing
    so the boundary is appropriately refined there.
    """
    h_max = size_field.h_max
    default_spacing = h_max * 0.85  # slightly finer than h_max for safety
    entry_x, entry_y = crack_entry

    def _march_edge(x0, y0, x1, y1) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-15:
            return pts
        ux = (x1 - x0) / L
        uy = (y1 - y0) / L
        travelled = 0.0
        while travelled < L:
            px = x0 + travelled * ux
            py = y0 + travelled * uy
            if not (0.0 <= px <= W and 0.0 <= py <= H):
                break
            pts.append((px, py))
            # near crack entry: use size field; elsewhere: uniform coarse
            d_entry = math.hypot(px - entry_x, py - entry_y)
            if L_entry is not None and d_entry < 2.0 * L_entry:
                spacing = size_field.evaluate(px, py)
                spacing = max(spacing, size_field.h_min * 0.5)
            else:
                spacing = default_spacing
            travelled += spacing
        # always include the far endpoint (corner)
        if len(pts) == 0 or (abs(pts[-1][0] - x1) > 1e-10 or abs(pts[-1][1] - y1) > 1e-10):
            pts.append((x1, y1))
        return pts

    # walk CCW  (each endpoint included only once by starting at the next corner)
    bottom = _march_edge(0.0, 0.0, W, 0.0)        # (0,0) -> (W,0)
    right  = _march_edge(W, 0.0, W, H)             # (W,0) -> (W,H)  -- but bottom already put (W,0)
    # remove duplicate at (W,0)
    if len(bottom) > 0 and len(right) > 0:
        bx, by = bottom[-1]
        rx, ry = right[0]
        if abs(bx - rx) < 1e-10 and abs(by - ry) < 1e-10:
            right = right[1:]
    top    = _march_edge(W, H, 0.0, H)
    if len(right) > 0 and len(top) > 0:
        rx, ry = right[-1]
        tx, ty = top[0]
        if abs(rx - tx) < 1e-10 and abs(ry - ty) < 1e-10:
            top = top[1:]
    left   = _march_edge(0.0, H, 0.0, 0.0)
    if len(top) > 0 and len(left) > 0:
        tx, ty = top[-1]
        lx, ly = left[0]
        if abs(tx - lx) < 1e-10 and abs(ty - ly) < 1e-10:
            left = left[1:]
    if len(left) > 0 and len(bottom) > 0:
        lx, ly = left[-1]
        bx, by = bottom[0]
        if abs(lx - bx) < 1e-10 and abs(ly - by) < 1e-10:
            left = left[:-1]

    all_pts = bottom + right + top + left
    return np.array(all_pts, dtype=float)


# ======================================================================
# Adaptive Poisson-disk  (Bridson's algorithm, per subdomain)
# ======================================================================
def _bridson_sample_subdomain(
    geometry,
    size_field,
    sx: float, sy: float,   # crack start
    nx: float, ny: float,   # crack normal
    side: int,
    rng: random.Random,
    k_candidates: int = 30,
    seed_attempts: int = 200,
) -> np.ndarray:
    """Sample one subdomain using adaptive Bridson."""

    W, H = geometry.W, geometry.H
    h_min = size_field.h_min

    # spatial grid  (cell = h_min / sqrt(2) guarantees <= 1 point per cell)
    cell_size = h_min / math.sqrt(2)
    grid = _SpatialGrid(cell_size, W, H)

    points: List[Tuple[float, float]] = []
    active: List[int] = []

    # ---- seed point ----
    seed = None
    for _ in range(seed_attempts):
        cx = rng.uniform(0.0, W)
        cy = rng.uniform(0.0, H)
        if geometry.inside(cx, cy) and _in_subdomain(cx, cy, sx, sy, nx, ny, side):
            seed = (cx, cy)
            break
    if seed is None:
        return np.empty((0, 2))  # subdomain too thin

    points.append(seed)
    active.append(0)
    grid.insert(0, seed[0], seed[1])

    # ---- main loop ----
    while active:
        i = active[rng.randrange(len(active))]
        px, py = points[i]
        r_i = size_field.evaluate(px, py)

        accepted = False
        for _ in range(k_candidates):
            # random point in annulus [r_i, 2*r_i]
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dist  = rng.uniform(r_i, 2.0 * r_i)
            cx = px + dist * math.cos(angle)
            cy = py + dist * math.sin(angle)

            # domain + subdomain checks
            if not geometry.inside(cx, cy):
                continue
            if not _in_subdomain(cx, cy, sx, sy, nx, ny, side):
                continue

            # adaptive radius at candidate
            r_c = size_field.evaluate(cx, cy)
            r_c = max(r_c, h_min * 0.5)  # floor to avoid collapse

            # distance check against neighbours
            neighbours = grid.query_radius(cx, cy, r_c)
            too_close = False
            for _, nx_p, ny_p in neighbours:
                d2 = (cx - nx_p) * (cx - nx_p) + (cy - ny_p) * (cy - ny_p)
                if d2 < r_c * r_c:
                    too_close = True
                    break

            if not too_close:
                idx = len(points)
                points.append((cx, cy))
                active.append(idx)
                grid.insert(idx, cx, cy)
                accepted = True
                break

        if not accepted:
            # swap-remove from active list
            last = active.pop()
            if last != i:
                active[active.index(i)] = last

    return np.array(points, dtype=float)


# ======================================================================
# Public API
# ======================================================================
class AdaptivePoissonSampler:
    """Generate an adaptive point cloud respecting a continuous size field.

    Parameters
    ----------
    geometry : EdgeCrackGeometry
        Module 1 geometry.
    size_field : ContinuousSizeField
        Module 2 size field.

    Usage
    -----
    >>> sampler = AdaptivePoissonSampler(geometry, size_field)
    >>> points = sampler.sample()
    >>> sampler.plot(points)
    """

    def __init__(self, geometry, size_field):
        self._geo = geometry
        self._sf  = size_field

    # ------------------------------------------------------------------
    def sample(
        self,
        k_candidates: int = 30,
        seed: int | None = None,
        n_crack_segments: int = 30,
        crack_spacing_factor: float = 0.90,
        n_layers: int = 3,
        n_tip_ring: int = 8,
        boundary_clearance_factor: float = 0.5,
        auxiliary_separation_factor: float = 0.5,
    ) -> np.ndarray:
        """Run the full sampling pipeline via the modular sub-modules.

        Delegates to ``sampling.sampling_pipeline.sample_all()``.
        """
        from sampling.sampling_pipeline import sample_all
        return sample_all(
            self._geo, self._sf,
            seed=seed,
            n_crack_segments=n_crack_segments,
            crack_spacing_factor=crack_spacing_factor,
            n_layers=n_layers,
            n_tip_ring=n_tip_ring,
            k_candidates=k_candidates,
            boundary_clearance_factor=boundary_clearance_factor,
            auxiliary_separation_factor=auxiliary_separation_factor,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Remove near-duplicate points.  Preserves order."""
        if len(pts) == 0:
            return pts
        tol2 = tol * tol
        keep = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            if not keep[i]:
                continue
            dx = pts[i+1:, 0] - pts[i, 0]
            dy = pts[i+1:, 1] - pts[i, 1]
            d2 = dx * dx + dy * dy
            keep[i+1:][d2 < tol2] = False
        return pts[keep]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self, points: np.ndarray) -> str:
        """One-line summary of the point cloud."""
        n = len(points)
        if n == 0:
            return "0 points"
        d = self._sf.evaluate_bulk(points)
        return (
            f"{n} points  "
            f"h: min={d.min():.5f}  median={np.median(d):.5f}  "
            f"max={d.max():.5f}"
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(self, points: np.ndarray, ax=None, markersize: float = 0.5):
        """Scatter-plot the point cloud, colour-coded by local size-field value."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.set_aspect("equal")
        W, H = self._geo.W, self._geo.H

        h_vals = self._sf.evaluate_bulk(points)

        sc = ax.scatter(
            points[:, 0], points[:, 1],
            c=h_vals, s=markersize,
            cmap="RdYlBu_r",
            vmin=self._sf.h_min, vmax=self._sf.h_max,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("h(x,y) at point", fontsize=8)

        # crack overlay
        sx, sy = self._geo.crack_start
        tx, ty = self._geo.crack_tip
        ax.plot([sx, tx], [sy, ty], "k-", linewidth=1.5)
        ax.plot(tx, ty, "ko", markersize=4, markerfacecolor="black")

        ax.set_xlim(-0.02 * W, W * 1.02)
        ax.set_ylim(-0.02 * H, H * 1.02)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Adaptive Point Cloud  ({len(points)} points)")
        return ax


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from edge_crack_geometry import EdgeCrackGeometry, generate_crack
    from size_field import ContinuousSizeField
    import matplotlib.pyplot as plt

    # ---- geometry ----
    start, tip = generate_crack("left", origin=1.0, length=0.30, theta=0.0)
    geo = EdgeCrackGeometry(
        W=1.0, H=2.0,
        crack_start=start, crack_tip=tip,
        crack_length=0.30, theta=0.0, edge_side="left",
    )

    # ---- size field ----
    sf = ContinuousSizeField(
        geo, h_min=0.004, h_max=0.030,
        L_tip=0.08, L_crack=0.05,
        L_boundary=0.03, w_boundary=0.25,
    )

    # ---- adaptive sampling ----
    sampler = AdaptivePoissonSampler(geo, sf)
    points = sampler.sample(k_candidates=30, seed=42, n_crack_segments=20)
    print(sampler.stats(points))

    # ---- plot ----
    sampler.plot(points, markersize=2.0)
    plt.savefig("/sessions/keen-brave-ritchie/mnt/3/adaptive_points.png",
                dpi=150, bbox_inches="tight")
    print("Saved: adaptive_points.png")

    # ---- angled crack demo ----
    print("\n--- Angled crack ---")
    start2, tip2 = generate_crack("left", origin=0.8, length=0.25,
                                   theta=math.radians(25))
    geo2 = EdgeCrackGeometry(
        W=1.0, H=2.0,
        crack_start=start2, crack_tip=tip2,
        crack_length=0.25, theta=math.radians(25), edge_side="left",
    )
    sf2 = ContinuousSizeField(geo2, h_min=0.004, h_max=0.030,
                               L_tip=0.08, L_crack=0.05)
    sampler2 = AdaptivePoissonSampler(geo2, sf2)
    pts2 = sampler2.sample(k_candidates=30, seed=99, n_crack_segments=20)
    print(sampler2.stats(pts2))

    sampler2.plot(pts2, markersize=2.0)
    plt.savefig("/sessions/keen-brave-ritchie/mnt/3/adaptive_points_angled.png",
                dpi=150, bbox_inches="tight")
    print("Saved: adaptive_points_angled.png")
