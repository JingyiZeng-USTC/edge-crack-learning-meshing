"""
Module 1: Edge Crack Geometry (arbitrary single-edge crack)
============================================================
Supports all four plate edges: left, right, top, bottom.
A crack is defined by its edge, origin position on that edge, length a, and
angle theta (measured from the inward-pointing axis into the plate).

A single ``generate_crack()`` function computes (start, tip) for any edge.

Random geometry generation via ``EdgeCrackGeometry.random_generate(sample_id)``
produces reproducible, diverse training geometries for downstream modules.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Literal

import numpy as np

EdgeSide = Literal["left", "right", "top", "bottom"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _clamp_in_plate(
    start: Tuple[float, float],
    tip: Tuple[float, float],
    W: float,
    H: float,
) -> bool:
    """Return True if the crack tip lies inside the rectangular domain."""
    x, y = tip
    return 0.0 <= x <= W and 0.0 <= y <= H


def generate_crack(
    edge_side: EdgeSide,
    origin: float,       # y0 for left/right, x0 for top/bottom
    length: float,        # a
    theta: float,         # radians, measured from inward axis
    W: float = 1.0,
    H: float = 2.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (crack_start, crack_tip) for an arbitrary single-edge crack.

    Parameters
    ----------
    edge_side : "left" | "right" | "top" | "bottom"
    origin : float
        Position along the edge: y-coordinate for left/right, x-coordinate
        for top/bottom.
    length : float
        Crack length a.
    theta : float
        Angle in radians.  For left/right edges, theta=0 means horizontal.
        For top/bottom edges, theta=0 means vertical.  Always measured
        relative to the inward normal of the edge.

    Returns
    -------
    (start, tip) : each a (float, float) tuple.
    """
    # Local convention used by every edge:
    #   theta = 0       -> inward boundary normal
    #   positive theta  -> positive coordinate direction along the edge
    # This keeps the same parameter meaning after rotating/reflection-mapping
    # any boundary to the canonical left edge.
    inward = {
        "left": np.asarray([1.0, 0.0]),
        "right": np.asarray([-1.0, 0.0]),
        "top": np.asarray([0.0, -1.0]),
        "bottom": np.asarray([0.0, 1.0]),
    }
    along = {
        "left": np.asarray([0.0, 1.0]),
        "right": np.asarray([0.0, 1.0]),
        "top": np.asarray([1.0, 0.0]),
        "bottom": np.asarray([1.0, 0.0]),
    }
    if edge_side not in inward:
        raise ValueError(f"Unknown edge_side: {edge_side}")
    direction = math.cos(theta) * inward[edge_side] + math.sin(theta) * along[edge_side]

    if edge_side == "left":
        start = (0.0, origin)

    elif edge_side == "right":
        start = (W, origin)

    elif edge_side == "top":
        start = (origin, H)

    elif edge_side == "bottom":
        start = (origin, 0.0)

    else:
        raise ValueError(f"Unknown edge_side: {edge_side}")

    tip = tuple(np.asarray(start, dtype=float) + float(length) * direction)

    return start, tip


# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------
@dataclass
class EdgeCrackGeometry:
    """Rectangular plate [0,W]x[0,H] with a single edge crack.

    All downstream modules (Module 2, 3, 4, ...) consume ONLY this object.
    They never need to know how the crack was generated.
    """

    W: float
    H: float
    crack_start:   Tuple[float, float]
    crack_tip:     Tuple[float, float]
    crack_length:  float
    theta:         float
    edge_side:     EdgeSide

    # ---- crack discretisation cache (private) ---------------------------
    _crack_vertices:   np.ndarray | None = None
    _crack_segments:   np.ndarray | None = None
    _crack_n_segments: int = 0

    # ---- precomputed directions (computed once in __post_init__) --------
    _tangent: Tuple[float, float] = (0.0, 0.0)
    _normal:  Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if self.W <= 0 or self.H <= 0:
            raise ValueError("W and H must be > 0")
        if self.crack_length <= 0 or self.crack_length >= min(self.W, self.H):
            raise ValueError(
                f"crack_length ({self.crack_length}) must be in (0, min(W,H)={min(self.W,self.H)})"
            )
        if not _clamp_in_plate(self.crack_start, self.crack_tip, self.W, self.H):
            raise ValueError("Crack tip lies outside the plate")

        # unit tangent  (tip - start) / length
        dx = self.crack_tip[0] - self.crack_start[0]
        dy = self.crack_tip[1] - self.crack_start[1]
        L = math.hypot(dx, dy)
        if L < 1e-15:
            raise ValueError("Crack has zero length")
        self._tangent = (dx / L, dy / L)
        # normal = rotate tangent 90 deg CCW
        self._normal = (-self._tangent[1], self._tangent[0])

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------
    @property
    def crack_tangent(self) -> Tuple[float, float]:
        """Unit vector along the crack direction (start -> tip)."""
        return self._tangent

    @property
    def crack_normal(self) -> Tuple[float, float]:
        """Unit normal to the crack (tangent rotated 90 deg CCW)."""
        return self._normal

    @property
    def boundary(self) -> List[Tuple[float, float]]:
        W, H = self.W, self.H
        return [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]

    # ------------------------------------------------------------------
    # Point queries
    # ------------------------------------------------------------------
    def inside(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.W and 0.0 <= y <= self.H

    def distance_to_tip(self, x: float, y: float) -> float:
        tx, ty = self.crack_tip
        return math.hypot(x - tx, y - ty)

    def distance_to_crack(self, x: float, y: float) -> float:
        """Signed distance from point to the infinite crack line.

        Positive on the normal side, but the absolute value is what
        Module 2 typically uses: ``abs(distance_to_crack(x, y))``.
        """
        sx, sy = self.crack_start
        nx, ny = self._normal
        # perpendicular distance = |(p - start) dot normal|
        return abs((x - sx) * nx + (y - sy) * ny)

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------
    def distance_to_tip_bulk(self, pts: np.ndarray) -> np.ndarray:
        tx, ty = self.crack_tip
        return np.hypot(pts[:, 0] - tx, pts[:, 1] - ty)

    def distance_to_crack_bulk(self, pts: np.ndarray) -> np.ndarray:
        sx, sy = self.crack_start
        nx, ny = self._normal
        return np.abs((pts[:, 0] - sx) * nx + (pts[:, 1] - sy) * ny)

    def inside_bulk(self, pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        return (0.0 <= x) & (x <= self.W) & (0.0 <= y) & (y <= self.H)

    # ------------------------------------------------------------------
    # Crack discretisation
    # ------------------------------------------------------------------
    def discretize_crack(self, n_segments: int = 10) -> None:
        """Generate equally-spaced vertices and index-pair segments."""
        if n_segments < 1:
            raise ValueError("n_segments >= 1 required")
        self._crack_n_segments = n_segments

        sx, sy = self.crack_start
        tx, ty = self.crack_tip
        xs = np.linspace(sx, tx, n_segments + 1)
        ys = np.linspace(sy, ty, n_segments + 1)
        self._crack_vertices = np.column_stack([xs, ys])

        ids = np.arange(n_segments + 1, dtype=np.int32)
        self._crack_segments = np.column_stack([ids[:-1], ids[1:]])

    @property
    def crack_vertices(self) -> np.ndarray | None:
        return self._crack_vertices

    @property
    def crack_segments(self) -> np.ndarray | None:
        return self._crack_segments

    @property
    def crack_n_segments(self) -> int:
        return self._crack_n_segments

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"Plate:          {self.W} x {self.H}\n"
            f"Edge:           {self.edge_side}\n"
            f"Crack length:   a = {self.crack_length:.4f}\n"
            f"Theta:          {math.degrees(self.theta):.2f} deg\n"
            f"Crack start:    {self.crack_start}\n"
            f"Crack tip:      {self.crack_tip}\n"
            f"Tangent:        {self._tangent}\n"
            f"Normal:         {self._normal}"
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(self, ax=None, show_crack_segments: bool = True):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.set_aspect("equal")
        W, H = self.W, self.H

        # plate boundary
        bx, by = [0, W, W, 0, 0], [0, 0, H, H, 0]
        ax.plot(bx, by, "k-", linewidth=1.5, label="Plate boundary")

        # crack
        sx, sy = self.crack_start
        tx, ty = self.crack_tip
        ax.plot([sx, tx], [sy, ty], "r-", linewidth=2.5, label="Crack")
        ax.plot(tx, ty, "ro", markersize=8, markerfacecolor="red",
                markeredgewidth=0)

        # discretised vertices
        if show_crack_segments and self._crack_vertices is not None:
            ax.scatter(
                self._crack_vertices[:, 0], self._crack_vertices[:, 1],
                c="blue", s=20, zorder=5, marker="o", edgecolors="none",
            )

        # annotations
        ax.annotate(
            "crack tip", xy=(tx, ty),
            xytext=(tx + 0.15 * W, ty + 0.10 * H),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red",
        )
        mid_x, mid_y = (sx + tx) / 2, (sy + ty) / 2
        ax.annotate(
            f"a={self.crack_length:.2f}",
            xy=(mid_x, mid_y),
            xytext=(mid_x + 0.08, mid_y + 0.06 * H),
            ha="center", fontsize=9, color="red",
        )

        ax.set_xlim(-0.15 * W, W + 0.25 * W)
        ax.set_ylim(-0.10 * H, H + 0.10 * H)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"Edge Crack  (edge={self.edge_side}, a={self.crack_length:.2f}, "
            f"theta={math.degrees(self.theta):.1f} deg)"
        )
        ax.legend(loc="upper right", fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Factory: random generation from a sample ID
    # ------------------------------------------------------------------
    @classmethod
    def random_generate(
        cls,
        sample_id: int,
        W: float = 1.0,
        H: float = 2.0,
        seed: int | None = None,
    ) -> "EdgeCrackGeometry":
        """Produce a random single-edge crack geometry.

        The ``sample_id`` is hashed into a random seed so that each ID
        always produces the same geometry (reproducibility).

        Parameters
        ----------
        sample_id : int
            Unique identifier for this geometry sample.
        W, H : float
            Plate dimensions.
        seed : int, optional
            Global seed mixed with sample_id.  Use for dataset-level
            reproducibility.
        """
        base = seed if seed is not None else 42
        rng = random.Random(f"{base}-{sample_id}")

        # --- random edge ---
        edge_side: EdgeSide = rng.choice(["left", "right", "top", "bottom"])

        # --- random crack length ---
        min_dim = min(W, H)
        a_min = 0.05 * min_dim
        a_max = 0.45 * min_dim
        crack_length = rng.uniform(a_min, a_max)

        # --- random theta ---
        # Keep crack mostly pointing into the plate (theta in [-pi/3, pi/3])
        theta = rng.uniform(-math.pi / 3, math.pi / 3)

        # --- random origin on the edge (with margin so tip stays inside) ---
        margin = crack_length * abs(math.sin(theta)) + 0.02 * min_dim

        if edge_side in ("left", "right"):
            origin = rng.uniform(margin, H - margin)
        else:
            origin = rng.uniform(margin, W - margin)

        start, tip = generate_crack(edge_side, origin, crack_length, theta, W, H)

        # If tip lands outside (corner case), retry with tighter theta
        retries = 0
        while not _clamp_in_plate(start, tip, W, H) and retries < 100:
            theta = rng.uniform(-math.pi / 6, math.pi / 6)
            margin2 = crack_length * abs(math.sin(theta)) + 0.05 * min_dim
            if edge_side in ("left", "right"):
                origin = rng.uniform(margin2, H - margin2)
            else:
                origin = rng.uniform(margin2, W - margin2)
            start, tip = generate_crack(edge_side, origin, crack_length, theta, W, H)
            retries += 1

        if not _clamp_in_plate(start, tip, W, H):
            raise RuntimeError(
                f"Failed to generate valid crack for sample_id={sample_id} "
                f"after {retries} retries"
            )

        return cls(
            W=W,
            H=H,
            crack_start=start,
            crack_tip=tip,
            crack_length=crack_length,
            theta=theta,
            edge_side=edge_side,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- fixed-geometry example ---
    print("=" * 55)
    print("Fixed geometry (left edge, horizontal)")
    print("=" * 55)
    start, tip = generate_crack("left", origin=1.0, length=0.30, theta=0.0)
    geo = EdgeCrackGeometry(
        W=1.0, H=2.0,
        crack_start=start, crack_tip=tip,
        crack_length=0.30, theta=0.0, edge_side="left",
    )
    print(geo.summary())
    geo.discretize_crack(n_segments=6)
    print(f"\nCrack segments ({geo.crack_segments.shape[0]} pairs):")
    print(geo.crack_segments)

    # distance tests
    print(f"\ndistance_to_tip(0.3, 1.0)      = {geo.distance_to_tip(0.3, 1.0):.6f}")
    print(f"distance_to_crack(0.1, 0.9)    = {geo.distance_to_crack(0.1, 0.9):.6f}")
    print(f"inside(0.5, 1.0)                = {geo.inside(0.5, 1.0)}")

    # bulk
    pts = np.array([[0.3, 1.0], [0.5, 1.0], [0.1, 1.2]])
    print(f"Bulk distance_to_crack: {geo.distance_to_crack_bulk(pts)}")
    print(f"Bulk inside:            {geo.inside_bulk(pts)}")

    geo.plot()
    plt.savefig("D:\\itl\\eg\\3\\edge_crack_left.png", dpi=120, bbox_inches="tight")
    print("\nSaved: edge_crack_left.png")

    # --- random generation example ---
    print("\n" + "=" * 55)
    print("Random geometries (sample_id = 1, 2, 3)")
    print("=" * 55)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for i, sid in enumerate([1, 2, 3]):
        geo_r = EdgeCrackGeometry.random_generate(sample_id=sid, seed=2026)
        print(f"\nSample {sid}:")
        print(geo_r.summary())
        geo_r.plot(ax=axes[i])

    fig.tight_layout()
    plt.savefig("D:\\itl\\eg\\3\\edge_crack_random.png", dpi=120, bbox_inches="tight")
    print("\nSaved: edge_crack_random.png")
