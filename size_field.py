"""
Module 2: Continuous Size Field  (v3)
======================================
Given a plate + edge-crack geometry, defines a smooth scalar field

    h(x, y)  ->  target element size at any spatial point.

Four physical drivers, fused via ``min``:

    1.  Crack-tip    distance    d_tip      (dominant)
    2.  Crack-line   distance    d_crack    (point-to-segment)
    3.  Crack-entry  distance    d_entry    (local, near crack-boundary junction)
    4.  Background               h_max      (far-field floor)

Each driver uses a continuous exponential ramp:

    h_factor(d) = h_min + (h_max - h_min) * (1 - exp(-d / L))

The final size is ``h = min(h_tip, h_crack, h_entry)``, floored at h_max.

No global boundary term.  Only the crack-entry junction gets local refinement;
the rest of the boundary stays coarse.  This matches fracture-mechanics
practice: compute budget goes to the crack, not the perimeter.

Anisotropic interface
---------------------
``metric(x, y)`` returns a 2x2 metric tensor M.  Currently isotropic
(M = I / h^2), but the signature is reserved for future anisotropic
size fields (different stretch along crack-tangent vs crack-normal).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Point-to-segment distance  (scalar + bulk)
# ---------------------------------------------------------------------------
def point_to_segment_distance(
    px: float, py: float,
    sx: float, sy: float,
    tx: float, ty: float,
) -> float:
    dx = tx - sx
    dy = ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-30:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / L2
    t = max(0.0, min(1.0, t))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_to_segment_distance_bulk(
    pts: np.ndarray,
    sx: float, sy: float,
    tx: float, ty: float,
) -> np.ndarray:
    dx = tx - sx
    dy = ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-30:
        return np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
    t = ((pts[:, 0] - sx) * dx + (pts[:, 1] - sy) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return np.hypot(pts[:, 0] - proj_x, pts[:, 1] - proj_y)


# ---------------------------------------------------------------------------
# Continuous size field
# ---------------------------------------------------------------------------
class ContinuousSizeField:
    """Smooth mesh-size field for a 2-D edge-cracked plate.

    Parameters
    ----------
    geometry : EdgeCrackGeometry
    h_min : float
        Minimum element size (at the crack tip / entry / on the crack line).
    h_max : float
        Far-field element size (background).
    L_tip : float
        Decay length for crack-tip refinement.
    L_crack : float
        Decay length for crack-line refinement.
    L_entry : float or None
        Decay length for crack-entry refinement (where crack meets plate edge).
        If None, entry refinement is disabled.
    mode : str
        ``"analytic"`` (default) or ``"AI"``.
    """

    def __init__(
        self,
        geometry,
        h_min: float = 0.004,
        h_max: float = 0.030,
        L_tip: float = 0.08,
        L_crack: float = 0.05,
        L_entry: float | None = 0.08,
        mode: str = "analytic",
    ):
        self._geo = geometry
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.L_tip = float(L_tip)
        self.L_crack = float(L_crack)
        self.L_entry = float(L_entry) if L_entry is not None else None

        if mode not in ("analytic", "AI"):
            raise ValueError(f"mode must be 'analytic' or 'AI', got '{mode}'")
        self.mode = mode

        # cache crack-start for entry distance
        self._entry = geometry.crack_start

        self._validate()

    def _validate(self) -> None:
        if self.h_min <= 0 or self.h_max <= 0:
            raise ValueError("h_min and h_max must be > 0")
        if self.h_min >= self.h_max:
            raise ValueError("h_min must be < h_max")
        if self.L_tip <= 0 or self.L_crack <= 0:
            raise ValueError("L_tip and L_crack must be > 0")
        if self.L_entry is not None and self.L_entry <= 0:
            raise ValueError("L_entry must be > 0")

    # ==================================================================
    # Exponential ramp
    # ==================================================================
    @staticmethod
    def _exp_ramp(d: float, L: float, lo: float, hi: float) -> float:
        if d <= 0.0:
            return lo
        return lo + (hi - lo) * (1.0 - math.exp(-d / L))

    @staticmethod
    def _exp_ramp_bulk(d: np.ndarray, L: float, lo: float, hi: float) -> np.ndarray:
        d = np.maximum(d, 0.0)
        return lo + (hi - lo) * (1.0 - np.exp(-d / L))

    # ==================================================================
    # Scalar evaluate
    # ==================================================================
    def evaluate(self, x: float, y: float) -> float:
        if self.mode == "analytic":
            return self._evaluate_analytic(x, y)
        else:
            return self._evaluate_ai(x, y)

    def _evaluate_analytic(self, x: float, y: float) -> float:
        # 1. crack tip
        d_tip = self._geo.distance_to_tip(x, y)
        h_tip = self._exp_ramp(d_tip, self.L_tip, self.h_min, self.h_max)

        # 2. crack segment
        sx, sy = self._geo.crack_start
        tx, ty = self._geo.crack_tip
        d_crack = point_to_segment_distance(x, y, sx, sy, tx, ty)
        h_crack = self._exp_ramp(d_crack, self.L_crack, self.h_min, self.h_max)

        candidates = [h_tip, h_crack]

        # 3. crack-entry refinement (local, near the crack-boundary junction)
        if self.L_entry is not None:
            ex, ey = self._entry
            d_entry = math.hypot(x - ex, y - ey)
            h_entry = self._exp_ramp(d_entry, self.L_entry, self.h_min, self.h_max)
            candidates.append(h_entry)

        h_base = min(candidates)

        # gradient limit: |grad h| <= 0.25 prevents abrupt size transitions
        GRAD = 0.25
        h_lim_tip = self.h_min + GRAD * d_tip
        h_lim_crack = self.h_min + GRAD * d_crack
        return max(self.h_min, min(h_base, h_lim_tip, h_lim_crack, self.h_max))

    # ------------------------------------------------------------------
    # AI stub
    # ------------------------------------------------------------------
    def _evaluate_ai(self, x: float, y: float) -> float:
        raise NotImplementedError(
            "AI mode is not yet implemented. "
            "Override _evaluate_ai() with your trained model."
        )

    # ==================================================================
    # Anisotropic metric tensor
    # ==================================================================
    def metric(self, x: float, y: float) -> np.ndarray:
        h = self.evaluate(x, y)
        inv_h2 = 1.0 / (h * h)
        return np.array([[inv_h2, 0.0], [0.0, inv_h2]])

    def metric_bulk(self, pts: np.ndarray) -> np.ndarray:
        h = self.evaluate_bulk(pts)
        inv_h2 = 1.0 / (h * h)
        M = np.zeros((len(pts), 2, 2))
        M[:, 0, 0] = inv_h2
        M[:, 1, 1] = inv_h2
        return M

    # ==================================================================
    # Bulk evaluate
    # ==================================================================
    def evaluate_bulk(self, pts: npt.NDArray[np.floating]) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if self.mode == "analytic":
            return self._evaluate_analytic_bulk(pts)
        else:
            return np.array([self._evaluate_ai(x, y) for x, y in pts])

    def _evaluate_analytic_bulk(self, pts: np.ndarray) -> np.ndarray:
        d_tip = self._geo.distance_to_tip_bulk(pts)
        h_tip = self._exp_ramp_bulk(d_tip, self.L_tip, self.h_min, self.h_max)

        sx, sy = self._geo.crack_start
        tx, ty = self._geo.crack_tip
        d_crack = point_to_segment_distance_bulk(pts, sx, sy, tx, ty)
        h_crack = self._exp_ramp_bulk(d_crack, self.L_crack, self.h_min, self.h_max)

        H = np.minimum(h_tip, h_crack)

        if self.L_entry is not None:
            ex, ey = self._entry
            d_entry = np.hypot(pts[:, 0] - ex, pts[:, 1] - ey)
            h_entry = self._exp_ramp_bulk(d_entry, self.L_entry, self.h_min, self.h_max)
            H = np.minimum(H, h_entry)

        # gradient limit: |grad h| <= 0.25
        GRAD = 0.25
        H = np.maximum(self.h_min, H)
        H = np.minimum(H, self.h_min + GRAD * d_tip)
        H = np.minimum(H, self.h_min + GRAD * d_crack)
        H = np.minimum(H, self.h_max)
        return H

    # ==================================================================
    # Grid evaluation
    # ==================================================================
    def evaluate_on_grid(
        self, nx: int = 200, ny: int = 400
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        W_plate, H_plate = self._geo.W, self._geo.H
        xs = np.linspace(0.0, W_plate, nx)
        ys = np.linspace(0.0, H_plate, ny)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        H = self.evaluate_bulk(pts).reshape(X.shape)
        return X, Y, H

    # ==================================================================
    # Visualisation
    # ==================================================================
    def plot_heatmap(self, nx: int = 200, ny: int = 400, ax=None):
        import matplotlib.pyplot as plt

        X, Y, H = self.evaluate_on_grid(nx, ny)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.set_aspect("equal")

        im = ax.pcolormesh(
            X, Y, H, cmap="RdYlBu_r", shading="auto",
            vmin=self.h_min, vmax=self.h_max,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Target element size  h(x, y)", fontsize=9)

        sx, sy = self._geo.crack_start
        tx, ty = self._geo.crack_tip
        ax.plot([sx, tx], [sy, ty], "k-", linewidth=2.0)
        ax.plot(tx, ty, "ko", markersize=5, markerfacecolor="black")
        ax.plot(sx, sy, "ks", markersize=5, markerfacecolor="black")

        lines = [
            f"h_min    = {self.h_min:.4f}",
            f"h_max    = {self.h_max:.3f}",
            f"L_tip    = {self.L_tip:.3f}",
            f"L_crack  = {self.L_crack:.3f}",
        ]
        if self.L_entry is not None:
            lines.append(f"L_entry  = {self.L_entry:.3f}")
        lines.append(f"mode     = {self.mode}")

        ax.text(0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes, verticalalignment="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Continuous Size Field  (mode={self.mode})")
        return ax

    # ------------------------------------------------------------------
    def summary(self) -> str:
        parts = [
            f"ContinuousSizeField  mode={self.mode}",
            f"  h_min       = {self.h_min}",
            f"  h_max       = {self.h_max}",
            f"  L_tip       = {self.L_tip}   (tip decay)",
            f"  L_crack     = {self.L_crack}   (crack-line decay)",
            f"  d_crack     = point-to-SEGMENT",
        ]
        if self.L_entry is not None:
            parts.append(f"  L_entry     = {self.L_entry}   (crack-entry decay)")
        parts.append("  metric()    = isotropic (reserved for anisotropic)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from edge_crack_geometry import EdgeCrackGeometry, generate_crack
    import matplotlib.pyplot as plt

    start, tip = generate_crack("left", origin=1.0, length=0.30, theta=0.0)
    geo = EdgeCrackGeometry(
        W=1.0, H=2.0,
        crack_start=start, crack_tip=tip,
        crack_length=0.30, theta=0.0, edge_side="left",
    )

    sf = ContinuousSizeField(
        geo, h_min=0.004, h_max=0.030,
        L_tip=0.08, L_crack=0.05, L_entry=0.08,
    )
    print(sf.summary())

    # key tests
    print("\n--- Size-field spot checks ---")
    print(f"h at tip     (0.30, 1.00)     = {sf.evaluate(0.30, 1.00):.6f}")
    print(f"h at entry   (0.00, 1.00)     = {sf.evaluate(0.00, 1.00):.6f}")
    print(f"h past tip   (0.50, 1.00)     = {sf.evaluate(0.50, 1.00):.6f}")
    print(f"h far right  (1.00, 1.00)     = {sf.evaluate(1.00, 1.00):.6f}")
    print(f"h above tip  (0.30, 1.50)     = {sf.evaluate(0.30, 1.50):.6f}")
    print(f"h corner     (1.00, 2.00)     = {sf.evaluate(1.00, 2.00):.6f}")
    print(f"h near entry (0.02, 1.02)     = {sf.evaluate(0.02, 1.02):.6f}")

    sf.plot_heatmap(nx=200, ny=400)
    plt.savefig("/sessions/keen-brave-ritchie/mnt/3/size_field_v3_heatmap.png",
                dpi=150, bbox_inches="tight")
    print("\nSaved: size_field_v3_heatmap.png")
