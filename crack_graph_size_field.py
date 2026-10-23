"""Deterministic safety size field for arbitrary crack graphs."""

from __future__ import annotations

import math

import numpy as np


class CrackGraphSizeField:
    """Continuous analytic baseline for polylines, multiple cracks and branches.

    This field is deliberately independent of training data.  A future learned
    residual can refine it, while clipping and complexity controls retain this
    deterministic safe fallback.
    """

    def __init__(
        self,
        geometry,
        h_min: float = 0.004,
        h_max: float = 0.030,
        L_tip: float = 0.08,
        L_crack: float = 0.05,
        L_entry: float | None = 0.08,
        L_branch: float | None = 0.08,
        curvature_strength: float = 0.35,
    ) -> None:
        self._geo = geometry
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.L_tip = float(L_tip)
        self.L_crack = float(L_crack)
        self.L_entry = None if L_entry is None else float(L_entry)
        self.L_branch = None if L_branch is None else float(L_branch)
        self.curvature_strength = float(curvature_strength)
        if not 0.0 < self.h_min < self.h_max:
            raise ValueError("Require 0 < h_min < h_max")
        for name in ("L_tip", "L_crack"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.L_entry is not None and self.L_entry <= 0.0:
            raise ValueError("L_entry must be positive")
        if self.L_branch is not None and self.L_branch <= 0.0:
            raise ValueError("L_branch must be positive")
        if self.curvature_strength < 0.0:
            raise ValueError("curvature_strength cannot be negative")

    def _ramp(self, distance: np.ndarray, length: float) -> np.ndarray:
        values = np.maximum(np.asarray(distance, dtype=float), 0.0)
        return self.h_min + (self.h_max - self.h_min) * (1.0 - np.exp(-values / length))

    def evaluate_bulk(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        local = self._geo.local_geometry_query(values)
        crack_distance = np.asarray(local["distance"], dtype=float)
        result = self._ramp(crack_distance, self.L_crack)
        tip_distance = np.asarray(local["tip_distance"], dtype=float)
        if np.any(np.isfinite(tip_distance)):
            result = np.minimum(result, self._ramp(tip_distance, self.L_tip))
        if self.L_entry is not None:
            entry_distance = np.asarray(local["entry_distance"], dtype=float)
            if np.any(np.isfinite(entry_distance)):
                result = np.minimum(result, self._ramp(entry_distance, self.L_entry))
        if self.L_branch is not None:
            branch_distance = np.asarray(local["branch_distance"], dtype=float)
            if np.any(np.isfinite(branch_distance)):
                result = np.minimum(result, self._ramp(branch_distance, self.L_branch))

        turning = np.asarray(local["turning"], dtype=float)
        if self.curvature_strength > 0.0:
            curvature_scale = np.clip(1.0 - self.curvature_strength * turning, 0.45, 1.0)
            result = np.minimum(result, self.h_min + curvature_scale * (result - self.h_min))

        # A Lipschitz-like limiter prevents abrupt changes next to any crack
        # segment, tip or branch even before mesh-quality optimization.
        gradient = 0.25
        result = np.minimum(result, self.h_min + gradient * crack_distance)
        if np.any(np.isfinite(tip_distance)):
            result = np.minimum(result, self.h_min + gradient * tip_distance)
        return np.clip(result, self.h_min, self.h_max)

    def evaluate(self, x: float, y: float) -> float:
        return float(self.evaluate_bulk(np.asarray([[x, y]], dtype=float))[0])

    def metric(self, x: float, y: float) -> np.ndarray:
        h = self.evaluate(x, y)
        return np.eye(2, dtype=float) / (h * h)

    def metric_bulk(self, points: np.ndarray) -> np.ndarray:
        h = self.evaluate_bulk(points)
        metric = np.zeros((len(h), 2, 2), dtype=float)
        metric[:, 0, 0] = 1.0 / (h * h)
        metric[:, 1, 1] = 1.0 / (h * h)
        return metric

    def summary(self) -> str:
        return "\n".join(
            (
                "CrackGraphSizeField",
                f"  h range       = [{self.h_min}, {self.h_max}]",
                f"  tip decay     = {self.L_tip}",
                f"  crack decay   = {self.L_crack}",
                f"  entry decay   = {self.L_entry}",
                f"  branch decay  = {self.L_branch}",
                f"  curvature     = {self.curvature_strength}",
            )
        )


__all__ = ["CrackGraphSizeField"]
