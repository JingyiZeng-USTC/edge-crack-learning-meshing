"""Runtime continuous size field backed by a trained pointwise MLP."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from size_field import ContinuousSizeField

from .features import FEATURE_NAMES, build_point_features
from .model import PointSizeFieldMLP


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@lru_cache(maxsize=8)
def _load_runtime(checkpoint_path: str, device_name: str):
    device = _resolve_device(device_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    supported_families = {
        "learned-residual-size-field-v1",
        "learned-residual-size-field-v2-mesh",
    }
    if checkpoint.get("model_family") not in supported_families:
        raise ValueError(
            f"{checkpoint_path} is not a supported learned size-field checkpoint.",
        )
    if list(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Size-field checkpoint feature schema does not match this code.")
    model = PointSizeFieldMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std, device


class LearnedResidualSizeField:
    """Continuous learned residual evaluated on a cached structured grid.

    Network inference is batched once per geometry.  The adaptive sampler then
    receives cheap bilinear interpolation values, avoiding thousands of tiny
    GPU kernel launches.
    """

    def __init__(
        self,
        geometry,
        checkpoint: str,
        h_min: float = 0.004,
        h_max: float = 0.030,
        L_tip: float = 0.08,
        L_crack: float = 0.05,
        L_entry: float | None = 0.08,
        blend: float = 1.0,
        grid_nx: int = 161,
        grid_ny: int = 321,
        smooth_sigma: float = 1.0,
        complexity_ratio: float = 1.0,
        complexity_tolerance: float = 1e-5,
        device: str = "auto",
        inference_batch_size: int = 65536,
    ):
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if grid_nx < 2 or grid_ny < 2:
            raise ValueError("grid_nx and grid_ny must be at least 2")
        if complexity_ratio <= 0.0:
            raise ValueError("complexity_ratio must be positive")
        if complexity_tolerance <= 0.0:
            raise ValueError("complexity_tolerance must be positive")
        self._geo = geometry
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.L_tip = float(L_tip)
        self.L_crack = float(L_crack)
        self.L_entry = float(L_entry) if L_entry is not None else None
        self.mode = "learned-residual"
        self.checkpoint = str(Path(checkpoint).resolve())
        self.blend = float(blend)
        self.grid_nx = int(grid_nx)
        self.grid_ny = int(grid_ny)
        self.smooth_sigma = float(smooth_sigma)
        self.complexity_ratio = float(complexity_ratio)
        self.complexity_tolerance = float(complexity_tolerance)
        self.base = ContinuousSizeField(
            geometry, h_min=h_min, h_max=h_max, L_tip=L_tip,
            L_crack=L_crack, L_entry=L_entry,
        )
        self._model, self._mean, self._std, self._device = _load_runtime(
            self.checkpoint, device,
        )
        self._build_grid(inference_batch_size)

    def _build_grid(self, batch_size: int) -> None:
        xs = np.linspace(0.0, float(self._geo.W), self.grid_nx)
        ys = np.linspace(0.0, float(self._geo.H), self.grid_ny)
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        analytic_h = self.base.evaluate_bulk(points)
        if self.blend == 0.0:
            learned_h = analytic_h
        else:
            features = build_point_features(
                points, self._geo, analytic_h, self.h_max,
            )
            normalized = (features - self._mean) / self._std
            corrections = []
            with torch.no_grad():
                for start in range(0, len(normalized), int(batch_size)):
                    tensor = torch.from_numpy(
                        normalized[start:start + int(batch_size)],
                    ).to(self._device)
                    corrections.append(self._model(tensor).cpu().numpy())
            log_correction = np.concatenate(corrections)
            learned_h = np.clip(
                analytic_h * np.exp(self.blend * log_correction),
                self.h_min,
                self.h_max,
            )
        grid = learned_h.reshape(self.grid_ny, self.grid_nx)
        if self.smooth_sigma > 0.0 and self.blend > 0.0:
            from scipy.ndimage import gaussian_filter

            grid = gaussian_filter(grid, sigma=self.smooth_sigma, mode="nearest")
            grid = np.clip(grid, self.h_min, self.h_max)
        base_grid = analytic_h.reshape(self.grid_ny, self.grid_nx)
        protected = base_grid <= 1.10 * self.h_min
        grid[protected] = np.minimum(grid[protected], base_grid[protected])
        self._grid = self._enforce_complexity_budget(grid, base_grid, protected)

    @staticmethod
    def _complexity(grid: np.ndarray) -> float:
        """Uniform-grid approximation of integral(1 / h^2) over the plate."""
        values = np.asarray(grid, dtype=float)
        return float(np.mean(1.0 / np.maximum(values, 1e-30) ** 2))

    def _enforce_complexity_budget(
        self,
        learned_grid: np.ndarray,
        base_grid: np.ndarray,
        protected: np.ndarray,
    ) -> np.ndarray:
        """Globally rescale h while preserving the learned spatial ranking."""
        base_complexity = self._complexity(base_grid)
        target = self.complexity_ratio * base_complexity

        def candidate(scale: float) -> np.ndarray:
            values = np.clip(
                np.asarray(learned_grid, dtype=float) * float(scale),
                self.h_min,
                self.h_max,
            )
            # The immediate crack/tip core is a hard geometric safeguard.
            values[protected] = np.minimum(values[protected], base_grid[protected])
            return values

        lower = 1e-3
        upper = 1e3
        lower_grid = candidate(lower)
        upper_grid = candidate(upper)
        lower_complexity = self._complexity(lower_grid)
        upper_complexity = self._complexity(upper_grid)
        self.minimum_feasible_complexity_ratio = (
            upper_complexity / max(base_complexity, 1e-30)
        )
        self.maximum_feasible_complexity_ratio = (
            lower_complexity / max(base_complexity, 1e-30)
        )
        feasible_tolerance = 5.0 * self.complexity_tolerance
        if target < upper_complexity * (1.0 - feasible_tolerance):
            raise ValueError(
                f"Requested complexity_ratio={self.complexity_ratio:.6f} is "
                f"below the feasible minimum "
                f"{self.minimum_feasible_complexity_ratio:.6f} for "
                f"h_max={self.h_max} and the protected crack core. Increase "
                "the ratio or allow a larger h_max."
            )
        if target > lower_complexity * (1.0 + feasible_tolerance):
            raise ValueError(
                f"Requested complexity_ratio={self.complexity_ratio:.6f} is "
                f"above the feasible maximum "
                f"{self.maximum_feasible_complexity_ratio:.6f} for "
                f"h_min={self.h_min}. Decrease the ratio or allow a smaller h_min."
            )

        # Complexity decreases monotonically as the size scale increases.
        result = candidate(1.0)
        scale = 1.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            trial = candidate(middle)
            trial_complexity = self._complexity(trial)
            result = trial
            scale = middle
            relative_gap = abs(trial_complexity - target) / max(target, 1e-30)
            if relative_gap <= self.complexity_tolerance:
                break
            if trial_complexity > target:
                lower = middle
            else:
                upper = middle

        self.base_complexity = base_complexity
        self.target_complexity = target
        self.realized_complexity = self._complexity(result)
        self.realized_complexity_ratio = self.realized_complexity / max(
            base_complexity, 1e-30,
        )
        self.complexity_scale = float(scale)
        return np.asarray(result, dtype=float)

    def evaluate_bulk(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        x = np.clip(pts[:, 0] / float(self._geo.W), 0.0, 1.0) * (self.grid_nx - 1)
        y = np.clip(pts[:, 1] / float(self._geo.H), 0.0, 1.0) * (self.grid_ny - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, self.grid_nx - 1)
        y1 = np.minimum(y0 + 1, self.grid_ny - 1)
        wx = x - x0
        wy = y - y0
        values = (
            (1.0 - wx) * (1.0 - wy) * self._grid[y0, x0]
            + wx * (1.0 - wy) * self._grid[y0, x1]
            + (1.0 - wx) * wy * self._grid[y1, x0]
            + wx * wy * self._grid[y1, x1]
        )
        base_values = self.base.evaluate_bulk(pts)
        protected = base_values <= 1.10 * self.h_min
        values[protected] = np.minimum(values[protected], base_values[protected])
        return np.clip(values, self.h_min, self.h_max)

    def evaluate(self, x: float, y: float) -> float:
        return float(self.evaluate_bulk(np.asarray([[x, y]], dtype=float))[0])

    def metric(self, x: float, y: float) -> np.ndarray:
        h = self.evaluate(x, y)
        return np.eye(2, dtype=float) / (h * h)

    def metric_bulk(self, points: np.ndarray) -> np.ndarray:
        h = self.evaluate_bulk(points)
        result = np.zeros((len(h), 2, 2), dtype=float)
        result[:, 0, 0] = 1.0 / (h * h)
        result[:, 1, 1] = 1.0 / (h * h)
        return result

    def evaluate_on_grid(self, nx: int = 200, ny: int = 400):
        xs = np.linspace(0.0, float(self._geo.W), nx)
        ys = np.linspace(0.0, float(self._geo.H), ny)
        X, Y = np.meshgrid(xs, ys)
        values = self.evaluate_bulk(np.column_stack((X.ravel(), Y.ravel())))
        return X, Y, values.reshape(ny, nx)

    def summary(self) -> str:
        return (
            "LearnedResidualSizeField\n"
            f"  checkpoint = {self.checkpoint}\n"
            f"  blend      = {self.blend}\n"
            f"  complexity = {self.realized_complexity_ratio:.6f} "
            f"(target {self.complexity_ratio:.6f})\n"
            f"  feasible C = [{self.minimum_feasible_complexity_ratio:.6f}, "
            f"{self.maximum_feasible_complexity_ratio:.6f}]\n"
            f"  size scale = {self.complexity_scale:.6f}\n"
            f"  h range    = [{self.h_min}, {self.h_max}]\n"
            f"  grid       = {self.grid_nx} x {self.grid_ny}\n"
            f"  device     = {self._device}"
        )
