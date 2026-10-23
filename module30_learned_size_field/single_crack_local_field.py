"""Runtime field for the straight-only local-invariant single-crack model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .model import DirectNormalizedLogSizeMLP
from .single_crack_local_features import (
    SINGLE_CRACK_LOCAL_FEATURE_NAMES,
    SINGLE_CRACK_LOCAL_MODEL_FAMILY,
    build_single_crack_local_features,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@lru_cache(maxsize=8)
def _load_runtime(checkpoint_path: str, device_name: str):
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != SINGLE_CRACK_LOCAL_MODEL_FAMILY:
        raise ValueError(f"{checkpoint_path} is not a single-crack local v4 checkpoint")
    if list(checkpoint.get("feature_names", ())) != SINGLE_CRACK_LOCAL_FEATURE_NAMES:
        raise ValueError("Local-v4 checkpoint feature schema does not match this code")
    if checkpoint.get("feature_normalization") != "fixed_physical_bounded_no_dataset_zscore":
        raise ValueError("Local-v4 checkpoint requests an unsupported normalization")
    if bool(checkpoint.get("runtime_uses_analytic_size_field", True)):
        raise ValueError("Checkpoint still declares an analytic runtime dependency")
    if checkpoint.get("output_parameterization") != "linear_normalized_log_h_clipped_at_runtime":
        raise ValueError("Local-v4 checkpoint has an unsupported output parameterization")
    model = DirectNormalizedLogSizeMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, checkpoint


class SingleCrackLocalSizeField:
    """Continuous absolute h(x,y) for mild changes of a single edge crack."""

    def __init__(
        self,
        geometry,
        checkpoint: str,
        h_min: float | None = None,
        h_max: float | None = None,
        grid_nx: int = 161,
        grid_ny: int = 321,
        smooth_sigma: float = 1.0,
        complexity_ratio: float = 1.0,
        complexity_tolerance: float = 1.0e-5,
        device: str = "auto",
        inference_batch_size: int = 65536,
    ) -> None:
        if grid_nx < 2 or grid_ny < 2:
            raise ValueError("grid dimensions must be at least 2")
        if complexity_ratio <= 0.0 or complexity_tolerance <= 0.0:
            raise ValueError("complexity settings must be positive")
        self._geo = geometry
        self.checkpoint = str(Path(checkpoint).resolve())
        self._model, self._device, saved = _load_runtime(self.checkpoint, device)
        saved_h_min = float(saved["h_min"])
        saved_h_max = float(saved["h_max"])
        self.h_min = saved_h_min if h_min is None else float(h_min)
        self.h_max = saved_h_max if h_max is None else float(h_max)
        if not 0.0 < self.h_min < self.h_max:
            raise ValueError("h_min and h_max must satisfy 0 < h_min < h_max")
        self.grid_nx = int(grid_nx)
        self.grid_ny = int(grid_ny)
        self.smooth_sigma = float(smooth_sigma)
        self.complexity_ratio = float(complexity_ratio)
        self.complexity_tolerance = float(complexity_tolerance)
        self.mode = "single-crack-local-size-field-v4"
        self._train_min = np.asarray(saved["feature_train_min"], dtype=np.float32)
        self._train_max = np.asarray(saved["feature_train_max"], dtype=np.float32)
        self._build_grid(int(inference_batch_size))

    @staticmethod
    def _complexity(grid: np.ndarray) -> float:
        values = np.asarray(grid, dtype=float)
        return float(np.mean(1.0 / np.maximum(values, 1.0e-30) ** 2))

    def _rescale_to_complexity(self, grid: np.ndarray, target: float) -> np.ndarray:
        def candidate(scale: float) -> np.ndarray:
            return np.clip(np.asarray(grid, dtype=float) * scale, self.h_min, self.h_max)

        lower, upper = 1.0e-3, 1.0e3
        lower_complexity = self._complexity(candidate(lower))
        upper_complexity = self._complexity(candidate(upper))
        if not upper_complexity <= target <= lower_complexity:
            raise ValueError("Requested complexity is infeasible under h_min/h_max")
        result = candidate(1.0)
        scale = 1.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            trial = candidate(middle)
            complexity = self._complexity(trial)
            result = trial
            scale = middle
            if abs(complexity - target) / max(target, 1.0e-30) <= self.complexity_tolerance:
                break
            if complexity > target:
                lower = middle
            else:
                upper = middle
        self.complexity_scale = float(scale)
        return np.asarray(result, dtype=float)

    def _feature_shift(self, features: np.ndarray) -> tuple[float, float]:
        span = np.maximum(self._train_max - self._train_min, 1.0e-3)
        below = np.maximum(self._train_min - features, 0.0) / span
        above = np.maximum(features - self._train_max, 0.0) / span
        excess = np.maximum(below, above)
        return float(excess.max(initial=0.0)), float(np.mean(excess > 0.05))

    def _build_grid(self, batch_size: int) -> None:
        xs = np.linspace(0.0, float(self._geo.W), self.grid_nx)
        ys = np.linspace(0.0, float(self._geo.H), self.grid_ny)
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        features = build_single_crack_local_features(
            points, self._geo, h_min=self.h_min,
        )
        self.max_feature_range_excess, self.fraction_feature_range_excess = (
            self._feature_shift(features)
        )
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                tensor = torch.from_numpy(features[start:start + batch_size]).to(self._device)
                predictions.append(self._model(tensor).cpu().numpy())
        normalized_log_h = np.clip(
            np.concatenate(predictions).astype(np.float64, copy=False),
            0.0,
            1.0,
        )
        log_min = float(np.log(self.h_min))
        log_range = float(np.log(self.h_max) - np.log(self.h_min))
        raw_grid = np.exp(
            log_min + normalized_log_h.reshape(self.grid_ny, self.grid_nx) * log_range,
        )
        raw_grid = np.clip(raw_grid, self.h_min, self.h_max)
        self.raw_network_complexity = self._complexity(raw_grid)
        target = self.complexity_ratio * self.raw_network_complexity
        grid = raw_grid
        if self.smooth_sigma > 0.0:
            from scipy.ndimage import gaussian_filter

            grid = gaussian_filter(grid, sigma=self.smooth_sigma, mode="nearest")
            grid = np.clip(grid, self.h_min, self.h_max)
        self._grid = self._rescale_to_complexity(grid, target)
        self.target_complexity = target
        self.realized_complexity = self._complexity(self._grid)
        self.realized_complexity_ratio = self.realized_complexity / max(
            self.raw_network_complexity, 1.0e-30,
        )

    def evaluate_bulk(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
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
            "SingleCrackLocalSizeField v4\n"
            f"  checkpoint = {self.checkpoint}\n"
            "  training shapes = straight single cracks only\n"
            "  analytic base = none\n"
            "  empirical z-score = none\n"
            f"  feature range excess = {self.max_feature_range_excess:.6g}\n"
            f"  h range = [{self.h_min}, {self.h_max}]\n"
            f"  grid = {self.grid_nx} x {self.grid_ny}\n"
            f"  device = {self._device}"
        )


__all__ = ["SingleCrackLocalSizeField"]
