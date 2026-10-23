"""Learned residual size field for arbitrary crack graphs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from crack_graph_size_field import CrackGraphSizeField

from .crack_graph_features import CRACK_GRAPH_FEATURE_NAMES, build_crack_graph_features
from .field import LearnedResidualSizeField, _resolve_device
from .model import PointSizeFieldMLP


@lru_cache(maxsize=8)
def _load_crack_graph_runtime(checkpoint_path: str, device_name: str):
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "learned-residual-crack-graph-size-field-v1":
        raise ValueError(f"{checkpoint_path} is not an arbitrary crack-graph checkpoint")
    if list(checkpoint["feature_names"]) != CRACK_GRAPH_FEATURE_NAMES:
        raise ValueError("Crack-graph checkpoint feature schema does not match this code")
    model = PointSizeFieldMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std, device


class LearnedCrackGraphSizeField(LearnedResidualSizeField):
    """Cached learned field using fixed-width local crack-graph features."""

    def __init__(
        self,
        geometry,
        checkpoint: str,
        h_min: float = 0.004,
        h_max: float = 0.030,
        L_tip: float = 0.08,
        L_crack: float = 0.05,
        L_entry: float | None = 0.08,
        L_branch: float | None = 0.08,
        curvature_strength: float = 0.35,
        blend: float = 1.0,
        grid_nx: int = 161,
        grid_ny: int = 321,
        smooth_sigma: float = 1.0,
        complexity_ratio: float = 1.0,
        complexity_tolerance: float = 1.0e-5,
        device: str = "auto",
        inference_batch_size: int = 65536,
    ) -> None:
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if grid_nx < 2 or grid_ny < 2:
            raise ValueError("grid dimensions must be at least 2")
        if complexity_ratio <= 0.0 or complexity_tolerance <= 0.0:
            raise ValueError("complexity settings must be positive")
        self._geo = geometry
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.L_tip = float(L_tip)
        self.L_crack = float(L_crack)
        self.L_entry = None if L_entry is None else float(L_entry)
        self.L_branch = None if L_branch is None else float(L_branch)
        self.curvature_strength = float(curvature_strength)
        self.mode = "learned-crack-graph-residual"
        self.checkpoint = str(Path(checkpoint).resolve())
        self.blend = float(blend)
        self.grid_nx = int(grid_nx)
        self.grid_ny = int(grid_ny)
        self.smooth_sigma = float(smooth_sigma)
        self.complexity_ratio = float(complexity_ratio)
        self.complexity_tolerance = float(complexity_tolerance)
        self.base = CrackGraphSizeField(
            geometry,
            h_min=h_min,
            h_max=h_max,
            L_tip=L_tip,
            L_crack=L_crack,
            L_entry=L_entry,
            L_branch=L_branch,
            curvature_strength=curvature_strength,
        )
        self._model, self._mean, self._std, self._device = _load_crack_graph_runtime(
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
            features = build_crack_graph_features(
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
            learned_h = np.clip(
                analytic_h * np.exp(self.blend * np.concatenate(corrections)),
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

    def summary(self) -> str:
        return (
            "LearnedCrackGraphSizeField\n"
            f"  checkpoint = {self.checkpoint}\n"
            f"  polylines  = {len(self._geo.crack_polylines)}\n"
            f"  tips       = {len(self._geo.crack_tips)}\n"
            f"  branches   = {len(self._geo.branch_points)}\n"
            f"  blend      = {self.blend}\n"
            f"  complexity = {self.realized_complexity_ratio:.6f} "
            f"(target {self.complexity_ratio:.6f})\n"
            f"  h range    = [{self.h_min}, {self.h_max}]\n"
            f"  grid       = {self.grid_nx} x {self.grid_ny}\n"
            f"  device     = {self._device}"
        )


__all__ = ["LearnedCrackGraphSizeField"]
