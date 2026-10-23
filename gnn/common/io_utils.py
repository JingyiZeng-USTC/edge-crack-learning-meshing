from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .geometry import EdgeCrack


def save_mesh(path: Path, nodes: np.ndarray, triangles: np.ndarray, gt_edges: np.ndarray, crack: EdgeCrack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, nodes=nodes, triangles=triangles, gt_edges=gt_edges, crack=json.dumps(crack.to_dict()))


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, EdgeCrack]:
    data = np.load(path)
    return data["nodes"], data["triangles"], data["gt_edges"], EdgeCrack(**json.loads(str(data["crack"])))


def split_files(folder: Path, seed: int = 7) -> dict[str, list[Path]]:
    files = sorted(folder.glob("mesh_*.npz"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    n = len(files)
    n_train, n_val = max(1, int(0.7 * n)), max(1, int(0.15 * n))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    return {"train": files[:n_train], "val": files[n_train:n_train+n_val], "test": files[n_train+n_val:]}

