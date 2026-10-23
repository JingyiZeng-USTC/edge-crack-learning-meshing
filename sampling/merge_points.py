"""Merge and deduplicate sampling results."""

import numpy as np


def merge_all(*point_arrays: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Concatenate point arrays and remove near-duplicates.

    First-occurrence wins (boundary/crack/layers before interior).
    """
    parts = [np.asarray(a, dtype=float) for a in point_arrays
             if a is not None and len(a) > 0]
    if not parts:
        return np.empty((0, 2))

    merged = np.vstack(parts)

    # greedy dedup: O(N^2) in worst case, but N is small (~2000)
    tol2 = tol * tol
    keep = np.ones(len(merged), dtype=bool)
    for i in range(len(merged)):
        if not keep[i]:
            continue
        dx = merged[i + 1:, 0] - merged[i, 0]
        dy = merged[i + 1:, 1] - merged[i, 1]
        d2 = dx * dx + dy * dy
        keep[i + 1:][d2 < tol2] = False

    return merged[keep]
