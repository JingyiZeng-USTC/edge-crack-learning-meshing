"""Tip-ring sampling: a small ring of nodes around the crack tip.

Prevents radial sliver triangles radiating from the tip.
"""

import math

import numpy as np


def sample_tip_ring(
    geometry,
    size_field,
    n_ring: int = 8,
    radius_factor: float = 2.0,
) -> np.ndarray:
    """Return (n_ring, 2) array of ring nodes around the crack tip.

    Parameters
    ----------
    geometry : EdgeCrackGeometry
    size_field : ContinuousSizeField
    n_ring : int
        Number of ring nodes (default 8).
    radius_factor : float
        Ring radius = radius_factor * h_tip (default 2.0).
    """
    points = []
    tips = (
        np.asarray(geometry.crack_tips, dtype=float)
        if getattr(geometry, "is_crack_graph", False)
        else np.asarray([geometry.crack_tip], dtype=float)
    )
    # Branches are deterministic constraint vertices and receive a ring as
    # well; this improves the transition around three or more incident faces.
    if getattr(geometry, "is_crack_graph", False) and len(geometry.branch_points):
        tips = np.vstack((tips, np.asarray(geometry.branch_points, dtype=float)))
    for tx, ty in tips:
        h_tip = size_field.evaluate(float(tx), float(ty))
        radius = radius_factor * h_tip
        if radius < 1e-15:
            continue
        for k in range(n_ring):
            angle = 2.0 * math.pi * k / n_ring
            px = tx + radius * math.cos(angle)
            py = ty + radius * math.sin(angle)
            if geometry.inside(px, py):
                points.append((px, py))

    if not points:
        return np.empty((0, 2))
    return np.array(points, dtype=float)
