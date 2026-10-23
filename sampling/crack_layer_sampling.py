"""Crack-layer sampling: structured offset nodes on both sides of the crack.

This is the key quality-improvement module.  2-3 layers of nodes are placed
at fixed fractions of the local size-field value away from the crack line,
creating a structured band that prevents sliver triangles near the crack.

Layer spacing
-------------
  Layer k offset = k * 0.8 * h(x)      (k = 1, 2, 3)
  Along-layer spacing = h(x)

The layers extend from crack_start to crack_tip on both the positive
and negative normal sides.
"""

import math

import numpy as np


def sample_crack_layers(
    geometry,
    size_field,
    n_layers: int = 3,
    layer_spacing_factor: float = 0.8,
) -> np.ndarray:
    """Return (M, 2) array of crack-layer nodes.

    Parameters
    ----------
    geometry : EdgeCrackGeometry
    size_field : ContinuousSizeField
    n_layers : int
        Number of offset layers per side (default 3).
    layer_spacing_factor : float
        Offset multiplier on h(x) per layer (default 0.8).
    """
    h_min = size_field.h_min
    all_points = []

    if getattr(geometry, "is_crack_graph", False):
        segments = np.asarray(geometry.segment_coordinates, dtype=float)
        for start, tip in segments:
            vector = tip - start
            crack_len = float(np.linalg.norm(vector))
            tangent = vector / crack_len
            normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
            for side_sign in (+1.0, -1.0):
                for layer_k in range(1, n_layers + 1):
                    travelled = 0.0
                    while travelled <= crack_len + 1.0e-10:
                        center = start + min(travelled, crack_len) * tangent
                        h_local = size_field.evaluate(float(center[0]), float(center[1]))
                        offset = side_sign * layer_k * layer_spacing_factor * h_local
                        point = center + offset * normal
                        if geometry.inside(float(point[0]), float(point[1])):
                            all_points.append(tuple(point))
                        travelled += max(h_local, h_min * 0.3)
        if not all_points:
            return np.empty((0, 2))
        return np.asarray(all_points, dtype=float)

    sx, sy = geometry.crack_start
    tx, ty = geometry.crack_tip
    nx, ny = geometry.crack_normal
    ux, uy = geometry.crack_tangent
    crack_len = geometry.crack_length

    for side_sign in [+1.0, -1.0]:     # both sides
        for layer_k in range(1, n_layers + 1):
            travelled = 0.0
            while True:
                # position on the crack line
                cx = sx + travelled * ux
                cy = sy + travelled * uy

                # check we haven't overshot the tip
                t_param = ((cx - sx) * ux + (cy - sy) * uy)
                if t_param > crack_len + 1e-10:
                    break

                h_local = size_field.evaluate(cx, cy)

                # offset this layer
                offset = layer_k * layer_spacing_factor * h_local

                lx = cx + side_sign * offset * nx
                ly = cy + side_sign * offset * ny

                if geometry.inside(lx, ly):
                    all_points.append((lx, ly))

                # step along the crack
                step = h_local
                step = max(step, h_min * 0.3)
                travelled += step

            # place a point at the tip-end of each layer
            h_tip = size_field.evaluate(tx, ty)
            offset_tip = layer_k * layer_spacing_factor * h_tip
            lx_tip = tx + side_sign * offset_tip * nx
            ly_tip = ty + side_sign * offset_tip * ny
            if geometry.inside(lx_tip, ly_tip):
                all_points.append((lx_tip, ly_tip))

            # place a point at the entry-end of each layer
            h_entry = size_field.evaluate(sx, sy)
            offset_entry = layer_k * layer_spacing_factor * h_entry
            lx_entry = sx + side_sign * offset_entry * nx
            ly_entry = sy + side_sign * offset_entry * ny
            if geometry.inside(lx_entry, ly_entry):
                all_points.append((lx_entry, ly_entry))

    if not all_points:
        return np.empty((0, 2))
    return np.array(all_points, dtype=float)
