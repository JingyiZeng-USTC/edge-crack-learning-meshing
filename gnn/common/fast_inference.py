"""Opt-in allocation-free scalar kernels; no thresholds or mesh rules changed."""
from __future__ import annotations
import math


def scalar_orientation(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def scalar_intersection(a, b, c, d, eps=1e-12):
    o1, o2 = scalar_orientation(a, b, c), scalar_orientation(a, b, d)
    o3, o4 = scalar_orientation(c, d, a), scalar_orientation(c, d, b)
    scale = max(abs(float(a[0])), abs(float(a[1])), abs(float(b[0])), abs(float(b[1])),
                abs(float(c[0])), abs(float(c[1])), abs(float(d[0])), abs(float(d[1])), 1.0)
    tolerance = float(eps) * scale * scale
    return (((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance))
            and ((o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)))


def scalar_overlap(a, b, c, d, eps=1e-10):
    if abs(scalar_orientation(a, b, c)) > eps or abs(scalar_orientation(a, b, d)) > eps:
        return False
    axis = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
    left = max(min(a[axis], b[axis]), min(c[axis], d[axis]))
    right = min(max(a[axis], b[axis]), max(c[axis], d[axis]))
    return bool(right - left > eps)


def scalar_size(self, x, y):
    # Same four-term bilinear interpolation, clipping and float64 grid as evaluate_bulk.
    gx = min(max(float(x) / float(self._geo.W), 0.0), 1.0) * (self.grid_nx - 1)
    gy = min(max(float(y) / float(self._geo.H), 0.0), 1.0) * (self.grid_ny - 1)
    x0, y0 = math.floor(gx), math.floor(gy)
    x1, y1 = min(x0 + 1, self.grid_nx - 1), min(y0 + 1, self.grid_ny - 1)
    wx, wy = gx - x0, gy - y0
    grid = self._grid
    value = ((1.0 - wx) * (1.0 - wy) * grid[y0, x0]
             + wx * (1.0 - wy) * grid[y0, x1]
             + (1.0 - wx) * wy * grid[y1, x0]
             + wx * wy * grid[y1, x1])
    return float(min(max(value, self.h_min), self.h_max))


def install_fast_kernels():
    from . import geometry, mesh_validation
    from module30_learned_size_field.single_crack_local_field import SingleCrackLocalSizeField
    geometry.orientation = scalar_orientation
    geometry.proper_segment_intersection = scalar_intersection
    geometry.collinear_interior_overlap = scalar_overlap
    mesh_validation.proper_segment_intersection = scalar_intersection
    mesh_validation.collinear_interior_overlap = scalar_overlap
    SingleCrackLocalSizeField.evaluate = scalar_size
