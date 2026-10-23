"""
Module 5: Ground Truth Mesh Generation
=======================================
Produces a high-quality, FEM-ready triangular mesh from the adaptive point
cloud, then labels every candidate edge in the topology search space.

Pipeline
--------
  Points  +  Constraints
        |
        v
  Constrained Delaunay (by subdomain splitting)
        |
        v
  Quality Optimization  (Delaunay edge flips + Laplacian smoothing)
        |
        v
  FEM Verification       (min angle, no crack crossing)
        |
        v
  Edge Labels            (1 = in final mesh, 0 = not)
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

_EPS = 1e-10


# ======================================================================
# Geometry helpers
# ======================================================================
def _orientation(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (min(i, j), max(i, j))


def _min_angle_of_triangle(pts, i0, i1, i2):
    a, b, c = pts[i0], pts[i1], pts[i2]
    v1, v2, v3 = b - a, c - a, c - b
    n1, n2, n3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    ang0 = math.acos(max(-1.0, min(1.0, np.dot(v1, v2) / max(n1 * n2, _EPS))))
    ang1 = math.acos(max(-1.0, min(1.0, np.dot(-v1, v3) / max(n1 * n3, _EPS))))
    ang2 = math.pi - ang0 - ang1
    return min(ang0, ang1, ang2)


def _max_angle_of_triangle(pts, i0, i1, i2):
    a, b, c = pts[i0], pts[i1], pts[i2]
    v1, v2, v3 = b - a, c - a, c - b
    n1, n2, n3 = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    ang0 = math.acos(max(-1.0, min(1.0, np.dot(v1, v2) / max(n1 * n2, _EPS))))
    ang1 = math.acos(max(-1.0, min(1.0, np.dot(-v1, v3) / max(n1 * n3, _EPS))))
    ang2 = math.pi - ang0 - ang1
    return max(ang0, ang1, ang2)


# ======================================================================
# Delaunay triangulation  (Bowyer-Watson)
# ======================================================================
class _Delaunay:
    def __init__(self, pts):
        self._pts = np.asarray(pts, dtype=float)
        self._N = len(self._pts)
        xmin, xmax = self._pts[:, 0].min(), self._pts[:, 0].max()
        ymin, ymax = self._pts[:, 1].min(), self._pts[:, 1].max()
        m = max(max(xmax - xmin, ymax - ymin, 1.0), 1.0) * 10.0
        mx, my = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        self._all = np.vstack([self._pts,
                               [(mx - m, my - m), (mx + m, my - m), (mx, my + m)]])
        self._tris: List[Tuple[int, int, int]] = []
        self._add_tri(self._N, self._N + 1, self._N + 2)
        for i in range(self._N):
            self._insert(i)

    def _cc(self, i0, i1, i2):
        a, b, c = self._all[i0], self._all[i1], self._all[i2]
        d = 2.0 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if abs(d) < 1e-20:
            return (0.0, 0.0, 1e20)
        a2, b2, c2 = a[0]**2+a[1]**2, b[0]**2+b[1]**2, c[0]**2+c[1]**2
        cx = (a2*(b[1]-c[1]) + b2*(c[1]-a[1]) + c2*(a[1]-b[1])) / d
        cy = (a2*(c[0]-b[0]) + b2*(a[0]-c[0]) + c2*(b[0]-a[0])) / d
        r2 = (cx - a[0])**2 + (cy - a[1])**2
        return (cx, cy, r2)

    def _add_tri(self, i0, i1, i2):
        self._tris.append((int(i0), int(i1), int(i2)))

    def _insert(self, idx):
        px, py = self._all[idx]
        bad = []
        for ti, t in enumerate(self._tris):
            cx, cy, r2 = self._cc(t[0], t[1], t[2])
            if (px - cx)**2 + (py - cy)**2 < r2 - 1e-12:
                bad.append(ti)
        if not bad:
            return
        ec: Dict[Tuple[int, int], int] = {}
        for ti in bad:
            i0, i1, i2 = self._tris[ti]
            for e in [(i0, i1), (i1, i2), (i2, i0)]:
                ek = (min(e), max(e))
                ec[ek] = ec.get(ek, 0) + 1
        cavity = [e for e, c in ec.items() if c == 1]
        for ti in sorted(bad, reverse=True):
            self._tris.pop(ti)
        for e0, e1 in cavity:
            self._add_tri(idx, e0, e1)

    def triangles(self):
        out = []
        for t in self._tris:
            if t[0] < self._N and t[1] < self._N and t[2] < self._N:
                out.append(list(t))
        return np.array(out, dtype=np.int32)


def _build_neighbors(tris, N_pts):
    """Return (K, 3) neighbour array.  -1 = no neighbour."""
    K = len(tris)
    em: Dict[Tuple[int, int], List[int]] = {}
    for ti, (i0, i1, i2) in enumerate(tris):
        for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
            em.setdefault(_edge_key(a, b), []).append(ti)
    ngh = np.full((K, 3), -1, dtype=np.int32)
    for ti, (i0, i1, i2) in enumerate(tris):
        for loc, (a, b) in enumerate([(i0, i1), (i1, i2), (i2, i0)]):
            for other in em.get(_edge_key(a, b), []):
                if other != ti:
                    ngh[ti, loc] = other
                    break
    return ngh


def _tri_edges_set(tris):
    s: Set[Tuple[int, int]] = set()
    for t in tris:
        i0, i1, i2 = int(t[0]), int(t[1]), int(t[2])
        s.add(_edge_key(i0, i1))
        s.add(_edge_key(i1, i2))
        s.add(_edge_key(i2, i0))
    return s


# ======================================================================
# Ground Truth Mesh Generator
# ======================================================================
class GroundTruthMeshGenerator:
    """Build a FEM-ready triangular mesh and label candidate edges.

    Strategy: split points by signed distance to crack, triangulate each
    subdomain separately, then merge.  This naturally prevents crack-crossing
    triangles.
    """

    def __init__(self, geometry, points, search_space_edges):
        self._geo = geometry
        self._pts = np.asarray(points, dtype=float).copy()
        self._N = len(self._pts)
        self._ss_edges = np.asarray(search_space_edges, dtype=np.int32)

        self._W, self._H = geometry.W, geometry.H
        self._sx, self._sy = geometry.crack_start
        self._tx, self._ty = geometry.crack_tip
        self._nx, self._ny = geometry.crack_normal

        # classify every point by signed distance to crack
        self._signed = np.array([
            (x - self._sx) * self._nx + (y - self._sy) * self._ny
            for x, y in self._pts
        ])

        # fixed points: boundary + crack
        x, y = self._pts[:, 0], self._pts[:, 1]
        on_bnd = (np.abs(x) < _EPS) | (np.abs(x - self._W) < _EPS) | \
                 (np.abs(y) < _EPS) | (np.abs(y - self._H) < _EPS)
        on_crack = np.abs(self._signed) < _EPS
        self._fixed = on_bnd | on_crack

    # ==================================================================
    def generate(self, quality_iters=5):
        # Step 1: subdomain CDT
        idx_pos = np.where(self._signed >= -_EPS)[0]
        idx_neg = np.where(self._signed <= _EPS)[0]
        tp = self._triangulate_subdomain(idx_pos, side_sign=1)
        tn = self._triangulate_subdomain(idx_neg, side_sign=-1)
        tris = np.vstack([tp, tn]) if len(tp)>0 and len(tn)>0 else (tp if len(tp)>0 else tn)
        tri_set: Set[Tuple[int,int,int]] = set()
        uniq = []
        for t in tris:
            key = tuple(sorted(int(x) for x in t))
            if key not in tri_set: tri_set.add(key); uniq.append(t)
        tris = np.array(uniq, dtype=np.int32)

        # Step 2: edge flip for quality (no smooth, no rebuild)
        ngh = _build_neighbors(tris, self._N)
        locked = self._build_locked_set()
        for _ in range(quality_iters):
            tris, ngh, changed = self._optimize_flips(tris, ngh, locked)
            if not changed: break

        # Step 3: verify + label
        quality = self._verify(tris)
        mesh_edges = _tri_edges_set(tris)
        labels = np.array([1 if _edge_key(int(e[0]),int(e[1])) in mesh_edges else 0
                          for e in self._ss_edges], dtype=np.int8)
        return {"triangles":tris,"points":self._pts.copy(),"labels":labels,"quality":quality,
                "stats":self._stats(tris,labels)}

    # ==================================================================
    # Subdomain triangulation
    # ==================================================================
    def _triangulate_subdomain(self, idx, side_sign):
        """Triangulate one subdomain.  idx are global indices."""
        if len(idx) < 3:
            return np.empty((0, 3), dtype=np.int32)

        sub_pts = self._pts[idx]
        del2d = _Delaunay(sub_pts)
        tris_local = del2d.triangles()
        if len(tris_local) == 0:
            return np.empty((0, 3), dtype=np.int32)

        # map local -> global indices
        tris_global = idx[tris_local]

        # filter: triangles must have all vertices on the correct side
        # (or on the crack, which belongs to both sides)
        s_vals = self._signed[tris_global]
        if side_sign > 0:
            keep = np.all(s_vals >= -_EPS, axis=1)
        else:
            keep = np.all(s_vals <= _EPS, axis=1)

        return tris_global[keep]

    # ==================================================================
    # Constrained edge detection
    # ==================================================================
    def _build_locked_set(self):
        """Return set of edge keys that must NOT be flipped."""
        locked: Set[Tuple[int, int]] = set()

        # boundary edges: connect consecutive boundary points on each side
        x, y = self._pts[:, 0], self._pts[:, 1]
        W, H = self._W, self._H

        for cond, axis in [
            (np.abs(y) < _EPS, 0),           # bottom
            (np.abs(x - W) < _EPS, 1),        # right
            (np.abs(y - H) < _EPS, 0),        # top
            (np.abs(x) < _EPS, 1),            # left
        ]:
            b_idx = np.where(cond)[0]
            if len(b_idx) > 1:
                order = np.argsort(self._pts[b_idx, axis])
                sorted_idx = b_idx[order]
                for k in range(len(sorted_idx) - 1):
                    locked.add(_edge_key(sorted_idx[k], sorted_idx[k + 1]))

        # crack edges: connect consecutive crack points along the crack
        c_idx = np.where(np.abs(self._signed) < _EPS)[0]
        if len(c_idx) > 1:
            dx = self._tx - self._sx
            dy = self._ty - self._sy
            L2 = dx * dx + dy * dy
            if L2 > 1e-20:
                t_vals = ((self._pts[c_idx, 0] - self._sx) * dx +
                          (self._pts[c_idx, 1] - self._sy) * dy) / L2
                order = np.argsort(t_vals)
                sorted_idx = c_idx[order]
                for k in range(len(sorted_idx) - 1):
                    locked.add(_edge_key(sorted_idx[k], sorted_idx[k + 1]))

        return locked

    # ==================================================================
    # Edge-flip quality optimization
    # ==================================================================
    def _optimize_flips(self, tris, ngh, locked):
        changed = False
        for ti in range(len(tris)):
            for loc in range(3):
                other = ngh[ti, loc]
                if other < 0 or other <= ti:
                    continue  # process each pair once

                i0, i1, i2 = tris[ti]
                a, b = [(i0, i1), (i1, i2), (i2, i0)][loc]
                if _edge_key(a, b) in locked:
                    continue

                opp_a = [i0, i1, i2][loc]
                o0, o1, o2 = tris[other]
                opp_b = None
                for lo2, (x, y) in enumerate([(o0, o1), (o1, o2), (o2, o0)]):
                    if _edge_key(x, y) == _edge_key(a, b):
                        opp_b = [o0, o1, o2][lo2]
                        break
                if opp_b is None:
                    continue

                cur = min(
                    _min_angle_of_triangle(self._pts, a, b, opp_a),
                    _min_angle_of_triangle(self._pts, a, b, opp_b),
                )
                cand_min = min(
                    _min_angle_of_triangle(self._pts, opp_a, opp_b, a),
                    _min_angle_of_triangle(self._pts, opp_a, opp_b, b),
                )
                # also consider max angle: Delaunay property = minimax angle
                cur_max = max(
                    _max_angle_of_triangle(self._pts, a, b, opp_a),
                    _max_angle_of_triangle(self._pts, a, b, opp_b),
                )
                cand_max = max(
                    _max_angle_of_triangle(self._pts, opp_a, opp_b, a),
                    _max_angle_of_triangle(self._pts, opp_a, opp_b, b),
                )
                # flip if min angle improves OR (min angle same but max angle improves)
                if not (cand_min > cur + 1e-8 or
                       (abs(cand_min - cur) < 1e-8 and cand_max < cur_max - 1e-8)):
                    continue

                # convexity check
                pa = self._pts[opp_a]; pb = self._pts[opp_b]
                pu = self._pts[a]; pv = self._pts[b]
                if _orientation(pa[0], pa[1], pu[0], pu[1], pb[0], pb[1]) * \
                   _orientation(pa[0], pa[1], pb[0], pb[1], pv[0], pv[1]) <= -_EPS:
                    continue

                tris[ti] = [opp_a, opp_b, a]
                tris[other] = [opp_a, opp_b, b]
                changed = True

        if changed:
            ngh = _build_neighbors(tris, self._N)
        return tris, ngh, changed

    # ==================================================================
    # Laplacian smoothing
    # ==================================================================
    def _laplacian_smooth_one_pass(self, tris):
        adj_sum = np.zeros((self._N, 2))
        adj_cnt = np.zeros(self._N, dtype=int)
        for t in tris:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                i, j = int(t[a]), int(t[b])
                adj_sum[i] += self._pts[j]; adj_cnt[i] += 1
                adj_sum[j] += self._pts[i]; adj_cnt[j] += 1
        for i in range(self._N):
            if self._fixed[i] or adj_cnt[i] == 0:
                continue
            self._pts[i] = adj_sum[i] / adj_cnt[i]

    def _rebuild_cdt(self):
        """Rebuild triangulation from scratch preserving subdomain split."""
        idx_pos = np.where(self._signed >= -_EPS)[0]
        idx_neg = np.where(self._signed <= _EPS)[0]
        tp = self._triangulate_subdomain(idx_pos, 1)
        tn = self._triangulate_subdomain(idx_neg, -1)
        tris = np.vstack([tp, tn]) if len(tp) > 0 and len(tn) > 0 \
               else (tp if len(tp) > 0 else tn)
        tri_set: Set[Tuple[int, int, int]] = set()
        uniq = []
        for t in tris:
            key = tuple(sorted(int(x) for x in t))
            if key not in tri_set:
                tri_set.add(key)
                uniq.append(t)
        # also recompute signed distances since points moved
        self._signed = np.array([
            (x - self._sx) * self._nx + (y - self._sy) * self._ny
            for x, y in self._pts
        ])
        return np.array(uniq, dtype=np.int32)

    # ==================================================================
    # Verification
    # ==================================================================
    def _verify(self, tris):
        angles = [_min_angle_of_triangle(self._pts, int(t[0]), int(t[1]), int(t[2]))
                  for t in tris]
        angles = np.array(angles) if angles else np.array([0.0])
        max_angles = [_max_angle_of_triangle(self._pts, int(t[0]), int(t[1]), int(t[2]))
                      for t in tris]
        max_angles = np.array(max_angles) if max_angles else np.array([0.0])

        # crack-crossing check
        crossings = 0
        for t in tris:
            s = self._signed[[int(t[0]), int(t[1]), int(t[2])]]
            has_pos = np.any(s > _EPS)
            has_neg = np.any(s < -_EPS)
            if has_pos and has_neg:
                on_crack_edge = False
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    if abs(s[a]) < _EPS and abs(s[b]) < _EPS:
                        on_crack_edge = True
                        break
                if not on_crack_edge:
                    crossings += 1

        return {
            "min_angle_deg": float(np.degrees(angles.min())),
            "max_angle_deg": float(np.degrees(max_angles.max())),
            "mean_angle_deg": float(np.degrees(angles.mean())),
            "n_triangles": len(tris),
            "crack_crossings": crossings,
            "fem_ready": bool(angles.min() > math.radians(20) and crossings == 0),
        }

    def _stats(self, tris, labels):
        n1 = int(labels.sum())
        n0 = len(labels) - n1
        return (
            f"Triangles: {len(tris)}  "
            f"GT edges: {n1}  Non-edges: {n0}  "
            f"Positive: {n1 / max(len(labels), 1):.1%}"
        )

    # ==================================================================
    # Visualization
    # ==================================================================
    def plot_mesh(self, result, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))
        ax.set_aspect("equal")
        ax.triplot(self._pts[:, 0], self._pts[:, 1], result["triangles"],
                   linewidth=0.3, color="gray")
        ax.plot([self._sx, self._tx], [self._sy, self._ty], "r-", linewidth=2)
        ax.plot(self._tx, self._ty, "ro", markersize=6, markerfacecolor="red")
        q = result["quality"]
        ax.set_title(f"Ground Truth  ({len(result['triangles'])} tris, "
                     f"min angle={q['min_angle_deg']:.1f} deg)")
        ax.set_xlim(-0.02 * self._W, self._W * 1.02)
        ax.set_ylim(-0.02 * self._H, self._H * 1.02)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return ax


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from edge_crack_geometry import EdgeCrackGeometry, generate_crack
    from size_field import ContinuousSizeField
    from adaptive_sampling import AdaptivePoissonSampler
    from topology_search_space import TopologySearchSpace
    import matplotlib.pyplot as plt

    start, tip = generate_crack("left", origin=1.0, length=0.30, theta=0.0)
    geo = EdgeCrackGeometry(W=1.0, H=2.0, crack_start=start, crack_tip=tip,
                            crack_length=0.30, theta=0.0, edge_side="left")
    sf = ContinuousSizeField(geo, h_min=0.004, h_max=0.030,
                              L_tip=0.08, L_crack=0.05,
                              L_boundary=0.03, w_boundary=0.25)
    sampler = AdaptivePoissonSampler(geo, sf)
    pts = sampler.sample(seed=42, n_crack_segments=30)

    tss = TopologySearchSpace(geo, sf, pts, K=10, radius_factor=2.0)
    graph = tss.build()
    print(f"M4: {tss.stats(graph)}")

    gt = GroundTruthMeshGenerator(geo, pts, graph["edges"])
    result = gt.generate(quality_iters=8, smooth_iters=2)

    print(f"\nM5: {result['stats']}")
    q = result["quality"]
    print(f"Quality: min_angle={q['min_angle_deg']:.1f} deg  "
          f"max_angle={q['max_angle_deg']:.1f} deg  "
          f"crack_crossings={q['crack_crossings']}  "
          f"fem_ready={q['fem_ready']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
