"""Generate adaptive nodes with the learned size field, then connect them with GraphSAGE.

This is the teacher-free inference bridge between the project's straight-trained
single-crack size-field model and its Line-Graph connectivity model.  The learned size
field controls node placement.  GraphSAGE-ranked candidate edges alone must form
all triangular faces: no CDT, Delaunay, Triangle, ear clipping, or inserted node
is allowed during recovery.  The existing ten-dimensional edge feature schema is
intentionally preserved so the trained GraphSAGE checkpoint remains valid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import torch


GNN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = GNN_ROOT.parent
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_sampling import AdaptivePoissonSampler
from crack_graph_geometry import CrackGraphGeometry
from edge_crack_geometry import generate_crack
from module30_learned_size_field.single_crack_local_field import (
    SingleCrackLocalSizeField,
)

from common.mesh_validation import (
    recover_strict_gnn_project_mesh,
    topology_metrics,
    triangle_quality,
)
from common.meshing import edge_features, line_graph_edge_index
from common.model import EdgeGraphSAGE
from common.project_data import project_candidate_edges
from common.quality_smoothing import constrained_quality_smoothing


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _entry_edge(geometry: CrackGraphGeometry) -> tuple[str, float]:
    if len(geometry.crack_entries) != 1 or len(geometry.crack_tips) != 1:
        raise ValueError("Inference requires exactly one boundary entry and one interior tip")
    x, y = map(float, geometry.crack_entries[0])
    tolerance = 1.0e-8 * max(geometry.W, geometry.H, 1.0)
    if abs(x) <= tolerance:
        return "left", y
    if abs(x - geometry.W) <= tolerance:
        return "right", y
    if abs(y) <= tolerance:
        return "bottom", x
    if abs(y - geometry.H) <= tolerance:
        return "top", x
    raise ValueError("The crack entry does not lie on the exterior boundary")


def _geometry(args: argparse.Namespace) -> tuple[CrackGraphGeometry, float, str]:
    if args.geometry_json is not None:
        geometry = CrackGraphGeometry.from_json(args.geometry_json)
        if len(geometry.crack_polylines) != 1 or geometry.has_branches:
            raise ValueError(
                "This generalization test supports one non-branching crack polyline only"
            )
        edge, origin = _entry_edge(geometry)
        return geometry, origin, edge
    origin = args.origin
    if origin is None:
        origin = 0.5 * (args.height if args.edge in ("left", "right") else args.width)
    start, tip = generate_crack(
        args.edge,
        float(origin),
        args.length,
        math.radians(args.angle_deg),
        args.width,
        args.height,
    )
    start_array = np.asarray(start, dtype=np.float64)
    tip_array = np.asarray(tip, dtype=np.float64)
    tolerance = 1.0e-10
    if not (
        tolerance < tip_array[0] < args.width - tolerance
        and tolerance < tip_array[1] < args.height - tolerance
    ):
        raise ValueError(
            "The crack tip must lie strictly inside the plate; adjust origin, length, or angle"
        )
    return (
        CrackGraphGeometry(args.width, args.height, [np.vstack((start_array, tip_array))]),
        float(origin),
        args.edge,
    )


def _validate_top_clearance(
    geometry: CrackGraphGeometry,
    minimum_clearance: float,
) -> float:
    """Require the complete crack polyline to stay below the loaded top edge."""
    if not (0.0 <= minimum_clearance < geometry.H):
        raise ValueError("min-top-clearance must lie in [0, plate height)")
    highest_y = max(
        float(np.max(np.asarray(polyline, dtype=np.float64)[:, 1]))
        for polyline in geometry.crack_polylines
    )
    actual_clearance = float(geometry.H - highest_y)
    tolerance = 1.0e-12 * max(float(geometry.W), float(geometry.H), 1.0)
    if actual_clearance < minimum_clearance - tolerance:
        raise ValueError(
            "Crack is too close to the loaded top edge: "
            f"clearance={actual_clearance:.6g} m, "
            f"required>={minimum_clearance:.6g} m. "
            "Adjust the crack origin/angle/length or --min-top-clearance."
        )
    return actual_clearance


def _load_connectivity_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[EdgeGraphSAGE, dict, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EdgeGraphSAGE(
        in_channels=int(checkpoint["in_channels"]),
        hidden_channels=int(checkpoint["hidden"]),
        layers=int(checkpoint["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, float(checkpoint["threshold"])


def _draw(
    path: Path,
    geometry: CrackGraphGeometry,
    size_field: SingleCrackLocalSizeField,
    input_points: np.ndarray,
    input_h: np.ndarray,
    recovered_points: np.ndarray,
    recovered_triangles: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 7.2), constrained_layout=True)
    X, Y, H = size_field.evaluate_on_grid(nx=181, ny=361)
    contour = axes[0].contourf(X, Y, H, levels=32, cmap="viridis")
    figure.colorbar(contour, ax=axes[0], fraction=0.046, pad=0.04, label="learned h(x,y) [m]")
    scatter = axes[1].scatter(
        input_points[:, 0], input_points[:, 1], c=input_h, s=2.0,
        cmap="viridis", vmin=size_field.h_min, vmax=size_field.h_max,
    )
    figure.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04, label="h at sampled node [m]")
    if len(recovered_triangles):
        axes[2].triplot(
            Triangulation(
                recovered_points[:, 0], recovered_points[:, 1], recovered_triangles,
            ),
            color="#2f6690", linewidth=0.30,
        )
    axes[2].scatter(recovered_points[:, 0], recovered_points[:, 1], s=0.7, color="#222222")
    titles = ("Learned size field", "Adaptive point distribution", "GraphSAGE recovered mesh")
    crack = np.asarray(geometry.crack_polylines[0])
    for axis, title in zip(axes, titles):
        axis.plot(crack[:, 0], crack[:, 1], color="#d1495b", linewidth=2.0, zorder=5)
        axis.set_xlim(-0.02 * geometry.W, 1.02 * geometry.W)
        axis.set_ylim(-0.01 * geometry.H, 1.01 * geometry.H)
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learned size field -> adaptive nodes -> Line Graph -> GraphSAGE -> "
            "confidence-ordered planar triangle recovery"
        ),
    )
    parser.add_argument("--edge", choices=("left", "right", "top", "bottom"), default="left")
    parser.add_argument(
        "--origin", type=float, default=None,
        help="y coordinate on left/right or x coordinate on top/bottom; default: edge midpoint",
    )
    parser.add_argument("--length", type=float, default=0.30)
    parser.add_argument("--angle-deg", type=float, default=0.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument(
        "--min-top-clearance", type=float, default=0.20,
        help=(
            "Minimum distance in metres from every point of the crack polyline "
            "to the loaded top edge; use 0 only to disable the restriction"
        ),
    )
    parser.add_argument(
        "--geometry-json",
        type=Path,
        default=None,
        help=(
            "Optional CrackGraphGeometry JSON containing one non-branching polyline. "
            "When supplied, edge/origin/length/angle/width/height arguments are ignored."
        ),
    )
    parser.add_argument(
        "--size-model", type=Path,
        default=PROJECT_ROOT / "module30_checkpoints" / "single_straight_crack_local_v4.pt",
    )
    parser.add_argument(
        "--gnn-model", type=Path,
        default=GNN_ROOT / "artifacts" / "single_straight_200_strict_model.pt",
    )
    parser.add_argument("--output", type=Path, default=GNN_ROOT / "artifacts" / "learned_size_gnn")
    parser.add_argument("--case-name", default="learned_size_single_crack")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=64031)
    parser.add_argument("--grid-nx", type=int, default=161)
    parser.add_argument("--grid-ny", type=int, default=321)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--complexity-ratio", type=float, default=1.0)
    parser.add_argument("--n-crack-segments", type=int, default=30)
    parser.add_argument(
        "--crack-spacing-factor", type=float, default=0.90,
        help=(
            "Target crack-constraint spacing as a multiple of learned h; "
            "n-crack-segments remains a lower bound"
        ),
    )
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-tip-ring", type=int, default=8)
    parser.add_argument("--sampling-k", type=int, default=30)
    parser.add_argument("--boundary-clearance-factor", type=float, default=0.5)
    parser.add_argument("--auxiliary-separation-factor", type=float, default=0.5)
    parser.add_argument("--quality-target-deg", type=float, default=20.0)
    parser.add_argument("--quality-weight", type=float, default=12.0)
    parser.add_argument("--quality-passes", type=int, default=12)
    parser.add_argument(
        "--node-smoothing", action=argparse.BooleanOptionalAction, default=True,
        help="Move only unconstrained interior nodes when local angles improve",
    )
    parser.add_argument("--smoothing-passes", type=int, default=10)
    parser.add_argument("--smoothing-move-factor", type=float, default=0.30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--figure", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width <= 0.0 or args.height <= 0.0 or args.length <= 0.0:
        raise ValueError("width, height, and crack length must be positive")
    if args.candidate_k < 2:
        raise ValueError("candidate-k must be at least 2")
    if args.crack_spacing_factor <= 0.0:
        raise ValueError("crack-spacing-factor must be positive")
    if args.smoothing_passes < 1 or args.smoothing_move_factor <= 0.0:
        raise ValueError("smoothing passes and move factor must be positive")
    if not (0.0 <= args.boundary_clearance_factor <= 2.0):
        raise ValueError("boundary-clearance-factor must lie in [0, 2]")
    if not (0.0 <= args.auxiliary_separation_factor <= 2.0):
        raise ValueError("auxiliary-separation-factor must lie in [0, 2]")
    if not args.size_model.is_file():
        raise FileNotFoundError(args.size_model)
    if not args.gnn_model.is_file():
        raise FileNotFoundError(args.gnn_model)

    if args.geometry_json is not None and not args.geometry_json.is_file():
        raise FileNotFoundError(args.geometry_json)
    geometry, origin, edge_side = _geometry(args)
    actual_top_clearance = _validate_top_clearance(
        geometry, float(args.min_top_clearance),
    )
    device = _device(args.device)
    size_field = SingleCrackLocalSizeField(
        geometry,
        checkpoint=str(args.size_model),
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        smooth_sigma=args.smooth_sigma,
        complexity_ratio=args.complexity_ratio,
        device=str(device),
    )
    sampler = AdaptivePoissonSampler(geometry, size_field)
    input_points = np.asarray(
        sampler.sample(
            seed=args.seed,
            n_crack_segments=args.n_crack_segments,
            crack_spacing_factor=args.crack_spacing_factor,
            n_layers=args.n_layers,
            n_tip_ring=args.n_tip_ring,
            k_candidates=args.sampling_k,
            boundary_clearance_factor=args.boundary_clearance_factor,
            auxiliary_separation_factor=args.auxiliary_separation_factor,
        ),
        dtype=np.float32,
    )
    if len(input_points) < 3:
        raise RuntimeError("Adaptive sampler produced fewer than three points")
    input_h = size_field.evaluate_bulk(input_points).astype(np.float32)
    crack_segments = np.asarray(geometry.segment_coordinates, dtype=np.float32)
    candidates = project_candidate_edges(
        input_points, geometry.W, geometry.H, crack_segments, k=args.candidate_k,
    )
    features = edge_features(input_points, candidates, geometry.W, geometry.H)
    line_edges = line_graph_edge_index(candidates)

    model, checkpoint, saved_threshold = _load_connectivity_model(args.gnn_model, device)
    if features.shape[1] != int(checkpoint["in_channels"]):
        raise RuntimeError(
            f"GraphSAGE expects {checkpoint['in_channels']} edge features, got {features.shape[1]}"
        )
    threshold = saved_threshold if args.threshold is None else float(args.threshold)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(features).to(device),
            torch.from_numpy(line_edges).to(device),
        )
        probability = logits.softmax(dim=1)[:, 1].cpu().numpy()
    positive = probability >= threshold
    predicted_edges = candidates[positive]
    recovered_points, recovered_triangles, selected_edges, recovery_stats = (
        recover_strict_gnn_project_mesh(
            input_points,
            candidates,
            probability,
            geometry.W,
            geometry.H,
            crack_segments,
            threshold=threshold,
            quality_target_deg=args.quality_target_deg,
            quality_weight=args.quality_weight,
            quality_passes=args.quality_passes,
        )
    )
    if args.node_smoothing:
        recovered_points, smoothing_stats = constrained_quality_smoothing(
            recovered_points,
            recovered_triangles,
            size_field.evaluate_bulk(recovered_points),
            geometry.W,
            geometry.H,
            crack_segments,
            target_angle_deg=args.quality_target_deg,
            max_passes=args.smoothing_passes,
            move_factor=args.smoothing_move_factor,
        )
    else:
        smoothing_stats = {"enabled": False}
    recovered_h = size_field.evaluate_bulk(recovered_points).astype(np.float32)
    quality = triangle_quality(recovered_points, recovered_triangles)
    topology = topology_metrics(
        recovered_points, recovered_triangles, geometry.W, geometry.H, crack_segments,
    )

    output = args.output.resolve()
    candidate_dir = output / "candidates"
    mesh_dir = output / "meshes"
    figure_dir = output / "figures"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    name = args.case_name
    crack_polyline = np.asarray(geometry.crack_polylines[0], dtype=np.float32)
    crack_family = "straight" if len(crack_segments) == 1 else "slight_curve"
    np.savez_compressed(
        candidate_dir / f"{name}.npz",
        nodes=input_points,
        candidates=candidates,
        features=features,
        line_edge_index=line_edges,
        probability=probability,
        predicted_edges=predicted_edges,
        learned_h=input_h,
        width=float(geometry.W),
        height=float(geometry.H),
        crack_segments=crack_segments,
        crack_polyline_points=crack_polyline,
        crack_polyline_offsets=np.asarray([0, len(crack_polyline)], dtype=np.int64),
        crack_family=crack_family,
        edge_side=edge_side,
    )
    np.savez_compressed(
        mesh_dir / f"{name}.npz",
        nodes=recovered_points,
        triangles=recovered_triangles,
        recovery_constraints=selected_edges,
        gnn_selected_edges=selected_edges,
        predicted_edges=predicted_edges,
        probability=probability,
        learned_h=recovered_h,
        crack_segments=crack_segments,
        crack_polyline_points=crack_polyline,
        crack_polyline_offsets=np.asarray([0, len(crack_polyline)], dtype=np.int64),
        width=float(geometry.W),
        height=float(geometry.H),
    )

    report = {
        "format": "learned-size-field-to-structured-line-graph-gnn-v6-polyline-generalization",
        "teacher_used_at_inference": False,
        "cdt_fallback_used": False,
        "geometric_triangulation_fallback_used": False,
        "scope": "one-nonbranching-polyline-single-edge-crack",
        "geometry": {
            "width": float(geometry.W),
            "height": float(geometry.H),
            "edge": edge_side,
            "origin": origin,
            "family": crack_family,
            "length": float(geometry.total_crack_length),
            "angle_deg": (
                float(args.angle_deg) if args.geometry_json is None else None
            ),
            "segments": int(len(crack_segments)),
            "path_length": float(geometry.total_crack_length),
            "chord_length": float(np.linalg.norm(
                np.asarray(geometry.crack_tips[0]) - np.asarray(geometry.crack_entries[0])
            )),
            "geometry_json": (
                str(args.geometry_json.resolve()) if args.geometry_json is not None else None
            ),
            "start": np.asarray(geometry.crack_entries[0]).tolist(),
            "tip": np.asarray(geometry.crack_tips[0]).tolist(),
            "polyline": crack_polyline.tolist(),
            "top_clearance_m": actual_top_clearance,
            "required_top_clearance_m": float(args.min_top_clearance),
        },
        "size_field": {
            "checkpoint": str(args.size_model.resolve()),
            "mode": size_field.mode,
            "h_min": float(size_field.h_min),
            "h_max": float(size_field.h_max),
            "sampled_h_min": float(input_h.min()),
            "sampled_h_median": float(np.median(input_h)),
            "sampled_h_max": float(input_h.max()),
            "grid": [int(args.grid_nx), int(args.grid_ny)],
            "smooth_sigma": float(args.smooth_sigma),
            "complexity_ratio": float(size_field.realized_complexity_ratio),
            "feature_range_excess": float(size_field.max_feature_range_excess),
            "boundary_clearance_factor": float(args.boundary_clearance_factor),
            "auxiliary_separation_factor": float(args.auxiliary_separation_factor),
            "crack_spacing_factor": float(args.crack_spacing_factor),
        },
        "connectivity": {
            "checkpoint": str(args.gnn_model.resolve()),
            "edge_feature_dimension": int(features.shape[1]),
            "candidate_k": int(args.candidate_k),
            "threshold": threshold,
            "input_nodes": int(len(input_points)),
            "candidates": int(len(candidates)),
            "line_graph_edges": int(line_edges.shape[1]),
            "predicted_positive_edges": int(np.count_nonzero(positive)),
            **recovery_stats,
        },
        "recovery": {
            "nodes": int(len(recovered_points)),
            "inserted_nodes": int(len(recovered_points) - len(input_points)),
            "triangles": int(len(recovered_triangles)),
            "quality": quality,
            "topology": topology,
            "node_smoothing": smoothing_stats,
        },
        "files": {
            "candidate_graph": str((candidate_dir / f"{name}.npz").resolve()),
            "recovered_mesh": str((mesh_dir / f"{name}.npz").resolve()),
            "figure": str((figure_dir / f"{name}.png").resolve()) if args.figure else None,
        },
    }
    report_path = output / f"{name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.figure:
        _draw(
            figure_dir / f"{name}.png",
            geometry,
            size_field,
            input_points,
            input_h,
            recovered_points,
            recovered_triangles,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved learned-size/GNN result to {output}")


if __name__ == "__main__":
    main()
