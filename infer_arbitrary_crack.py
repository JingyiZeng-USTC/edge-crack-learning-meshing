"""CLI: arbitrary crack graph -> density field -> constrained FEM mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arbitrary_crack_mesh import generate_arbitrary_crack_mesh
from crack_graph_geometry import CrackGraphGeometry


def _polyline_arrays(geometry: CrackGraphGeometry) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    points = []
    for polyline in geometry.crack_polylines:
        points.append(np.asarray(polyline, dtype=np.float64))
        offsets.append(offsets[-1] + len(polyline))
    return np.vstack(points), np.asarray(offsets, dtype=np.int64)


def _edge_array(values) -> np.ndarray:
    edges = np.asarray(sorted(values), dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return edges.reshape(-1, 2)


def _ordered_edge_chains(
    edges: np.ndarray,
    points: np.ndarray,
    geometry: CrackGraphGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one or more non-branching face-edge components into ordered paths."""
    if len(edges) == 0:
        return np.empty(0, dtype=np.int64), np.asarray([0], dtype=np.int64)
    adjacency: dict[int, set[int]] = {}
    for first, second in np.asarray(edges, dtype=np.int64):
        first, second = int(first), int(second)
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError("A non-branched crack-face edge chain has degree greater than two")

    remaining = set(adjacency)
    chains = []
    entries = np.asarray(geometry.crack_entries, dtype=float)
    while remaining:
        seed = next(iter(remaining))
        component = set()
        stack = [seed]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        endpoints = [node for node in component if len(adjacency[node]) == 1]
        if len(endpoints) != 2:
            raise ValueError("A crack-face component is not an open path")
        if len(entries):
            distance = [
                np.min(np.linalg.norm(entries - points[node], axis=1))
                for node in endpoints
            ]
            start = endpoints[int(np.argmin(distance))]
        else:
            start = min(endpoints)
        chain = [start]
        previous = None
        current = start
        while True:
            candidates = adjacency[current] - ({previous} if previous is not None else set())
            if not candidates:
                break
            if len(candidates) != 1:
                raise ValueError("Crack-face traversal encountered an ambiguous next node")
            following = next(iter(candidates))
            chain.append(following)
            previous, current = current, following
        if len(chain) != len(component):
            raise ValueError("Crack-face edge component is disconnected during traversal")
        chains.append(chain)
        remaining -= component
    chains.sort(key=lambda chain: tuple(np.asarray(points[chain[0]], dtype=float)))
    offsets = np.cumsum([0] + [len(chain) for chain in chains]).astype(np.int64)
    values = np.asarray([node for chain in chains for node in chain], dtype=np.int64)
    return values, offsets


def save_mesh_npz(path: Path, mesh, geometry: CrackGraphGeometry) -> None:
    polyline_points, polyline_offsets = _polyline_arrays(geometry)
    info = mesh.crack_info
    points = np.asarray(mesh.points, dtype=np.float64)
    face_edges = _edge_array(info.get("crack_face_edges", ()))
    upper_edges = _edge_array(info.get("crack_upper", ()))
    lower_edges = _edge_array(info.get("crack_lower", ()))
    upper_ordered, upper_offsets = _ordered_edge_chains(
        upper_edges, points, geometry,
    )
    lower_ordered, lower_offsets = _ordered_edge_chains(
        lower_edges, points, geometry,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points,
        triangles=np.asarray(mesh.triangles, dtype=np.int64),
        W=float(geometry.W),
        H=float(geometry.H),
        crack_graph_vertices=np.asarray(geometry.graph_vertices, dtype=np.float64),
        crack_graph_segments=np.asarray(geometry.graph_segments, dtype=np.int64),
        crack_polyline_points=polyline_points,
        crack_polyline_offsets=polyline_offsets,
        crack_tips=np.asarray(geometry.crack_tips, dtype=np.float64),
        crack_entries=np.asarray(geometry.crack_entries, dtype=np.float64),
        branch_points=np.asarray(geometry.branch_points, dtype=np.float64),
        crack_upper_nodes=np.asarray(sorted(info.get("crack_upper_nodes", ())), dtype=np.int64),
        crack_lower_nodes=np.asarray(sorted(info.get("crack_lower_nodes", ())), dtype=np.int64),
        crack_upper_edges=upper_edges,
        crack_lower_edges=lower_edges,
        crack_upper_ordered_nodes=upper_ordered,
        crack_upper_ordered_offsets=upper_offsets,
        crack_lower_ordered_nodes=lower_ordered,
        crack_lower_ordered_offsets=lower_offsets,
        crack_tip_nodes=np.asarray(sorted(info.get("crack_tip_nodes", ())), dtype=np.int64),
        crack_face_nodes=np.asarray(sorted(info.get("crack_face_nodes", ())), dtype=np.int64),
        crack_face_edges=face_edges,
        branch_sector_nodes=np.asarray(
            sorted(info.get("branch_sector_nodes", ())), dtype=np.int64,
        ),
        crack_node_copy_sources=np.asarray(
            info.get("node_copy_sources", ()), dtype=np.int64,
        ),
        crack_node_copy_offsets=np.asarray(
            info.get("node_copy_offsets", ()), dtype=np.int64,
        ),
        crack_node_copy_values=np.asarray(
            info.get("node_copy_values", ()), dtype=np.int64,
        ),
        has_branches=bool(geometry.has_branches),
        crack_faces_open=not bool(info.get("shared_crack_nodes", False)),
        mesh_minimum_angle_deg=float(mesh.quality["minimum_angle_deg"]),
        mesh_angle_p05_deg=float(mesh.quality["angle_p05_deg"]),
        mesh_steiner_points=int(mesh.quality.get("steiner_points", 0)),
        requested_minimum_angle_deg=float(
            mesh.quality.get("requested_minimum_angle_deg", 0.0)
        ),
        quality_target_met=bool(mesh.quality.get("quality_target_met", True)),
    )


def save_mesh_vtu(path: Path, mesh, geometry: CrackGraphGeometry) -> None:
    import meshio

    points3 = np.column_stack((mesh.points, np.zeros(len(mesh.points))))
    info = mesh.crack_info
    # VTU has no native point-set representation.  Writing point_sets makes
    # meshio perform a lossy conversion and emit warnings for unassigned
    # points.  Independent 0/1 arrays preserve overlapping roles (a crack tip
    # can also be on a crack face) and are read directly by ParaView.
    point_data = {}
    for name, key in (
        ("CRACK_FACE_POSITIVE", "crack_upper_nodes"),
        ("CRACK_FACE_NEGATIVE", "crack_lower_nodes"),
        ("CRACK_FACE", "crack_face_nodes"),
        ("BRANCH_SECTORS", "branch_sector_nodes"),
        ("CRACK_TIPS", "crack_tip_nodes"),
    ):
        marker = np.zeros(len(mesh.points), dtype=np.uint8)
        node_ids = np.asarray(sorted(info.get(key, ())), dtype=np.int64)
        if len(node_ids):
            if int(node_ids.min()) < 0 or int(node_ids.max()) >= len(marker):
                raise ValueError(f"{key} contains a node ID outside the mesh")
            marker[node_ids] = 1
        point_data[name] = marker
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.Mesh(
        points3,
        [("triangle", np.asarray(mesh.triangles, dtype=int))],
        point_data=point_data,
    ).write(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a constrained mesh from arbitrary crack polylines",
    )
    parser.add_argument("--geometry", required=True, help="CrackGraphGeometry JSON")
    parser.add_argument("--output", default="output/arbitrary_crack_mesh")
    parser.add_argument("--format", choices=("npz", "vtu", "both"), default="both")
    parser.add_argument(
        "--size-field-model", default=None,
        help=(
            "Optional residual-v1, analytic-independent absolute-v2, or "
            "straight-only local-v4 "
            "crack-graph size-field checkpoint"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size-field-blend", type=float, default=1.0)
    parser.add_argument("--size-field-grid-nx", type=int, default=161)
    parser.add_argument("--size-field-grid-ny", type=int, default=321)
    parser.add_argument("--size-field-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--size-field-complexity-ratio", type=float, default=1.0)
    parser.add_argument("--h-min", type=float, default=0.004)
    parser.add_argument("--h-max", type=float, default=0.030)
    parser.add_argument("--L-tip", type=float, default=0.08)
    parser.add_argument("--L-crack", type=float, default=0.05)
    parser.add_argument("--L-entry", type=float, default=0.08)
    parser.add_argument("--L-branch", type=float, default=0.08)
    parser.add_argument("--curvature-strength", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-crack-segments", type=int, default=30)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-tip-ring", type=int, default=8)
    parser.add_argument("--k-candidates", type=int, default=30)
    parser.add_argument("--quality-iterations", type=int, default=2)
    parser.add_argument(
        "--triangulation-min-angle", type=float, default=20.0,
        help="Triangle quality target in degrees; use 0 to disable refinement",
    )
    parser.add_argument(
        "--max-steiner-points", type=int, default=20000,
        help="Maximum quality-refinement points added by Triangle",
    )
    parser.add_argument(
        "--keep-shared-crack-nodes", action="store_true",
        help="Topology-only branch test; resulting crack faces are not open for FEM",
    )
    parser.add_argument("--run-fem", action="store_true")
    parser.add_argument("--fem-output", default=None)
    parser.add_argument("--fem-E", type=float, default=2.10e11)
    parser.add_argument("--fem-nu", type=float, default=0.30)
    parser.add_argument("--fem-thickness", type=float, default=0.01)
    parser.add_argument(
        "--fem-behavior", choices=("plane-stress", "plane-strain"),
        default="plane-stress",
    )
    parser.add_argument("--fem-top-uy", type=float, default=2.0e-4)
    parser.add_argument("--fem-bottom-uy", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geometry = CrackGraphGeometry.from_json(args.geometry)
    print(geometry.summary())
    if args.run_fem and args.keep_shared_crack_nodes:
        raise ValueError("FEM is disabled when crack nodes are intentionally shared")
    field_arguments = dict(
        h_min=args.h_min,
        h_max=args.h_max,
        L_tip=args.L_tip,
        L_crack=args.L_crack,
        L_entry=args.L_entry,
        L_branch=args.L_branch,
        curvature_strength=args.curvature_strength,
    )
    if args.size_field_model:
        import torch

        checkpoint_path = Path(args.size_field_model)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Size-field checkpoint not found: {checkpoint_path}. "
                "Train the requested model before running inference."
            )
        checkpoint_header = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        model_family = checkpoint_header.get("model_family")
        if model_family == "single-crack-local-size-field-v4":
            if abs(args.size_field_blend - 1.0) > 1.0e-12:
                raise ValueError(
                    "--size-field-blend is only defined for residual models; "
                    "use 1.0 with the local v4 model"
                )
            from module30_learned_size_field.single_crack_local_field import (
                SingleCrackLocalSizeField,
            )

            field = SingleCrackLocalSizeField(
                geometry,
                checkpoint=args.size_field_model,
                h_min=args.h_min,
                h_max=args.h_max,
                grid_nx=args.size_field_grid_nx,
                grid_ny=args.size_field_grid_ny,
                smooth_sigma=args.size_field_smooth_sigma,
                complexity_ratio=args.size_field_complexity_ratio,
                device=args.device,
            )
        elif model_family == "absolute-crack-graph-size-field-v2":
            if abs(args.size_field_blend - 1.0) > 1.0e-12:
                raise ValueError(
                    "--size-field-blend is only defined for residual models; "
                    "use 1.0 with the absolute v2 model"
                )
            from module30_learned_size_field.absolute_crack_graph_field import (
                AbsoluteLearnedCrackGraphSizeField,
            )

            field = AbsoluteLearnedCrackGraphSizeField(
                geometry,
                checkpoint=args.size_field_model,
                h_min=args.h_min,
                h_max=args.h_max,
                grid_nx=args.size_field_grid_nx,
                grid_ny=args.size_field_grid_ny,
                smooth_sigma=args.size_field_smooth_sigma,
                complexity_ratio=args.size_field_complexity_ratio,
                device=args.device,
            )
        elif model_family == "learned-residual-crack-graph-size-field-v1":
            from module30_learned_size_field.crack_graph_field import (
                LearnedCrackGraphSizeField,
            )

            field = LearnedCrackGraphSizeField(
                geometry,
                checkpoint=args.size_field_model,
                blend=args.size_field_blend,
                grid_nx=args.size_field_grid_nx,
                grid_ny=args.size_field_grid_ny,
                smooth_sigma=args.size_field_smooth_sigma,
                complexity_ratio=args.size_field_complexity_ratio,
                device=args.device,
                **field_arguments,
            )
        else:
            raise ValueError(
                f"Unsupported arbitrary-crack size-field checkpoint: {model_family!r}"
            )
    else:
        from crack_graph_size_field import CrackGraphSizeField

        field = CrackGraphSizeField(geometry, **field_arguments)
    print(field.summary())
    mesh = generate_arbitrary_crack_mesh(
        geometry,
        field,
        seed=args.seed,
        n_crack_segments=args.n_crack_segments,
        n_layers=args.n_layers,
        n_tip_ring=args.n_tip_ring,
        k_candidates=args.k_candidates,
        quality_iterations=args.quality_iterations,
        triangulation_minimum_angle=args.triangulation_min_angle,
        max_steiner_points=args.max_steiner_points,
        open_crack_faces=not args.keep_shared_crack_nodes,
    )
    print(
        f"Mesh: nodes={len(mesh.points)}, triangles={len(mesh.triangles)}, "
        f"min_angle={mesh.quality['minimum_angle_deg']:.3f}, "
        f"p05_angle={mesh.quality['angle_p05_deg']:.3f}, "
        f"steiner={mesh.quality['steiner_points']}, "
        f"degenerate={mesh.quality['degenerate_elements']}"
    )
    if not mesh.quality["quality_target_met"]:
        print(
            "Warning: the requested minimum angle was not reached. "
            "Increase --max-steiner-points or lower --triangulation-min-angle."
        )
    destination = Path(args.output)
    if args.format in ("npz", "both"):
        save_mesh_npz(destination.with_suffix(".npz"), mesh, geometry)
        print(f"Pure mesh: {destination.with_suffix('.npz')}")
    if args.format in ("vtu", "both"):
        save_mesh_vtu(destination.with_suffix(".vtu"), mesh, geometry)
        print(f"Visualization: {destination.with_suffix('.vtu')}")

    if args.run_fem:
        from module10_fem.mesh_io import save_npz as save_fem_npz, save_vtu
        from module10_fem.pipeline import run_linear_elastic_analysis

        analysis = run_linear_elastic_analysis(
            mesh.points,
            mesh.triangles,
            young_modulus=args.fem_E,
            poisson_ratio=args.fem_nu,
            thickness=args.fem_thickness,
            behavior=args.fem_behavior,
            top_displacement=args.fem_top_uy,
            bottom_displacement=args.fem_bottom_uy,
        )
        fem_output = Path(args.fem_output or (str(destination) + "_fem"))
        fem_output.parent.mkdir(parents=True, exist_ok=True)
        save_fem_npz(
            fem_output.with_suffix(".npz"), analysis.system, analysis.boundary,
            analysis.solution, analysis.fields,
        )
        save_vtu(
            fem_output.with_suffix(".vtu"), analysis.system,
            analysis.solution, analysis.fields,
        )
        rhs = max(float(np.linalg.norm(analysis.solution.rhs_free)), 1.0e-30)
        residual = float(np.linalg.norm(analysis.solution.residual_free)) / rhs
        print(
            f"Module 10: relative_residual={residual:.6e}, "
            f"max_von_mises={float(analysis.fields.element_von_mises.max()):.6e}"
        )
        print(f"FEM result: {fem_output.with_suffix('.npz')}, {fem_output.with_suffix('.vtu')}")


if __name__ == "__main__":
    main()
