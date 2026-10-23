"""Controlled size-field efficiency experiment on one straight edge crack.

Only the prescribed size function changes.  Both methods use the same Gmsh
geometry, meshing algorithm, crack opening operation, quality options, CST
finite-element assembly, line load and sparse direct solver.

The final figure contains two curves only:
    node count -> relative strain-energy error

This script is an offline ablation experiment.  Using Gmsh here as the common
meshing operator does not change the paper's separate online pure-GNN workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Callable
from xml.etree import ElementTree as ET
import re

import gmsh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)


class ScaledField:
    def __init__(self, base, scale: float):
        if scale <= 0.0:
            raise ValueError("size scale must be positive")
        self.base = base
        self.scale = float(scale)

    def evaluate(self, x: float, y: float) -> float:
        return self.scale * float(self.base.evaluate(x, y))

    def evaluate_bulk(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(self.base.evaluate_bulk(points), dtype=float)


class MinimumField:
    def __init__(self, first, second, scale: float):
        self.first = first
        self.second = second
        self.scale = float(scale)

    def evaluate(self, x: float, y: float) -> float:
        return self.scale * min(
            float(self.first.evaluate(x, y)), float(self.second.evaluate(x, y))
        )


class ConstantField:
    """Uniform Gmsh size with the same continuous complexity as a base field."""

    def __init__(self, value: float):
        if value <= 0.0:
            raise ValueError("constant size must be positive")
        self.value = float(value)

    def evaluate(self, x: float, y: float) -> float:
        del x, y
        return self.value

    def evaluate_bulk(self, points: np.ndarray) -> np.ndarray:
        return np.full(len(points), self.value, dtype=float)


def _equivalent_uniform_size(field, width: float, height: float) -> float:
    """Match mean(1/h_uniform^2) to mean(1/h(x)^2)."""
    xs = np.linspace(0.0, width, 301)
    ys = np.linspace(0.0, height, 601)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    sizes = np.asarray(field.evaluate_bulk(points), dtype=float)
    return float(1.0 / math.sqrt(np.mean(1.0 / np.maximum(sizes, 1.0e-30) ** 2)))


def _triangle_quality(points: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    vertices = points[triangles]
    lengths = np.stack(
        (
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ),
        axis=1,
    )
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    area = 0.5 * cross
    quality = 4.0 * math.sqrt(3.0) * area / np.maximum(
        np.sum(lengths * lengths, axis=1), 1.0e-30
    )
    cosine = np.empty((len(triangles), 3), dtype=float)
    cosine[:, 0] = (
        lengths[:, 0] ** 2 + lengths[:, 2] ** 2 - lengths[:, 1] ** 2
    ) / np.maximum(2.0 * lengths[:, 0] * lengths[:, 2], 1.0e-30)
    cosine[:, 1] = (
        lengths[:, 0] ** 2 + lengths[:, 1] ** 2 - lengths[:, 2] ** 2
    ) / np.maximum(2.0 * lengths[:, 0] * lengths[:, 1], 1.0e-30)
    cosine[:, 2] = (
        lengths[:, 1] ** 2 + lengths[:, 2] ** 2 - lengths[:, 0] ** 2
    ) / np.maximum(2.0 * lengths[:, 1] * lengths[:, 2], 1.0e-30)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    minimum_angles = np.min(angles, axis=1)
    return {
        "minimum_angle_deg": float(np.min(minimum_angles)),
        "angle_p05_deg": float(np.quantile(minimum_angles, 0.05)),
        "quality_mean": float(np.mean(quality)),
        "quality_p05": float(np.quantile(quality, 0.05)),
    }


def _open_embedded_crack(
    points: np.ndarray,
    triangles: np.ndarray,
    crack_node_indices: np.ndarray,
    start: np.ndarray,
    tip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Duplicate embedded-line nodes on one side to create a zero-thickness slit."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64).copy()
    crack_node_indices = np.unique(np.asarray(crack_node_indices, dtype=np.int64))
    tip_distance = np.linalg.norm(points[crack_node_indices] - tip[None, :], axis=1)
    tip_index = int(crack_node_indices[int(np.argmin(tip_distance))])
    scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    if np.linalg.norm(points[tip_index] - tip) > 1.0e-8 * scale:
        raise RuntimeError("Gmsh crack-line nodes do not contain the crack tip")
    duplicated_source = crack_node_indices[crack_node_indices != tip_index]
    duplicate_map: dict[int, int] = {}
    output_points = points.tolist()
    for source in duplicated_source:
        duplicate_map[int(source)] = len(output_points)
        output_points.append(points[source].tolist())
    output_points_array = np.asarray(output_points, dtype=float)

    tangent = tip - start
    tangent /= np.linalg.norm(tangent)
    centroids = np.mean(points[triangles], axis=1)
    signed_side = (
        tangent[0] * (centroids[:, 1] - start[1])
        - tangent[1] * (centroids[:, 0] - start[0])
    )
    negative_side = signed_side < 0.0
    for element_index in np.flatnonzero(negative_side):
        for local_index in range(3):
            source = int(triangles[element_index, local_index])
            if source in duplicate_map:
                triangles[element_index, local_index] = duplicate_map[source]

    used = np.unique(triangles)
    if len(used) != len(output_points_array):
        raise RuntimeError(
            f"Opening the crack produced {len(output_points_array) - len(used)} isolated nodes"
        )
    return output_points_array, triangles, len(duplicated_source)


def generate_gmsh_mesh(
    field,
    *,
    width: float,
    height: float,
    start: np.ndarray,
    tip: np.ndarray,
    algorithm: int,
    optimize: bool,
    model_name: str,
    open_crack: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    gmsh.model.mesh.removeSizeCallback()
    gmsh.clear()
    gmsh.model.add(model_name)
    geometry = gmsh.model.geo
    tolerance = 1.0e-12
    if abs(float(start[0])) > tolerance:
        raise ValueError("This controlled experiment currently supports a left-edge crack")
    if not 0.0 < float(start[1]) < height:
        raise ValueError("Crack entry must lie strictly inside the left boundary")

    p0 = geometry.addPoint(0.0, 0.0, 0.0)
    p1 = geometry.addPoint(width, 0.0, 0.0)
    p2 = geometry.addPoint(width, height, 0.0)
    p3 = geometry.addPoint(0.0, height, 0.0)
    entry = geometry.addPoint(float(start[0]), float(start[1]), 0.0)
    crack_tip = geometry.addPoint(float(tip[0]), float(tip[1]), 0.0)
    bottom = geometry.addLine(p0, p1)
    right = geometry.addLine(p1, p2)
    top = geometry.addLine(p2, p3)
    left_upper = geometry.addLine(p3, entry)
    left_lower = geometry.addLine(entry, p0)
    crack_line = geometry.addLine(entry, crack_tip)
    loop = geometry.addCurveLoop([bottom, right, top, left_upper, left_lower])
    surface = geometry.addPlaneSurface([loop])
    geometry.synchronize()
    gmsh.model.mesh.embed(1, [crack_line], 2, surface)

    gmsh.option.setNumber("Mesh.Algorithm", float(algorithm))
    gmsh.option.setNumber("Mesh.ElementOrder", 1.0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)
    gmsh.option.setNumber("Mesh.Optimize", 1.0 if optimize else 0.0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0.0)

    def callback(dim, tag, x, y, z, lc):
        del dim, tag, z, lc
        value = float(field.evaluate(float(x), float(y)))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Size-field callback produced a non-positive value")
        return value

    if hasattr(field, 'install_gmsh'):
        field.install_gmsh(crack_line, crack_tip)
    else:
        gmsh.model.mesh.setSizeCallback(callback)
    gmsh.model.mesh.generate(2)
    if optimize:
        gmsh.model.mesh.optimize("Relocate2D")

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=float).reshape(-1, 3)[:, :2]
    tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}

    element_types, _, element_node_tags = gmsh.model.mesh.getElements(2, surface)
    triangles = []
    for element_type, flattened_tags in zip(element_types, element_node_tags):
        properties = gmsh.model.mesh.getElementProperties(int(element_type))
        element_name = properties[0]
        nodes_per_element = int(properties[3])
        if nodes_per_element != 3 or "Triangle" not in element_name:
            continue
        tags = np.asarray(flattened_tags, dtype=np.int64).reshape(-1, 3)
        triangles.append(
            np.asarray(
                [[tag_to_index[int(tag)] for tag in element] for element in tags],
                dtype=np.int64,
            )
        )
    if not triangles:
        raise RuntimeError("Gmsh did not generate linear triangles")
    triangles_array = np.vstack(triangles)

    crack_tags, _, _ = gmsh.model.mesh.getNodes(1, crack_line, includeBoundary=True)
    crack_indices = np.asarray(
        [tag_to_index[int(tag)] for tag in np.asarray(crack_tags, dtype=np.int64)],
        dtype=np.int64,
    )
    if open_crack:
        points, triangles_array, duplicated = _open_embedded_crack(
            points, triangles_array, crack_indices, start, tip
        )
    else:
        # Fixed-node connectivity comparisons use the single-node crack-line
        # representation of the existing GNN test meshes.  Never report this
        # un-opened representation as a mechanical slit mesh.
        duplicated = 0
    first = points[triangles_array[:, 1]] - points[triangles_array[:, 0]]
    second = points[triangles_array[:, 2]] - points[triangles_array[:, 0]]
    signed_area = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    clockwise = signed_area < 0.0
    triangles_array[clockwise, 1], triangles_array[clockwise, 2] = (
        triangles_array[clockwise, 2].copy(),
        triangles_array[clockwise, 1].copy(),
    )
    quality = _triangle_quality(points, triangles_array)
    quality["duplicated_crack_nodes"] = int(duplicated)
    quality["domain_area"] = float(np.sum(np.abs(signed_area)))
    return points, triangles_array, quality


def solve_strain_energy(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    young_modulus: float,
    poisson_ratio: float,
    thickness: float,
    line_load: float,
) -> tuple[float, float]:
    from module10_fem.assembly import assemble_stiffness
    from module10_fem.boundary import top_uniform_line_load_boundary
    from module10_fem.material import plane_stress_matrix
    from module10_fem.solver import solve_linear_system

    constitutive = plane_stress_matrix(young_modulus, poisson_ratio)
    system = assemble_stiffness(points, triangles, constitutive, thickness)
    boundary = top_uniform_line_load_boundary(points, line_load_y=line_load)
    solution = solve_linear_system(system, boundary)
    residual_scale = max(float(np.linalg.norm(solution.rhs_free)), 1.0e-30)
    relative_residual = float(np.linalg.norm(solution.residual_free)) / residual_scale
    return float(solution.strain_energy), relative_residual


def _crossing_node_count(rows: list[dict], threshold_percent: float) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row["nodes"]))
    nodes = np.asarray([row["nodes"] for row in ordered], dtype=float)
    errors = np.asarray([row["strain_energy_error_percent"] for row in ordered], dtype=float)
    envelope = np.minimum.accumulate(errors)
    if envelope[0] <= threshold_percent:
        return float(nodes[0])
    crossing = np.flatnonzero(envelope <= threshold_percent)
    if not len(crossing):
        return None
    upper = int(crossing[0])
    lower = upper - 1
    x0, x1 = nodes[lower], nodes[upper]
    y0, y1 = envelope[lower], envelope[upper]
    if abs(y1 - y0) < 1.0e-15:
        return float(x1)
    fraction = (threshold_percent - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    options = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    fig.savefig(base.with_suffix(".svg"), **options)
    fig.savefig(base.with_suffix(".pdf"), **options)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, **options)
    fig.savefig(base.with_suffix(".png"), dpi=300, **options)


def plot_curves(
    output: Path,
    rows: list[dict],
    threshold_percent: float,
    traditional_required: float | None,
    learned_required: float | None,
    traditional_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(88.0 / 25.4, 68.0 / 25.4), constrained_layout=True)
    styles = {
        "traditional": {
            "label": traditional_label,
            "color": "#6F6F6F",
            "marker": "s",
            "linestyle": "--",
        },
        "learned": {
            "label": "Learned size field",
            "color": "#2878B5",
            "marker": "o",
            "linestyle": "-",
        },
    }
    for method in ("traditional", "learned"):
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["nodes"]),
        )
        ax.plot(
            [row["nodes"] for row in selected],
            [row["strain_energy_error_percent"] for row in selected],
            linewidth=1.35,
            markersize=4.2,
            markeredgewidth=0.75,
            markerfacecolor="white" if method == "traditional" else styles[method]["color"],
            **styles[method],
        )
    ax.axhline(
        threshold_percent, color="#B64342", linewidth=0.9, linestyle=":",
        label=fr"Target error = {threshold_percent:g}%",
    )
    for required, color in (
        (traditional_required, styles["traditional"]["color"]),
        (learned_required, styles["learned"]["color"]),
    ):
        if required is not None:
            ax.vlines(
                required, ymin=0.0, ymax=threshold_percent,
                color=color, linewidth=0.8, linestyle=":",
            )
    if traditional_required is not None and learned_required is not None:
        change = 100.0 * (learned_required - traditional_required) / traditional_required
        if change >= 0.0:
            comparison = f"Learned node increase at {threshold_percent:g}% error: {change:.1f}%"
        else:
            comparison = f"Learned node reduction at {threshold_percent:g}% error: {-change:.1f}%"
        ax.text(
            0.98, 0.04, comparison,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        )
    ax.set_xlabel("Number of nodes, $N$")
    ax.set_ylabel("Relative strain-energy error (%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax.legend(loc="best", fontsize=6.2)
    _save_figure(fig, output / "Figure_size_field_node_efficiency")
    plt.close(fig)


def validate_figure(output: Path) -> None:
    base = output / "Figure_size_field_node_efficiency"
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    tiff_path = base.with_suffix(".tiff")
    svg = ET.parse(svg_path).getroot()
    text_count = len(svg.findall(".//{http://www.w3.org/2000/svg}text"))
    pdf = pdf_path.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", pdf))
    with Image.open(tiff_path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
    if text_count == 0 or pages != 1 or min(dpi) < 590:
        raise RuntimeError(
            f"Figure QA failed: SVG text={text_count}, PDF pages={pages}, TIFF dpi={dpi}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "module30_checkpoints/single_straight_crack_local_v4.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/size_field_efficiency_gmsh",
    )
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=2.0)
    parser.add_argument("--origin", type=float, default=1.07)
    parser.add_argument("--crack-length", type=float, default=0.32)
    parser.add_argument("--angle-deg", type=float, default=11.0)
    parser.add_argument("--h-min", type=float, default=0.004)
    parser.add_argument("--h-max", type=float, default=0.030)
    parser.add_argument("--L-tip", type=float, default=0.08)
    parser.add_argument("--L-crack", type=float, default=0.05)
    parser.add_argument("--L-entry", type=float, default=0.08)
    parser.add_argument(
        "--traditional-mode",
        choices=("uniform", "analytic"),
        default="uniform",
        help=(
            "uniform isolates adaptive allocation against equal-complexity fixed "
            "sizing; analytic uses the stronger hand-designed crack-distance field"
        ),
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[2.40, 2.00, 1.60, 1.30, 1.00, 0.80, 0.65],
    )
    parser.add_argument("--reference-scale", type=float, default=0.48)
    parser.add_argument("--target-error-percent", type=float, default=1.0)
    parser.add_argument("--gmsh-algorithm", type=int, default=6)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--grid-nx", type=int, default=161)
    parser.add_argument("--grid-ny", type=int, default=321)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--E", type=float, default=2.10e11)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--top-line-load", type=float, default=5.0e5)
    parser.add_argument("--save-meshes", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if any(scale <= 0.0 for scale in args.scales):
        raise ValueError("All size scales must be positive")
    if not 0.0 < args.reference_scale < min(args.scales):
        raise ValueError("reference-scale must be positive and finer than all reported levels")
    sys.path.insert(0, str(project_root))

    from crack_graph_geometry import CrackGraphGeometry
    from edge_crack_geometry import EdgeCrackGeometry, generate_crack
    from module30_learned_size_field.single_crack_local_field import (
        SingleCrackLocalSizeField,
    )
    from size_field import ContinuousSizeField

    theta = math.radians(args.angle_deg)
    start_tuple, tip_tuple = generate_crack(
        "left", args.origin, args.crack_length, theta, args.width, args.height
    )
    start = np.asarray(start_tuple, dtype=float)
    tip = np.asarray(tip_tuple, dtype=float)
    if not (0.0 < tip[0] < args.width and 0.0 < tip[1] < args.height):
        raise ValueError("Crack tip must lie strictly inside the plate")
    analytic_geometry = EdgeCrackGeometry(
        W=args.width,
        H=args.height,
        crack_start=start_tuple,
        crack_tip=tip_tuple,
        crack_length=args.crack_length,
        theta=theta,
        edge_side="left",
    )
    learned_geometry = CrackGraphGeometry(
        args.width, args.height, [np.vstack((start, tip))]
    )
    analytic_base = ContinuousSizeField(
        analytic_geometry,
        h_min=args.h_min,
        h_max=args.h_max,
        L_tip=args.L_tip,
        L_crack=args.L_crack,
        L_entry=args.L_entry,
    )
    learned_base = SingleCrackLocalSizeField(
        learned_geometry,
        checkpoint=str(checkpoint),
        h_min=args.h_min,
        h_max=args.h_max,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        smooth_sigma=args.smooth_sigma,
        complexity_ratio=1.0,
        device=args.device,
    )
    equivalent_uniform_h = _equivalent_uniform_size(
        learned_base, args.width, args.height
    )
    if args.traditional_mode == "uniform":
        traditional_base = ConstantField(equivalent_uniform_h)
        traditional_description = "uniform size matched to learned-field complexity"
        traditional_label = "Uniform Gmsh sizing"
    else:
        traditional_base = analytic_base
        traditional_description = "analytic crack-distance size field"
        traditional_label = "Rule-based Gmsh sizing"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir = output / "meshes"
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0.0)
    try:
        reference_field = MinimumField(
            traditional_base, learned_base, args.reference_scale
        )
        reference_points, reference_triangles, reference_quality = generate_gmsh_mesh(
            reference_field,
            width=args.width,
            height=args.height,
            start=start,
            tip=tip,
            algorithm=args.gmsh_algorithm,
            optimize=not args.no_optimize,
            model_name="method_neutral_reference",
        )
        reference_energy, reference_residual = solve_strain_energy(
            reference_points,
            reference_triangles,
            young_modulus=args.E,
            poisson_ratio=args.nu,
            thickness=args.thickness,
            line_load=args.top_line_load,
        )
        if args.save_meshes:
            np.savez_compressed(
                mesh_dir / "reference.npz",
                nodes=reference_points,
                triangles=reference_triangles,
                crack_start=start,
                crack_tip=tip,
            )

        rows: list[dict] = []
        for method, base in (
            ("traditional", traditional_base),
            ("learned", learned_base),
        ):
            for level, scale in enumerate(sorted(args.scales, reverse=True), start=1):
                points, triangles, quality = generate_gmsh_mesh(
                    ScaledField(base, scale),
                    width=args.width,
                    height=args.height,
                    start=start,
                    tip=tip,
                    algorithm=args.gmsh_algorithm,
                    optimize=not args.no_optimize,
                    model_name=f"{method}_{level}",
                )
                energy, residual = solve_strain_energy(
                    points,
                    triangles,
                    young_modulus=args.E,
                    poisson_ratio=args.nu,
                    thickness=args.thickness,
                    line_load=args.top_line_load,
                )
                error_percent = 100.0 * abs(energy - reference_energy) / max(
                    abs(reference_energy), 1.0e-30
                )
                row = {
                    "method": method,
                    "level": level,
                    "size_scale": scale,
                    "nodes": len(points),
                    "triangles": len(triangles),
                    "strain_energy_j": energy,
                    "reference_strain_energy_j": reference_energy,
                    "strain_energy_error_percent": error_percent,
                    "relative_residual": residual,
                    **quality,
                }
                rows.append(row)
                if args.save_meshes:
                    np.savez_compressed(
                        mesh_dir / f"{method}_level_{level}.npz",
                        nodes=points,
                        triangles=triangles,
                        crack_start=start,
                        crack_tip=tip,
                        size_scale=scale,
                    )
                print(
                    f"{method:11s} level={level} scale={scale:.3f} "
                    f"N={len(points)} Ne={len(triangles)} "
                    f"eU={error_percent:.4f}% min_angle={quality['minimum_angle_deg']:.3f}"
                )
    finally:
        gmsh.finalize()

    _write_csv(output / "size_field_efficiency_results.csv", rows)
    traditional_rows = [row for row in rows if row["method"] == "traditional"]
    learned_rows = [row for row in rows if row["method"] == "learned"]
    traditional_required = _crossing_node_count(
        traditional_rows, args.target_error_percent
    )
    learned_required = _crossing_node_count(learned_rows, args.target_error_percent)
    saving = None
    if traditional_required is not None and learned_required is not None:
        saving = 100.0 * (
            traditional_required - learned_required
        ) / traditional_required
    summary = {
        "experiment": "controlled-size-field-efficiency-with-common-gmsh",
        "only_changed_variable": "size_field",
        "traditional_field": traditional_description,
        "traditional_mode": args.traditional_mode,
        "equivalent_uniform_h_m": equivalent_uniform_h,
        "learned_field": "single-crack-local-size-field-v4",
        "common_connector": f"Gmsh 2-D algorithm {args.gmsh_algorithm}",
        "online_gnn_used": False,
        "fcnn_used": False,
        "geometry": {
            "width_m": args.width,
            "height_m": args.height,
            "crack_start_m": start.tolist(),
            "crack_tip_m": tip.tolist(),
            "crack_length_m": args.crack_length,
            "angle_deg": args.angle_deg,
        },
        "loading": {
            "type": "uniform-top-line-load",
            "value_n_per_m": args.top_line_load,
            "plane_condition": "plane-stress",
        },
        "reference": {
            "nodes": len(reference_points),
            "triangles": len(reference_triangles),
            "strain_energy_j": reference_energy,
            "relative_residual": reference_residual,
            **reference_quality,
        },
        "target_error_percent": args.target_error_percent,
        "traditional_required_nodes_interpolated": traditional_required,
        "learned_required_nodes_interpolated": learned_required,
        "learned_node_saving_percent": saving,
        "interpretation": (
            "A positive node saving means that the learned size field reaches the "
            "same strain-energy error with fewer nodes under a controlled common "
            "Gmsh meshing and FEM pipeline."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_curves(
        output,
        rows,
        args.target_error_percent,
        traditional_required,
        learned_required,
        traditional_label,
    )
    validate_figure(output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Complete. Results: {output}")


if __name__ == "__main__":
    main()
