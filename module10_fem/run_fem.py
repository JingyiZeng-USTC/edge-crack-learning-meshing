"""Command-line runner for the fixed crack mesh FEM stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mesh_io import load_mesh, save_npz, save_vtu
from .pipeline import run_linear_elastic_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse CST FEM for a 2-D crack mesh")
    parser.add_argument("mesh", help="Input .inp/.npz/.msh/.vtk/.vtu mesh")
    parser.add_argument("--E", type=float, default=210e9, help="Young's modulus")
    parser.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--behavior", choices=("plane-stress", "plane-strain"), default="plane-stress")
    parser.add_argument("--top-uy", type=float, default=2e-4)
    parser.add_argument("--bottom-uy", type=float, default=0.0)
    parser.add_argument("--pin-node", type=int, default=None,
                        help="Zero-based node index for UX=0; default is bottom-left")
    parser.add_argument("--output", default="module10_fem_result", help="Output path without extension")
    args = parser.parse_args()

    points, triangles, _ = load_mesh(args.mesh)
    analysis = run_linear_elastic_analysis(
        points,
        triangles,
        young_modulus=args.E,
        poisson_ratio=args.nu,
        thickness=args.thickness,
        behavior=args.behavior,
        top_displacement=args.top_uy,
        bottom_displacement=args.bottom_uy,
        pin_node=args.pin_node,
    )
    system = analysis.system
    bc = analysis.boundary
    solution = analysis.solution
    fields = analysis.fields

    output = Path(args.output)
    save_npz(output.with_suffix(".npz"), system, bc, solution, fields)
    save_vtu(output.with_suffix(".vtu"), system, solution, fields)

    rhs_norm = max(float(np.linalg.norm(solution.rhs_free)), 1e-30)
    relative_residual = float(np.linalg.norm(solution.residual_free)) / rhs_norm
    print(f"Nodes: {system.n_nodes}")
    print(f"Triangles: {len(system.triangles)}")
    print(f"DOFs: {system.n_dofs} ({len(solution.free_dofs)} free)")
    print(f"Strain energy: {solution.strain_energy:.6e}")
    print(f"Relative free-DOF residual: {relative_residual:.6e}")
    print(f"Max displacement: {np.linalg.norm(solution.displacement_xy, axis=1).max():.6e}")
    print(f"Max element von Mises: {fields.element_von_mises.max():.6e}")
    print(f"Saved: {output.with_suffix('.npz')}, {output.with_suffix('.vtu')}")


if __name__ == "__main__":
    main()
