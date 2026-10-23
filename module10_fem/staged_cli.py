"""Independent CLI stages: mesh -> K/operators -> U -> strain/stress."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .assembly import assemble_stiffness
from .boundary import vertical_tension_boundary
from .material import plane_strain_matrix, plane_stress_matrix
from .mesh_io import load_mesh, save_npz, save_vtu
from .recovery import recover_fields
from .solver import solve_linear_system
from .stage_io import (
    load_solution,
    load_system,
    save_fields,
    save_solution,
    save_system,
)


def _npz_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")


def _constitutive(behavior: str, young_modulus: float, poisson_ratio: float):
    if behavior == "plane-stress":
        return plane_stress_matrix(young_modulus, poisson_ratio)
    if behavior == "plane-strain":
        return plane_strain_matrix(young_modulus, poisson_ratio)
    raise ValueError("Unknown material behavior")


def assemble_stage(args) -> Path:
    points, triangles, metadata = load_mesh(args.mesh)
    crack_faces_open = metadata.get("crack_faces_open")
    if crack_faces_open is not None:
        faces_are_open = bool(np.asarray(crack_faces_open).reshape(()).item())
        if not faces_are_open and not getattr(args, "allow_closed_crack_faces", False):
            raise ValueError(
                "The mesh declares crack_faces_open=False. Its crack-face topology "
                "is not ready for physical analysis. Use a face-separated mesh.",
            )
    D = _constitutive(args.behavior, args.E, args.nu)
    system = assemble_stiffness(points, triangles, D, args.thickness)
    output = _npz_path(args.output)
    identifier = save_system(output, system, source_mesh=str(Path(args.mesh).resolve()))
    antisymmetric = system.stiffness - system.stiffness.T
    symmetry_error = float(np.linalg.norm(antisymmetric.data)) / max(
        float(np.linalg.norm(system.stiffness.data)), 1.0e-30,
    )
    print(
        f"ASSEMBLE: nodes={system.n_nodes}, triangles={len(system.triangles)}, "
        f"DOFs={system.n_dofs}, nnz={system.stiffness.nnz}",
    )
    print(
        f"ASSEMBLE: symmetry_error={symmetry_error:.6e}, "
        f"min_area={float(system.element_areas.min()):.6e}",
    )
    print(f"ASSEMBLE: system_id={identifier}")
    print(f"ASSEMBLE: saved={output}")
    return output


def _load_force(path: str | None, n_dofs: int) -> np.ndarray | None:
    if not path:
        return None
    source = Path(path)
    if source.suffix.lower() == ".npy":
        force = np.load(source, allow_pickle=False)
    else:
        with np.load(source, allow_pickle=False) as data:
            if "force" not in data.files:
                raise ValueError(f"{source} does not contain a 'force' array")
            force = data["force"]
    force = np.asarray(force, dtype=float).reshape(-1)
    if force.shape != (n_dofs,):
        raise ValueError(f"force must have shape ({n_dofs},)")
    return force


def solve_stage(args) -> Path:
    system, identifier, _ = load_system(args.system)
    force = _load_force(args.force_file, system.n_dofs)
    boundary = vertical_tension_boundary(
        system.points,
        top_displacement=args.top_uy,
        bottom_displacement=args.bottom_uy,
        pin_node=args.pin_node,
        force=force,
    )
    solution = solve_linear_system(system, boundary)
    output = _npz_path(args.output)
    save_solution(output, system, boundary, solution)
    rhs = max(float(np.linalg.norm(solution.rhs_free)), 1.0e-30)
    relative_residual = float(np.linalg.norm(solution.residual_free)) / rhs
    print(
        f"SOLVE: system_id={identifier}, free_DOFs={len(solution.free_dofs)}, "
        f"prescribed_DOFs={len(solution.prescribed_dofs)}",
    )
    print(
        f"SOLVE: relative_residual={relative_residual:.6e}, "
        f"strain_energy={solution.strain_energy:.6e}",
    )
    print(f"SOLVE: saved={output}")
    return output


def recover_stage(args) -> Path:
    system, identifier, _ = load_system(args.system)
    boundary, solution = load_solution(args.solution, system)
    fields = recover_fields(system, solution.displacement)
    output = _npz_path(args.output)
    save_fields(output, system, fields)
    if args.vtu:
        vtu = Path(args.vtu)
        vtu.parent.mkdir(parents=True, exist_ok=True)
        save_vtu(vtu, system, solution, fields)
        print(f"RECOVER: visualization={vtu}")
    if args.combined_output:
        combined = _npz_path(args.combined_output)
        combined.parent.mkdir(parents=True, exist_ok=True)
        save_npz(combined, system, boundary, solution, fields)
        print(f"RECOVER: combined={combined}")
    print(
        f"RECOVER: system_id={identifier}, "
        f"max_von_mises={float(fields.element_von_mises.max()):.6e}",
    )
    print(f"RECOVER: saved={output}")
    return output


def _add_material_arguments(parser) -> None:
    parser.add_argument("--E", type=float, default=2.10e11)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument(
        "--behavior", choices=("plane-stress", "plane-strain"),
        default="plane-stress",
    )


def _add_load_arguments(parser) -> None:
    parser.add_argument("--top-uy", type=float, default=2.0e-4)
    parser.add_argument("--bottom-uy", type=float, default=0.0)
    parser.add_argument("--pin-node", type=int, default=None)
    parser.add_argument(
        "--force-file", default=None,
        help="Optional .npy vector or .npz containing 'force' with 2N entries",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged deterministic Module 10 FEM pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    assemble = commands.add_parser("assemble", help="pure mesh -> K, B, areas, DOF map")
    assemble.add_argument("--mesh", required=True)
    assemble.add_argument("--output", required=True)
    _add_material_arguments(assemble)
    assemble.add_argument(
        "--allow-closed-crack-faces", action="store_true",
        help="Override the crack-topology safety check (not recommended)",
    )

    solve = commands.add_parser("solve", help="system + BC/load -> displacement")
    solve.add_argument("--system", required=True)
    solve.add_argument("--output", required=True)
    _add_load_arguments(solve)

    recover = commands.add_parser("recover", help="system + displacement -> strain/stress")
    recover.add_argument("--system", required=True)
    recover.add_argument("--solution", required=True)
    recover.add_argument("--output", required=True)
    recover.add_argument("--vtu", default=None)
    recover.add_argument("--combined-output", default=None)

    all_stages = commands.add_parser("all", help="run all three stages and save each product")
    all_stages.add_argument("--mesh", required=True)
    all_stages.add_argument("--output-prefix", required=True)
    _add_material_arguments(all_stages)
    _add_load_arguments(all_stages)
    all_stages.add_argument(
        "--allow-closed-crack-faces", action="store_true",
        help="Override the crack-topology safety check (not recommended)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "assemble":
        assemble_stage(args)
        return
    if args.command == "solve":
        solve_stage(args)
        return
    if args.command == "recover":
        recover_stage(args)
        return
    if args.command == "all":
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        assemble_args = argparse.Namespace(
            mesh=args.mesh,
            output=str(prefix) + "_system.npz",
            E=args.E,
            nu=args.nu,
            thickness=args.thickness,
            behavior=args.behavior,
            allow_closed_crack_faces=args.allow_closed_crack_faces,
        )
        system_path = assemble_stage(assemble_args)
        solve_args = argparse.Namespace(
            system=system_path,
            output=str(prefix) + "_solution.npz",
            top_uy=args.top_uy,
            bottom_uy=args.bottom_uy,
            pin_node=args.pin_node,
            force_file=args.force_file,
        )
        solution_path = solve_stage(solve_args)
        recover_args = argparse.Namespace(
            system=system_path,
            solution=solution_path,
            output=str(prefix) + "_fields.npz",
            vtu=str(prefix) + ".vtu",
            combined_output=str(prefix) + "_complete.npz",
        )
        recover_stage(recover_args)
        print("STAGED MODULE 10 COMPLETED")
        return
    raise RuntimeError("Unreachable command")


if __name__ == "__main__":
    main()
