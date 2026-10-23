"""Public in-memory entry point for the deterministic FEM stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assembly import FEMSystem, assemble_stiffness
from .boundary import BoundaryConditions, vertical_tension_boundary
from .material import plane_strain_matrix, plane_stress_matrix
from .recovery import FieldResults, recover_fields
from .solver import Solution, solve_linear_system


@dataclass(frozen=True)
class FEMAnalysis:
    """Products of assembly, displacement solution, and field recovery."""

    system: FEMSystem
    boundary: BoundaryConditions
    solution: Solution
    fields: FieldResults


def run_linear_elastic_analysis(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    young_modulus: float = 210e9,
    poisson_ratio: float = 0.3,
    thickness: float = 0.01,
    behavior: str = "plane-stress",
    top_displacement: float = 2e-4,
    bottom_displacement: float = 0.0,
    pin_node: int | None = None,
) -> FEMAnalysis:
    """Run Module 10 directly from an in-memory pure mesh.

    The function has no dataset or sample dependency.  It first assembles the
    stiffness matrix, then solves ``K u = f``, and finally recovers strain and
    stress from the solved displacement.
    """
    if behavior == "plane-stress":
        constitutive = plane_stress_matrix(young_modulus, poisson_ratio)
    elif behavior == "plane-strain":
        constitutive = plane_strain_matrix(young_modulus, poisson_ratio)
    else:
        raise ValueError("behavior must be 'plane-stress' or 'plane-strain'")
    system = assemble_stiffness(points, triangles, constitutive, thickness)
    boundary = vertical_tension_boundary(
        system.points,
        top_displacement=top_displacement,
        bottom_displacement=bottom_displacement,
        pin_node=pin_node,
    )
    solution = solve_linear_system(system, boundary)
    fields = recover_fields(system, solution.displacement)
    return FEMAnalysis(system, boundary, solution, fields)


__all__ = ["FEMAnalysis", "run_linear_elastic_analysis"]
