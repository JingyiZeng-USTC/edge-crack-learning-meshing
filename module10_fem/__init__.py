"""Sparse 2-D linear-elastic FEM for the generated crack meshes."""

from .assembly import FEMSystem, assemble_stiffness
from .boundary import BoundaryConditions, vertical_tension_boundary
from .material import plane_stress_matrix, plane_strain_matrix
from .pipeline import FEMAnalysis, run_linear_elastic_analysis
from .recovery import FieldResults, recover_fields
from .solver import Solution, solve_linear_system
from .stage_io import (
    load_fields,
    load_solution,
    load_system,
    save_fields,
    save_solution,
    save_system,
    system_identifier,
)

__all__ = [
    "BoundaryConditions",
    "FEMSystem",
    "FieldResults",
    "FEMAnalysis",
    "Solution",
    "assemble_stiffness",
    "plane_stress_matrix",
    "plane_strain_matrix",
    "recover_fields",
    "run_linear_elastic_analysis",
    "solve_linear_system",
    "load_fields",
    "load_solution",
    "load_system",
    "save_fields",
    "save_solution",
    "save_system",
    "system_identifier",
    "vertical_tension_boundary",
]
