"""Mesh loading and FEM result export."""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from .assembly import FEMSystem
from .boundary import BoundaryConditions
from .recovery import FieldResults
from .solver import Solution


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load points/triangles from .inp, .msh, .vtk/.vtu or project .npz."""
    source = Path(path)
    if source.suffix.lower() == ".npz":
        data = np.load(source, allow_pickle=True)
        points = np.asarray(data["points"], dtype=float)[:, :2]
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        metadata = {k: data[k] for k in data.files if k not in {"points", "triangles"}}
        return points, triangles, metadata

    mesh = meshio.read(source)
    blocks = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type in {"triangle", "triangle3"}]
    if not blocks:
        raise ValueError(f"No three-node triangle cells found in {source}.")
    return np.asarray(mesh.points, dtype=float)[:, :2], np.vstack(blocks), {
        "point_sets": mesh.point_sets,
        "cell_sets": mesh.cell_sets,
    }


def save_npz(
    path: str | Path,
    system: FEMSystem,
    bc: BoundaryConditions,
    solution: Solution,
    fields: FieldResults,
) -> None:
    np.savez_compressed(
        path,
        points=system.points,
        triangles=system.triangles,
        stiffness_data=system.stiffness.data,
        stiffness_indices=system.stiffness.indices,
        stiffness_indptr=system.stiffness.indptr,
        stiffness_shape=np.asarray(system.stiffness.shape),
        constitutive=system.constitutive,
        thickness=system.thickness,
        element_areas=system.element_areas,
        element_B=system.element_B,
        element_dof_map=system.element_dof_map,
        prescribed_dofs=bc.prescribed_dofs,
        prescribed_values=bc.prescribed_values,
        force=bc.force,
        free_dofs=solution.free_dofs,
        displacement=solution.displacement_xy,
        reactions=solution.reactions,
        residual_free=solution.residual_free,
        strain_energy=solution.strain_energy,
        element_strain=fields.element_strain,
        element_stress=fields.element_stress,
        element_von_mises=fields.element_von_mises,
        nodal_strain=fields.nodal_strain,
        nodal_stress=fields.nodal_stress,
        nodal_von_mises=fields.nodal_von_mises,
    )


def save_vtu(
    path: str | Path,
    system: FEMSystem,
    solution: Solution,
    fields: FieldResults,
) -> None:
    points3d = np.column_stack((system.points, np.zeros(system.n_nodes)))
    displacement3d = np.column_stack((solution.displacement_xy, np.zeros(system.n_nodes)))
    meshio.Mesh(
        points3d,
        [("triangle", system.triangles)],
        point_data={
            "displacement": displacement3d,
            "strain": fields.nodal_strain,
            "stress": fields.nodal_stress,
            "von_mises": fields.nodal_von_mises,
        },
        cell_data={
            "strain": [fields.element_strain],
            "stress": [fields.element_stress],
            "von_mises": [fields.element_von_mises],
        },
    ).write(path)
