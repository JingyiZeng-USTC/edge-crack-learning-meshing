"""Per-mesh, zero-label coordinate FCNN-SM solver for variable crack meshes."""

from .features import FEATURE_NAMES, build_node_features, geometry_from_mesh_metadata
from .model import CoordinateFCNNSM

__all__ = [
    "CoordinateFCNNSM",
    "FEATURE_NAMES",
    "build_node_features",
    "geometry_from_mesh_metadata",
]

