import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
from typing import Tuple, List


def read_stl_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """returns: (unique_vertices, triangles_vertex_indices)"""
    stl_mesh = mesh.Mesh.from_file(filename)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
    triangles = inverse_indices.reshape(-1, 3)
    return unique_vertices, triangles


def center_of_mass(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """returns: center of mass coordinates [x, y, z]"""
    total_volume = 0.0
    weighted_centroid = np.zeros(3)
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        volume_contribution = np.dot(v0, np.cross(v1, v2)) / 6.0
        tetrahedron_centroid = (v0 + v1 + v2) / 4.0
        total_volume += volume_contribution
        weighted_centroid += volume_contribution * tetrahedron_centroid
    return weighted_centroid / total_volume