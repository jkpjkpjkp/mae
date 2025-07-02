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

if __name__ == "__main__":
    points, triangles = read_stl_file("data/A6/11.stl")
    print(len(points))
    print(len(triangles))
    print(points)
    print(triangles)
    print(points[triangles[0]])