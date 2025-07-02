import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional, Union
import healpy as hp
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb

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


def ray_triangle_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, 
                            triangle_vertices: np.ndarray, epsilon: float = 1e-8) -> Optional[float]:
    """returns distance along ray to intersection, or None if no intersection."""
    assert np.isclose(np.linalg.norm(ray_direction), 1), "ray_direction must be a unit vector"
    v0, v1, v2 = triangle_vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return None
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if u < 0.0 or v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        return t
    return None

# [testing]
def surface_distances_healpix(vertices: np.ndarray, triangles: np.ndarray, 
                              com: np.ndarray, nside: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert nside == 2**int(np.log2(nside)), "nside must be a power of 2"
    npix = hp.nside2npix(nside)
    pixel_indices = np.arange(npix)
    pixel_directions = np.array(hp.pix2vec(nside, pixel_indices, nest=True)).T
    assert np.all(np.isclose(np.linalg.norm(pixel_directions, axis=1), 1)), "hp.pix2vec must be unit vectors"

    pixel_distances = np.zeros(npix)
    
    for i, ray_dir in enumerate(pixel_directions):
        for triangle_indices in triangles:
            triangle_verts = vertices[triangle_indices]
            distance = ray_triangle_intersection(com, ray_dir, triangle_verts)
            
            if distance is not None:
                pixel_distances[i] = distance
                break

    assert np.all(pixel_distances > 0), "No intersection found for some pixels"

    return pixel_directions, pixel_distances, pixel_indices