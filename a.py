import numpy as np
from stl import mesh
from scipy.spatial import ConvexHull
from typing import Tuple, List, Optional, Union


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


# [claude] Fibonacci spiral sampling
def sample_directions(n_samples: int) -> np.ndarray:
    """returns array of shape (n_samples, 3) of unit vectors."""
    points = np.zeros((n_samples, 3))
    golden_ratio = (1 + 5**0.5) / 2
    
    # Fibonacci spiral
    for i in range(n_samples):
        # Latitude (y-coordinate)
        y = 1 - (i / (n_samples - 1)) * 2  # y goes from 1 to -1
        # Longitude 
        theta = 2 * np.pi * i / golden_ratio
        
        radius = np.sqrt(1 - y * y)
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points[i] = [x, y, z]
    
    return points

# [claude]
def ray_triangle_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, 
                            triangle_vertices: np.ndarray, epsilon: float = 1e-8) -> Optional[float]:
    """returns distance along ray to intersection, or None if no intersection."""
    v0, v1, v2 = triangle_vertices
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < epsilon:
        return None  # Ray is parallel to triangle
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return None
    
    t = f * np.dot(edge2, q)
    
    if t > epsilon:
        return t
    
    return None

# [claude]
def surface_distances_uniform_rays(vertices: np.ndarray, triangles: np.ndarray, 
                                 com: np.ndarray, n_rays: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """returns: (ray_directions, distances)

    ray_directions: array of shape (n_rays, 3) with unit direction vectors
    distances: array of shape (n_rays,) with distances to surface (0 if no intersection)
    
    com = center of mass"""
    ray_directions = sample_directions(n_rays)
    distances = np.zeros(n_rays)
    
    # TODO: parallelize this
    for i, ray_dir in enumerate(ray_directions):
        min_distance = float('inf')
        found_intersection = False
        
        # TODO: parallelize this
        for triangle_indices in triangles:
            triangle_verts = vertices[triangle_indices]
            distance = ray_triangle_intersection(com, ray_dir, triangle_verts)
            
            if distance is not None:
                min_distance = min(min_distance, distance)
                found_intersection = True
        
        distances[i] = min_distance if found_intersection else 0.0
    
    return ray_directions, distances