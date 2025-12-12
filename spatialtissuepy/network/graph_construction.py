"""
Graph construction methods for spatial tissue data.

Provides multiple algorithms for constructing cell graphs from spatial coordinates:
- Proximity (radius-based): Connect cells within a distance threshold
- k-Nearest Neighbors (kNN): Connect each cell to its k nearest neighbors
- Delaunay Triangulation: Natural tessellation based on Voronoi regions
- Gabriel Graph: Subset of Delaunay where edge sphere contains no other points
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np
from scipy.spatial import cKDTree, Delaunay

if TYPE_CHECKING:
    import networkx as nx

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GraphMethod(Enum):
    """Graph construction methods."""
    PROXIMITY = 'proximity'
    KNN = 'knn'
    DELAUNAY = 'delaunay'
    GABRIEL = 'gabriel'


def _check_networkx():
    """Raise ImportError if networkx is not available."""
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for graph operations. "
            "Install with: pip install networkx"
        )


def build_proximity_graph(
    coordinates: np.ndarray,
    radius: float,
    cell_types: Optional[np.ndarray] = None,
    cell_ids: Optional[np.ndarray] = None,
) -> 'nx.Graph':
    """
    Build a graph connecting cells within a distance threshold.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates, shape (n_cells, n_dims).
    radius : float
        Maximum distance for edge creation.
    cell_types : np.ndarray, optional
        Cell type labels for node attributes.
    cell_ids : np.ndarray, optional
        Cell identifiers. If None, uses integer indices.
    
    Returns
    -------
    nx.Graph
        Graph with cells as nodes and proximity edges.
    """
    _check_networkx()
    
    n_cells = len(coordinates)
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_cells):
        node_attrs = {'pos': tuple(coordinates[i])}
        if cell_types is not None:
            node_attrs['cell_type'] = cell_types[i]
        if cell_ids is not None:
            node_attrs['cell_id'] = cell_ids[i]
        G.add_node(i, **node_attrs)
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(coordinates)
    
    # Find all pairs within radius
    pairs = tree.query_pairs(radius)
    
    # Add edges with distance as weight
    for i, j in pairs:
        dist = np.linalg.norm(coordinates[i] - coordinates[j])
        G.add_edge(i, j, weight=dist, distance=dist)
    
    return G


def build_knn_graph(
    coordinates: np.ndarray,
    k: int,
    cell_types: Optional[np.ndarray] = None,
    cell_ids: Optional[np.ndarray] = None,
    mutual: bool = False,
) -> 'nx.Graph':
    """
    Build a k-nearest neighbor graph.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates, shape (n_cells, n_dims).
    k : int
        Number of nearest neighbors per cell.
    cell_types : np.ndarray, optional
        Cell type labels for node attributes.
    cell_ids : np.ndarray, optional
        Cell identifiers.
    mutual : bool, default False
        If True, only include edges where both nodes are in each other's
        k-nearest neighbors (mutual kNN graph).
    
    Returns
    -------
    nx.Graph
        k-NN graph.
    """
    _check_networkx()
    
    n_cells = len(coordinates)
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_cells):
        node_attrs = {'pos': tuple(coordinates[i])}
        if cell_types is not None:
            node_attrs['cell_type'] = cell_types[i]
        if cell_ids is not None:
            node_attrs['cell_id'] = cell_ids[i]
        G.add_node(i, **node_attrs)
    
    # Build KD-tree
    tree = cKDTree(coordinates)
    
    # Query k+1 neighbors (first is self)
    distances, indices = tree.query(coordinates, k=min(k + 1, n_cells))
    
    if mutual:
        # Build neighbor sets for mutual check
        neighbor_sets = [set(indices[i, 1:]) for i in range(n_cells)]
        
        for i in range(n_cells):
            for j_idx, j in enumerate(indices[i, 1:]):
                if i in neighbor_sets[j]:  # Mutual neighbors
                    dist = distances[i, j_idx + 1]
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=dist, distance=dist)
    else:
        # Standard kNN (asymmetric made symmetric)
        for i in range(n_cells):
            for j_idx, j in enumerate(indices[i, 1:]):
                dist = distances[i, j_idx + 1]
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=dist, distance=dist)
    
    return G


def build_delaunay_graph(
    coordinates: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    cell_ids: Optional[np.ndarray] = None,
    max_edge_length: Optional[float] = None,
) -> 'nx.Graph':
    """
    Build a Delaunay triangulation graph.
    
    The Delaunay triangulation connects cells such that no cell lies
    inside the circumcircle of any triangle.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates, shape (n_cells, 2) for 2D.
    cell_types : np.ndarray, optional
        Cell type labels for node attributes.
    cell_ids : np.ndarray, optional
        Cell identifiers.
    max_edge_length : float, optional
        Maximum edge length to include. Useful for removing long edges
        at tissue boundaries.
    
    Returns
    -------
    nx.Graph
        Delaunay triangulation graph.
    """
    _check_networkx()
    
    n_cells = len(coordinates)
    
    if coordinates.shape[1] > 2:
        # Use only x, y for 2D Delaunay
        coords_2d = coordinates[:, :2]
    else:
        coords_2d = coordinates
    
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_cells):
        node_attrs = {'pos': tuple(coordinates[i])}
        if cell_types is not None:
            node_attrs['cell_type'] = cell_types[i]
        if cell_ids is not None:
            node_attrs['cell_id'] = cell_ids[i]
        G.add_node(i, **node_attrs)
    
    # Compute Delaunay triangulation
    if n_cells < 3:
        return G  # Not enough points for triangulation
    
    try:
        tri = Delaunay(coords_2d)
    except Exception:
        # Fall back to proximity if Delaunay fails (e.g., collinear points)
        return G
    
    # Extract edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    
    # Add edges
    for i, j in edges:
        dist = np.linalg.norm(coordinates[i] - coordinates[j])
        
        if max_edge_length is not None and dist > max_edge_length:
            continue
        
        G.add_edge(i, j, weight=dist, distance=dist)
    
    return G


def build_gabriel_graph(
    coordinates: np.ndarray,
    cell_types: Optional[np.ndarray] = None,
    cell_ids: Optional[np.ndarray] = None,
) -> 'nx.Graph':
    """
    Build a Gabriel graph.
    
    The Gabriel graph is a subgraph of the Delaunay triangulation where
    an edge (i, j) exists only if no other point lies within the circle
    having (i, j) as diameter.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates, shape (n_cells, n_dims).
    cell_types : np.ndarray, optional
        Cell type labels for node attributes.
    cell_ids : np.ndarray, optional
        Cell identifiers.
    
    Returns
    -------
    nx.Graph
        Gabriel graph.
    """
    _check_networkx()
    
    n_cells = len(coordinates)
    
    # Start with Delaunay graph
    G = build_delaunay_graph(coordinates, cell_types, cell_ids)
    
    if n_cells < 3:
        return G
    
    # Build KD-tree for point queries
    tree = cKDTree(coordinates)
    
    # Check each edge for Gabriel property
    edges_to_remove = []
    
    for i, j in G.edges():
        # Midpoint of edge
        midpoint = (coordinates[i] + coordinates[j]) / 2
        
        # Radius of circle with (i, j) as diameter
        radius = np.linalg.norm(coordinates[i] - coordinates[j]) / 2
        
        # Find points within this circle
        points_in_circle = tree.query_ball_point(midpoint, radius - 1e-10)
        
        # Remove i and j from the list
        other_points = [p for p in points_in_circle if p != i and p != j]
        
        if len(other_points) > 0:
            edges_to_remove.append((i, j))
    
    G.remove_edges_from(edges_to_remove)
    
    return G


def build_graph(
    coordinates: np.ndarray,
    method: Union[str, GraphMethod] = 'proximity',
    cell_types: Optional[np.ndarray] = None,
    cell_ids: Optional[np.ndarray] = None,
    radius: float = 50.0,
    k: int = 6,
    mutual_knn: bool = False,
    max_edge_length: Optional[float] = None,
) -> 'nx.Graph':
    """
    Build a cell graph using the specified method.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cell coordinates, shape (n_cells, n_dims).
    method : str or GraphMethod, default 'proximity'
        Graph construction method:
        - 'proximity': Connect cells within radius
        - 'knn': k-nearest neighbors
        - 'delaunay': Delaunay triangulation
        - 'gabriel': Gabriel graph (subset of Delaunay)
    cell_types : np.ndarray, optional
        Cell type labels for node attributes.
    cell_ids : np.ndarray, optional
        Cell identifiers.
    radius : float, default 50.0
        Radius for proximity graph.
    k : int, default 6
        Number of neighbors for kNN graph.
    mutual_knn : bool, default False
        Use mutual kNN (both nodes must be in each other's k-NN).
    max_edge_length : float, optional
        Maximum edge length (for Delaunay graph pruning).
    
    Returns
    -------
    nx.Graph
        Cell graph.
    
    Examples
    --------
    >>> G = build_graph(coordinates, method='proximity', radius=30)
    >>> G = build_graph(coordinates, method='knn', k=8)
    >>> G = build_graph(coordinates, method='delaunay')
    >>> G = build_graph(coordinates, method='gabriel')
    """
    if isinstance(method, str):
        method = GraphMethod(method.lower())
    
    if method == GraphMethod.PROXIMITY:
        return build_proximity_graph(
            coordinates, radius, cell_types, cell_ids
        )
    elif method == GraphMethod.KNN:
        return build_knn_graph(
            coordinates, k, cell_types, cell_ids, mutual=mutual_knn
        )
    elif method == GraphMethod.DELAUNAY:
        return build_delaunay_graph(
            coordinates, cell_types, cell_ids, max_edge_length
        )
    elif method == GraphMethod.GABRIEL:
        return build_gabriel_graph(
            coordinates, cell_types, cell_ids
        )
    else:
        raise ValueError(f"Unknown graph method: {method}")
