"""
Distance metrics and nearest neighbor operations for spatial tissue analysis.

This module provides efficient spatial distance computations using scipy's
cKDTree for O(log n) queries. Supports multiple distance metrics and both
radius-based and k-nearest neighbor queries.

References
----------
.. [1] Bentley, J. L. (1975). Multidimensional binary search trees used for 
       associative searching. Communications of the ACM.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, TYPE_CHECKING
from enum import Enum
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


class DistanceMetric(Enum):
    """Supported distance metrics."""
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'cityblock'
    CHEBYSHEV = 'chebyshev'
    MINKOWSKI = 'minkowski'


# -----------------------------------------------------------------------------
# Distance Matrix Computations
# -----------------------------------------------------------------------------

def pairwise_distances(
    coordinates: np.ndarray,
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """
    Compute pairwise distances between all points.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    metric : str, default 'euclidean'
        Distance metric. Supported: 'euclidean', 'manhattan', 'chebyshev', 
        'minkowski', or any metric supported by scipy.spatial.distance.cdist.
    **kwargs
        Additional arguments passed to cdist (e.g., p for Minkowski).

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_points, n_points).

    Notes
    -----
    For large datasets (>10,000 points), consider using sparse representations
    or query-based methods instead of full distance matrices.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [0, 1]])
    >>> pairwise_distances(coords)
    array([[0.        , 1.        , 1.        ],
           [1.        , 0.        , 1.41421356],
           [1.        , 1.41421356, 0.        ]])
    """
    # Map manhattan to cityblock for scipy
    if metric == 'manhattan':
        metric = 'cityblock'
    return cdist(coordinates, coordinates, metric=metric, **kwargs)


def pairwise_distances_between(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """
    Compute pairwise distances between two sets of points.

    Parameters
    ----------
    coords_a : np.ndarray
        First set of coordinates, shape (n_a, n_dims).
    coords_b : np.ndarray
        Second set of coordinates, shape (n_b, n_dims).
    metric : str, default 'euclidean'
        Distance metric.
    **kwargs
        Additional arguments passed to cdist.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_a, n_b).

    Examples
    --------
    >>> a = np.array([[0, 0], [1, 1]])
    >>> b = np.array([[2, 0], [0, 2]])
    >>> pairwise_distances_between(a, b)
    array([[2.        , 2.        ],
           [1.41421356, 1.41421356]])
    """
    # Map manhattan to cityblock for scipy
    if metric == 'manhattan':
        metric = 'cityblock'
    return cdist(coords_a, coords_b, metric=metric, **kwargs)


def condensed_distances(
    coordinates: np.ndarray,
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """
    Compute condensed (upper triangular) distance vector.

    More memory-efficient than full distance matrix for large datasets.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    metric : str, default 'euclidean'
        Distance metric.
    **kwargs
        Additional arguments passed to pdist.

    Returns
    -------
    np.ndarray
        Condensed distance vector of length n*(n-1)/2.

    Notes
    -----
    Use scipy.spatial.distance.squareform to convert to full matrix if needed.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [0, 1]])
    >>> condensed_distances(coords)
    array([1.        , 1.        , 1.41421356])
    """
    # Map manhattan to cityblock for scipy
    if metric == 'manhattan':
        metric = 'cityblock'
    return pdist(coordinates, metric=metric, **kwargs)


# -----------------------------------------------------------------------------
# Nearest Neighbor Queries
# -----------------------------------------------------------------------------

def build_kdtree(
    coordinates: np.ndarray,
    leafsize: int = 16
) -> cKDTree:
    """
    Build a KD-tree for efficient spatial queries.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    leafsize : int, default 16
        Number of points at which to switch to brute-force search.

    Returns
    -------
    cKDTree
        KD-tree for spatial queries.

    Notes
    -----
    KD-trees provide O(log n) query time for nearest neighbor searches,
    compared to O(n) for brute force.

    Examples
    --------
    >>> coords = np.random.rand(1000, 2)
    >>> tree = build_kdtree(coords)
    >>> distances, indices = tree.query([0.5, 0.5], k=5)
    """
    return cKDTree(coordinates, leafsize=leafsize)


def nearest_neighbors(
    coordinates: np.ndarray,
    k: int = 5,
    include_self: bool = False,
    return_distances: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Find k nearest neighbors for all points.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    k : int, default 5
        Number of neighbors to find.
    include_self : bool, default False
        Whether to include each point as its own neighbor.
    return_distances : bool, default True
        Whether to return distances along with indices.

    Returns
    -------
    distances : np.ndarray, optional
        Distances to neighbors of shape (n_points, k). Only if return_distances=True.
    indices : np.ndarray
        Neighbor indices of shape (n_points, k).

    Examples
    --------
    >>> coords = np.random.rand(100, 2)
    >>> distances, indices = nearest_neighbors(coords, k=5)
    >>> indices.shape
    (100, 5)
    """
    tree = build_kdtree(coordinates)
    
    # Query k+1 if excluding self (first neighbor is self)
    k_query = k if include_self else k + 1
    k_query = min(k_query, len(coordinates))
    
    distances, indices = tree.query(coordinates, k=k_query)
    
    # Handle 1D case
    if k_query == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    
    # Remove self if needed
    if not include_self and k_query > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    if return_distances:
        return distances, indices
    return indices


def radius_neighbors(
    coordinates: np.ndarray,
    radius: float,
    return_distances: bool = False,
    sort_results: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Find all neighbors within a radius for all points.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    radius : float
        Search radius.
    return_distances : bool, default False
        Whether to return distances along with indices.
    sort_results : bool, default False
        Whether to sort neighbors by distance.

    Returns
    -------
    indices : list of np.ndarray
        List where indices[i] contains neighbor indices for point i.
    distances : list of np.ndarray, optional
        List of distances. Only if return_distances=True.

    Notes
    -----
    Returns variable-length arrays since different points may have
    different numbers of neighbors within the radius.

    Examples
    --------
    >>> coords = np.array([[0, 0], [0.5, 0], [2, 0]])
    >>> indices = radius_neighbors(coords, radius=1.0)
    >>> len(indices[0])  # Point 0 has 2 neighbors within radius 1
    2
    """
    tree = build_kdtree(coordinates)
    indices_list = tree.query_ball_tree(tree, radius)
    
    # Remove self from each neighborhood
    for i, neighbors in enumerate(indices_list):
        if i in neighbors:
            neighbors.remove(i)
    
    # Convert to numpy arrays
    indices = [np.array(idx, dtype=int) for idx in indices_list]
    
    if return_distances or sort_results:
        distances = []
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                dists = np.linalg.norm(
                    coordinates[idx] - coordinates[i], axis=1
                )
                if sort_results:
                    order = np.argsort(dists)
                    indices[i] = idx[order]
                    dists = dists[order]
                distances.append(dists)
            else:
                distances.append(np.array([], dtype=float))
        
        if return_distances:
            return distances, indices
    
    return indices


def nearest_neighbor_distances(
    coordinates: np.ndarray,
    k: int = 1
) -> np.ndarray:
    """
    Compute distance to k-th nearest neighbor for all points.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).
    k : int, default 1
        Which neighbor (1 = nearest, 2 = second nearest, etc.).

    Returns
    -------
    np.ndarray
        Distances to k-th nearest neighbor for each point.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [3, 0]])
    >>> nearest_neighbor_distances(coords, k=1)
    array([1., 1., 2.])
    """
    distances, _ = nearest_neighbors(coordinates, k=k, include_self=False)
    return distances[:, -1]  # k-th column


def mean_nearest_neighbor_distance(
    coordinates: np.ndarray,
    k: int = 1
) -> float:
    """
    Compute mean distance to k-th nearest neighbor.

    This is a common spatial statistic for assessing point pattern regularity.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    k : int, default 1
        Which neighbor.

    Returns
    -------
    float
        Mean distance to k-th nearest neighbor.

    Notes
    -----
    Lower values indicate clustered patterns; higher values indicate
    regular/dispersed patterns (relative to random expectations).

    References
    ----------
    .. [1] Clark, P. J., & Evans, F. C. (1954). Distance to nearest neighbor
           as a measure of spatial relationships in populations. Ecology.
    """
    return float(np.mean(nearest_neighbor_distances(coordinates, k)))


# -----------------------------------------------------------------------------
# Distance to Cell Types
# -----------------------------------------------------------------------------

def distance_to_type(
    data: 'SpatialTissueData',
    target_type: str,
    from_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute distance from each cell to nearest cell of a target type.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    target_type : str
        Target cell type.
    from_indices : np.ndarray, optional
        Compute distances only for these cell indices. Default: all cells.

    Returns
    -------
    np.ndarray
        Distance to nearest cell of target type. Shape (n_cells,) or
        (len(from_indices),) if from_indices provided.

    Examples
    --------
    >>> # Distance from all cells to nearest tumor cell
    >>> distances = distance_to_type(data, 'Tumor')
    >>> 
    >>> # Distance from T cells to nearest tumor cell
    >>> t_cell_idx = data.get_cells_by_type('T_cell')
    >>> distances = distance_to_type(data, 'Tumor', from_indices=t_cell_idx)
    """
    target_idx = data.get_cells_by_type(target_type)
    
    if len(target_idx) == 0:
        raise ValueError(f"No cells of type '{target_type}' found")
    
    target_coords = data._coordinates[target_idx]
    target_tree = build_kdtree(target_coords)
    
    if from_indices is None:
        query_coords = data._coordinates
    else:
        query_coords = data._coordinates[from_indices]
    
    distances, _ = target_tree.query(query_coords, k=1)
    return np.asarray(distances)


def distance_to_nearest_different_type(
    data: 'SpatialTissueData'
) -> np.ndarray:
    """
    Compute distance from each cell to nearest cell of a different type.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.

    Returns
    -------
    np.ndarray
        Distance to nearest different-type cell for each cell.

    Notes
    -----
    Useful for detecting interface regions where different cell types meet.
    """
    n_cells = data.n_cells
    distances = np.full(n_cells, np.inf)
    
    for cell_type in data.cell_types_unique:
        # Get indices of this type and other types
        this_type_idx = data.get_cells_by_type(cell_type)
        other_idx = np.where(data._cell_types != cell_type)[0]
        
        if len(other_idx) == 0:
            continue
        
        # Build tree for other types
        other_tree = build_kdtree(data._coordinates[other_idx])
        
        # Query from this type to others
        this_coords = data._coordinates[this_type_idx]
        dists, _ = other_tree.query(this_coords, k=1)
        
        distances[this_type_idx] = dists
    
    return distances


def distance_matrix_by_type(
    data: 'SpatialTissueData',
    metric: str = 'mean'
) -> Dict[Tuple[str, str], float]:
    """
    Compute summary distance statistics between all cell type pairs.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    metric : str, default 'mean'
        Summary statistic: 'mean', 'median', 'min', or 'max'.

    Returns
    -------
    dict
        Dictionary mapping (type_a, type_b) to distance statistic.

    Examples
    --------
    >>> dist_matrix = distance_matrix_by_type(data, metric='mean')
    >>> dist_matrix[('T_cell', 'Tumor')]
    45.2
    """
    agg_func = {
        'mean': np.mean,
        'median': np.median,
        'min': np.min,
        'max': np.max,
    }.get(metric)
    
    if agg_func is None:
        raise ValueError(f"Unknown metric: {metric}. Use mean, median, min, max.")
    
    result = {}
    cell_types = data.cell_types_unique
    
    for type_a in cell_types:
        idx_a = data.get_cells_by_type(type_a)
        coords_a = data._coordinates[idx_a]
        
        for type_b in cell_types:
            idx_b = data.get_cells_by_type(type_b)
            coords_b = data._coordinates[idx_b]
            
            if type_a == type_b:
                # Exclude self-distances
                if len(idx_a) > 1:
                    dists = pdist(coords_a)
                    result[(type_a, type_b)] = float(agg_func(dists))
                else:
                    result[(type_a, type_b)] = np.nan
            else:
                dists = cdist(coords_a, coords_b)
                result[(type_a, type_b)] = float(agg_func(dists))
    
    return result


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the centroid (mean position) of points.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates of shape (n_points, n_dims).

    Returns
    -------
    np.ndarray
        Centroid coordinates of shape (n_dims,).
    """
    return np.mean(coordinates, axis=0)


def centroid_by_type(data: 'SpatialTissueData') -> Dict[str, np.ndarray]:
    """
    Compute centroid for each cell type.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.

    Returns
    -------
    dict
        Dictionary mapping cell type to centroid coordinates.
    """
    return {
        ct: centroid(data._coordinates[data.get_cells_by_type(ct)])
        for ct in data.cell_types_unique
    }


def bounding_box(
    coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.

    Returns
    -------
    min_coords : np.ndarray
        Minimum coordinates for each dimension.
    max_coords : np.ndarray
        Maximum coordinates for each dimension.
    """
    return coordinates.min(axis=0), coordinates.max(axis=0)


def convex_hull_area(coordinates: np.ndarray) -> float:
    """
    Compute the area of the convex hull of points (2D).

    Parameters
    ----------
    coordinates : np.ndarray
        2D point coordinates of shape (n_points, 2).

    Returns
    -------
    float
        Convex hull area.

    Notes
    -----
    Requires at least 3 non-collinear points.
    """
    from scipy.spatial import ConvexHull
    
    if coordinates.shape[1] != 2:
        raise ValueError("convex_hull_area requires 2D coordinates")
    if coordinates.shape[0] < 3:
        return 0.0
    
    try:
        hull = ConvexHull(coordinates)
        return float(hull.volume)  # In 2D, 'volume' is area
    except Exception:
        return 0.0


def point_density(
    coordinates: np.ndarray,
    method: str = 'bounding_box'
) -> float:
    """
    Compute point density (points per unit area).

    Parameters
    ----------
    coordinates : np.ndarray
        Point coordinates.
    method : str, default 'bounding_box'
        Area calculation method: 'bounding_box' or 'convex_hull'.

    Returns
    -------
    float
        Point density.
    """
    n_points = len(coordinates)
    
    if method == 'bounding_box':
        min_c, max_c = bounding_box(coordinates)
        ranges = max_c - min_c
        area = np.prod(ranges[ranges > 0])  # Handle 1D case
    elif method == 'convex_hull':
        area = convex_hull_area(coordinates)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if area == 0:
        return np.inf
    return n_points / area
