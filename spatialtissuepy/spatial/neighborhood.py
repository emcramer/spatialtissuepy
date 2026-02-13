"""
Neighborhood analysis for spatial tissue data.

This module provides functions for computing cell neighborhoods, neighborhood
compositions (cell type proportions), and adjacency relationships. These form
the foundation for spatial statistics and Spatial LDA analyses.

Key Concepts
------------
- **Neighborhood**: Set of cells within a specified distance of a focal cell
- **Composition**: Proportions of each cell type within a neighborhood
- **Adjacency**: Binary or weighted matrix of cell-cell spatial relationships

References
----------
.. [1] Schürch, C. M. et al. (2020). Coordinated cellular neighborhoods
       orchestrate antitumoral immunity at the colorectal cancer invasive front.
       Cell.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Dict, List, TYPE_CHECKING
from enum import Enum
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


class NeighborhoodMethod(Enum):
    """Methods for defining neighborhoods."""
    RADIUS = 'radius'
    KNN = 'knn'


# -----------------------------------------------------------------------------
# Neighborhood Computation
# -----------------------------------------------------------------------------

def compute_neighborhoods(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    include_self: bool = False,
    **kwargs
) -> List[np.ndarray]:
    """
    Compute neighborhoods for all cells.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Neighborhood method: 'radius' or 'knn'.
    radius : float, optional
        Search radius (required if method='radius').
    k : int, optional
        Number of neighbors (required if method='knn').
    include_self : bool, default False
        Whether to include the focal cell in its own neighborhood.
    **kwargs
        Additional arguments.

    Returns
    -------
    list of np.ndarray
        List where neighborhoods[i] contains neighbor indices for cell i.

    Examples
    --------
    >>> # Radius-based neighborhoods
    >>> neighborhoods = compute_neighborhoods(data, method='radius', radius=30)
    >>> 
    >>> # k-nearest neighbor neighborhoods
    >>> neighborhoods = compute_neighborhoods(data, method='knn', k=10)
    """
    tree = data.kdtree
    coords = data._coordinates
    
    if method == 'radius':
        if radius is None:
            raise ValueError("radius required for method='radius'")
        
        # Query all neighbors within radius
        indices_list = tree.query_ball_tree(tree, radius)
        
        # Process results
        neighborhoods = []
        for i, neighbors in enumerate(indices_list):
            neighbors = np.array(neighbors, dtype=int)
            if not include_self:
                neighbors = neighbors[neighbors != i]
            neighborhoods.append(neighbors)
    
    elif method == 'knn':
        if k is None:
            raise ValueError("k required for method='knn'")
        
        k_query = k if include_self else k + 1
        k_query = min(k_query, len(coords))
        
        _, indices = tree.query(coords, k=k_query)
        
        if k_query == 1:
            indices = indices.reshape(-1, 1)
        
        neighborhoods = []
        for i, neighbors in enumerate(indices):
            if not include_self:
                neighbors = neighbors[neighbors != i]
            neighborhoods.append(np.array(neighbors, dtype=int))
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'radius' or 'knn'.")
    
    return neighborhoods


def neighborhood_counts(
    data: 'SpatialTissueData',
    neighborhoods: List[np.ndarray]
) -> np.ndarray:
    """
    Compute cell type counts for each neighborhood.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    neighborhoods : list of np.ndarray
        Neighborhood indices from compute_neighborhoods().

    Returns
    -------
    np.ndarray
        Count matrix of shape (n_cells, n_cell_types).
        Columns are ordered by data.cell_types_unique.

    Examples
    --------
    >>> neighborhoods = compute_neighborhoods(data, method='radius', radius=30)
    >>> counts = neighborhood_counts(data, neighborhoods)
    >>> counts.shape
    (1000, 5)  # 1000 cells, 5 cell types
    """
    cell_types = data._cell_types
    unique_types = data.cell_types_unique
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    
    n_cells = data.n_cells
    n_types = len(unique_types)
    
    counts = np.zeros((n_cells, n_types), dtype=int)
    
    for i, neighbors in enumerate(neighborhoods):
        if len(neighbors) > 0:
            neighbor_types = cell_types[neighbors]
            for ct in neighbor_types:
                counts[i, type_to_idx[ct]] += 1
    
    return counts


def neighborhood_composition(
    data: 'SpatialTissueData',
    neighborhoods: Optional[List[np.ndarray]] = None,
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    include_self: bool = False,
    normalize: bool = True,
    pseudocount: float = 0.0,
    **kwargs
) -> np.ndarray:
    """
    Compute cell type composition (proportions) for each neighborhood.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    neighborhoods : list of np.ndarray, optional
        Precomputed neighborhoods. If None, computed using method/radius/k.
    method : str, default 'radius'
        Neighborhood method if neighborhoods not provided.
    radius : float, optional
        Search radius for radius method.
    k : int, optional
        Number of neighbors for knn method.
    include_self : bool, default False
        Whether to include the focal cell in its own neighborhood.
    normalize : bool, default True
        Whether to normalize to proportions (sum to 1).
    pseudocount : float, default 0.0
        Small value added to counts before normalization to avoid zeros.
    **kwargs
        Additional arguments passed to compute_neighborhoods.

    Returns
    -------
    np.ndarray
        Composition matrix of shape (n_cells, n_cell_types).
        If normalize=True, rows sum to 1.

    Notes
    -----
    This is the key input for Spatial LDA and other neighborhood-based analyses.

    Examples
    --------
    >>> composition = neighborhood_composition(
    ...     data, method='radius', radius=50
    ... )
    >>> composition.shape
    (1000, 5)
    >>> np.allclose(composition.sum(axis=1), 1.0)
    True
    """
    if neighborhoods is None:
        neighborhoods = compute_neighborhoods(
            data, method=method, radius=radius, k=k, include_self=include_self, **kwargs
        )
    
    counts = neighborhood_counts(data, neighborhoods)
    
    if include_self:
        # Add focal cell type to counts
        unique_types = data.cell_types_unique
        type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
        for i in range(data.n_cells):
            ct = data._cell_types[i]
            counts[i, type_to_idx[ct]] += 1
    
    if pseudocount > 0:
        counts = counts.astype(float) + pseudocount
    
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero for cells with no neighbors
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return counts / row_sums
    
    return counts.astype(float)


def window_composition(
    data: 'SpatialTissueData',
    window_size: float,
    grid_step: Optional[float] = None,
    min_cells: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cell type composition in sliding windows across the tissue.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    window_size : float
        Size of square windows (diameter in coordinate units).
    grid_step : float, optional
        Step size between window centers. Default: window_size/2 (50% overlap).
    min_cells : int, default 1
        Minimum cells required in window to include.

    Returns
    -------
    compositions : np.ndarray
        Composition matrix of shape (n_windows, n_cell_types).
    centers : np.ndarray
        Window center coordinates of shape (n_windows, 2).

    Notes
    -----
    Useful for creating ROIs for Spatial LDA or analyzing spatial gradients.

    Examples
    --------
    >>> compositions, centers = window_composition(data, window_size=100)
    """
    if grid_step is None:
        grid_step = window_size / 2
    
    bounds = data.bounds
    half_window = window_size / 2
    
    # Generate grid of window centers
    x_centers = np.arange(
        bounds['x'][0] + half_window,
        bounds['x'][1] - half_window + grid_step,
        grid_step
    )
    y_centers = np.arange(
        bounds['y'][0] + half_window,
        bounds['y'][1] - half_window + grid_step,
        grid_step
    )
    
    xx, yy = np.meshgrid(x_centers, y_centers)
    grid_centers = np.column_stack([xx.ravel(), yy.ravel()])
    
    # For each window, count cell types
    tree = data.kdtree
    unique_types = data.cell_types_unique
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    n_types = len(unique_types)
    
    compositions = []
    valid_centers = []
    
    for center in grid_centers:
        # Find cells in window (Chebyshev distance = infinity norm)
        # Use radius query with sqrt(2)*half_window for circumscribed circle
        # then filter to square window
        candidate_idx = tree.query_ball_point(center, half_window * np.sqrt(2))
        
        # Filter to actual square window
        cell_coords = data._coordinates[candidate_idx]
        in_window = np.all(np.abs(cell_coords - center) <= half_window, axis=1)
        window_idx = np.array(candidate_idx)[in_window]
        
        if len(window_idx) >= min_cells:
            counts = np.zeros(n_types)
            for idx in window_idx:
                ct = data._cell_types[idx]
                counts[type_to_idx[ct]] += 1
            
            # Normalize to proportions
            compositions.append(counts / counts.sum())
            valid_centers.append(center)
    
    return np.array(compositions), np.array(valid_centers)


# -----------------------------------------------------------------------------
# Adjacency Matrices
# -----------------------------------------------------------------------------

def adjacency_matrix(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    weighted: bool = False,
    sparse: bool = True
) -> Union[np.ndarray, csr_matrix]:
    """
    Compute cell-cell adjacency matrix.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Method: 'radius' for distance threshold, 'knn' for k-nearest neighbors.
    radius : float, optional
        Distance threshold (required for method='radius').
    k : int, optional
        Number of neighbors (required for method='knn').
    weighted : bool, default False
        If True, use inverse distance weights instead of binary adjacency.
    sparse : bool, default True
        If True, return scipy sparse matrix; else dense numpy array.

    Returns
    -------
    np.ndarray or csr_matrix
        Adjacency matrix of shape (n_cells, n_cells).

    Examples
    --------
    >>> # Binary radius-based adjacency
    >>> adj = adjacency_matrix(data, method='radius', radius=30)
    >>> 
    >>> # Distance-weighted k-NN adjacency
    >>> adj = adjacency_matrix(data, method='knn', k=10, weighted=True)
    """
    n_cells = data.n_cells
    coords = data._coordinates
    tree = data.kdtree
    
    # Use lil_matrix for efficient construction
    adj = lil_matrix((n_cells, n_cells), dtype=float)
    
    if method == 'radius':
        if radius is None:
            raise ValueError("radius required for method='radius'")
        
        pairs = tree.query_pairs(radius, output_type='ndarray')
        
        if weighted:
            for i, j in pairs:
                dist = np.linalg.norm(coords[i] - coords[j])
                w = 1.0 / max(dist, 1e-10)
                adj[i, j] = w
                adj[j, i] = w
        else:
            for i, j in pairs:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    
    elif method == 'knn':
        if k is None:
            raise ValueError("k required for method='knn'")
        
        k_query = min(k + 1, n_cells)
        distances, indices = tree.query(coords, k=k_query)
        
        if k_query == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        for i in range(n_cells):
            for j_idx in range(1, indices.shape[1]):  # Skip self (index 0)
                j = indices[i, j_idx]
                if weighted:
                    dist = distances[i, j_idx]
                    w = 1.0 / max(dist, 1e-10)
                    adj[i, j] = w
                else:
                    adj[i, j] = 1.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    adj = adj.tocsr()
    
    if sparse:
        return adj
    return adj.toarray()


def type_adjacency_matrix(
    data: 'SpatialTissueData',
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None,
    normalize: str = 'none'
) -> pd.DataFrame:
    """
    Compute adjacency counts between cell types.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    method : str, default 'radius'
        Neighborhood method.
    radius, k : float, int
        Method parameters.
    normalize : str, default 'none'
        Normalization: 'none', 'row', 'total', or 'expected'.
        - 'none': Raw edge counts
        - 'row': Each row sums to 1
        - 'total': All entries sum to 1
        - 'expected': Divide by expected count under random mixing

    Returns
    -------
    pd.DataFrame
        Type-by-type adjacency matrix.

    Examples
    --------
    >>> type_adj = type_adjacency_matrix(data, method='radius', radius=30)
    >>> print(type_adj)
                    T_cell  Tumor  Stromal
    T_cell           1234    456      789
    Tumor             456   5678      321
    Stromal           789    321     2345
    """
    neighborhoods = compute_neighborhoods(
        data, method=method, radius=radius, k=k
    )
    
    cell_types = data._cell_types
    unique_types = list(data.cell_types_unique)
    n_types = len(unique_types)
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    
    # Count type-type edges
    counts = np.zeros((n_types, n_types), dtype=float)
    
    for i, neighbors in enumerate(neighborhoods):
        type_i = type_to_idx[cell_types[i]]
        for j in neighbors:
            type_j = type_to_idx[cell_types[j]]
            counts[type_i, type_j] += 1
    
    # Normalize if requested
    if normalize == 'row':
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        counts = counts / row_sums
    elif normalize == 'total':
        total = counts.sum()
        if total > 0:
            counts = counts / total
    elif normalize == 'expected':
        # Expected under random: (n_i * n_j) / n_total * (total_edges / n_total)
        type_counts = np.array([
            np.sum(cell_types == ct) for ct in unique_types
        ])
        n_total = data.n_cells
        total_edges = counts.sum()
        
        expected = np.outer(type_counts, type_counts) / (n_total ** 2) * total_edges
        counts = counts / np.where(expected > 0, expected, 1)
    elif normalize != 'none':
        raise ValueError(f"Unknown normalize: {normalize}")
    
    return pd.DataFrame(counts, index=unique_types, columns=unique_types)


# -----------------------------------------------------------------------------
# Neighborhood Statistics
# -----------------------------------------------------------------------------

def neighborhood_size(
    neighborhoods: List[np.ndarray]
) -> np.ndarray:
    """
    Compute the number of neighbors for each cell.

    Parameters
    ----------
    neighborhoods : list of np.ndarray
        Neighborhood indices.

    Returns
    -------
    np.ndarray
        Number of neighbors for each cell.
    """
    return np.array([len(n) for n in neighborhoods])


def neighborhood_diversity(
    data: 'SpatialTissueData',
    neighborhoods: List[np.ndarray],
    metric: str = 'shannon'
) -> np.ndarray:
    """
    Compute diversity of cell types in each neighborhood.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    neighborhoods : list of np.ndarray
        Neighborhood indices.
    metric : str, default 'shannon'
        Diversity metric: 'shannon', 'simpson', or 'richness'.

    Returns
    -------
    np.ndarray
        Diversity score for each cell's neighborhood.

    Notes
    -----
    - Shannon: -sum(p * log(p)), higher = more diverse
    - Simpson: 1 - sum(p^2), higher = more diverse
    - Richness: number of unique types present
    """
    composition = neighborhood_composition(
        data, neighborhoods=neighborhoods, normalize=True
    )
    
    if metric == 'shannon':
        # Shannon entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.log(composition)
            log_p = np.where(np.isfinite(log_p), log_p, 0)
        diversity = -np.sum(composition * log_p, axis=1)
    
    elif metric == 'simpson':
        # Simpson's diversity index (1 - dominance)
        diversity = 1 - np.sum(composition ** 2, axis=1)
    
    elif metric == 'richness':
        # Number of types present
        diversity = np.sum(composition > 0, axis=1).astype(float)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return diversity


def neighborhood_enrichment(
    data: 'SpatialTissueData',
    neighborhoods: List[np.ndarray],
    target_type: str
) -> np.ndarray:
    """
    Compute enrichment of a target cell type in each neighborhood.

    Enrichment is the ratio of observed to expected proportion.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    neighborhoods : list of np.ndarray
        Neighborhood indices.
    target_type : str
        Cell type to compute enrichment for.

    Returns
    -------
    np.ndarray
        Enrichment scores (>1 = enriched, <1 = depleted).
    """
    composition = neighborhood_composition(
        data, neighborhoods=neighborhoods, normalize=True
    )
    
    # Get column index for target type
    unique_types = list(data.cell_types_unique)
    if target_type not in unique_types:
        raise ValueError(f"Unknown cell type: {target_type}")
    
    type_idx = unique_types.index(target_type)
    observed = composition[:, type_idx]
    
    # Expected proportion (global)
    expected = np.sum(data._cell_types == target_type) / data.n_cells
    
    # Enrichment = observed / expected
    return observed / max(expected, 1e-10)


def interface_cells(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float,
    min_neighbors: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find cells at the interface between two cell types.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    type_a : str
        First cell type.
    type_b : str
        Second cell type.
    radius : float
        Neighborhood radius.
    min_neighbors : int, default 1
        Minimum number of opposite-type neighbors to be considered interface.

    Returns
    -------
    type_a_interface : np.ndarray
        Indices of type_a cells at interface.
    type_b_interface : np.ndarray
        Indices of type_b cells at interface.

    Examples
    --------
    >>> tumor_interface, immune_interface = interface_cells(
    ...     data, 'Tumor', 'CD8_T_cell', radius=30
    ... )
    """
    neighborhoods = compute_neighborhoods(data, method='radius', radius=radius)
    cell_types = data._cell_types
    
    idx_a = data.get_cells_by_type(type_a)
    idx_b = data.get_cells_by_type(type_b)
    
    idx_b_set = set(idx_b)
    idx_a_set = set(idx_a)
    
    # Type A cells with >= min_neighbors of type B
    type_a_interface = []
    for i in idx_a:
        n_type_b = sum(1 for j in neighborhoods[i] if j in idx_b_set)
        if n_type_b >= min_neighbors:
            type_a_interface.append(i)
    
    # Type B cells with >= min_neighbors of type A
    type_b_interface = []
    for i in idx_b:
        n_type_a = sum(1 for j in neighborhoods[i] if j in idx_a_set)
        if n_type_a >= min_neighbors:
            type_b_interface.append(i)
    
    return np.array(type_a_interface), np.array(type_b_interface)


# -----------------------------------------------------------------------------
# Neighborhood as DataFrame
# -----------------------------------------------------------------------------

def neighborhood_to_dataframe(
    data: 'SpatialTissueData',
    neighborhoods: Optional[List[np.ndarray]] = None,
    method: str = 'radius',
    radius: Optional[float] = None,
    k: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert neighborhood composition to DataFrame with cell type columns.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    neighborhoods : list of np.ndarray, optional
        Precomputed neighborhoods.
    method, radius, k
        Parameters for computing neighborhoods if not provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each cell type proportion.
    """
    composition = neighborhood_composition(
        data, neighborhoods=neighborhoods, method=method, radius=radius, k=k
    )
    
    df = pd.DataFrame(
        composition,
        columns=data.cell_types_unique
    )
    
    # Add metadata columns
    df['cell_type'] = data._cell_types
    if data._sample_ids is not None:
        df['sample_id'] = data._sample_ids
    
    return df
