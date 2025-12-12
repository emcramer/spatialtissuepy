"""
Filter functions for Mapper algorithm.

This module provides standard (non-spatial) filter functions for the Mapper
algorithm. For spatial-aware filters, see spatial_filters.py.

Filter Function Protocol
------------------------
All filter functions must accept:
    - coordinates: np.ndarray of shape (n_cells, n_dims)
    - neighborhoods: np.ndarray of shape (n_cells, n_cell_types)
    - data: SpatialTissueData object

And return:
    - np.ndarray of shape (n_cells,) with filter values

Available Filters
-----------------
density_filter : Local point density
pca_filter : Projection onto principal components
eccentricity_filter : Distance from data centroid
linfinity_centrality_filter : Maximum distance to any other point
sum_filter : Sum of neighborhood vector (total neighbors)
entropy_filter : Shannon entropy of neighborhood composition
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData

# Type alias for filter functions
FilterFunction = Callable[[np.ndarray, np.ndarray, 'SpatialTissueData'], np.ndarray]


def density_filter(
    radius: float = 50.0,
    normalize: bool = True
) -> FilterFunction:
    """
    Create a density-based filter function.
    
    Computes local cell density as the number of cells within a given radius.
    
    Parameters
    ----------
    radius : float, default 50.0
        Radius for density calculation (in coordinate units).
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes local density.
    
    Examples
    --------
    >>> filter_fn = density_filter(radius=100)
    >>> values = filter_fn(coordinates, neighborhoods, data)
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        tree = cKDTree(coordinates)
        # Count neighbors within radius for each point
        counts = np.array([
            len(tree.query_ball_point(coord, radius)) - 1  # Exclude self
            for coord in coordinates
        ], dtype=float)
        
        if normalize and counts.max() > counts.min():
            counts = (counts - counts.min()) / (counts.max() - counts.min())
        
        return counts
    
    return _filter


def pca_filter(
    n_components: int = 1,
    component_index: int = 0
) -> FilterFunction:
    """
    Create a PCA-based filter function.
    
    Projects neighborhood vectors onto principal components.
    
    Parameters
    ----------
    n_components : int, default 1
        Number of PCA components to compute.
    component_index : int, default 0
        Which component to use as filter (0 = first PC).
    
    Returns
    -------
    FilterFunction
        Filter function that computes PCA projection.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        from sklearn.decomposition import PCA
        
        # Handle edge case of single-column neighborhoods
        if neighborhoods.shape[1] <= n_components:
            # Can't do PCA, just return first column or zeros
            if neighborhoods.shape[1] > component_index:
                return neighborhoods[:, component_index]
            return np.zeros(len(neighborhoods))
        
        pca = PCA(n_components=min(n_components, neighborhoods.shape[1]))
        transformed = pca.fit_transform(neighborhoods)
        
        if component_index >= transformed.shape[1]:
            raise ValueError(
                f"component_index {component_index} >= n_components {transformed.shape[1]}"
            )
        
        return transformed[:, component_index]
    
    return _filter


def eccentricity_filter(normalize: bool = True) -> FilterFunction:
    """
    Create an eccentricity-based filter function.
    
    Computes distance from each point to the centroid of all points
    in the neighborhood feature space.
    
    Parameters
    ----------
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes eccentricity.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        centroid = neighborhoods.mean(axis=0)
        distances = np.linalg.norm(neighborhoods - centroid, axis=1)
        
        if normalize and distances.max() > distances.min():
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        return distances
    
    return _filter


def linfinity_centrality_filter(
    sample_size: Optional[int] = None,
    normalize: bool = True
) -> FilterFunction:
    """
    Create an L-infinity centrality filter function.
    
    Computes the maximum distance to any other point in neighborhood space.
    For large datasets, samples a subset for efficiency.
    
    Parameters
    ----------
    sample_size : int, optional
        Number of points to sample for distance computation.
        If None, uses all points (can be slow for large datasets).
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes L-infinity centrality.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        n_points = len(neighborhoods)
        
        if sample_size is not None and sample_size < n_points:
            # Sample indices for efficiency
            sample_idx = np.random.choice(n_points, sample_size, replace=False)
            sample_points = neighborhoods[sample_idx]
        else:
            sample_points = neighborhoods
        
        # Compute max distance to sampled points for each point
        max_distances = np.zeros(n_points)
        for i, point in enumerate(neighborhoods):
            distances = np.linalg.norm(sample_points - point, axis=1)
            max_distances[i] = distances.max()
        
        if normalize and max_distances.max() > max_distances.min():
            max_distances = (
                (max_distances - max_distances.min()) / 
                (max_distances.max() - max_distances.min())
            )
        
        return max_distances
    
    return _filter


def sum_filter(normalize: bool = True) -> FilterFunction:
    """
    Create a sum-based filter function.
    
    Computes the sum of neighborhood vector components (total neighbor count
    if neighborhoods are counts, or 1.0 if normalized).
    
    Parameters
    ----------
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes neighborhood sum.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        sums = neighborhoods.sum(axis=1)
        
        if normalize and sums.max() > sums.min():
            sums = (sums - sums.min()) / (sums.max() - sums.min())
        
        return sums
    
    return _filter


def entropy_filter(normalize: bool = True) -> FilterFunction:
    """
    Create an entropy-based filter function.
    
    Computes Shannon entropy of neighborhood composition, measuring
    diversity of cell types in each cell's neighborhood.
    
    Parameters
    ----------
    normalize : bool, default True
        If True, normalize by maximum possible entropy.
    
    Returns
    -------
    FilterFunction
        Filter function that computes neighborhood entropy.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        # Normalize rows to proportions
        row_sums = neighborhoods.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        props = neighborhoods / row_sums
        
        # Compute entropy: -sum(p * log(p))
        # Handle zeros by using np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            log_props = np.where(props > 0, np.log(props), 0)
        entropy = -np.sum(props * log_props, axis=1)
        
        if normalize:
            # Maximum entropy is log(n_categories)
            max_entropy = np.log(neighborhoods.shape[1])
            if max_entropy > 0:
                entropy = entropy / max_entropy
        
        return entropy
    
    return _filter


def constant_filter(value: float = 0.0) -> FilterFunction:
    """
    Create a constant filter function (useful for testing).
    
    Parameters
    ----------
    value : float, default 0.0
        Constant value to return for all points.
    
    Returns
    -------
    FilterFunction
        Filter function that returns constant values.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        return np.full(len(coordinates), value)
    
    return _filter
