"""
Spatial filter functions for Mapper algorithm.

This module provides filter functions that leverage spatial coordinates
of cells, which is a key innovation for tissue biology applications.

These filters enable Mapper to capture spatial gradients and tissue
organization that would be missed by purely phenotype-based filters.

Available Spatial Filters
-------------------------
spatial_coordinate_filter : Direct x, y, or z coordinate projection
radial_filter : Distance from a reference point
distance_to_type_filter : Distance to nearest cell of a specified type
distance_to_boundary_filter : Distance from tissue/ROI edge
spatial_density_filter : Local cell density in spatial coordinates
density_gradient_filter : Direction of steepest density change
gaussian_smoothed_filter : Spatially smoothed version of any filter
composite_filter : Weighted combination of multiple filters
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData

# Type alias for filter functions
FilterFunction = Callable[[np.ndarray, np.ndarray, 'SpatialTissueData'], np.ndarray]


def spatial_coordinate_filter(
    axis: Union[str, int] = 'x',
    normalize: bool = True
) -> FilterFunction:
    """
    Create a filter based on spatial coordinates.
    
    Uses the raw x, y, or z coordinate as the filter value. This captures
    spatial gradients along tissue axes (e.g., epithelium-to-stroma transitions).
    
    Parameters
    ----------
    axis : str or int, default 'x'
        Which axis to use: 'x' (0), 'y' (1), or 'z' (2).
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that returns coordinate values.
    
    Examples
    --------
    >>> # Capture left-right gradient
    >>> filter_fn = spatial_coordinate_filter(axis='x')
    >>> 
    >>> # Capture top-bottom gradient  
    >>> filter_fn = spatial_coordinate_filter(axis='y')
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis, axis) if isinstance(axis, str) else axis
    
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        if axis_idx >= coordinates.shape[1]:
            raise ValueError(
                f"Axis {axis_idx} not available in {coordinates.shape[1]}D data"
            )
        
        values = coordinates[:, axis_idx].copy()
        
        if normalize and values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min())
        
        return values
    
    return _filter


def radial_filter(
    center: Optional[np.ndarray] = None,
    normalize: bool = True
) -> FilterFunction:
    """
    Create a radial distance filter from a reference point.
    
    Computes Euclidean distance from each cell to a reference point.
    Useful for analyzing radial organization around a tumor center,
    blood vessel, or other landmark.
    
    Parameters
    ----------
    center : np.ndarray, optional
        Reference point coordinates [x, y] or [x, y, z].
        If None, uses the centroid of all cells.
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes radial distances.
    
    Examples
    --------
    >>> # Distance from tumor center
    >>> tumor_center = np.array([500, 500])
    >>> filter_fn = radial_filter(center=tumor_center)
    >>>
    >>> # Distance from data centroid (auto-computed)
    >>> filter_fn = radial_filter()
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        ref_point = center
        if ref_point is None:
            ref_point = coordinates.mean(axis=0)
        
        ref_point = np.asarray(ref_point)
        if len(ref_point) != coordinates.shape[1]:
            raise ValueError(
                f"Center has {len(ref_point)} dims but data has {coordinates.shape[1]}"
            )
        
        distances = np.linalg.norm(coordinates - ref_point, axis=1)
        
        if normalize and distances.max() > distances.min():
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        return distances
    
    return _filter


def distance_to_type_filter(
    cell_type: str,
    normalize: bool = True,
    max_distance: Optional[float] = None
) -> FilterFunction:
    """
    Create a filter based on distance to nearest cell of a specified type.
    
    For each cell, computes the distance to the nearest cell of the target
    type. This is powerful for analyzing spatial relationships like
    immune proximity to tumor.
    
    Parameters
    ----------
    cell_type : str
        Target cell type to measure distance to.
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    max_distance : float, optional
        Maximum distance to consider. Distances beyond this are capped.
        Useful for preventing outliers from dominating normalization.
    
    Returns
    -------
    FilterFunction
        Filter function that computes distance to cell type.
    
    Examples
    --------
    >>> # Distance to nearest tumor cell
    >>> filter_fn = distance_to_type_filter('Tumor')
    >>>
    >>> # Distance to nearest T cell, capped at 500 µm
    >>> filter_fn = distance_to_type_filter('T_cell', max_distance=500)
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        # Find cells of target type
        target_mask = data._cell_types == cell_type
        
        if not np.any(target_mask):
            raise ValueError(f"No cells of type '{cell_type}' found in data")
        
        target_coords = data._coordinates[target_mask]
        
        # Build KD-tree for target cells
        target_tree = cKDTree(target_coords)
        
        # Query distance to nearest target for all cells
        distances, _ = target_tree.query(coordinates, k=1)
        
        # Cap distances if specified
        if max_distance is not None:
            distances = np.minimum(distances, max_distance)
        
        if normalize and distances.max() > distances.min():
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        return distances
    
    return _filter


def distance_to_boundary_filter(
    boundary_method: str = 'convex_hull',
    normalize: bool = True
) -> FilterFunction:
    """
    Create a filter based on distance to tissue boundary.
    
    Computes distance from each cell to the edge of the tissue region.
    Useful for identifying core vs. peripheral cells.
    
    Parameters
    ----------
    boundary_method : str, default 'convex_hull'
        Method for determining boundary:
        - 'convex_hull': Use convex hull of all points
        - 'bbox': Use bounding box
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes boundary distances.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        if boundary_method == 'bbox':
            # Distance to nearest bounding box edge
            mins = coordinates.min(axis=0)
            maxs = coordinates.max(axis=0)
            
            # For each point, find distance to nearest edge
            dist_to_min = coordinates - mins
            dist_to_max = maxs - coordinates
            distances = np.minimum(dist_to_min, dist_to_max).min(axis=1)
            
        elif boundary_method == 'convex_hull':
            from scipy.spatial import ConvexHull, Delaunay
            
            if coordinates.shape[1] != 2:
                raise ValueError("Convex hull boundary only supported for 2D data")
            
            hull = ConvexHull(coordinates)
            hull_points = coordinates[hull.vertices]
            
            # Approximate: distance to nearest hull point
            # (True distance to hull boundary is more complex)
            hull_tree = cKDTree(hull_points)
            distances, _ = hull_tree.query(coordinates, k=1)
            
            # Invert: points on boundary have 0, interior points positive
            # We want interior points to have high values
            max_dist = distances.max()
            distances = max_dist - distances
        else:
            raise ValueError(f"Unknown boundary_method: {boundary_method}")
        
        if normalize and distances.max() > distances.min():
            distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        return distances
    
    return _filter


def spatial_density_filter(
    radius: float = 50.0,
    normalize: bool = True
) -> FilterFunction:
    """
    Create a filter based on local spatial cell density.
    
    Unlike the generic density_filter (which operates on neighborhood
    feature space), this operates directly on spatial coordinates.
    
    Parameters
    ----------
    radius : float, default 50.0
        Radius for counting neighbors (in coordinate units).
    normalize : bool, default True
        If True, normalize to [0, 1] range.
    
    Returns
    -------
    FilterFunction
        Filter function that computes spatial density.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        tree = cKDTree(coordinates)
        
        # Count neighbors within radius (excluding self)
        counts = np.array([
            len(tree.query_ball_point(coord, radius)) - 1
            for coord in coordinates
        ], dtype=float)
        
        if normalize and counts.max() > counts.min():
            counts = (counts - counts.min()) / (counts.max() - counts.min())
        
        return counts
    
    return _filter


def gaussian_smoothed_filter(
    base_filter: FilterFunction,
    sigma: float = 50.0,
    n_neighbors: int = 20
) -> FilterFunction:
    """
    Create a spatially smoothed version of any filter.
    
    Applies Gaussian-weighted spatial smoothing to the output of another
    filter function, reducing noise and emphasizing regional trends.
    
    Parameters
    ----------
    base_filter : FilterFunction
        The filter function to smooth.
    sigma : float, default 50.0
        Spatial scale of Gaussian smoothing (in coordinate units).
    n_neighbors : int, default 20
        Number of nearest neighbors to use for smoothing.
    
    Returns
    -------
    FilterFunction
        Spatially smoothed filter function.
    
    Examples
    --------
    >>> # Smooth PCA filter spatially
    >>> base = pca_filter(n_components=1)
    >>> smoothed = gaussian_smoothed_filter(base, sigma=100)
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        # Get base filter values
        base_values = base_filter(coordinates, neighborhoods, data)
        
        # Build spatial KD-tree
        tree = cKDTree(coordinates)
        
        # For each point, compute weighted average of neighbors
        smoothed = np.zeros_like(base_values)
        
        for i, coord in enumerate(coordinates):
            # Find nearest neighbors
            distances, indices = tree.query(coord, k=min(n_neighbors, len(coordinates)))
            
            # Gaussian weights
            weights = np.exp(-0.5 * (distances / sigma) ** 2)
            weights /= weights.sum()
            
            # Weighted average
            smoothed[i] = np.sum(weights * base_values[indices])
        
        return smoothed
    
    return _filter


def composite_filter(
    filters: List[FilterFunction],
    weights: Optional[List[float]] = None,
    normalize_components: bool = True
) -> FilterFunction:
    """
    Create a weighted combination of multiple filters.
    
    Combines multiple filter functions into a single filter by weighted
    averaging. Useful for creating multi-scale or multi-aspect filters.
    
    Parameters
    ----------
    filters : list of FilterFunction
        Filter functions to combine.
    weights : list of float, optional
        Weights for each filter. If None, uses equal weights.
    normalize_components : bool, default True
        If True, normalize each component to [0, 1] before combining.
    
    Returns
    -------
    FilterFunction
        Combined filter function.
    
    Examples
    --------
    >>> # Combine PCA with spatial x-coordinate
    >>> combined = composite_filter(
    ...     [pca_filter(1), spatial_coordinate_filter('x')],
    ...     weights=[0.7, 0.3]
    ... )
    """
    if weights is None:
        weights = [1.0 / len(filters)] * len(filters)
    
    if len(weights) != len(filters):
        raise ValueError("Number of weights must match number of filters")
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        components = []
        
        for f in filters:
            values = f(coordinates, neighborhoods, data)
            
            if normalize_components:
                if values.max() > values.min():
                    values = (values - values.min()) / (values.max() - values.min())
            
            components.append(values)
        
        # Weighted sum
        combined = np.zeros(len(coordinates))
        for w, c in zip(weights, components):
            combined += w * c
        
        return combined
    
    return _filter


def multiscale_spatial_filter(
    radii: List[float],
    method: str = 'density',
    weights: Optional[List[float]] = None
) -> FilterFunction:
    """
    Create a multi-scale spatial filter.
    
    Computes a spatial measure (density) at multiple scales and combines
    them. This captures both local and regional spatial patterns.
    
    Parameters
    ----------
    radii : list of float
        Radii to use for multi-scale analysis.
    method : str, default 'density'
        Spatial measure to use at each scale ('density').
    weights : list of float, optional
        Weights for each scale. If None, uses equal weights.
    
    Returns
    -------
    FilterFunction
        Multi-scale spatial filter.
    
    Examples
    --------
    >>> # Density at multiple scales
    >>> multiscale = multiscale_spatial_filter(radii=[25, 50, 100, 200])
    """
    if method != 'density':
        raise ValueError(f"Unknown method: {method}. Currently only 'density' supported.")
    
    # Create density filters at each scale
    filters = [spatial_density_filter(radius=r) for r in radii]
    
    return composite_filter(filters, weights=weights)


def type_proportion_filter(
    cell_type: str,
    radius: float = 50.0,
    normalize: bool = True
) -> FilterFunction:
    """
    Create a filter based on local proportion of a cell type.
    
    For each cell, computes the proportion of nearby cells that are
    of the specified type. Useful for identifying regions enriched
    for specific cell populations.
    
    Parameters
    ----------
    cell_type : str
        Cell type to compute proportion for.
    radius : float, default 50.0
        Radius for neighborhood (in coordinate units).
    normalize : bool, default True
        If True, output is already in [0, 1] (proportion).
    
    Returns
    -------
    FilterFunction
        Filter that computes local type proportion.
    """
    def _filter(
        coordinates: np.ndarray,
        neighborhoods: np.ndarray,
        data: 'SpatialTissueData'
    ) -> np.ndarray:
        tree = cKDTree(coordinates)
        type_mask = data._cell_types == cell_type
        
        proportions = np.zeros(len(coordinates))
        
        for i, coord in enumerate(coordinates):
            neighbor_idx = tree.query_ball_point(coord, radius)
            if len(neighbor_idx) > 1:  # Exclude self only case
                n_type = type_mask[neighbor_idx].sum()
                proportions[i] = n_type / len(neighbor_idx)
        
        # Already in [0, 1], normalization is optional
        return proportions
    
    return _filter
