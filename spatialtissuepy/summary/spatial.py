"""
Spatial statistics for spatial summary.

These metrics describe the spatial organization of cells.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from .registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


@register_metric(
    name='mean_nearest_neighbor_distance',
    category='spatial',
    description='Mean nearest neighbor distance (global and per type)',
    dynamic_columns=True
)
def mean_nearest_neighbor_distance(
    data: 'SpatialTissueData'
) -> Dict[str, float]:
    """
    Compute mean nearest neighbor distance.
    
    Returns
    -------
    dict
        Keys: 'mean_nnd', 'mean_nnd_{type}' for each type.
    """
    coords = data.coordinates
    
    if len(coords) < 2:
        return {'mean_nnd': np.nan}
    
    tree = cKDTree(coords)
    
    # Global mean NND (k=2 because first is self with distance 0)
    distances, _ = tree.query(coords, k=2)
    global_nnd = distances[:, 1].mean()
    
    result = {'mean_nnd': global_nnd}
    
    # Per-type NND
    for cell_type in data.cell_types_unique:
        mask = data.cell_types == cell_type
        if mask.sum() < 2:
            result[f'mean_nnd_{cell_type}'] = np.nan
            continue
        
        type_coords = coords[mask]
        type_tree = cKDTree(type_coords)
        type_distances, _ = type_tree.query(type_coords, k=2)
        result[f'mean_nnd_{cell_type}'] = type_distances[:, 1].mean()
    
    return result


@register_metric(
    name='cross_type_nnd',
    category='spatial',
    description='Mean distance from cells of type A to nearest cell of type B',
    parameters={'type_from': str, 'type_to': str}
)
def cross_type_nnd(
    data: 'SpatialTissueData',
    type_from: str,
    type_to: str,
) -> Dict[str, float]:
    """
    Compute mean distance from type A to nearest type B.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    type_from : str
        Source cell type.
    type_to : str
        Target cell type.
    
    Returns
    -------
    dict
        Key: 'nnd_{type_from}_to_{type_to}'.
    """
    coords = data.coordinates
    
    mask_from = data.cell_types == type_from
    mask_to = data.cell_types == type_to
    
    if mask_from.sum() == 0 or mask_to.sum() == 0:
        return {f'nnd_{type_from}_to_{type_to}': np.nan}
    
    coords_from = coords[mask_from]
    coords_to = coords[mask_to]
    
    tree_to = cKDTree(coords_to)
    distances, _ = tree_to.query(coords_from, k=1)
    
    return {f'nnd_{type_from}_to_{type_to}': distances.mean()}


@register_metric(
    name='clark_evans_index',
    category='spatial',
    description='Clark-Evans aggregation index R (R<1: clustered, R>1: dispersed)',
    dynamic_columns=True
)
def clark_evans_index(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute Clark-Evans index R.
    
    R = mean_observed_NND / mean_expected_NND
    
    Where expected NND under CSR = 0.5 * sqrt(area/n)
    
    R < 1: clustered
    R = 1: random
    R > 1: dispersed
    
    Returns
    -------
    dict
        Keys: 'clark_evans_R', 'clark_evans_R_{type}'.
    """
    coords = data.coordinates
    n = len(coords)
    
    if n < 2:
        return {'clark_evans_R': np.nan}
    
    extent = data.extent
    area = extent['x'] * extent['y']
    
    if area == 0:
        return {'clark_evans_R': np.nan}
    
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    observed_nnd = distances[:, 1].mean()
    
    # Expected NND under CSR
    expected_nnd = 0.5 * np.sqrt(area / n)
    
    R = observed_nnd / expected_nnd if expected_nnd > 0 else np.nan
    
    result = {'clark_evans_R': R}
    
    # Per-type Clark-Evans
    for cell_type in data.cell_types_unique:
        mask = data.cell_types == cell_type
        n_type = mask.sum()
        
        if n_type < 2:
            result[f'clark_evans_R_{cell_type}'] = np.nan
            continue
        
        type_coords = coords[mask]
        type_tree = cKDTree(type_coords)
        type_distances, _ = type_tree.query(type_coords, k=2)
        obs_nnd_type = type_distances[:, 1].mean()
        exp_nnd_type = 0.5 * np.sqrt(area / n_type)
        
        result[f'clark_evans_R_{cell_type}'] = obs_nnd_type / exp_nnd_type
    
    return result


@register_metric(
    name='ripleys_k',
    category='spatial',
    description="Ripley's K function at specified radii",
    parameters={'radii': list, 'edge_correction': bool},
    dynamic_columns=True
)
def ripleys_k(
    data: 'SpatialTissueData',
    radii: Optional[List[float]] = None,
    edge_correction: bool = True,
) -> Dict[str, float]:
    """
    Compute Ripley's K function at specified radii.
    
    K(r) = (A/n^2) * sum(I(d_ij < r))
    
    Where A is area, n is number of points, d_ij is distance between i and j.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    radii : list of float, optional
        Radii at which to compute K. Default: [25, 50, 100, 200].
    edge_correction : bool, default True
        Apply Ripley's edge correction.
    
    Returns
    -------
    dict
        Keys: 'K_r{radius}' for each radius.
    """
    if radii is None:
        radii = [25, 50, 100, 200]
    
    coords = data.coordinates
    n = len(coords)
    
    if n < 2:
        return {f'K_r{r}': np.nan for r in radii}
    
    extent = data.extent
    area = extent['x'] * extent['y']
    bounds = data.bounds
    
    if area == 0:
        return {f'K_r{r}': np.nan for r in radii}
    
    tree = cKDTree(coords)
    
    result = {}
    
    for r in radii:
        # Count pairs within distance r
        count = 0
        weight_sum = 0
        
        for i, coord in enumerate(coords):
            neighbors = tree.query_ball_point(coord, r)
            n_neighbors = len(neighbors) - 1  # Exclude self
            
            if edge_correction:
                # Ripley's isotropic edge correction
                # Weight by proportion of circle inside study region
                weight = _edge_correction_weight(
                    coord, r, 
                    bounds['x'][0], bounds['x'][1],
                    bounds['y'][0], bounds['y'][1]
                )
                count += n_neighbors * weight
                weight_sum += weight
            else:
                count += n_neighbors
        
        if edge_correction and weight_sum > 0:
            K = (area / (n * weight_sum)) * count
        else:
            K = (area / (n * (n - 1))) * count
        
        result[f'K_r{int(r)}'] = K
    
    return result


def _edge_correction_weight(
    point: np.ndarray,
    r: float,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> float:
    """
    Compute Ripley's isotropic edge correction weight.
    
    Approximates the proportion of the circle of radius r centered
    at point that lies within the rectangular study region.
    """
    x, y = point[0], point[1]
    
    # Distances to boundaries
    d_left = x - x_min
    d_right = x_max - x
    d_bottom = y - y_min
    d_top = y_max - y
    
    # Minimum distance to boundary
    d_min = min(d_left, d_right, d_bottom, d_top)
    
    if d_min >= r:
        # Circle fully inside
        return 1.0
    
    # Simplified approximation: based on nearest edge
    # More accurate would consider corner effects
    if d_min <= 0:
        return 0.5
    
    # Proportion of circumference inside (approximate)
    # Using simplified arc correction
    prop_inside = 0.5 + 0.5 * (d_min / r)
    
    return max(0.5, min(1.0, prop_inside))


@register_metric(
    name='l_function',
    category='spatial',
    description="Ripley's L function (variance-stabilized K)",
    parameters={'radii': list},
    dynamic_columns=True
)
def l_function(
    data: 'SpatialTissueData',
    radii: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute Ripley's L function.
    
    L(r) = sqrt(K(r) / pi) - r
    
    L > 0: clustered
    L = 0: random (CSR)
    L < 0: dispersed
    
    Returns
    -------
    dict
        Keys: 'L_r{radius}' for each radius.
    """
    K_values = ripleys_k(data, radii=radii)
    
    if radii is None:
        radii = [25, 50, 100, 200]
    
    result = {}
    
    for r in radii:
        K = K_values.get(f'K_r{int(r)}', np.nan)
        
        if np.isnan(K) or K < 0:
            L = np.nan
        else:
            L = np.sqrt(K / np.pi) - r
        
        result[f'L_r{int(r)}'] = L
    
    return result


@register_metric(
    name='g_function_summary',
    category='spatial',
    description='Summary statistics of the G function (nearest neighbor CDF)',
    parameters={'radii': list}
)
def g_function_summary(
    data: 'SpatialTissueData',
    radii: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute summary of the G function (empirical CDF of NND).
    
    Returns G(r) values and the median NND.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    radii : list of float, optional
        Radii at which to evaluate G.
    
    Returns
    -------
    dict
        Keys: 'G_r{radius}', 'median_nnd'.
    """
    if radii is None:
        radii = [10, 25, 50, 100]
    
    coords = data.coordinates
    n = len(coords)
    
    if n < 2:
        result = {f'G_r{int(r)}': np.nan for r in radii}
        result['median_nnd'] = np.nan
        return result
    
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nnd = distances[:, 1]
    
    result = {'median_nnd': np.median(nnd)}
    
    for r in radii:
        # G(r) = proportion of NNDs <= r
        G_r = np.mean(nnd <= r)
        result[f'G_r{int(r)}'] = G_r
    
    return result


@register_metric(
    name='spatial_autocorrelation',
    category='spatial',
    description="Moran's I spatial autocorrelation for cell types",
    parameters={'cell_type': str, 'radius': float}
)
def spatial_autocorrelation(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float = 50.0,
) -> Dict[str, float]:
    """
    Compute Moran's I for a binary cell type indicator.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    cell_type : str
        Cell type to analyze.
    radius : float
        Radius for spatial weights.
    
    Returns
    -------
    dict
        Key: 'morans_I_{cell_type}'.
    """
    coords = data.coordinates
    n = len(coords)
    
    if n < 3:
        return {f'morans_I_{cell_type}': np.nan}
    
    # Binary indicator
    y = (data.cell_types == cell_type).astype(float)
    y_mean = y.mean()
    y_centered = y - y_mean
    
    if np.var(y) == 0:
        return {f'morans_I_{cell_type}': np.nan}
    
    # Spatial weights (binary within radius)
    tree = cKDTree(coords)
    
    numerator = 0.0
    W = 0.0
    
    for i in range(n):
        neighbors = tree.query_ball_point(coords[i], radius)
        for j in neighbors:
            if i != j:
                numerator += y_centered[i] * y_centered[j]
                W += 1
    
    if W == 0:
        return {f'morans_I_{cell_type}': np.nan}
    
    denominator = np.sum(y_centered ** 2)
    
    if denominator == 0:
        return {f'morans_I_{cell_type}': np.nan}
    
    I = (n / W) * (numerator / denominator)
    
    return {f'morans_I_{cell_type}': I}


@register_metric(
    name='convex_hull_metrics',
    category='morphology',
    description='Convex hull area and compactness',
)
def convex_hull_metrics(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute convex hull metrics.
    
    Returns
    -------
    dict
        Keys: 'convex_hull_area', 'compactness'.
    """
    from scipy.spatial import ConvexHull
    
    coords = data.coordinates[:, :2]  # Use 2D
    
    if len(coords) < 3:
        return {'convex_hull_area': np.nan, 'compactness': np.nan}
    
    try:
        hull = ConvexHull(coords)
        hull_area = hull.volume  # In 2D, 'volume' is area
        
        # Compactness: ratio of actual spread to hull
        extent = data.extent
        bbox_area = extent['x'] * extent['y']
        
        compactness = hull_area / bbox_area if bbox_area > 0 else np.nan
        
        return {
            'convex_hull_area': hull_area,
            'compactness': compactness,
        }
    except Exception:
        return {'convex_hull_area': np.nan, 'compactness': np.nan}
