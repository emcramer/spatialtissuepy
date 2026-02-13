"""
Spatial hotspot detection.

This module provides methods for identifying statistically significant
spatial clusters (hotspots) and voids (coldspots) of cell types or
marker expression values.

Key Methods
-----------
- Getis-Ord Gi*: Local clustering statistic
- Local Moran's I: Local spatial autocorrelation
- LISA: Local indicators of spatial association

References
----------
.. [1] Getis, A., & Ord, J. K. (1992). The analysis of spatial association
       by use of distance statistics. Geographical Analysis.
.. [2] Anselin, L. (1995). Local indicators of spatial association—LISA.
       Geographical Analysis.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Getis-Ord Gi* Statistic
# -----------------------------------------------------------------------------

def getis_ord_gi_star(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float,
    standardize: bool = True,
    return_dict: bool = False
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute Getis-Ord Gi* statistic for each cell.

    The Gi* statistic identifies local clusters of high or low values.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze (e.g., marker expression, cell type indicator).
    radius : float
        Neighborhood radius for spatial weights.
    standardize : bool, default True
        If True, return z-scores; if False, return raw Gi*.
    return_dict : bool, default False
        If True, return a dictionary with key 'gi_star'.

    Returns
    -------
    np.ndarray or dict
        Gi* z-scores for each cell.
        - High positive values: Hotspots (clusters of high values)
        - High negative values: Coldspots (clusters of low values)

    Notes
    -----
    The Gi* statistic includes the focal cell in its own neighborhood,
    unlike the Gi statistic which excludes it.

    Examples
    --------
    >>> # Find hotspots of marker expression
    >>> marker_values = data.markers['CD8'].values
    >>> gi_star = getis_ord_gi_star(data, marker_values, radius=50)
    >>> hotspots = np.where(gi_star > 1.96)[0]  # p < 0.05
    """
    n = len(values)
    values = np.asarray(values, dtype=float)
    
    # Global statistics
    x_bar = np.mean(values)
    s = np.std(values, ddof=0)
    
    if s == 0:
        return np.zeros(n)
    
    # Build KD-tree
    tree = cKDTree(data._coordinates)
    
    gi_star = np.zeros(n)
    
    for i in range(n):
        # Get neighbors including self
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        
        if len(neighbors) == 0:
            gi_star[i] = 0
            continue
        
        # Sum of neighbor values (including self)
        neighbor_sum = np.sum(values[neighbors])
        w_i = len(neighbors)  # Number of neighbors (binary weights)
        
        # Gi* statistic
        numerator = neighbor_sum - x_bar * w_i
        
        # Denominator (standard error)
        denominator = s * np.sqrt((n * w_i - w_i**2) / (n - 1))
        
        if denominator > 0:
            gi_star[i] = numerator / denominator
        else:
            gi_star[i] = 0
    
    if not standardize:
        # Return raw Gi* (not z-score)
        result = gi_star * s / x_bar if x_bar != 0 else gi_star
    else:
        result = gi_star
    
    if return_dict:
        return {'gi_star': result}
    return result


def getis_ord_gi(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Compute Getis-Ord Gi statistic (excludes focal cell).

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze.
    radius : float
        Neighborhood radius.

    Returns
    -------
    np.ndarray
        Gi z-scores for each cell.

    Notes
    -----
    Unlike Gi*, the Gi statistic excludes the focal cell from its
    neighborhood. This is useful when the focal cell's value should
    not influence its own hotspot designation.
    """
    n = len(values)
    values = np.asarray(values, dtype=float)
    
    x_bar = np.mean(values)
    s = np.std(values, ddof=0)
    
    if s == 0:
        return np.zeros(n)
    
    tree = cKDTree(data._coordinates)
    gi = np.zeros(n)
    
    for i in range(n):
        # Get neighbors excluding self
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            gi[i] = 0
            continue
        
        neighbor_sum = np.sum(values[neighbors])
        w_i = len(neighbors)
        
        numerator = neighbor_sum - x_bar * w_i
        denominator = s * np.sqrt((n * w_i - w_i**2) / (n - 1))
        
        if denominator > 0:
            gi[i] = numerator / denominator
        else:
            gi[i] = 0
    
    return gi


# -----------------------------------------------------------------------------
# Local Moran's I
# -----------------------------------------------------------------------------

def local_morans_i(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float,
    permutations: int = 0,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute Local Moran's I (LISA) for each cell.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze.
    radius : float
        Neighborhood radius.
    permutations : int, default 0
        Number of permutations for pseudo p-values.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - 'I': Local Moran's I values
        - 'zscore': Z-scores
        - 'pvalue': P-values (analytical or permutation-based)
        - 'quadrant': LISA quadrant (1=HH, 2=LH, 3=LL, 4=HL)

    Notes
    -----
    Quadrants indicate the type of spatial association:
    - HH (1): High value surrounded by high values (cluster)
    - LH (2): Low value surrounded by high values (spatial outlier)
    - LL (3): Low value surrounded by low values (cluster)
    - HL (4): High value surrounded by low values (spatial outlier)
    """
    n = len(values)
    values = np.asarray(values, dtype=float)
    
    # Standardize values
    z = (values - np.mean(values)) / np.std(values, ddof=0)
    z = np.nan_to_num(z, nan=0)
    
    tree = cKDTree(data._coordinates)
    
    I_local = np.zeros(n)
    lag = np.zeros(n)  # Spatial lag
    
    for i in range(n):
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            I_local[i] = 0
            lag[i] = 0
            continue
        
        # Spatial lag (mean of neighbors' standardized values)
        lag[i] = np.mean(z[neighbors])
        
        # Local Moran's I
        I_local[i] = z[i] * lag[i]
    
    # Determine quadrants
    quadrant = np.zeros(n, dtype=int)
    quadrant[(z > 0) & (lag > 0)] = 1  # HH
    quadrant[(z < 0) & (lag > 0)] = 2  # LH
    quadrant[(z < 0) & (lag < 0)] = 3  # LL
    quadrant[(z > 0) & (lag < 0)] = 4  # HL
    
    # Analytical p-values (simplified)
    # Variance approximation
    E_I = -1 / (n - 1)
    
    zscore = np.zeros(n)
    pvalue = np.ones(n)
    
    for i in range(n):
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) == 0:
            continue
        
        w_i = len(neighbors)
        var_I = (w_i * (n - 3)) / ((n - 1) * (n + 1))
        
        if var_I > 0:
            zscore[i] = (I_local[i] - E_I) / np.sqrt(var_I)
            pvalue[i] = 2 * (1 - stats.norm.cdf(abs(zscore[i])))
    
    # Permutation p-values if requested
    if permutations > 0:
        rng = np.random.default_rng(seed)
        pvalue = np.zeros(n)
        
        for i in range(n):
            neighbors = tree.query_ball_point(data._coordinates[i], radius)
            neighbors = [j for j in neighbors if j != i]
            
            if len(neighbors) == 0:
                pvalue[i] = 1.0
                continue
            
            observed_I = I_local[i]
            count_extreme = 0
            
            for _ in range(permutations):
                perm_idx = rng.choice(n, size=len(neighbors), replace=False)
                perm_lag = np.mean(z[perm_idx])
                perm_I = z[i] * perm_lag
                
                if abs(perm_I) >= abs(observed_I):
                    count_extreme += 1
            
            pvalue[i] = (count_extreme + 1) / (permutations + 1)
    
    return {
        'I': I_local,
        'zscore': zscore,
        'pvalue': pvalue,
        'quadrant': quadrant,
        'lag': lag,
    }


# -----------------------------------------------------------------------------
# Hotspot Detection Functions
# -----------------------------------------------------------------------------

def detect_hotspots(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float,
    method: str = 'gi_star',
    alpha: float = 0.05,
    correction: str = 'fdr',
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Detect statistically significant hotspots and coldspots.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze.
    radius : float
        Neighborhood radius.
    method : str, default 'gi_star'
        Detection method: 'gi_star', 'gi', or 'local_moran'.
    alpha : float, default 0.05
        Significance level.
    correction : str, default 'fdr'
        Multiple testing correction: 'none', 'bonferroni', or 'fdr'.
    **kwargs
        Additional arguments, including 'significance' (alias for alpha).

    Returns
    -------
    dict
        Dictionary with:
        - 'statistic': Test statistic values
        - 'pvalue': P-values
        - 'is_hotspot': Boolean mask for hotspots
        - 'is_coldspot': Boolean mask for coldspots
        - 'hotspot_idx': Indices of hotspot cells
        - 'coldspot_idx': Indices of coldspot cells
    """
    if 'significance' in kwargs:
        alpha = kwargs.pop('significance')
    
    n = len(values)
    
    if method == 'gi_star':
        statistic = getis_ord_gi_star(data, values, radius)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(statistic)))
    elif method == 'gi':
        statistic = getis_ord_gi(data, values, radius)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(statistic)))
    elif method == 'local_moran':
        result = local_morans_i(data, values, radius)
        statistic = result['zscore']
        pvalue = result['pvalue']
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply multiple testing correction
    if correction == 'bonferroni':
        adjusted_alpha = alpha / n
    elif correction == 'fdr':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(pvalue)
        sorted_pval = pvalue[sorted_idx]
        
        adjusted_pval = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            rank = i + 1
            adjusted_pval[idx] = sorted_pval[i] * n / rank
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            if adjusted_pval[sorted_idx[i]] > adjusted_pval[sorted_idx[i + 1]]:
                adjusted_pval[sorted_idx[i]] = adjusted_pval[sorted_idx[i + 1]]
        
        pvalue = np.minimum(adjusted_pval, 1.0)
        adjusted_alpha = alpha
    else:
        adjusted_alpha = alpha
    
    # Identify hotspots and coldspots
    is_hotspot = (statistic > 0) & (pvalue < adjusted_alpha)
    is_coldspot = (statistic < 0) & (pvalue < adjusted_alpha)
    
    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'is_hotspot': is_hotspot,
        'is_coldspot': is_coldspot,
        'hotspot_idx': np.where(is_hotspot)[0],
        'coldspot_idx': np.where(is_coldspot)[0],
    }


def cell_type_hotspots(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float,
    method: str = 'gi_star',
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Detect hotspots and coldspots for a specific cell type.

    Creates a binary indicator (1 for target type, 0 otherwise) and
    identifies where the cell type is spatially clustered.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    cell_type : str
        Cell type to analyze.
    radius : float
        Neighborhood radius.
    method : str, default 'gi_star'
        Detection method.
    alpha : float, default 0.05
        Significance level.
    **kwargs
        Additional arguments for detect_hotspots.

    Returns
    -------
    dict
        Hotspot detection results.

    Examples
    --------
    >>> result = cell_type_hotspots(data, 'Tumor', radius=50)
    >>> tumor_hotspots = result['hotspot_idx']
    >>> print(f"Found {len(tumor_hotspots)} cells in tumor hotspots")
    """
    # Create binary indicator for cell type
    indicator = (data._cell_types == cell_type).astype(float)
    
    return detect_hotspots(data, indicator, radius, method, alpha, **kwargs)


def marker_hotspots(
    data: 'SpatialTissueData',
    marker: str,
    radius: float,
    method: str = 'gi_star',
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Detect hotspots and coldspots for marker expression.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    marker : str
        Marker name.
    radius : float
        Neighborhood radius.
    method : str, default 'gi_star'
        Detection method.
    alpha : float, default 0.05
        Significance level.
    **kwargs
        Additional arguments for detect_hotspots.

    Returns
    -------
    dict
        Hotspot detection results.
    """
    if data.markers is None:
        raise ValueError("No marker data available")
    
    if marker not in data.marker_names:
        raise ValueError(f"Marker not found: {marker}")
    
    values = data.markers[marker].values
    
    return detect_hotspots(data, values, radius, method, alpha, **kwargs)


# -----------------------------------------------------------------------------
# Hotspot Statistics
# -----------------------------------------------------------------------------

def hotspot_statistics(
    data: 'SpatialTissueData',
    hotspot_result: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute summary statistics for hotspot detection results.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    hotspot_result : dict
        Result from detect_hotspots() or similar.

    Returns
    -------
    dict
        Summary statistics including:
        - n_hotspots: Number of hotspot cells
        - n_coldspots: Number of coldspot cells
        - hotspot_fraction: Fraction of cells in hotspots
        - coldspot_fraction: Fraction of cells in coldspots
        - hotspot_types: Cell type composition of hotspots
    """
    n_total = data.n_cells
    
    hotspot_idx = hotspot_result.get('hotspot_idx', np.array([]))
    coldspot_idx = hotspot_result.get('coldspot_idx', np.array([]))
    
    n_hotspots = len(hotspot_idx)
    n_coldspots = len(coldspot_idx)
    
    # Cell type composition of hotspots
    if n_hotspots > 0:
        hotspot_types, counts = np.unique(
            data._cell_types[hotspot_idx], return_counts=True
        )
        hotspot_composition = dict(zip(hotspot_types, counts.astype(int)))
    else:
        hotspot_composition = {}
    
    # Cell type composition of coldspots
    if n_coldspots > 0:
        coldspot_types, counts = np.unique(
            data._cell_types[coldspot_idx], return_counts=True
        )
        coldspot_composition = dict(zip(coldspot_types, counts.astype(int)))
    else:
        coldspot_composition = {}
    
    return {
        'n_hotspots': n_hotspots,
        'n_coldspots': n_coldspots,
        'hotspot_fraction': n_hotspots / n_total if n_total > 0 else 0,
        'coldspot_fraction': n_coldspots / n_total if n_total > 0 else 0,
        'hotspot_composition': hotspot_composition,
        'coldspot_composition': coldspot_composition,
    }


def hotspot_regions(
    data: 'SpatialTissueData',
    hotspot_result: Dict[str, np.ndarray],
    merge_radius: float
) -> np.ndarray:
    """
    Merge nearby hotspot cells into contiguous regions.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    hotspot_result : dict
        Result from detect_hotspots().
    merge_radius : float
        Maximum distance to merge hotspot cells.

    Returns
    -------
    np.ndarray
        Region labels for each cell (-1 for non-hotspot cells).
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    
    n_cells = data.n_cells
    hotspot_idx = hotspot_result.get('hotspot_idx', np.array([]))
    
    if len(hotspot_idx) == 0:
        return np.full(n_cells, -1, dtype=int)
    
    # Build graph of hotspot cells
    hotspot_coords = data._coordinates[hotspot_idx]
    tree = cKDTree(hotspot_coords)
    
    # Find connected pairs
    pairs = tree.query_pairs(merge_radius, output_type='ndarray')
    
    # Create adjacency matrix
    n_hotspots = len(hotspot_idx)
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data_vals = np.ones(len(row))
    
    adj = csr_matrix((data_vals, (row, col)), shape=(n_hotspots, n_hotspots))
    
    # Find connected components
    n_components, component_labels = connected_components(adj, directed=False)
    
    # Map back to full cell array
    region_labels = np.full(n_cells, -1, dtype=int)
    region_labels[hotspot_idx] = component_labels
    
    return region_labels


# -----------------------------------------------------------------------------
# Summary Functions
# -----------------------------------------------------------------------------

def hotspot_summary_by_type(
    data: 'SpatialTissueData',
    radius: float,
    method: str = 'gi_star',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute hotspot statistics for all cell types.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Neighborhood radius.
    method : str, default 'gi_star'
        Detection method.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Summary with columns for each cell type.
    """
    results = []
    
    for cell_type in data.cell_types_unique:
        result = cell_type_hotspots(data, cell_type, radius, method, alpha)
        stats = hotspot_statistics(data, result)
        
        results.append({
            'cell_type': cell_type,
            'n_cells': len(data.get_cells_by_type(cell_type)),
            'n_hotspots': stats['n_hotspots'],
            'n_coldspots': stats['n_coldspots'],
            'hotspot_fraction': stats['hotspot_fraction'],
            'coldspot_fraction': stats['coldspot_fraction'],
        })
    
    return pd.DataFrame(results).set_index('cell_type')
