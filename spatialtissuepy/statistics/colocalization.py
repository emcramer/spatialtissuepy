"""
Cell type co-localization analysis.

This module provides methods for quantifying spatial associations between
cell types, including co-localization quotients, neighborhood enrichment
tests, and interaction matrices.

Key Concepts
------------
- **Co-localization**: Two cell types occurring together more than expected by chance
- **Segregation**: Two cell types avoiding each other spatially
- **Enrichment**: Ratio of observed to expected neighbors

References
----------
.. [1] Schapiro, D. et al. (2017). histoCAT: analysis of cell phenotypes and
       interactions in multiplex image cytometry data. Nature Methods.
.. [2] Palla, G. et al. (2022). Squidpy: a scalable framework for spatial
       omics analysis. Nature Methods.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Co-localization Quotient
# -----------------------------------------------------------------------------

def colocalization_quotient(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float
) -> float:
    """
    Compute co-localization quotient (CLQ) between two cell types.

    CLQ = (observed A-B neighbors) / (expected A-B neighbors under CSR)

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

    Returns
    -------
    float
        Co-localization quotient.
        - CLQ > 1: Types are co-localized (attracted)
        - CLQ < 1: Types are segregated (repelled)
        - CLQ = 1: Random spatial association

    Examples
    --------
    >>> clq = colocalization_quotient(data, 'T_cell', 'Tumor', radius=50)
    >>> if clq > 1:
    ...     print("T cells are enriched near tumor cells")
    """
    idx_a = data.get_cells_by_type(type_a)
    idx_b = data.get_cells_by_type(type_b)
    
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan
    
    coords_a = data._coordinates[idx_a]
    coords_b = data._coordinates[idx_b]
    
    # Build tree for type B
    tree_b = cKDTree(coords_b)
    
    # Count B neighbors for each A cell
    observed = 0
    for coord in coords_a:
        neighbors = tree_b.query_ball_point(coord, radius)
        observed += len(neighbors)
    
    # Expected under random distribution
    # Expected = n_a * n_b * (π * r²) / area
    bounds = data.bounds
    area = (bounds['x'][1] - bounds['x'][0]) * (bounds['y'][1] - bounds['y'][0])
    
    if area <= 0:
        return np.nan
    
    expected = len(idx_a) * len(idx_b) * (np.pi * radius**2) / area
    
    if expected <= 0:
        return np.nan
    
    return observed / expected


def colocalization_matrix(
    data: 'SpatialTissueData',
    radius: float,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute co-localization quotient for all cell type pairs.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Neighborhood radius.
    normalize : bool, default True
        If True, return CLQ; if False, return raw counts.

    Returns
    -------
    pd.DataFrame
        Matrix of co-localization values.

    Examples
    --------
    >>> clq_matrix = colocalization_matrix(data, radius=50)
    >>> print(clq_matrix)
    """
    cell_types = list(data.cell_types_unique)
    n_types = len(cell_types)
    
    matrix = np.zeros((n_types, n_types))
    
    for i, type_a in enumerate(cell_types):
        for j, type_b in enumerate(cell_types):
            if normalize:
                matrix[i, j] = colocalization_quotient(data, type_a, type_b, radius)
            else:
                # Raw counts
                idx_a = data.get_cells_by_type(type_a)
                idx_b = data.get_cells_by_type(type_b)
                
                if len(idx_a) == 0 or len(idx_b) == 0:
                    matrix[i, j] = 0
                else:
                    coords_a = data._coordinates[idx_a]
                    tree_b = cKDTree(data._coordinates[idx_b])
                    
                    count = sum(
                        len(tree_b.query_ball_point(c, radius))
                        for c in coords_a
                    )
                    matrix[i, j] = count
    
    return pd.DataFrame(matrix, index=cell_types, columns=cell_types)


# -----------------------------------------------------------------------------
# Neighborhood Enrichment Analysis
# -----------------------------------------------------------------------------

def neighborhood_enrichment_score(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float
) -> Tuple[float, float]:
    """
    Compute neighborhood enrichment score with z-score.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    type_a : str
        Focal cell type.
    type_b : str
        Neighbor cell type.
    radius : float
        Neighborhood radius.

    Returns
    -------
    enrichment : float
        Enrichment score (observed / expected).
    zscore : float
        Z-score for statistical significance.
    """
    idx_a = data.get_cells_by_type(type_a)
    idx_b = data.get_cells_by_type(type_b)
    
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan, np.nan
    
    coords_a = data._coordinates[idx_a]
    coords_b = data._coordinates[idx_b]
    n_b = len(idx_b)
    
    # Count B neighbors for each A cell
    tree_b = cKDTree(coords_b)
    counts = np.array([
        len(tree_b.query_ball_point(c, radius)) for c in coords_a
    ])
    
    observed_mean = np.mean(counts)
    
    # Expected under random: proportion * total possible neighbors
    bounds = data.bounds
    area = (bounds['x'][1] - bounds['x'][0]) * (bounds['y'][1] - bounds['y'][0])
    
    if area <= 0:
        return np.nan, np.nan
    
    # Expected count based on density
    density_b = n_b / area
    expected_mean = density_b * np.pi * radius**2
    
    # Variance under Poisson assumption
    expected_var = expected_mean
    
    if expected_mean <= 0:
        return np.nan, np.nan
    
    enrichment = observed_mean / expected_mean
    zscore = (observed_mean - expected_mean) / np.sqrt(expected_var / len(idx_a))
    
    return enrichment, zscore


def neighborhood_enrichment_test(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float,
    n_permutations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Permutation test for neighborhood enrichment.

    Tests whether type_a cells have more type_b neighbors than expected
    by randomly permuting cell type labels.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    type_a : str
        Focal cell type.
    type_b : str
        Neighbor cell type.
    radius : float
        Neighborhood radius.
    n_permutations : int, default 1000
        Number of permutations.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - 'observed': Observed mean neighbor count
        - 'expected': Mean of permutation distribution
        - 'std': Std of permutation distribution
        - 'zscore': Standardized score
        - 'pvalue': Two-sided p-value
        - 'enrichment': Observed / expected ratio

    Examples
    --------
    >>> result = neighborhood_enrichment_test(
    ...     data, 'CD8_T_cell', 'Tumor', radius=30, n_permutations=999
    ... )
    >>> if result['pvalue'] < 0.05:
    ...     if result['enrichment'] > 1:
    ...         print("Significant co-localization")
    ...     else:
    ...         print("Significant segregation")
    """
    rng = np.random.default_rng(seed)
    
    idx_a = data.get_cells_by_type(type_a)
    idx_b = data.get_cells_by_type(type_b)
    
    if len(idx_a) == 0 or len(idx_b) == 0:
        return {
            'observed': np.nan,
            'expected': np.nan,
            'std': np.nan,
            'zscore': np.nan,
            'pvalue': np.nan,
            'enrichment': np.nan,
        }
    
    coords = data._coordinates
    cell_types = data._cell_types.copy()
    n_cells = len(cell_types)
    
    # Observed count
    coords_a = coords[idx_a]
    tree_b = cKDTree(coords[idx_b])
    observed_counts = np.array([
        len(tree_b.query_ball_point(c, radius)) for c in coords_a
    ])
    observed = np.sum(observed_counts)
    
    # Permutation distribution
    perm_counts = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle cell type labels
        perm_types = rng.permutation(cell_types)
        
        # Get new indices
        perm_idx_a = np.where(perm_types == type_a)[0]
        perm_idx_b = np.where(perm_types == type_b)[0]
        
        if len(perm_idx_a) == 0 or len(perm_idx_b) == 0:
            perm_counts[i] = 0
            continue
        
        # Count neighbors
        perm_tree_b = cKDTree(coords[perm_idx_b])
        perm_counts[i] = sum(
            len(perm_tree_b.query_ball_point(coords[j], radius))
            for j in perm_idx_a
        )
    
    # Statistics
    expected = np.mean(perm_counts)
    std = np.std(perm_counts)
    
    if std > 0:
        zscore = (observed - expected) / std
    else:
        zscore = 0 if observed == expected else np.inf
    
    # Two-sided p-value
    more_extreme = np.sum(np.abs(perm_counts - expected) >= np.abs(observed - expected))
    pvalue = (more_extreme + 1) / (n_permutations + 1)
    
    enrichment = observed / expected if expected > 0 else np.nan
    
    return {
        'observed': float(observed),
        'expected': float(expected),
        'std': float(std),
        'zscore': float(zscore),
        'pvalue': float(pvalue),
        'enrichment': float(enrichment),
    }


def neighborhood_enrichment_matrix(
    data: 'SpatialTissueData',
    radius: float,
    n_permutations: int = 1000,
    seed: Optional[int] = None,
    return_pvalues: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Compute neighborhood enrichment for all cell type pairs.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Neighborhood radius.
    n_permutations : int, default 1000
        Number of permutations per test.
    seed : int, optional
        Random seed.
    return_pvalues : bool, default False
        If True, also return p-value matrix.

    Returns
    -------
    enrichment_matrix : pd.DataFrame
        Matrix of enrichment scores.
    pvalue_matrix : pd.DataFrame, optional
        Matrix of p-values (if return_pvalues=True).
    """
    cell_types = list(data.cell_types_unique)
    n_types = len(cell_types)
    
    enrichment = np.zeros((n_types, n_types))
    pvalues = np.zeros((n_types, n_types))
    
    rng = np.random.default_rng(seed)
    
    for i, type_a in enumerate(cell_types):
        for j, type_b in enumerate(cell_types):
            result = neighborhood_enrichment_test(
                data, type_a, type_b, radius, 
                n_permutations, 
                seed=rng.integers(0, 2**31)
            )
            enrichment[i, j] = result['enrichment']
            pvalues[i, j] = result['pvalue']
    
    enrichment_df = pd.DataFrame(enrichment, index=cell_types, columns=cell_types)
    
    if return_pvalues:
        pvalue_df = pd.DataFrame(pvalues, index=cell_types, columns=cell_types)
        return enrichment_df, pvalue_df
    
    return enrichment_df


# -----------------------------------------------------------------------------
# Interaction Analysis
# -----------------------------------------------------------------------------

def spatial_interaction_matrix(
    data: 'SpatialTissueData',
    radius: float,
    method: str = 'log_ratio'
) -> pd.DataFrame:
    """
    Compute spatial interaction matrix between cell types.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float
        Neighborhood radius.
    method : str, default 'log_ratio'
        Interaction measure:
        - 'log_ratio': log2(observed/expected)
        - 'zscore': (observed - expected) / std
        - 'count': Raw neighbor counts

    Returns
    -------
    pd.DataFrame
        Interaction matrix.

    Notes
    -----
    For 'log_ratio':
    - Positive values indicate attraction
    - Negative values indicate repulsion
    - Zero indicates random mixing
    """
    cell_types = list(data.cell_types_unique)
    n_types = len(cell_types)
    
    # Compute type adjacency counts
    from spatialtissuepy.spatial.neighborhood import type_adjacency_matrix
    
    observed = type_adjacency_matrix(
        data, method='radius', radius=radius, normalize='none'
    ).values
    
    # Expected under random mixing
    type_counts = np.array([
        len(data.get_cells_by_type(ct)) for ct in cell_types
    ])
    total_edges = observed.sum()
    n_total = data.n_cells
    
    expected = np.outer(type_counts, type_counts) / (n_total ** 2) * total_edges
    
    if method == 'count':
        matrix = observed
    elif method == 'log_ratio':
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.log2(observed / expected)
            matrix = np.where(np.isfinite(matrix), matrix, 0)
    elif method == 'zscore':
        # Approximate variance (Poisson)
        std = np.sqrt(expected)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = (observed - expected) / std
            matrix = np.where(np.isfinite(matrix), matrix, 0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pd.DataFrame(matrix, index=cell_types, columns=cell_types)


# -----------------------------------------------------------------------------
# Spatial Correlation
# -----------------------------------------------------------------------------

def spatial_cross_correlation(
    data: 'SpatialTissueData',
    marker_a: str,
    marker_b: str,
    radius: float,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """
    Compute spatial cross-correlation between marker expressions.

    For each cell, compares its marker_a expression to the mean marker_b
    expression of its neighbors.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data with marker expressions.
    marker_a : str
        First marker name.
    marker_b : str
        Second marker name.
    radius : float
        Neighborhood radius.
    method : str, default 'pearson'
        Correlation method: 'pearson' or 'spearman'.

    Returns
    -------
    correlation : float
        Correlation coefficient.
    pvalue : float
        P-value for the correlation.
    """
    if data.markers is None:
        raise ValueError("No marker data available")
    
    if marker_a not in data.marker_names or marker_b not in data.marker_names:
        raise ValueError(f"Markers not found: {marker_a}, {marker_b}")
    
    expr_a = data.markers[marker_a].values
    expr_b = data.markers[marker_b].values
    
    # Build KD-tree
    tree = cKDTree(data._coordinates)
    
    # For each cell, compute mean neighbor expression of marker_b
    neighbor_mean_b = np.zeros(data.n_cells)
    
    for i in range(data.n_cells):
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) > 0:
            neighbor_mean_b[i] = np.mean(expr_b[neighbors])
        else:
            neighbor_mean_b[i] = np.nan
    
    # Remove cells with no neighbors
    valid = ~np.isnan(neighbor_mean_b)
    
    if np.sum(valid) < 3:
        return np.nan, np.nan
    
    if method == 'pearson':
        corr, pval = stats.pearsonr(expr_a[valid], neighbor_mean_b[valid])
    elif method == 'spearman':
        corr, pval = stats.spearmanr(expr_a[valid], neighbor_mean_b[valid])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(corr), float(pval)


def morans_i(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float,
    permutations: int = 0,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to test for spatial autocorrelation (e.g., marker expression).
    radius : float
        Neighborhood radius for spatial weights.
    permutations : int, default 0
        Number of permutations for p-value. If 0, use analytical p-value.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - 'I': Moran's I statistic
        - 'expected': Expected I under no autocorrelation
        - 'variance': Variance of I
        - 'zscore': Z-score
        - 'pvalue': P-value

    Notes
    -----
    - I ≈ 1: Strong positive autocorrelation (clustering of similar values)
    - I ≈ 0: No autocorrelation (random)
    - I ≈ -1: Strong negative autocorrelation (checkerboard pattern)
    """
    n = len(values)
    if n < 3:
        return {
            'I': np.nan, 'expected': np.nan, 'variance': np.nan,
            'zscore': np.nan, 'pvalue': np.nan
        }
    
    # Standardize values
    z = values - np.mean(values)
    
    # Build spatial weights
    tree = cKDTree(data._coordinates)
    
    # Compute Moran's I
    numerator = 0.0
    W = 0.0  # Sum of weights
    
    for i in range(n):
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        for j in neighbors:
            w = 1.0  # Binary weights
            numerator += w * z[i] * z[j]
            W += w
    
    if W == 0:
        return {
            'I': np.nan, 'expected': np.nan, 'variance': np.nan,
            'zscore': np.nan, 'pvalue': np.nan
        }
    
    denominator = np.sum(z**2)
    
    if denominator == 0:
        return {
            'I': np.nan, 'expected': np.nan, 'variance': np.nan,
            'zscore': np.nan, 'pvalue': np.nan
        }
    
    I = (n / W) * (numerator / denominator)
    
    # Expected value under null
    expected = -1.0 / (n - 1)
    
    # Analytical variance (simplified)
    variance = (n**2 * W - n * W + 3 * W**2) / ((n - 1) * (n + 1) * W**2)
    variance = max(variance - expected**2, 1e-10)
    
    zscore = (I - expected) / np.sqrt(variance)
    pvalue = 2 * (1 - stats.norm.cdf(abs(zscore)))
    
    # Permutation test if requested
    if permutations > 0:
        rng = np.random.default_rng(seed)
        perm_I = np.zeros(permutations)
        
        for p in range(permutations):
            perm_z = rng.permutation(z)
            perm_num = 0.0
            
            for i in range(n):
                neighbors = tree.query_ball_point(data._coordinates[i], radius)
                neighbors = [j for j in neighbors if j != i]
                
                for j in neighbors:
                    perm_num += perm_z[i] * perm_z[j]
            
            perm_I[p] = (n / W) * (perm_num / denominator)
        
        pvalue = (np.sum(np.abs(perm_I) >= np.abs(I)) + 1) / (permutations + 1)
    
    return {
        'I': float(I),
        'expected': float(expected),
        'variance': float(variance),
        'zscore': float(zscore),
        'pvalue': float(pvalue),
    }


def gearys_c(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float
) -> Dict[str, float]:
    """
    Compute Geary's C spatial autocorrelation statistic.

    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to test.
    radius : float
        Neighborhood radius.

    Returns
    -------
    dict
        Dictionary with 'C', 'expected', 'zscore', 'pvalue'.

    Notes
    -----
    - C ≈ 0: Strong positive autocorrelation
    - C ≈ 1: No autocorrelation
    - C > 1: Negative autocorrelation
    """
    n = len(values)
    if n < 3:
        return {'C': np.nan, 'expected': np.nan, 'zscore': np.nan, 'pvalue': np.nan}
    
    z = values - np.mean(values)
    tree = cKDTree(data._coordinates)
    
    # Compute Geary's C
    numerator = 0.0
    W = 0.0
    
    for i in range(n):
        neighbors = tree.query_ball_point(data._coordinates[i], radius)
        neighbors = [j for j in neighbors if j != i]
        
        for j in neighbors:
            numerator += (values[i] - values[j])**2
            W += 1
    
    if W == 0:
        return {'C': np.nan, 'expected': np.nan, 'zscore': np.nan, 'pvalue': np.nan}
    
    denominator = 2 * W * np.sum(z**2) / (n - 1)
    
    if denominator == 0:
        return {'C': np.nan, 'expected': np.nan, 'zscore': np.nan, 'pvalue': np.nan}
    
    C = numerator / denominator
    expected = 1.0
    
    # Simplified variance
    variance = (2 * W + W**2) / (W**2 * (n - 1))
    
    zscore = (C - expected) / np.sqrt(max(variance, 1e-10))
    pvalue = 2 * (1 - stats.norm.cdf(abs(zscore)))
    
    return {
        'C': float(C),
        'expected': float(expected),
        'zscore': float(zscore),
        'pvalue': float(pvalue),
    }
