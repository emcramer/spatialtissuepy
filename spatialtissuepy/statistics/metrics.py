"""
Spatial statistics metrics for integration with summary module.

This module registers spatial statistics metrics with the StatisticsPanel
for standardized computation across samples.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

from spatialtissuepy.summary.registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# -----------------------------------------------------------------------------
# Ripley's K-function Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='ripleys_h_max',
    category='spatial_stats',
    description="Maximum Ripley's H(r) value, indicating clustering strength",
    parameters={'max_radius': float, 'n_radii': int}
)
def _ripleys_h_max(
    data: 'SpatialTissueData',
    max_radius: float = 100.0,
    n_radii: int = 20
) -> Dict[str, float]:
    """Compute maximum H(r) as summary of clustering."""
    from .spatial_stats import ripleys_h
    
    radii = np.linspace(1, max_radius, n_radii)
    coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        return {'ripleys_h_max': np.nan}
    
    H = ripleys_h(coords, radii)
    return {'ripleys_h_max': float(np.max(H))}


@register_metric(
    name='ripleys_h_by_type',
    category='spatial_stats',
    description="Maximum Ripley's H(r) for each cell type",
    parameters={'max_radius': float, 'n_radii': int}
)
def _ripleys_h_by_type(
    data: 'SpatialTissueData',
    max_radius: float = 100.0,
    n_radii: int = 20
) -> Dict[str, float]:
    """Compute max H(r) per cell type."""
    from .spatial_stats import ripleys_h
    
    radii = np.linspace(1, max_radius, n_radii)
    result = {}
    
    for cell_type in data.cell_types_unique:
        idx = data.get_cells_by_type(cell_type)
        coords = data._coordinates[idx, :2]
        
        if len(coords) >= 3:
            H = ripleys_h(coords, radii)
            result[f'ripleys_h_max_{cell_type}'] = float(np.max(H))
        else:
            result[f'ripleys_h_max_{cell_type}'] = np.nan
    
    return result


@register_metric(
    name='ripleys_h_auc',
    category='spatial_stats',
    description="Area under Ripley's H(r) curve (positive = clustered)",
    parameters={'max_radius': float, 'n_radii': int}
)
def _ripleys_h_auc(
    data: 'SpatialTissueData',
    max_radius: float = 100.0,
    n_radii: int = 20
) -> Dict[str, float]:
    """Compute AUC of H(r) curve."""
    from .spatial_stats import ripleys_h
    
    radii = np.linspace(1, max_radius, n_radii)
    coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        return {'ripleys_h_auc': np.nan}
    
    H = ripleys_h(coords, radii)
    # Trapezoidal integration
    auc = np.trapz(H, radii)
    return {'ripleys_h_auc': float(auc)}


@register_metric(
    name='pair_correlation_peak',
    category='spatial_stats',
    description="Peak value of pair correlation function g(r)",
    parameters={'max_radius': float, 'n_radii': int}
)
def _pcf_peak(
    data: 'SpatialTissueData',
    max_radius: float = 100.0,
    n_radii: int = 20
) -> Dict[str, float]:
    """Compute peak of pair correlation function."""
    from .spatial_stats import pair_correlation_function
    
    radii = np.linspace(1, max_radius, n_radii)
    coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        return {'pcf_peak': np.nan, 'pcf_peak_radius': np.nan}
    
    g = pair_correlation_function(coords, radii)
    peak_idx = np.argmax(g)
    
    return {
        'pcf_peak': float(g[peak_idx]),
        'pcf_peak_radius': float(radii[peak_idx]),
    }


# -----------------------------------------------------------------------------
# Cross-type Spatial Statistics
# -----------------------------------------------------------------------------

@register_metric(
    name='cross_h_max',
    category='spatial_stats',
    description="Maximum cross-type H(r) between two cell types",
    parameters={'type_a': str, 'type_b': str, 'max_radius': float}
)
def _cross_h_max(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    max_radius: float = 100.0
) -> Dict[str, float]:
    """Compute max cross-type H(r)."""
    from .spatial_stats import cross_h
    
    coords_a = data._coordinates[data.get_cells_by_type(type_a), :2]
    coords_b = data._coordinates[data.get_cells_by_type(type_b), :2]
    
    if len(coords_a) < 1 or len(coords_b) < 1:
        return {f'cross_h_max_{type_a}_{type_b}': np.nan}
    
    radii = np.linspace(1, max_radius, 20)
    H = cross_h(coords_a, coords_b, radii)
    
    return {f'cross_h_max_{type_a}_{type_b}': float(np.max(H))}


@register_metric(
    name='cross_g_function',
    category='spatial_stats',
    description="Cross-type G-function at specified radius",
    parameters={'type_a': str, 'type_b': str, 'radius': float}
)
def _cross_g_at_radius(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute cross G-function at specific radius."""
    from .spatial_stats import g_function_cross
    
    coords_a = data._coordinates[data.get_cells_by_type(type_a), :2]
    coords_b = data._coordinates[data.get_cells_by_type(type_b), :2]
    
    if len(coords_a) < 1 or len(coords_b) < 1:
        return {f'cross_g_{type_a}_{type_b}': np.nan}
    
    G = g_function_cross(coords_a, coords_b, np.array([radius]))
    
    return {f'cross_g_{type_a}_{type_b}': float(G[0])}


# -----------------------------------------------------------------------------
# Co-localization Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='colocalization_quotient',
    category='colocalization',
    description="Co-localization quotient between two cell types",
    parameters={'type_a': str, 'type_b': str, 'radius': float}
)
def _coloc_quotient(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute CLQ between two types."""
    from .colocalization import colocalization_quotient
    
    clq = colocalization_quotient(data, type_a, type_b, radius)
    return {f'clq_{type_a}_{type_b}': clq}


@register_metric(
    name='neighborhood_enrichment',
    category='colocalization',
    description="Permutation-based neighborhood enrichment z-score",
    parameters={'type_a': str, 'type_b': str, 'radius': float, 'n_permutations': int}
)
def _neighborhood_enrichment(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0,
    n_permutations: int = 100
) -> Dict[str, float]:
    """Compute neighborhood enrichment with permutation test."""
    from .colocalization import neighborhood_enrichment_test
    
    result = neighborhood_enrichment_test(
        data, type_a, type_b, radius, n_permutations
    )
    
    return {
        f'enrichment_{type_a}_{type_b}': result['enrichment'],
        f'enrichment_zscore_{type_a}_{type_b}': result['zscore'],
        f'enrichment_pvalue_{type_a}_{type_b}': result['pvalue'],
    }


@register_metric(
    name='morans_i',
    category='colocalization',
    description="Moran's I spatial autocorrelation for marker expression",
    parameters={'marker': str, 'radius': float}
)
def _morans_i_metric(
    data: 'SpatialTissueData',
    marker: str,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute Moran's I for marker."""
    from .colocalization import morans_i
    
    if data.markers is None or marker not in data.marker_names:
        return {
            f'morans_i_{marker}': np.nan,
            f'morans_i_pvalue_{marker}': np.nan,
        }
    
    values = data.markers[marker].values
    result = morans_i(data, values, radius)
    
    return {
        f'morans_i_{marker}': result['I'],
        f'morans_i_pvalue_{marker}': result['pvalue'],
    }


@register_metric(
    name='spatial_interaction_score',
    category='colocalization',
    description="Log-ratio spatial interaction score between two types",
    parameters={'type_a': str, 'type_b': str, 'radius': float}
)
def _spatial_interaction(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute log-ratio interaction score."""
    from .colocalization import spatial_interaction_matrix
    
    matrix = spatial_interaction_matrix(data, radius, method='log_ratio')
    
    if type_a in matrix.index and type_b in matrix.columns:
        score = matrix.loc[type_a, type_b]
    else:
        score = np.nan
    
    return {f'interaction_{type_a}_{type_b}': float(score)}


# -----------------------------------------------------------------------------
# Hotspot Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='hotspot_fraction',
    category='hotspots',
    description="Fraction of cells in significant hotspots for a cell type",
    parameters={'cell_type': str, 'radius': float, 'alpha': float}
)
def _hotspot_fraction(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float = 50.0,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Compute fraction of cells in hotspots."""
    from .hotspots import cell_type_hotspots, hotspot_statistics
    
    result = cell_type_hotspots(data, cell_type, radius, alpha=alpha)
    stats = hotspot_statistics(data, result)
    
    return {
        f'hotspot_fraction_{cell_type}': stats['hotspot_fraction'],
        f'coldspot_fraction_{cell_type}': stats['coldspot_fraction'],
    }


@register_metric(
    name='n_hotspot_cells',
    category='hotspots',
    description="Number of cells in significant hotspots for a cell type",
    parameters={'cell_type': str, 'radius': float, 'alpha': float}
)
def _n_hotspot_cells(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float = 50.0,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Count cells in hotspots."""
    from .hotspots import cell_type_hotspots, hotspot_statistics
    
    result = cell_type_hotspots(data, cell_type, radius, alpha=alpha)
    stats = hotspot_statistics(data, result)
    
    return {
        f'n_hotspot_cells_{cell_type}': stats['n_hotspots'],
        f'n_coldspot_cells_{cell_type}': stats['n_coldspots'],
    }


@register_metric(
    name='mean_gi_star',
    category='hotspots',
    description="Mean Getis-Ord Gi* statistic for a cell type",
    parameters={'cell_type': str, 'radius': float}
)
def _mean_gi_star(
    data: 'SpatialTissueData',
    cell_type: str,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute mean Gi* for cell type indicator."""
    from .hotspots import getis_ord_gi_star
    
    indicator = (data._cell_types == cell_type).astype(float)
    gi_star = getis_ord_gi_star(data, indicator, radius)
    
    return {
        f'mean_gi_star_{cell_type}': float(np.mean(gi_star)),
        f'max_gi_star_{cell_type}': float(np.max(gi_star)),
    }


@register_metric(
    name='marker_hotspot_fraction',
    category='hotspots',
    description="Fraction of cells in marker expression hotspots",
    parameters={'marker': str, 'radius': float, 'alpha': float}
)
def _marker_hotspot_fraction(
    data: 'SpatialTissueData',
    marker: str,
    radius: float = 50.0,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Compute fraction of cells in marker hotspots."""
    from .hotspots import marker_hotspots, hotspot_statistics
    
    if data.markers is None or marker not in data.marker_names:
        return {
            f'marker_hotspot_fraction_{marker}': np.nan,
            f'marker_coldspot_fraction_{marker}': np.nan,
        }
    
    result = marker_hotspots(data, marker, radius, alpha=alpha)
    stats = hotspot_statistics(data, result)
    
    return {
        f'marker_hotspot_fraction_{marker}': stats['hotspot_fraction'],
        f'marker_coldspot_fraction_{marker}': stats['coldspot_fraction'],
    }


# -----------------------------------------------------------------------------
# G and F Function Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='g_function_median',
    category='spatial_stats',
    description="Median radius where G-function reaches 0.5 (typical NN distance)",
    parameters={'max_radius': float}
)
def _g_function_median(
    data: 'SpatialTissueData',
    max_radius: float = 100.0
) -> Dict[str, float]:
    """Compute median of G-function (typical nearest neighbor distance)."""
    from .spatial_stats import g_function
    
    radii = np.linspace(0.1, max_radius, 100)
    coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        return {'g_function_median': np.nan}
    
    G = g_function(coords, radii)
    
    # Find radius where G crosses 0.5
    idx = np.searchsorted(G, 0.5)
    if idx >= len(radii):
        median_r = max_radius
    elif idx == 0:
        median_r = radii[0]
    else:
        # Linear interpolation
        median_r = radii[idx-1] + (0.5 - G[idx-1]) * (radii[idx] - radii[idx-1]) / (G[idx] - G[idx-1] + 1e-10)
    
    return {'g_function_median': float(median_r)}


@register_metric(
    name='j_function_summary',
    category='spatial_stats',
    description="J-function summary (mean deviation from 1)",
    parameters={'max_radius': float}
)
def _j_function_summary(
    data: 'SpatialTissueData',
    max_radius: float = 100.0
) -> Dict[str, float]:
    """Compute J-function summary statistics."""
    from .spatial_stats import j_function
    
    radii = np.linspace(1, max_radius, 30)
    coords = data._coordinates[:, :2]
    
    if len(coords) < 2:
        return {'j_function_mean': np.nan, 'j_function_min': np.nan}
    
    J = j_function(coords, radii)
    
    return {
        'j_function_mean': float(np.mean(J)),
        'j_function_min': float(np.min(J)),  # < 1 indicates clustering
    }
