"""
Population-level statistics for spatial summary.

These metrics describe the overall cell population without considering
spatial relationships.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import numpy as np

from .registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData


@register_metric(
    name='cell_counts',
    category='population',
    description='Total cell count and count per cell type',
    dynamic_columns=True
)
def cell_counts(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute total and per-type cell counts.
    
    Returns
    -------
    dict
        Keys: 'n_cells', 'n_{type}' for each cell type.
    """
    result = {'n_cells': float(data.n_cells)}
    
    for cell_type, count in data.cell_type_counts.items():
        result[f'n_{cell_type}'] = float(count)
    
    return result


@register_metric(
    name='cell_proportions',
    category='population',
    description='Proportion of each cell type',
    dynamic_columns=True
)
def cell_proportions(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute proportion of each cell type.
    
    Returns
    -------
    dict
        Keys: 'prop_{type}' for each cell type.
    """
    total = data.n_cells
    result = {}
    
    for cell_type, count in data.cell_type_counts.items():
        prop = count / total if total > 0 else 0.0
        result[f'prop_{cell_type}'] = prop
    
    return result


@register_metric(
    name='cell_type_ratio',
    category='population',
    description='Ratio of two cell type counts',
    parameters={'numerator': str, 'denominator': str}
)
def cell_type_ratio(
    data: 'SpatialTissueData',
    numerator: str,
    denominator: str,
) -> Dict[str, float]:
    """
    Compute ratio of two cell type counts.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    numerator : str
        Cell type for numerator.
    denominator : str
        Cell type for denominator.
    
    Returns
    -------
    dict
        Key: '{numerator}_{denominator}_ratio'.
    """
    counts = data.cell_type_counts
    
    num = counts.get(numerator, 0)
    denom = counts.get(denominator, 0)
    
    if denom == 0:
        ratio = np.nan if num == 0 else np.inf
    else:
        ratio = num / denom
    
    return {f'{numerator}_{denominator}_ratio': ratio}


@register_metric(
    name='cell_density',
    category='population',
    description='Cell density (cells per unit area)',
    dynamic_columns=True
)
def cell_density(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute cell density (cells per unit area).
    
    Returns
    -------
    dict
        Keys: 'density_total', 'density_{type}' for each type.
    """
    extent = data.extent
    area = extent['x'] * extent['y']
    
    if area == 0:
        return {'density_total': np.nan}
    
    result = {'density_total': data.n_cells / area}
    
    for cell_type, count in data.cell_type_counts.items():
        result[f'density_{cell_type}'] = count / area
    
    return result


@register_metric(
    name='shannon_diversity',
    category='diversity',
    description='Shannon diversity index of cell type distribution',
    parameters={'normalize': bool}
)
def shannon_diversity(
    data: 'SpatialTissueData',
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute Shannon diversity index.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    normalize : bool, default True
        If True, normalize by maximum possible diversity.
    
    Returns
    -------
    dict
        Key: 'shannon_diversity'.
    """
    counts = data.cell_type_counts
    total = counts.sum()
    
    if total == 0:
        return {'shannon_diversity': np.nan}
    
    props = counts / total
    props = props[props > 0]  # Remove zeros
    
    entropy = -np.sum(props * np.log(props))
    
    if normalize:
        max_entropy = np.log(len(counts))
        if max_entropy > 0:
            entropy = entropy / max_entropy
    
    return {'shannon_diversity': entropy}


@register_metric(
    name='simpson_diversity',
    category='diversity',
    description='Simpson diversity index (Gini-Simpson)',
    parameters={}
)
def simpson_diversity(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute Simpson diversity index (1 - sum(p_i^2)).
    
    Returns
    -------
    dict
        Key: 'simpson_diversity'.
    """
    counts = data.cell_type_counts
    total = counts.sum()
    
    if total == 0:
        return {'simpson_diversity': np.nan}
    
    props = counts / total
    simpson = 1 - np.sum(props ** 2)
    
    return {'simpson_diversity': simpson}


@register_metric(
    name='marker_statistics',
    category='population',
    description='Statistics of marker expression per cell type',
    parameters={'markers': list, 'stats': list},
    dynamic_columns=True
)
def marker_statistics(
    data: 'SpatialTissueData',
    markers: Optional[List[str]] = None,
    stats: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute marker expression statistics.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data with marker data.
    markers : list of str, optional
        Markers to include. If None, uses all.
    stats : list of str, optional
        Statistics to compute: 'mean', 'std', 'median', 'p25', 'p75'.
        Default: ['mean'].
    
    Returns
    -------
    dict
        Keys: '{marker}_{stat}' or '{marker}_{type}_{stat}'.
    """
    if data.markers is None:
        return {}
    
    if markers is None:
        markers = data.marker_names
    
    if stats is None:
        stats = ['mean']
    
    result = {}
    
    stat_funcs = {
        'mean': np.nanmean,
        'std': np.nanstd,
        'median': np.nanmedian,
        'p25': lambda x: np.nanpercentile(x, 25),
        'p75': lambda x: np.nanpercentile(x, 75),
        'min': np.nanmin,
        'max': np.nanmax,
    }
    
    for marker in markers:
        if marker not in data.marker_names:
            continue
        
        values = data.markers[marker].values
        
        # Global statistics
        for stat in stats:
            if stat in stat_funcs:
                key = f'{marker}_{stat}'
                result[key] = stat_funcs[stat](values)
    
    return result


@register_metric(
    name='marker_statistics_by_type',
    category='population',
    description='Marker expression statistics per cell type',
    parameters={'markers': list, 'stats': list, 'cell_types': list},
    dynamic_columns=True
)
def marker_statistics_by_type(
    data: 'SpatialTissueData',
    markers: Optional[List[str]] = None,
    stats: Optional[List[str]] = None,
    cell_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute marker expression statistics stratified by cell type.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    markers : list of str, optional
        Markers to include.
    stats : list of str, optional
        Statistics: 'mean', 'std', 'median'.
    cell_types : list of str, optional
        Cell types to include.
    
    Returns
    -------
    dict
        Keys: '{marker}_{type}_{stat}'.
    """
    if data.markers is None:
        return {}
    
    if markers is None:
        markers = data.marker_names
    
    if stats is None:
        stats = ['mean']
    
    if cell_types is None:
        cell_types = list(data.cell_types_unique)
    
    stat_funcs = {
        'mean': np.nanmean,
        'std': np.nanstd,
        'median': np.nanmedian,
    }
    
    result = {}
    
    for marker in markers:
        if marker not in data.marker_names:
            continue
        
        for cell_type in cell_types:
            mask = data.cell_types == cell_type
            if mask.sum() == 0:
                continue
            
            values = data.markers.loc[mask, marker].values
            
            for stat in stats:
                if stat in stat_funcs:
                    key = f'{marker}_{cell_type}_{stat}'
                    result[key] = stat_funcs[stat](values)
    
    return result


@register_metric(
    name='spatial_extent',
    category='morphology',
    description='Spatial extent of the tissue sample',
)
def spatial_extent(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute spatial extent metrics.
    
    Returns
    -------
    dict
        Keys: 'extent_x', 'extent_y', 'extent_area', 'extent_z' (if 3D).
    """
    extent = data.extent
    
    result = {
        'extent_x': extent['x'],
        'extent_y': extent['y'],
        'extent_area': extent['x'] * extent['y'],
    }
    
    if 'z' in extent:
        result['extent_z'] = extent['z']
        result['extent_volume'] = extent['x'] * extent['y'] * extent['z']
    
    return result


@register_metric(
    name='centroid',
    category='morphology',
    description='Centroid of all cells',
)
def centroid(data: 'SpatialTissueData') -> Dict[str, float]:
    """
    Compute centroid of all cells.
    
    Returns
    -------
    dict
        Keys: 'centroid_x', 'centroid_y', 'centroid_z' (if 3D).
    """
    coords = data.coordinates
    center = coords.mean(axis=0)
    
    result = {
        'centroid_x': center[0],
        'centroid_y': center[1],
    }
    
    if coords.shape[1] > 2:
        result['centroid_z'] = center[2]
    
    return result
