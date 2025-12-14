"""
Statistics visualization functions.

This module provides functions for visualizing spatial statistics,
including Ripley's functions, pair correlation functions, co-localization
analysis, and hotspot detection.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    get_axes, get_cell_type_colors, get_sequential_cmap,
    get_diverging_cmap, despine, _check_matplotlib
)

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    import matplotlib.pyplot as plt


def plot_ripleys_curve(
    data: 'SpatialTissueData',
    radii: Optional[np.ndarray] = None,
    cell_type: Optional[str] = None,
    statistic: str = 'H',
    n_simulations: int = 99,
    confidence_level: float = 0.95,
    show_envelope: bool = True,
    show_csr: bool = True,
    color: str = '#1f77b4',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot Ripley's K, L, or H function with confidence envelope.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radii : np.ndarray, optional
        Radii to evaluate. If None, auto-determined.
    cell_type : str, optional
        Cell type to analyze. If None, uses all cells.
    statistic : str, default 'H'
        Statistic to plot: 'K', 'L', or 'H'.
    n_simulations : int, default 99
        Number of CSR simulations for envelope.
    confidence_level : float, default 0.95
        Confidence level for envelope.
    show_envelope : bool, default True
        Show confidence envelope from CSR simulations.
    show_csr : bool, default True
        Show expected value under CSR.
    color : str, default '#1f77b4'
        Line color.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to plot().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.statistics import ripleys_k, ripleys_l, ripleys_h
    
    ax = get_axes(ax)
    
    # Get coordinates
    if cell_type is not None:
        mask = data._cell_types == cell_type
        coords = data._coordinates[mask]
        title_suffix = f' ({cell_type})'
    else:
        coords = data._coordinates
        title_suffix = ''
    
    # Determine radii
    if radii is None:
        max_dist = min(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) / 4
        radii = np.linspace(0, max_dist, 50)
    
    # Compute statistic
    if statistic == 'K':
        values = ripleys_k(coords, radii)
        ylabel = "Ripley's K(r)"
        csr_values = np.pi * radii**2
    elif statistic == 'L':
        values = ripleys_l(coords, radii)
        ylabel = "Ripley's L(r)"
        csr_values = radii
    elif statistic == 'H':
        values = ripleys_h(coords, radii)
        ylabel = "Ripley's H(r)"
        csr_values = np.zeros_like(radii)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Plot CSR expectation
    if show_csr:
        ax.plot(radii, csr_values, '--', color='gray', alpha=0.7, label='CSR')
    
    # Plot observed values
    ax.plot(radii, values, color=color, linewidth=2, label='Observed', **kwargs)
    
    ax.set_xlabel('Distance r (um)')
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}{title_suffix}")
    ax.legend(frameon=False)
    if statistic == 'H':
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    despine(ax)
    
    return ax


def plot_pcf_curve(
    data: 'SpatialTissueData',
    radii: Optional[np.ndarray] = None,
    cell_type: Optional[str] = None,
    color: str = '#1f77b4',
    show_csr: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot pair correlation function g(r).
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radii : np.ndarray, optional
        Radii to evaluate.
    cell_type : str, optional
        Cell type to analyze.
    color : str, default '#1f77b4'
        Line color.
    show_csr : bool, default True
        Show g(r) = 1 reference line.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to plot().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.statistics import pair_correlation_function
    
    ax = get_axes(ax)
    
    # Get coordinates
    if cell_type is not None:
        mask = data._cell_types == cell_type
        coords = data._coordinates[mask]
        title_suffix = f' ({cell_type})'
    else:
        coords = data._coordinates
        title_suffix = ''
    
    # Determine radii
    if radii is None:
        max_dist = min(coords[:, 0].ptp(), coords[:, 1].ptp()) / 4
        radii = np.linspace(1, max_dist, 50)
    
    # Compute PCF
    g_values = pair_correlation_function(coords, radii)
    
    # Plot
    if show_csr:
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='CSR (g=1)')
    
    ax.plot(radii, g_values, color=color, linewidth=2, label='Observed', **kwargs)
    
    ax.set_xlabel('Distance r (um)')
    ax.set_ylabel('g(r)')
    ax.set_title(f"Pair Correlation Function{title_suffix}")
    ax.legend(frameon=False)
    despine(ax)
    
    return ax


def plot_colocalization_heatmap(
    data: 'SpatialTissueData',
    radius: float = 50.0,
    metric: str = 'clq',
    cell_types: Optional[List[str]] = None,
    cmap: str = 'RdBu_r',
    center: float = 1.0,
    annot: bool = True,
    fmt: str = '.2f',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot co-localization heatmap between cell types.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    radius : float, default 50.0
        Radius for co-localization analysis.
    metric : str, default 'clq'
        Metric: 'clq' (co-localization quotient) or 'enrichment'.
    cell_types : list of str, optional
        Cell types to include. If None, uses all.
    cmap : str, default 'RdBu_r'
        Colormap.
    center : float, default 1.0
        Center value for diverging colormap.
    annot : bool, default True
        Annotate cells with values.
    fmt : str, default '.2f'
        Number format.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to imshow().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.statistics import colocalization_quotient
    
    ax = get_axes(ax)
    
    if cell_types is None:
        cell_types = list(data.cell_types_unique)
    
    n_types = len(cell_types)
    matrix = np.zeros((n_types, n_types))
    
    # Compute co-localization for each pair
    for i, type_a in enumerate(cell_types):
        for j, type_b in enumerate(cell_types):
            try:
                matrix[i, j] = colocalization_quotient(data, type_a, type_b, radius)
            except Exception:
                matrix[i, j] = np.nan
    
    # Determine color limits
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) > 0:
        vmax = max(abs(valid_values.max() - center), abs(valid_values.min() - center)) + center
        vmin = 2 * center - vmax
    else:
        vmin, vmax = 0, 2
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, **kwargs)
    plt.colorbar(im, ax=ax, label='CLQ' if metric == 'clq' else 'Enrichment')
    
    # Add annotations
    if annot:
        for i in range(n_types):
            for j in range(n_types):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                color = 'white' if abs(value - center) > (vmax - center) * 0.5 else 'black'
                ax.text(j, i, format(value, fmt), ha='center', va='center', color=color)
    
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_yticklabels(cell_types)
    ax.set_xlabel('Cell Type B')
    ax.set_ylabel('Cell Type A')
    ax.set_title(f'Co-localization (r={radius}um)')
    
    return ax


def plot_neighborhood_enrichment(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    radius: float = 50.0,
    n_permutations: int = 999,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot neighborhood enrichment test result.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    type_a : str
        First cell type.
    type_b : str
        Second cell type.
    radius : float, default 50.0
        Neighborhood radius.
    n_permutations : int, default 999
        Number of permutations.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to hist().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.statistics import neighborhood_enrichment_test
    
    ax = get_axes(ax)
    
    # Run enrichment test
    result = neighborhood_enrichment_test(data, type_a, type_b, radius, n_permutations)
    
    # Plot null distribution
    ax.hist(result['null_distribution'], bins=30, alpha=0.7, color='gray', 
            label='Null distribution', **kwargs)
    
    # Plot observed value
    ax.axvline(result['observed'], color='red', linewidth=2,
               label=f"Observed ({result['observed']:.2f})")
    
    # Add p-value annotation
    pval = result['pvalue']
    pval_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
    
    ax.text(0.95, 0.95, pval_str, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Neighborhood Enrichment: {type_a} -> {type_b}')
    ax.legend(frameon=False)
    despine(ax)
    
    return ax


def plot_hotspot_map(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float = 50.0,
    alpha_level: float = 0.05,
    size: float = 10,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot hotspot analysis results.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze for hotspots.
    radius : float, default 50.0
        Neighborhood radius.
    alpha_level : float, default 0.05
        Significance level.
    size : float, default 10
        Point size.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.statistics import detect_hotspots
    
    ax = get_axes(ax)
    coords = data._coordinates
    
    # Detect hotspots
    result = detect_hotspots(data, values, radius, alpha_level)
    
    # Plot non-significant points
    ns_mask = result['classification'] == 'not_significant'
    ax.scatter(coords[ns_mask, 0], coords[ns_mask, 1], c='#cccccc', s=size * 0.5, 
               alpha=0.5, label='Not significant', rasterized=True)
    
    # Plot hotspots
    hot_mask = result['classification'] == 'hotspot'
    if np.any(hot_mask):
        ax.scatter(coords[hot_mask, 0], coords[hot_mask, 1], 
                   c=result['z_scores'][hot_mask], cmap='Reds', s=size, vmin=0,
                   label=f'Hotspots (n={np.sum(hot_mask)})', rasterized=True, **kwargs)
    
    # Plot coldspots
    cold_mask = result['classification'] == 'coldspot'
    if np.any(cold_mask):
        ax.scatter(coords[cold_mask, 0], coords[cold_mask, 1],
                   c=result['z_scores'][cold_mask], cmap='Blues_r', s=size, vmax=0,
                   label=f'Coldspots (n={np.sum(cold_mask)})', rasterized=True, **kwargs)
    
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'Hotspot Analysis (alpha={alpha_level})')
    ax.legend(frameon=False, loc='upper left')
    ax.set_aspect('equal')
    despine(ax)
    
    return ax


def plot_morans_scatter(
    data: 'SpatialTissueData',
    values: np.ndarray,
    radius: float = 50.0,
    standardize: bool = True,
    color: str = '#1f77b4',
    show_regression: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot Moran's I scatter plot (spatial lag vs. value).
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    values : np.ndarray
        Values to analyze.
    radius : float, default 50.0
        Neighborhood radius for spatial weights.
    standardize : bool, default True
        Standardize values to z-scores.
    color : str, default '#1f77b4'
        Point color.
    show_regression : bool, default True
        Show regression line.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from scipy import stats
    
    ax = get_axes(ax)
    coords = data._coordinates
    
    # Standardize values
    if standardize:
        values = (values - np.mean(values)) / np.std(values)
    
    # Compute spatial lag
    tree = cKDTree(coords)
    spatial_lag = np.zeros_like(values)
    
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, radius)
        neighbors = [n for n in neighbors if n != i]
        if neighbors:
            spatial_lag[i] = np.mean(values[neighbors])
    
    # Plot scatter
    ax.scatter(values, spatial_lag, c=color, s=10, alpha=0.5, rasterized=True, **kwargs)
    
    # Add regression line
    if show_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(values, spatial_lag)
        x_line = np.array([values.min(), values.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"Slope={slope:.3f} (Moran's I)")
        ax.legend(frameon=False)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Value (z-score)' if standardize else 'Value')
    ax.set_ylabel('Spatial Lag')
    ax.set_title("Moran's I Scatter Plot")
    despine(ax)
    
    return ax
