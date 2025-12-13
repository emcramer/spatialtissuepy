"""
Quality control and diagnostic visualization functions.

This module provides functions for visualizing model quality, stability,
convergence, and data coverage diagnostics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import (
    get_axes, get_categorical_palette, despine, _check_matplotlib
)

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    import matplotlib.pyplot as plt


def plot_cell_count_summary(
    data: 'SpatialTissueData',
    sort_by: str = 'count',
    horizontal: bool = True,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot cell count summary bar chart.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    sort_by : str, default 'count'
        Sort by: 'count', 'name', or None.
    horizontal : bool, default True
        Use horizontal bars.
    colors : dict, optional
        Custom color mapping.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to bar/barh.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from .config import get_cell_type_colors
    
    ax = get_axes(ax)
    
    cell_types = data._cell_types
    unique_types, counts = np.unique(cell_types, return_counts=True)
    
    # Sort
    if sort_by == 'count':
        sort_idx = np.argsort(counts)[::-1]
    elif sort_by == 'name':
        sort_idx = np.argsort(unique_types)
    else:
        sort_idx = np.arange(len(unique_types))
    
    unique_types = unique_types[sort_idx]
    counts = counts[sort_idx]
    
    if colors is None:
        colors = get_cell_type_colors(list(unique_types))
    
    bar_colors = [colors.get(ct, '#888888') for ct in unique_types]
    
    if horizontal:
        ax.barh(range(len(unique_types)), counts, color=bar_colors, **kwargs)
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.set_xlabel('Count')
        ax.invert_yaxis()
    else:
        ax.bar(range(len(unique_types)), counts, color=bar_colors, **kwargs)
        ax.set_xticks(range(len(unique_types)))
        ax.set_xticklabels(unique_types, rotation=45, ha='right')
        ax.set_ylabel('Count')
    
    ax.set_title(f'Cell Type Distribution (n={data.n_cells})')
    despine(ax)
    
    return ax


def plot_spatial_coverage(
    data: 'SpatialTissueData',
    resolution: int = 50,
    cell_type: Optional[str] = None,
    cmap: str = 'viridis',
    show_points: bool = False,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot spatial coverage heatmap.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    resolution : int, default 50
        Grid resolution.
    cell_type : str, optional
        Cell type to analyze.
    cmap : str, default 'viridis'
        Colormap.
    show_points : bool, default False
        Overlay cell positions.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to imshow.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    # Get coordinates
    if cell_type is not None:
        mask = data._cell_types == cell_type
        coords = data._coordinates[mask]
        title_suffix = f' ({cell_type})'
    else:
        coords = data._coordinates
        title_suffix = ''
    
    bounds = data.bounds
    
    # Create grid
    x_edges = np.linspace(bounds['x'][0], bounds['x'][1], resolution + 1)
    y_edges = np.linspace(bounds['y'][0], bounds['y'][1], resolution + 1)
    
    # Count cells in each grid cell
    hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_edges, y_edges])
    
    # Plot heatmap
    extent = [bounds['x'][0], bounds['x'][1], bounds['y'][0], bounds['y'][1]]
    im = ax.imshow(hist.T, origin='lower', extent=extent, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax, label='Cell Count')
    
    # Overlay points
    if show_points:
        ax.scatter(coords[:, 0], coords[:, 1], c='white', s=1, alpha=0.3, rasterized=True)
    
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'Spatial Coverage{title_suffix}')
    
    return ax


def plot_model_selection(
    metrics_df: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    x_col: str = 'n_topics',
    highlight_best: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot model selection metrics (e.g., for choosing number of LDA topics).
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with model selection metrics.
    metric_cols : list of str, optional
        Columns to plot. If None, uses all numeric columns except x_col.
    x_col : str, default 'n_topics'
        Column for x-axis.
    highlight_best : bool, default True
        Highlight optimal values.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to plot.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    if metric_cols is None:
        metric_cols = [c for c in metrics_df.select_dtypes(include=[np.number]).columns 
                       if c != x_col]
    
    x = metrics_df[x_col].values
    colors = get_categorical_palette(len(metric_cols))
    
    for i, col in enumerate(metric_cols):
        y = metrics_df[col].values
        ax.plot(x, y, 'o-', color=colors[i], label=col, **kwargs)
        
        if highlight_best:
            # For perplexity-like metrics (lower is better), find min
            # For coherence-like metrics (higher is better), find max
            if 'perplexity' in col.lower() or 'loss' in col.lower():
                best_idx = np.argmin(y)
            else:
                best_idx = np.argmax(y)
            
            ax.scatter([x[best_idx]], [y[best_idx]], s=100, c=colors[i], 
                       marker='*', zorder=10, edgecolor='black')
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Selection')
    ax.legend(frameon=False)
    despine(ax)
    
    return ax


def plot_stability_analysis(
    stability_df: pd.DataFrame,
    metric: str = 'n_nodes',
    x_col: str = 'run',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot stability analysis results from repeated runs.
    
    Parameters
    ----------
    stability_df : pd.DataFrame
        DataFrame with results from multiple runs.
    metric : str, default 'n_nodes'
        Metric to plot.
    x_col : str, default 'run'
        Column for x-axis (run index).
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    if x_col in stability_df.columns:
        x = stability_df[x_col].values
    else:
        x = np.arange(len(stability_df))
    
    y = stability_df[metric].values
    mean_val = np.mean(y)
    std_val = np.std(y)
    cv = std_val / mean_val if mean_val > 0 else 0
    
    # Plot points
    ax.scatter(x, y, alpha=0.6, s=50, **kwargs)
    
    # Plot mean and std bands
    ax.axhline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean={mean_val:.2f}')
    ax.axhline(mean_val + std_val, color='red', linestyle='--', alpha=0.5)
    ax.axhline(mean_val - std_val, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(ax.get_xlim(), mean_val - std_val, mean_val + std_val, 
                    alpha=0.1, color='red')
    
    ax.set_xlabel('Run' if x_col == 'run' else x_col)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Stability Analysis: CV={cv:.3f}')
    ax.legend(frameon=False)
    despine(ax)
    
    return ax


def plot_convergence(
    history: Union[List[float], np.ndarray],
    metric_name: str = 'Loss',
    log_scale: bool = False,
    show_best: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot convergence history.
    
    Parameters
    ----------
    history : array-like
        Sequence of metric values over iterations.
    metric_name : str, default 'Loss'
        Name of the metric.
    log_scale : bool, default False
        Use log scale for y-axis.
    show_best : bool, default True
        Highlight best value.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to plot.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    history = np.asarray(history)
    iterations = np.arange(len(history))
    
    ax.plot(iterations, history, 'b-', linewidth=1.5, **kwargs)
    
    if show_best:
        if 'loss' in metric_name.lower() or 'perplexity' in metric_name.lower():
            best_idx = np.argmin(history)
        else:
            best_idx = np.argmax(history)
        
        ax.scatter([best_idx], [history[best_idx]], s=100, c='red', 
                   marker='*', zorder=10, label=f'Best: {history[best_idx]:.4f}')
        ax.legend(frameon=False)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name)
    ax.set_title('Convergence')
    despine(ax)
    
    return ax


def plot_parameter_sweep(
    results_df: pd.DataFrame,
    param_col: str,
    metric_col: str,
    group_by: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot results of parameter sweep.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with sweep results.
    param_col : str
        Column with parameter values.
    metric_col : str
        Column with metric values.
    group_by : str, optional
        Column for grouping (multiple lines).
    colors : dict, optional
        Custom color mapping for groups.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to plot.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    if group_by is not None:
        groups = results_df[group_by].unique()
        if colors is None:
            palette = get_categorical_palette(len(groups))
            colors = {g: palette[i] for i, g in enumerate(groups)}
        
        for group in groups:
            mask = results_df[group_by] == group
            subset = results_df[mask].sort_values(param_col)
            ax.plot(subset[param_col], subset[metric_col], 'o-',
                    color=colors.get(group, '#888888'), label=str(group), **kwargs)
        
        ax.legend(title=group_by, frameon=False)
    else:
        subset = results_df.sort_values(param_col)
        ax.plot(subset[param_col], subset[metric_col], 'o-', **kwargs)
    
    ax.set_xlabel(param_col.replace('_', ' ').title())
    ax.set_ylabel(metric_col.replace('_', ' ').title())
    ax.set_title('Parameter Sweep')
    despine(ax)
    
    return ax


def plot_sample_qc_summary(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    sample_col: str = 'sample_id',
    figsize: Tuple[float, float] = (12, 8),
    **kwargs
) -> 'plt.Figure':
    """
    Plot QC summary for multiple samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with QC metrics per sample.
    metrics : list of str, optional
        Metrics to include.
    thresholds : dict, optional
        Dict of metric -> (min, max) acceptable range.
    sample_col : str, default 'sample_id'
        Column identifying samples.
    figsize : tuple, default (12, 8)
        Figure size.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    plt.Figure
        Figure with QC panels.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    if metrics is None:
        metrics = ['n_cells', 'n_cell_types', 'density']
        metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        metrics = df.select_dtypes(include=[np.number]).columns[:4].tolist()
    
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    for idx, metric in enumerate(metrics):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        values = df[metric].values
        samples = df[sample_col].values if sample_col in df.columns else np.arange(len(df))
        
        colors = ['steelblue'] * len(values)
        
        # Check thresholds
        if thresholds and metric in thresholds:
            min_val, max_val = thresholds[metric]
            for i, v in enumerate(values):
                if v < min_val or v > max_val:
                    colors[i] = 'red'
        
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xlabel('Sample')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        
        if thresholds and metric in thresholds:
            min_val, max_val = thresholds[metric]
            ax.axhline(min_val, color='red', linestyle='--', alpha=0.5)
            ax.axhline(max_val, color='red', linestyle='--', alpha=0.5)
        
        despine(ax)
    
    # Hide empty panels
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    
    fig.tight_layout()
    
    return fig
