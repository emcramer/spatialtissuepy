"""
Comparison visualization functions.

This module provides functions for comparing metrics across multiple samples,
including violin plots, heatmaps, PCA/UMAP projections, and trajectory plots.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .config import (
    get_axes, get_categorical_palette, get_sequential_cmap,
    get_diverging_cmap, despine, _check_matplotlib
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    group_by: str,
    kind: str = 'box',
    colors: Optional[Dict[str, str]] = None,
    show_points: bool = True,
    order: Optional[List[str]] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot metric comparison across groups (box/violin/bar plot).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics.
    metric : str
        Column name of metric to plot.
    group_by : str
        Column name for grouping.
    kind : str, default 'box'
        Plot type: 'box', 'violin', 'bar', 'strip'.
    colors : dict, optional
        Custom color mapping for groups.
    show_points : bool, default True
        Overlay individual data points.
    order : list of str, optional
        Order of groups on x-axis.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to seaborn plot function.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")
    if group_by not in df.columns:
        raise ValueError(f"Group column '{group_by}' not found in DataFrame")
    
    groups = df[group_by].unique() if order is None else order
    n_groups = len(groups)
    
    if colors is None:
        palette = get_categorical_palette(n_groups)
        colors = {g: palette[i] for i, g in enumerate(groups)}
    
    # Prepare data
    data_by_group = [df[df[group_by] == g][metric].dropna().values for g in groups]
    positions = range(n_groups)
    
    if kind == 'box':
        bp = ax.boxplot(data_by_group, positions=positions, patch_artist=True, **kwargs)
        for patch, group in zip(bp['boxes'], groups):
            patch.set_facecolor(colors.get(group, '#888888'))
            patch.set_alpha(0.7)
            
    elif kind == 'violin':
        parts = ax.violinplot(data_by_group, positions=positions, showmeans=True, **kwargs)
        for i, (pc, group) in enumerate(zip(parts['bodies'], groups)):
            pc.set_facecolor(colors.get(group, '#888888'))
            pc.set_alpha(0.7)
            
    elif kind == 'bar':
        means = [np.mean(d) for d in data_by_group]
        stds = [np.std(d) for d in data_by_group]
        bars = ax.bar(positions, means, yerr=stds, capsize=3,
                      color=[colors.get(g, '#888888') for g in groups], alpha=0.7, **kwargs)
        
    elif kind == 'strip':
        for i, (data, group) in enumerate(zip(data_by_group, groups)):
            jitter = np.random.uniform(-0.2, 0.2, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data, 
                       c=colors.get(group, '#888888'), alpha=0.6, s=20, **kwargs)
    
    # Overlay points for box/violin
    if show_points and kind in ['box', 'violin']:
        for i, (data, group) in enumerate(zip(data_by_group, groups)):
            jitter = np.random.uniform(-0.15, 0.15, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data, 
                       c='black', alpha=0.4, s=10, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlabel(group_by)
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by {group_by}')
    despine(ax)
    
    return ax


def plot_metric_heatmap(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    sample_col: str = 'sample_id',
    standardize: bool = True,
    cmap: str = 'RdBu_r',
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    annot: bool = False,
    figsize: Tuple[float, float] = (12, 8),
    **kwargs
) -> 'plt.Figure':
    """
    Plot heatmap of metrics across samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples as rows, metrics as columns.
    metrics : list of str, optional
        Metrics to include. If None, uses all numeric columns.
    sample_col : str, default 'sample_id'
        Column identifying samples.
    standardize : bool, default True
        Standardize each metric to z-scores.
    cmap : str, default 'RdBu_r'
        Colormap.
    cluster_rows : bool, default True
        Hierarchically cluster rows (samples).
    cluster_cols : bool, default True
        Hierarchically cluster columns (metrics).
    annot : bool, default False
        Annotate cells with values.
    figsize : tuple, default (12, 8)
        Figure size.
    **kwargs
        Additional arguments to imshow().
        
    Returns
    -------
    plt.Figure
        Figure with heatmap.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    
    # Select metrics
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        if sample_col in metrics:
            metrics.remove(sample_col)
    
    # Prepare data matrix
    if sample_col in df.columns:
        sample_labels = df[sample_col].values
        data_matrix = df[metrics].values
    else:
        sample_labels = np.arange(len(df))
        data_matrix = df[metrics].values
    
    # Standardize
    if standardize:
        data_matrix = (data_matrix - np.nanmean(data_matrix, axis=0)) / np.nanstd(data_matrix, axis=0)
    
    # Handle NaN
    data_matrix = np.nan_to_num(data_matrix, nan=0)
    
    # Clustering
    if cluster_rows and data_matrix.shape[0] > 2:
        row_linkage = hierarchy.linkage(pdist(data_matrix), method='ward')
        row_order = hierarchy.leaves_list(row_linkage)
    else:
        row_order = np.arange(data_matrix.shape[0])
    
    if cluster_cols and data_matrix.shape[1] > 2:
        col_linkage = hierarchy.linkage(pdist(data_matrix.T), method='ward')
        col_order = hierarchy.leaves_list(col_linkage)
    else:
        col_order = np.arange(data_matrix.shape[1])
    
    # Reorder
    data_ordered = data_matrix[row_order][:, col_order]
    metrics_ordered = [metrics[i] for i in col_order]
    samples_ordered = sample_labels[row_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color limits
    vmax = np.percentile(np.abs(data_ordered), 95)
    
    im = ax.imshow(data_ordered, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax, **kwargs)
    plt.colorbar(im, ax=ax, label='Z-score' if standardize else 'Value')
    
    # Add annotations
    if annot:
        for i in range(data_ordered.shape[0]):
            for j in range(data_ordered.shape[1]):
                value = data_ordered[i, j]
                color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=6)
    
    ax.set_xticks(range(len(metrics_ordered)))
    ax.set_xticklabels(metrics_ordered, rotation=90, ha='center')
    ax.set_yticks(range(len(samples_ordered)))
    ax.set_yticklabels(samples_ordered)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Sample')
    ax.set_title('Sample-Metric Heatmap')
    
    fig.tight_layout()
    
    return fig


def plot_violin_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    group_by: str,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    **kwargs
) -> 'plt.Figure':
    """
    Create faceted violin plots for multiple metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics.
    metrics : list of str
        Metrics to plot.
    group_by : str
        Column for grouping.
    ncols : int, default 3
        Number of columns.
    figsize_per_panel : tuple, default (4, 4)
        Size per panel.
    **kwargs
        Additional arguments to violinplot.
        
    Returns
    -------
    plt.Figure
        Figure with faceted violin plots.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    n_metrics = len(metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False
    )
    
    groups = sorted(df[group_by].unique())
    n_groups = len(groups)
    palette = get_categorical_palette(n_groups)
    colors = {g: palette[i] for i, g in enumerate(groups)}
    
    for idx, metric in enumerate(metrics):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{metric}\nnot found', ha='center', va='center')
            continue
        
        data_by_group = [df[df[group_by] == g][metric].dropna().values for g in groups]
        
        parts = ax.violinplot(data_by_group, showmeans=True, **kwargs)
        for i, (pc, group) in enumerate(zip(parts['bodies'], groups)):
            pc.set_facecolor(colors[group])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(1, n_groups + 1))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_title(metric)
        despine(ax)
    
    # Hide empty panels
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    
    fig.tight_layout()
    
    return fig


def plot_pca_samples(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    label_col: Optional[str] = None,
    n_components: int = 2,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot PCA of samples based on metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples and metrics.
    metrics : list of str, optional
        Metrics to use for PCA. If None, uses all numeric columns.
    color_by : str, optional
        Column to use for coloring points.
    label_col : str, optional
        Column to use for point labels.
    n_components : int, default 2
        Number of PCA components.
    colors : dict, optional
        Custom color mapping.
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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    ax = get_axes(ax)
    
    # Select metrics
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare data
    X = df[metrics].values
    X = np.nan_to_num(X, nan=0)
    
    # Standardize and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    if color_by is not None and color_by in df.columns:
        groups = df[color_by].unique()
        if colors is None:
            palette = get_categorical_palette(len(groups))
            colors = {g: palette[i] for i, g in enumerate(groups)}
        
        for group in groups:
            mask = df[color_by] == group
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors.get(group, '#888888'), label=group, **kwargs)
        ax.legend(frameon=False)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], **kwargs)
    
    # Add labels
    if label_col is not None and label_col in df.columns:
        for i, label in enumerate(df[label_col]):
            ax.annotate(str(label), (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA of Samples')
    despine(ax)
    
    return ax


def plot_umap_samples(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    label_col: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot UMAP of samples based on metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with samples and metrics.
    metrics : list of str, optional
        Metrics to use. If None, uses all numeric columns.
    color_by : str, optional
        Column for coloring.
    label_col : str, optional
        Column for labels.
    n_neighbors : int, default 15
        UMAP n_neighbors parameter.
    min_dist : float, default 0.1
        UMAP min_dist parameter.
    colors : dict, optional
        Custom color mapping.
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
    
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn required for UMAP plots. Install with: pip install umap-learn")
    
    from sklearn.preprocessing import StandardScaler
    
    ax = get_axes(ax)
    
    # Select metrics
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare data
    X = df[metrics].values
    X = np.nan_to_num(X, nan=0)
    
    # Standardize and UMAP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    # Plot
    if color_by is not None and color_by in df.columns:
        groups = df[color_by].unique()
        if colors is None:
            palette = get_categorical_palette(len(groups))
            colors = {g: palette[i] for i, g in enumerate(groups)}
        
        for group in groups:
            mask = df[color_by] == group
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=colors.get(group, '#888888'), label=group, **kwargs)
        ax.legend(frameon=False)
    else:
        ax.scatter(X_umap[:, 0], X_umap[:, 1], **kwargs)
    
    # Add labels
    if label_col is not None and label_col in df.columns:
        for i, label in enumerate(df[label_col]):
            ax.annotate(str(label), (X_umap[i, 0], X_umap[i, 1]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP of Samples')
    despine(ax)
    
    return ax


def plot_sample_correlation(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    method: str = 'pearson',
    cmap: str = 'RdBu_r',
    annot: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot correlation matrix between samples.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics (samples as rows).
    metrics : list of str, optional
        Metrics to use for correlation.
    method : str, default 'pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'.
    cmap : str, default 'RdBu_r'
        Colormap.
    annot : bool, default True
        Annotate with values.
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
    
    ax = get_axes(ax)
    
    # Select metrics
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute correlation between samples (transpose to get samples as columns)
    data = df[metrics].T
    corr = data.corr(method=method)
    
    # Plot heatmap
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, **kwargs)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add annotations
    if annot:
        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    ax.set_title(f'Sample Correlation ({method})')
    
    return ax


def plot_trajectory(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, List[str]],
    group_by: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    show_points: bool = True,
    show_error: bool = True,
    error_type: str = 'std',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot metric trajectories over time or another continuous variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trajectory data.
    x_col : str
        Column for x-axis (e.g., 'time').
    y_cols : str or list of str
        Column(s) for y-axis.
    group_by : str, optional
        Column for grouping (separate lines).
    colors : dict, optional
        Custom color mapping.
    show_points : bool, default True
        Show data points.
    show_error : bool, default True
        Show error bands.
    error_type : str, default 'std'
        Error type: 'std', 'sem', 'ci95'.
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
    
    ax = get_axes(ax)
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    # Determine groups
    if group_by is not None:
        groups = sorted(df[group_by].unique())
    else:
        groups = ['all']
    
    if colors is None:
        palette = get_categorical_palette(len(groups) * len(y_cols))
        color_idx = 0
    
    for y_col in y_cols:
        for group in groups:
            if group_by is not None:
                mask = df[group_by] == group
                subset = df[mask]
                label = f'{y_col} ({group})' if len(y_cols) > 1 else str(group)
            else:
                subset = df
                label = y_col
            
            x = subset[x_col].values
            y = subset[y_col].values
            
            # Get color
            if colors is not None:
                color = colors.get(group, colors.get(y_col, None))
            else:
                color = palette[color_idx]
                color_idx += 1
            
            # Plot line
            sort_idx = np.argsort(x)
            ax.plot(x[sort_idx], y[sort_idx], color=color, label=label, **kwargs)
            
            # Show points
            if show_points:
                ax.scatter(x, y, color=color, s=20, alpha=0.6)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_cols[0] if len(y_cols) == 1 else 'Value')
    ax.set_title('Trajectory')
    ax.legend(frameon=False)
    despine(ax)
    
    return ax
