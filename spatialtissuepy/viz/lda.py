"""
Spatial LDA visualization functions.

This module provides functions for visualizing Spatial LDA results,
including topic compositions, spatial distributions, enrichment heatmaps,
and model diagnostics.
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
    from spatialtissuepy.core import SpatialTissueData
    from spatialtissuepy.lda import SpatialLDA
    import matplotlib.pyplot as plt


def plot_topic_composition(
    model: 'SpatialLDA',
    n_top: int = 5,
    normalize: bool = True,
    cmap: str = 'viridis',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot topic-cell type composition matrix.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    n_top : int, default 5
        Number of top cell types to highlight per topic.
    normalize : bool, default True
        Normalize rows to sum to 1.
    cmap : str, default 'viridis'
        Colormap.
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
    
    # Get topic-word matrix (topics x cell types)
    topic_word = model.model.components_
    
    if normalize:
        topic_word = topic_word / topic_word.sum(axis=1, keepdims=True)
    
    # Get cell type names
    cell_types = model._cell_types_order
    
    # Plot heatmap
    im = ax.imshow(topic_word, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax, label='Weight' if not normalize else 'Proportion')
    
    ax.set_yticks(range(model.n_topics))
    ax.set_yticklabels([f'Topic {i}' for i in range(model.n_topics)])
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Topic')
    ax.set_title('Topic-Cell Type Composition')
    
    return ax


def plot_topic_spatial(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    topic: int = 0,
    size: float = 5,
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot spatial distribution of a single topic.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    data : SpatialTissueData
        Spatial tissue data.
    topic : int, default 0
        Topic index to visualize.
    size : float, default 5
        Point size.
    cmap : str, default 'viridis'
        Colormap.
    show_colorbar : bool, default True
        Show colorbar.
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
    
    ax = get_axes(ax)
    
    # Get topic weights
    topic_weights = model.transform(data)
    
    if topic >= topic_weights.shape[1]:
        raise ValueError(f"Topic {topic} not found. Model has {model.n_topics} topics.")
    
    coords = data._coordinates
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=topic_weights[:, topic], s=size, cmap=cmap,
        rasterized=True, **kwargs
    )
    
    if show_colorbar:
        plt.colorbar(scatter, ax=ax, label=f'Topic {topic} weight')
    
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'Topic {topic} Spatial Distribution')
    ax.set_aspect('equal')
    despine(ax)
    
    return ax


def plot_topic_enrichment_heatmap(
    model: 'SpatialLDA',
    cmap: str = 'RdBu_r',
    center: float = 0.0,
    annot: bool = True,
    fmt: str = '.2f',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot topic-cell type enrichment heatmap (log2 fold change).
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    cmap : str, default 'RdBu_r'
        Colormap.
    center : float, default 0.0
        Center value for diverging colormap.
    annot : bool, default True
        Annotate with values.
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
    from spatialtissuepy.lda import topic_enrichment
    
    ax = get_axes(ax)
    
    # Compute enrichment
    enrichment = topic_enrichment(model)
    
    cell_types = model._cell_types_order
    
    # Determine color limits
    vmax = np.nanpercentile(np.abs(enrichment), 95)
    
    # Plot heatmap
    im = ax.imshow(enrichment, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax, **kwargs)
    plt.colorbar(im, ax=ax, label='Log2 Fold Enrichment')
    
    # Add annotations
    if annot:
        for i in range(enrichment.shape[0]):
            for j in range(enrichment.shape[1]):
                value = enrichment[i, j]
                if np.isnan(value):
                    continue
                color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, format(value, fmt), ha='center', va='center', color=color, fontsize=8)
    
    ax.set_yticks(range(model.n_topics))
    ax.set_yticklabels([f'Topic {i}' for i in range(model.n_topics)])
    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Topic')
    ax.set_title('Topic-Cell Type Enrichment')
    
    return ax


def plot_topic_transition_matrix(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    radius: float = 50.0,
    normalize: bool = True,
    cmap: str = 'Blues',
    annot: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot topic transition/co-occurrence matrix.
    
    Shows how often topics co-occur in neighboring cells.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    data : SpatialTissueData
        Spatial tissue data.
    radius : float, default 50.0
        Neighborhood radius.
    normalize : bool, default True
        Normalize to proportions.
    cmap : str, default 'Blues'
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
    from scipy.spatial import cKDTree
    
    ax = get_axes(ax)
    
    # Get dominant topics
    dominant = model.predict(data)
    coords = data._coordinates
    
    # Build spatial neighbor graph
    tree = cKDTree(coords)
    
    # Count topic transitions
    n_topics = model.n_topics
    transition_matrix = np.zeros((n_topics, n_topics))
    
    for i, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, radius)
        for j in neighbors:
            if i != j:
                transition_matrix[dominant[i], dominant[j]] += 1
    
    if normalize:
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
    
    # Plot heatmap
    im = ax.imshow(transition_matrix, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax, label='Proportion' if normalize else 'Count')
    
    # Add annotations
    if annot:
        for i in range(n_topics):
            for j in range(n_topics):
                value = transition_matrix[i, j]
                color = 'white' if value > transition_matrix.max() * 0.5 else 'black'
                fmt = '.2f' if normalize else '.0f'
                ax.text(j, i, format(value, fmt), ha='center', va='center', color=color, fontsize=8)
    
    ax.set_xticks(range(n_topics))
    ax.set_yticks(range(n_topics))
    ax.set_xticklabels([f'Topic {i}' for i in range(n_topics)])
    ax.set_yticklabels([f'Topic {i}' for i in range(n_topics)])
    ax.set_xlabel('Neighbor Topic')
    ax.set_ylabel('Cell Topic')
    ax.set_title('Topic Transition Matrix')
    
    return ax


def plot_lda_diagnostics(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    figsize: Tuple[float, float] = (14, 10),
    **kwargs
) -> 'plt.Figure':
    """
    Create comprehensive LDA diagnostic plots.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    data : SpatialTissueData
        Spatial tissue data.
    figsize : tuple, default (14, 10)
        Figure size.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    plt.Figure
        Figure with diagnostic panels.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from spatialtissuepy.lda import (
        topic_assignment_uncertainty,
        topic_spatial_distribution,
    )
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Topic composition heatmap
    plot_topic_composition(model, ax=axes[0, 0])
    
    # 2. Topic enrichment heatmap
    plot_topic_enrichment_heatmap(model, ax=axes[0, 1])
    
    # 3. Topic prevalence bar chart
    topic_weights = model.transform(data)
    topic_means = topic_weights.mean(axis=0)
    
    ax = axes[0, 2]
    palette = get_categorical_palette(model.n_topics)
    ax.bar(range(model.n_topics), topic_means, color=palette)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Topic Prevalence')
    ax.set_xticks(range(model.n_topics))
    despine(ax)
    
    # 4. Assignment uncertainty distribution
    ax = axes[1, 0]
    uncertainty = topic_assignment_uncertainty(model, data)
    ax.hist(uncertainty, bins=50, alpha=0.7, color='steelblue')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Assignment Uncertainty')
    ax.axvline(np.mean(uncertainty), color='red', linestyle='--', label=f'Mean={np.mean(uncertainty):.2f}')
    ax.legend(frameon=False)
    despine(ax)
    
    # 5. Dominant topic spatial plot
    ax = axes[1, 1]
    coords = data._coordinates
    dominant = model.predict(data)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=dominant, s=3, cmap='tab10', rasterized=True)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title('Dominant Topic')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Topic')
    despine(ax)
    
    # 6. Perplexity info
    ax = axes[1, 2]
    ax.axis('off')
    
    try:
        perplexity = model.perplexity(data)
        perplexity_text = f'{perplexity:.2f}'
    except Exception:
        perplexity_text = 'N/A'
    
    info_text = [
        'Model Summary',
        '=' * 30,
        f'Number of topics: {model.n_topics}',
        f'Number of cell types: {len(model._cell_types_order)}',
        f'Perplexity: {perplexity_text}',
        f'Mean assignment entropy: {np.mean(uncertainty):.3f}',
        f'Max topic weight (mean): {topic_means.max():.3f}',
        '',
        'Cell Types',
        '=' * 30,
        ', '.join(model._cell_types_order[:10]),
        '...' if len(model._cell_types_order) > 10 else '',
    ]
    
    ax.text(0.1, 0.9, '\n'.join(info_text), transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top')
    
    fig.tight_layout()
    
    return fig


def plot_topic_proportions_bar(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    by_cell_type: bool = False,
    stacked: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot topic proportion bar chart.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    data : SpatialTissueData
        Spatial tissue data.
    by_cell_type : bool, default False
        If True, show topic proportions by cell type.
    stacked : bool, default True
        Use stacked bars.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to bar().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    topic_weights = model.transform(data)
    palette = get_categorical_palette(model.n_topics)
    
    if by_cell_type:
        # Group by cell type
        cell_types = data._cell_types
        unique_types = data.cell_types_unique
        n_types = len(unique_types)
        
        # Compute mean topic weights per cell type
        type_topic_means = np.zeros((n_types, model.n_topics))
        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            type_topic_means[i] = topic_weights[mask].mean(axis=0)
        
        x = np.arange(n_types)
        
        if stacked:
            bottom = np.zeros(n_types)
            for t in range(model.n_topics):
                ax.bar(x, type_topic_means[:, t], bottom=bottom,
                       label=f'Topic {t}', color=palette[t], **kwargs)
                bottom += type_topic_means[:, t]
        else:
            width = 0.8 / model.n_topics
            for t in range(model.n_topics):
                ax.bar(x + t * width, type_topic_means[:, t],
                       width=width, label=f'Topic {t}', color=palette[t], **kwargs)
        
        ax.set_xticks(x + (0 if stacked else width * (model.n_topics - 1) / 2))
        ax.set_xticklabels(unique_types, rotation=45, ha='right')
        ax.set_xlabel('Cell Type')
        
    else:
        # Overall topic proportions
        topic_means = topic_weights.mean(axis=0)
        topic_stds = topic_weights.std(axis=0)
        
        x = np.arange(model.n_topics)
        ax.bar(x, topic_means, yerr=topic_stds, capsize=3,
               color=palette[:model.n_topics], **kwargs)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'Topic {i}' for i in range(model.n_topics)])
        ax.set_xlabel('Topic')
    
    ax.set_ylabel('Mean Weight')
    ax.set_title('Topic Proportions')
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    despine(ax)
    
    return ax


def plot_topic_spatial_grid(
    model: 'SpatialLDA',
    data: 'SpatialTissueData',
    ncols: int = 3,
    size: float = 3,
    cmap: str = 'viridis',
    figsize_per_panel: Tuple[float, float] = (4, 4),
    **kwargs
) -> 'plt.Figure':
    """
    Plot all topics in a faceted grid.
    
    Parameters
    ----------
    model : SpatialLDA
        Fitted Spatial LDA model.
    data : SpatialTissueData
        Spatial tissue data.
    ncols : int, default 3
        Number of columns.
    size : float, default 3
        Point size.
    cmap : str, default 'viridis'
        Colormap.
    figsize_per_panel : tuple, default (4, 4)
        Size per panel.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Figure
        Figure with topic panels.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    n_topics = model.n_topics
    nrows = int(np.ceil(n_topics / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False
    )
    
    topic_weights = model.transform(data)
    coords = data._coordinates
    
    for t in range(n_topics):
        row = t // ncols
        col = t % ncols
        ax = axes[row, col]
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=topic_weights[:, t], s=size, cmap=cmap,
            rasterized=True, **kwargs
        )
        
        ax.set_title(f'Topic {t}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        despine(ax, left=True, bottom=True)
    
    # Hide empty panels
    for idx in range(n_topics, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    
    fig.tight_layout()
    
    return fig
