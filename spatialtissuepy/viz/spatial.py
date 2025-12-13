"""
Spatial visualization functions.

This module provides functions for visualizing cells in spatial coordinates,
including scatter plots, density maps, Voronoi diagrams, and neighborhood
visualizations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    get_axes, get_cell_type_colors, get_sequential_cmap,
    despine, add_scalebar, _check_matplotlib
)

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    import matplotlib.pyplot as plt


def plot_spatial_scatter(
    data: 'SpatialTissueData',
    color_by: str = 'cell_type',
    marker: Optional[str] = None,
    size: float = 5,
    alpha: float = 0.7,
    colors: Optional[Dict[str, str]] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    xlabel: str = 'X (µm)',
    ylabel: str = 'Y (µm)',
    scalebar: Optional[float] = None,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Basic spatial scatter plot of cells.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    color_by : str, default 'cell_type'
        Coloring scheme: 'cell_type', a marker name, or 'density'.
    marker : str, optional
        Marker name for continuous coloring. Overrides color_by.
    size : float, default 5
        Point size.
    alpha : float, default 0.7
        Point transparency.
    colors : dict, optional
        Custom color mapping for cell types.
    cmap : str, default 'viridis'
        Colormap for continuous values.
    vmin, vmax : float, optional
        Value range for colormap.
    show_colorbar : bool, default True
        Show colorbar for continuous coloring.
    show_legend : bool, default True
        Show legend for categorical coloring.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    scalebar : float, optional
        If provided, add scale bar of this length.
    ax : plt.Axes, optional
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
        
    Examples
    --------
    >>> # Color by cell type
    >>> plot_spatial_scatter(data, color_by='cell_type')
    >>> 
    >>> # Color by marker expression
    >>> plot_spatial_scatter(data, marker='Ki67', cmap='magma')
    >>> 
    >>> # Custom colors
    >>> colors = {'Tumor': 'red', 'T_cell': 'blue'}
    >>> plot_spatial_scatter(data, colors=colors)
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    ax = get_axes(ax)
    
    coords = data._coordinates
    
    # Determine coloring
    if marker is not None:
        color_by = 'marker'
        
    if color_by == 'cell_type':
        # Categorical coloring by cell type
        cell_types = data._cell_types
        unique_types = data.cell_types_unique
        
        if colors is None:
            colors = get_cell_type_colors(list(unique_types))
        
        for ct in unique_types:
            mask = cell_types == ct
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors.get(ct, '#888888'),
                s=size,
                alpha=alpha,
                label=ct,
                **kwargs
            )
        
        if show_legend:
            ax.legend(
                bbox_to_anchor=(1.02, 1), loc='upper left',
                frameon=False, markerscale=2
            )
            
    elif color_by == 'marker' and marker is not None:
        # Continuous coloring by marker
        if data.markers is None or marker not in data.markers.columns:
            raise ValueError(f"Marker '{marker}' not found in data")
        
        values = data.markers[marker].values
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values,
            s=size,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        
        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label=marker)
            
    elif color_by == 'density':
        # Color by local density
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        
        # Count neighbors within adaptive radius
        radius = np.sqrt((coords[:, 0].ptp() * coords[:, 1].ptp()) / len(coords)) * 2
        counts = np.array([len(tree.query_ball_point(c, radius)) for c in coords])
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=counts,
            s=size,
            alpha=alpha,
            cmap=get_sequential_cmap('density'),
            **kwargs
        )
        
        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label='Local density')
            
    else:
        # Single color
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=size,
            alpha=alpha,
            **kwargs
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title)
    
    if scalebar is not None:
        add_scalebar(ax, scalebar)
    
    despine(ax)
    
    return ax


def plot_cell_types(
    data: 'SpatialTissueData',
    cell_types: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    size: float = 5,
    alpha: float = 0.7,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    **kwargs
) -> 'plt.Figure':
    """
    Create faceted plot with one panel per cell type.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    cell_types : list of str, optional
        Cell types to plot. If None, plots all.
    colors : dict, optional
        Custom color mapping.
    size : float, default 5
        Point size.
    alpha : float, default 0.7
        Transparency.
    ncols : int, default 3
        Number of columns in grid.
    figsize_per_panel : tuple, default (4, 4)
        Size of each panel.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Figure
        Figure with faceted panels.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    if cell_types is None:
        cell_types = list(data.cell_types_unique)
    
    n_types = len(cell_types)
    nrows = int(np.ceil(n_types / ncols))
    
    if colors is None:
        colors = get_cell_type_colors(cell_types)
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False
    )
    
    coords = data._coordinates
    all_types = data._cell_types
    
    for idx, ct in enumerate(cell_types):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Plot background (other cells)
        mask_other = all_types != ct
        ax.scatter(
            coords[mask_other, 0], coords[mask_other, 1],
            c='#e0e0e0', s=size * 0.5, alpha=0.3, rasterized=True
        )
        
        # Plot highlighted cell type
        mask = all_types == ct
        n_cells = np.sum(mask)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors.get(ct, '#1f77b4'), s=size, alpha=alpha,
            label=f'{ct} (n={n_cells})', rasterized=True, **kwargs
        )
        
        ax.set_title(f'{ct} (n={n_cells})')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        despine(ax, left=True, bottom=True)
    
    # Hide empty panels
    for idx in range(n_types, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    fig.tight_layout()
    
    return fig


def plot_marker_expression(
    data: 'SpatialTissueData',
    markers: List[str],
    ncols: int = 3,
    cmap: str = 'magma',
    size: float = 3,
    vmin_percentile: float = 1,
    vmax_percentile: float = 99,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    **kwargs
) -> 'plt.Figure':
    """
    Create faceted plot of marker expression.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    markers : list of str
        Marker names to plot.
    ncols : int, default 3
        Number of columns.
    cmap : str, default 'magma'
        Colormap.
    size : float, default 3
        Point size.
    vmin_percentile, vmax_percentile : float
        Percentiles for color scaling.
    figsize_per_panel : tuple, default (4, 4)
        Size of each panel.
    **kwargs
        Additional arguments to scatter().
        
    Returns
    -------
    plt.Figure
        Figure with marker panels.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    
    if data.markers is None:
        raise ValueError("Data has no markers")
    
    n_markers = len(markers)
    nrows = int(np.ceil(n_markers / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False
    )
    
    coords = data._coordinates
    
    for idx, marker in enumerate(markers):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        if marker not in data.markers.columns:
            ax.text(0.5, 0.5, f'{marker}\nnot found', ha='center', va='center')
            ax.set_title(marker)
            continue
        
        values = data.markers[marker].values
        vmin = np.percentile(values, vmin_percentile)
        vmax = np.percentile(values, vmax_percentile)
        
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, s=size, cmap=cmap,
            vmin=vmin, vmax=vmax, rasterized=True, **kwargs
        )
        
        ax.set_title(marker)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.8)
        despine(ax, left=True, bottom=True)
    
    # Hide empty panels
    for idx in range(n_markers, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    fig.tight_layout()
    
    return fig


def plot_density_map(
    data: 'SpatialTissueData',
    cell_type: Optional[str] = None,
    method: str = 'kde',
    bandwidth: Optional[float] = None,
    resolution: int = 100,
    cmap: str = 'viridis',
    show_points: bool = False,
    point_size: float = 1,
    point_alpha: float = 0.3,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot cell density map.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    cell_type : str, optional
        If provided, compute density only for this cell type.
    method : str, default 'kde'
        Density estimation method: 'kde' or 'histogram'.
    bandwidth : float, optional
        Bandwidth for KDE. If None, estimated automatically.
    resolution : int, default 100
        Grid resolution.
    cmap : str, default 'viridis'
        Colormap.
    show_points : bool, default False
        Overlay cell positions.
    point_size, point_alpha : float
        Point display parameters.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to contourf() or imshow().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy import stats
    
    ax = get_axes(ax)
    
    # Get coordinates
    if cell_type is not None:
        mask = data._cell_types == cell_type
        coords = data._coordinates[mask]
        title_suffix = f' ({cell_type})'
    else:
        coords = data._coordinates
        title_suffix = ''
    
    if len(coords) < 10:
        ax.text(0.5, 0.5, 'Not enough cells', ha='center', va='center')
        return ax
    
    # Create grid
    bounds = data.bounds
    x_grid = np.linspace(bounds['x'][0], bounds['x'][1], resolution)
    y_grid = np.linspace(bounds['y'][0], bounds['y'][1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    if method == 'kde':
        # Kernel density estimation
        if bandwidth is None:
            bandwidth = np.sqrt(coords[:, 0].var() + coords[:, 1].var()) / 10
        
        kernel = stats.gaussian_kde(coords.T, bw_method=bandwidth)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kernel(positions).reshape(xx.shape)
        
    elif method == 'histogram':
        density, _, _ = np.histogram2d(
            coords[:, 0], coords[:, 1],
            bins=[x_grid, y_grid]
        )
        density = density.T  # Transpose for correct orientation
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot density
    im = ax.contourf(xx, yy, density, levels=20, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax, label='Density')
    
    # Overlay points
    if show_points:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=point_size, c='white', alpha=point_alpha, rasterized=True
        )
    
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Cell Density{title_suffix}')
    ax.set_aspect('equal')
    
    return ax


def plot_voronoi(
    data: 'SpatialTissueData',
    color_by: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.1,
    alpha: float = 0.7,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot Voronoi tessellation of cells.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    color_by : str, default 'cell_type'
        Coloring scheme.
    colors : dict, optional
        Custom color mapping.
    edge_color : str, default 'black'
        Voronoi edge color.
    edge_width : float, default 0.1
        Edge line width.
    alpha : float, default 0.7
        Fill transparency.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to PolyCollection.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from scipy.spatial import Voronoi
    
    ax = get_axes(ax)
    
    coords = data._coordinates[:, :2]  # 2D only
    
    # Compute Voronoi
    vor = Voronoi(coords)
    
    if colors is None and color_by == 'cell_type':
        colors = get_cell_type_colors(list(data.cell_types_unique))
    
    # Get cell colors
    if color_by == 'cell_type':
        cell_colors = [colors.get(ct, '#888888') for ct in data._cell_types]
    else:
        cell_colors = ['#1f77b4'] * data.n_cells
    
    # Create polygons for finite regions
    polygons = []
    poly_colors = []
    
    bounds = data.bounds
    
    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            continue
        
        vertices = vor.vertices[region]
        
        # Clip to bounds
        if (np.any(vertices[:, 0] < bounds['x'][0] - 100) or 
            np.any(vertices[:, 0] > bounds['x'][1] + 100) or
            np.any(vertices[:, 1] < bounds['y'][0] - 100) or 
            np.any(vertices[:, 1] > bounds['y'][1] + 100)):
            continue
        
        polygons.append(vertices)
        poly_colors.append(cell_colors[idx])
    
    # Add polygon collection
    collection = PolyCollection(
        polygons,
        facecolors=poly_colors,
        edgecolors=edge_color,
        linewidths=edge_width,
        alpha=alpha,
        **kwargs
    )
    ax.add_collection(collection)
    
    ax.set_xlim(bounds['x'])
    ax.set_ylim(bounds['y'])
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title('Voronoi Tessellation')
    ax.set_aspect('equal')
    
    return ax


def plot_spatial_domains(
    data: 'SpatialTissueData',
    domain_labels: np.ndarray,
    colors: Optional[List[str]] = None,
    size: float = 5,
    alpha: float = 0.7,
    show_boundaries: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot spatial domain assignments.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    domain_labels : np.ndarray
        Domain assignment for each cell.
    colors : list of str, optional
        Colors for each domain.
    size : float, default 5
        Point size.
    alpha : float, default 0.7
        Transparency.
    show_boundaries : bool, default True
        Show domain boundaries.
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
    from .config import get_categorical_palette
    
    ax = get_axes(ax)
    
    coords = data._coordinates
    unique_domains = np.unique(domain_labels[domain_labels >= 0])  # Exclude -1
    n_domains = len(unique_domains)
    
    if colors is None:
        colors = get_categorical_palette(n_domains)
    
    # Plot each domain
    for i, domain in enumerate(unique_domains):
        mask = domain_labels == domain
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[i % len(colors)], s=size, alpha=alpha,
            label=f'Domain {domain}', rasterized=True, **kwargs
        )
    
    # Plot unclustered cells
    mask_unclustered = domain_labels < 0
    if np.any(mask_unclustered):
        ax.scatter(
            coords[mask_unclustered, 0], coords[mask_unclustered, 1],
            c='#cccccc', s=size * 0.5, alpha=0.3,
            label='Unclustered', rasterized=True
        )
    
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Spatial Domains (n={n_domains})')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    despine(ax)
    
    return ax


def plot_cell_neighborhoods(
    data: 'SpatialTissueData',
    cell_indices: Union[int, List[int]],
    radius: float = 50.0,
    highlight_color: str = 'red',
    neighbor_color: str = 'blue',
    background_color: str = '#e0e0e0',
    size: float = 20,
    alpha: float = 0.7,
    show_radius: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Visualize the neighborhood of specific cells.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    cell_indices : int or list of int
        Index/indices of focal cells to highlight.
    radius : float, default 50.0
        Neighborhood radius.
    highlight_color : str, default 'red'
        Color for focal cells.
    neighbor_color : str, default 'blue'
        Color for neighbors.
    background_color : str, default '#e0e0e0'
        Color for non-neighbor cells.
    size : float, default 20
        Point size.
    alpha : float, default 0.7
        Transparency.
    show_radius : bool, default True
        Draw circle showing neighborhood radius.
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
    from matplotlib.patches import Circle
    from scipy.spatial import cKDTree
    
    ax = get_axes(ax)
    
    if isinstance(cell_indices, int):
        cell_indices = [cell_indices]
    
    coords = data._coordinates
    tree = cKDTree(coords)
    
    # Find all neighbors
    all_neighbors = set()
    for idx in cell_indices:
        neighbors = tree.query_ball_point(coords[idx], radius)
        all_neighbors.update(neighbors)
    
    all_neighbors.discard(set(cell_indices))
    
    # Plot background cells
    background_mask = np.ones(data.n_cells, dtype=bool)
    background_mask[list(all_neighbors)] = False
    for idx in cell_indices:
        background_mask[idx] = False
    
    ax.scatter(
        coords[background_mask, 0], coords[background_mask, 1],
        c=background_color, s=size * 0.3, alpha=0.3, rasterized=True
    )
    
    # Plot neighbors
    neighbor_list = list(all_neighbors)
    if neighbor_list:
        ax.scatter(
            coords[neighbor_list, 0], coords[neighbor_list, 1],
            c=neighbor_color, s=size, alpha=alpha, label='Neighbors', **kwargs
        )
    
    # Plot focal cells and radius circles
    for idx in cell_indices:
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=highlight_color, s=size * 2, marker='*',
            edgecolor='black', linewidth=0.5, zorder=10,
            label='Focal cell' if idx == cell_indices[0] else None
        )
        
        if show_radius:
            circle = Circle(
                (coords[idx, 0], coords[idx, 1]), radius,
                fill=False, edgecolor=highlight_color, linestyle='--',
                linewidth=1.5, alpha=0.7
            )
            ax.add_patch(circle)
    
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Cell Neighborhood (r={radius})')
    ax.set_aspect('equal')
    ax.legend(frameon=False)
    despine(ax)
    
    return ax
