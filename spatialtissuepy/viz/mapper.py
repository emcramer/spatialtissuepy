"""
Mapper/TDA visualization functions.

This module provides functions for visualizing Mapper algorithm results,
including graph visualizations, spatial embeddings, filter distributions,
and diagnostic plots.

Note: These functions consolidate and extend the visualization code
previously in spatialtissuepy.topology.visualization.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    get_axes, get_categorical_palette, get_cell_type_colors,
    despine, _check_matplotlib
)

if TYPE_CHECKING:
    from spatialtissuepy.topology import MapperResult
    from spatialtissuepy.core import SpatialTissueData
    import matplotlib.pyplot as plt


def plot_mapper_graph(
    result: 'MapperResult',
    layout: str = 'spring',
    node_size_scale: float = 50.0,
    color_by: str = 'size',
    cmap: str = 'viridis',
    show_labels: bool = False,
    edge_alpha: float = 0.5,
    edge_width: float = 1.0,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot the Mapper graph.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result to visualize.
    layout : str, default 'spring'
        Graph layout: 'spring', 'kamada_kawai', 'circular', 'spectral', 'spatial'.
    node_size_scale : float, default 50.0
        Scale factor for node sizes.
    color_by : str, default 'size'
        Node coloring: 'size', 'filter_mean', 'component', or a cell type name.
    cmap : str, default 'viridis'
        Colormap for continuous coloring.
    show_labels : bool, default False
        Show node labels.
    edge_alpha : float, default 0.5
        Edge transparency.
    edge_width : float, default 1.0
        Edge line width.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to networkx.draw().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import networkx as nx
    
    ax = get_axes(ax)
    
    if result.graph is None:
        ax.text(0.5, 0.5, 'No graph available', ha='center', va='center')
        return ax
    
    G = result.graph
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        if G.number_of_nodes() > 2:
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
    elif layout == 'spatial':
        pos = {node.node_id: node.spatial_centroid[:2] for node in result.nodes}
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Compute node sizes
    sizes = np.array([G.nodes[n].get('size', 1) for n in G.nodes()])
    sizes = sizes * node_size_scale
    
    # Compute node colors
    if color_by == 'size':
        colors = [G.nodes[n].get('size', 1) for n in G.nodes()]
    elif color_by == 'filter_mean':
        colors = []
        for node in result.nodes:
            filter_vals = result.filter_values[node.members]
            colors.append(np.mean(filter_vals))
    elif color_by == 'component':
        components = list(nx.connected_components(G))
        node_to_comp = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = i
        colors = [node_to_comp.get(n, 0) for n in G.nodes()]
        cmap = 'tab10'
    else:
        # Assume cell type name
        colors = []
        for n in G.nodes():
            comp = G.nodes[n].get('composition', {})
            total = sum(comp.values()) if comp else 1
            colors.append(comp.get(color_by, 0) / total)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, width=edge_width)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=colors,
        cmap=cmap, **kwargs
    )
    
    if color_by not in ['component']:
        plt.colorbar(nodes, ax=ax, label=color_by.replace('_', ' ').title())
    
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    
    ax.set_title(f'Mapper Graph ({result.n_nodes} nodes, {result.n_edges} edges)')
    ax.axis('off')
    
    return ax


def plot_mapper_spatial(
    result: 'MapperResult',
    data: 'SpatialTissueData',
    color_by: str = 'component',
    show_edges: bool = True,
    show_centroids: bool = True,
    point_size: float = 5,
    alpha: float = 0.6,
    centroid_size: float = 50,
    edge_alpha: float = 0.3,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot Mapper results in spatial coordinates.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result to visualize.
    data : SpatialTissueData
        Original spatial data.
    color_by : str, default 'component'
        Coloring: 'component', 'filter', 'n_nodes'.
    show_edges : bool, default True
        Draw edges between connected node centroids.
    show_centroids : bool, default True
        Show node centroids.
    point_size : float, default 5
        Cell point size.
    alpha : float, default 0.6
        Cell transparency.
    centroid_size : float, default 50
        Centroid marker size.
    edge_alpha : float, default 0.3
        Edge transparency.
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
    import networkx as nx
    
    ax = get_axes(ax)
    
    coords = data._coordinates
    
    # Compute cell colors
    if color_by == 'component':
        if result.graph is not None:
            components = list(nx.connected_components(result.graph))
            node_to_comp = {}
            for i, comp in enumerate(components):
                for node in comp:
                    node_to_comp[node] = i
        else:
            node_to_comp = {n.node_id: 0 for n in result.nodes}
        
        cell_colors = np.full(data.n_cells, -1)
        for cell_idx, node_ids in result.cell_node_map.items():
            if node_ids:
                cell_colors[cell_idx] = node_to_comp.get(node_ids[0], 0)
        cmap = 'tab10'
        
    elif color_by == 'filter':
        cell_colors = result.filter_values
        cmap = 'viridis'
        
    elif color_by == 'n_nodes':
        cell_colors = np.array([
            len(result.cell_node_map.get(i, [])) for i in range(data.n_cells)
        ])
        cmap = 'viridis'
        
    else:
        cell_colors = result.filter_values
        cmap = 'viridis'
    
    # Plot cells
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cell_colors, s=point_size, alpha=alpha, cmap=cmap,
        rasterized=True, **kwargs
    )
    
    if color_by != 'component':
        plt.colorbar(scatter, ax=ax, label=color_by.replace('_', ' ').title())
    
    # Plot edges
    if show_edges and result.edges:
        centroids = {node.node_id: node.spatial_centroid for node in result.nodes}
        
        for edge in result.edges:
            c1 = centroids.get(edge.source)
            c2 = centroids.get(edge.target)
            if c1 is not None and c2 is not None:
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                        'k-', alpha=edge_alpha, linewidth=1)
    
    # Plot centroids
    if show_centroids:
        for node in result.nodes:
            ax.scatter(
                node.spatial_centroid[0], node.spatial_centroid[1],
                c='red', s=centroid_size, marker='x', zorder=10
            )
    
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'Mapper Spatial ({result.n_components} components)')
    ax.set_aspect('equal')
    despine(ax)
    
    return ax


def plot_filter_distribution(
    result: 'MapperResult',
    show_cover: bool = True,
    bins: int = 50,
    color: str = 'steelblue',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot filter value distribution with cover elements.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    show_cover : bool, default True
        Show cover element boundaries.
    bins : int, default 50
        Number of histogram bins.
    color : str, default 'steelblue'
        Histogram color.
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
    
    ax = get_axes(ax)
    
    # Plot histogram
    ax.hist(result.filter_values, bins=bins, alpha=0.7, density=True, color=color, **kwargs)
    
    # Show cover elements
    if show_cover and result.cover is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(result.cover.elements)))
        
        for i, element in enumerate(result.cover.elements):
            ax.axvspan(element.lower, element.upper,
                       alpha=0.1, color=colors[i % len(colors)])
            ax.axvline(element.lower, color='gray', alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Filter Value')
    ax.set_ylabel('Density')
    ax.set_title('Filter Distribution with Cover')
    despine(ax)
    
    return ax


def plot_node_composition(
    result: 'MapperResult',
    cell_types: Optional[List[str]] = None,
    normalize: bool = True,
    sort_by: str = 'filter',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot cell type composition of each Mapper node.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    cell_types : list of str, optional
        Cell types to include.
    normalize : bool, default True
        Show proportions instead of counts.
    sort_by : str, default 'filter'
        Sort nodes by: 'filter', 'size', 'id'.
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
    
    if result.graph is None:
        ax.text(0.5, 0.5, 'No graph available', ha='center', va='center')
        return ax
    
    G = result.graph
    nodes = list(G.nodes())
    
    # Determine cell types
    if cell_types is None:
        all_types = set()
        for n in nodes:
            comp = G.nodes[n].get('composition', {})
            all_types.update(comp.keys())
        cell_types = sorted(list(all_types))
    
    if not cell_types:
        ax.text(0.5, 0.5, 'No composition data', ha='center', va='center')
        return ax
    
    # Sort nodes
    if sort_by == 'filter':
        filter_means = [(node.node_id, np.mean(result.filter_values[node.members])) 
                        for node in result.nodes]
        filter_means.sort(key=lambda x: x[1])
        nodes = [n for n, _ in filter_means]
    elif sort_by == 'size':
        nodes = sorted(nodes, key=lambda n: G.nodes[n].get('size', 0), reverse=True)
    
    # Build composition matrix
    n_nodes = len(nodes)
    n_types = len(cell_types)
    type_to_idx = {t: i for i, t in enumerate(cell_types)}
    
    comp_matrix = np.zeros((n_nodes, n_types))
    
    for i, node_id in enumerate(nodes):
        comp = G.nodes[node_id].get('composition', {})
        for ct, count in comp.items():
            if ct in type_to_idx:
                comp_matrix[i, type_to_idx[ct]] = count
    
    if normalize:
        row_sums = comp_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        comp_matrix = comp_matrix / row_sums
    
    # Create stacked bar chart
    x = np.arange(n_nodes)
    bottom = np.zeros(n_nodes)
    colors = get_cell_type_colors(cell_types)
    
    for j, ct in enumerate(cell_types):
        ax.bar(x, comp_matrix[:, j], bottom=bottom, label=ct,
               color=colors.get(ct, f'C{j}'), width=0.8, **kwargs)
        bottom += comp_matrix[:, j]
    
    ax.set_xlabel('Node (sorted)')
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_title('Node Composition')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    despine(ax)
    
    return ax


def plot_mapper_diagnostics(
    result: 'MapperResult',
    data: 'SpatialTissueData',
    figsize: Tuple[float, float] = (16, 12),
    **kwargs
) -> 'plt.Figure':
    """
    Create comprehensive Mapper diagnostic plots.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    data : SpatialTissueData
        Spatial tissue data.
    figsize : tuple, default (16, 12)
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
    import networkx as nx
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Mapper graph
    try:
        plot_mapper_graph(result, ax=axes[0, 0], layout='spring')
    except Exception as e:
        axes[0, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 2. Spatial view
    try:
        plot_mapper_spatial(result, data, ax=axes[0, 1])
    except Exception as e:
        axes[0, 1].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 3. Filter distribution
    try:
        plot_filter_distribution(result, ax=axes[0, 2])
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 4. Node composition
    try:
        plot_node_composition(result, ax=axes[1, 0])
    except Exception as e:
        axes[1, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
    
    # 5. Node size distribution
    ax = axes[1, 1]
    if result.nodes:
        sizes = [node.size for node in result.nodes]
        ax.hist(sizes, bins=20, alpha=0.7, color='steelblue')
        ax.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean={np.mean(sizes):.1f}')
        ax.set_xlabel('Node Size')
        ax.set_ylabel('Count')
        ax.set_title('Node Size Distribution')
        ax.legend(frameon=False)
        despine(ax)
    else:
        ax.text(0.5, 0.5, 'No nodes', ha='center', va='center')
    
    # 6. Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    
    stats = result.statistics
    
    stats_text = [
        'Mapper Summary',
        '=' * 35,
        f'Nodes: {result.n_nodes}',
        f'Edges: {result.n_edges}',
        f'Connected components: {result.n_components}',
        '',
        'Graph Statistics',
        '=' * 35,
        f"Mean degree: {stats.get('mean_degree', 'N/A'):.2f}" if 'mean_degree' in stats else '',
        f"Density: {stats.get('density', 'N/A'):.4f}" if 'density' in stats else '',
        f"Clustering coef: {stats.get('avg_clustering', 'N/A'):.3f}" if 'avg_clustering' in stats else '',
        '',
        'Parameters',
        '=' * 35,
        f"Filter: {result.parameters.get('filter_fn', 'N/A')}",
        f"Intervals: {result.parameters.get('n_intervals', 'N/A')}",
        f"Overlap: {result.parameters.get('overlap', 'N/A')}",
        f"Clustering: {result.parameters.get('clustering', 'N/A')}",
        f"Radius: {result.parameters.get('neighborhood_radius', 'N/A')}",
    ]
    
    ax.text(0.05, 0.95, '\n'.join([s for s in stats_text if s]),
            transform=ax.transAxes, fontfamily='monospace', fontsize=9,
            verticalalignment='top')
    
    fig.tight_layout()
    
    return fig


def create_mapper_report(
    result: 'MapperResult',
    data: 'SpatialTissueData',
    output_path: Optional[str] = None,
    **kwargs
) -> 'plt.Figure':
    """
    Create and optionally save a comprehensive Mapper report.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    data : SpatialTissueData
        Spatial tissue data.
    output_path : str, optional
        If provided, save figure to this path.
    **kwargs
        Additional arguments to plot_mapper_diagnostics.
        
    Returns
    -------
    plt.Figure
        Figure with report.
    """
    fig = plot_mapper_diagnostics(result, data, **kwargs)
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
