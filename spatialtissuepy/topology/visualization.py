"""
Visualization utilities for Mapper algorithm results.

This module provides functions for visualizing Mapper graphs and their
spatial embeddings in tissue context.

Note: Requires matplotlib for plotting. Install with `pip install matplotlib`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
import numpy as np

if TYPE_CHECKING:
    from .mapper import MapperResult
    from spatialtissuepy.core.spatial_data import SpatialTissueData


def plot_mapper_graph(
    result: 'MapperResult',
    layout: str = 'spring',
    node_size_scale: float = 50.0,
    color_by: str = 'size',
    cmap: str = 'viridis',
    show_labels: bool = False,
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
        Graph layout algorithm: 'spring', 'kamada_kawai', 'circular', 'spectral'.
    node_size_scale : float, default 50.0
        Scale factor for node sizes.
    color_by : str, default 'size'
        Node coloring: 'size', 'filter_mean', 'component', or a cell type name.
    cmap : str, default 'viridis'
        Colormap for node colors.
    show_labels : bool, default False
        If True, show node labels.
    ax : plt.Axes, optional
        Matplotlib axes. If None, creates new figure.
    **kwargs
        Additional arguments passed to networkx.draw().
    
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("matplotlib and networkx required for visualization")
    
    if result.graph is None:
        raise ValueError("MapperResult has no graph (networkx not available during fit)")
    
    G = result.graph
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'spatial':
        # Use spatial centroids
        pos = {
            node.node_id: node.spatial_centroid[:2]  # x, y only
            for node in result.nodes
        }
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
    else:
        # Assume it's a cell type name
        colors = []
        for n in G.nodes():
            comp = G.nodes[n].get('composition', {})
            total = sum(comp.values()) if comp else 1
            colors.append(comp.get(color_by, 0) / total)
    
    # Draw graph
    nx.draw(
        G, pos,
        ax=ax,
        node_size=sizes,
        node_color=colors,
        cmap=cmap,
        with_labels=show_labels,
        **kwargs
    )
    
    ax.set_title(f"Mapper Graph ({result.n_nodes} nodes, {result.n_edges} edges)")
    
    return ax


def plot_mapper_spatial(
    result: 'MapperResult',
    data: 'SpatialTissueData',
    color_by: str = 'component',
    show_edges: bool = True,
    alpha: float = 0.6,
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
        Coloring scheme: 'component', 'filter', 'dominant_topic'.
    show_edges : bool, default True
        If True, draw edges between connected node centroids.
    alpha : float, default 0.6
        Transparency for cell points.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments passed to scatter().
    
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
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
        
        cell_colors = np.full(data.n_cells, -1)  # -1 for unclustered
        for cell_idx, node_ids in result.cell_node_map.items():
            if node_ids:
                cell_colors[cell_idx] = node_to_comp.get(node_ids[0], 0)
                
    elif color_by == 'filter':
        cell_colors = result.filter_values
        
    elif color_by == 'n_nodes':
        # Number of nodes each cell belongs to
        cell_colors = np.array([
            len(result.cell_node_map.get(i, []))
            for i in range(data.n_cells)
        ])
    else:
        cell_colors = result.filter_values
    
    # Plot cells
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=cell_colors,
        alpha=alpha,
        s=5,
        **kwargs
    )
    
    # Plot edges between node centroids
    if show_edges and result.edges:
        centroids = {node.node_id: node.spatial_centroid for node in result.nodes}
        
        for edge in result.edges:
            c1 = centroids.get(edge.source)
            c2 = centroids.get(edge.target)
            if c1 is not None and c2 is not None:
                ax.plot(
                    [c1[0], c2[0]], [c1[1], c2[1]],
                    'k-', alpha=0.3, linewidth=1
                )
    
    # Plot node centroids
    for node in result.nodes:
        ax.scatter(
            node.spatial_centroid[0], node.spatial_centroid[1],
            c='red', s=30, marker='x', zorder=10
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Mapper Spatial View ({result.n_components} components)")
    
    return ax


def plot_filter_distribution(
    result: 'MapperResult',
    ax: Optional['plt.Axes'] = None,
    show_cover: bool = True,
    bins: int = 50,
) -> 'plt.Axes':
    """
    Plot filter value distribution with cover elements.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    ax : plt.Axes, optional
        Matplotlib axes.
    show_cover : bool, default True
        If True, show cover element boundaries.
    bins : int, default 50
        Number of histogram bins.
    
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot histogram
    ax.hist(result.filter_values, bins=bins, alpha=0.7, density=True)
    
    # Show cover elements
    if show_cover and result.cover is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(result.cover.elements)))
        
        ymax = ax.get_ylim()[1]
        for i, element in enumerate(result.cover.elements):
            ax.axvspan(
                element.lower, element.upper,
                alpha=0.1, color=colors[i % len(colors)]
            )
            # Mark boundaries
            ax.axvline(element.lower, color='gray', alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Filter Value')
    ax.set_ylabel('Density')
    ax.set_title('Filter Value Distribution with Cover Elements')
    
    return ax


def plot_node_composition(
    result: 'MapperResult',
    cell_types: Optional[List[str]] = None,
    normalize: bool = True,
    sort_by: str = 'filter',
    ax: Optional['plt.Axes'] = None,
) -> 'plt.Axes':
    """
    Plot cell type composition of each Mapper node.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    cell_types : list of str, optional
        Cell types to include. If None, uses all.
    normalize : bool, default True
        If True, show proportions instead of counts.
    sort_by : str, default 'filter'
        Sort nodes by: 'filter' (mean filter value), 'size', or 'id'.
    ax : plt.Axes, optional
        Matplotlib axes.
    
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    if result.graph is None:
        raise ValueError("MapperResult has no graph")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get node compositions
    G = result.graph
    nodes = list(G.nodes())
    
    # Determine cell types
    if cell_types is None:
        all_types = set()
        for n in nodes:
            comp = G.nodes[n].get('composition', {})
            all_types.update(comp.keys())
        cell_types = sorted(list(all_types))
    
    # Sort nodes
    if sort_by == 'filter':
        filter_means = []
        for node in result.nodes:
            filter_mean = np.mean(result.filter_values[node.members])
            filter_means.append((node.node_id, filter_mean))
        filter_means.sort(key=lambda x: x[1])
        nodes = [n for n, _ in filter_means]
    elif sort_by == 'size':
        nodes = sorted(nodes, key=lambda n: G.nodes[n].get('size', 0), reverse=True)
    # else: keep original order (by id)
    
    # Build composition matrix
    n_nodes = len(nodes)
    n_types = len(cell_types)
    type_to_idx = {t: i for i, t in enumerate(cell_types)}
    
    comp_matrix = np.zeros((n_nodes, n_types))
    
    for i, node_id in enumerate(nodes):
        comp = G.nodes[node_id].get('composition', {})
        for cell_type, count in comp.items():
            if cell_type in type_to_idx:
                comp_matrix[i, type_to_idx[cell_type]] = count
    
    if normalize:
        row_sums = comp_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        comp_matrix = comp_matrix / row_sums
    
    # Create stacked bar chart
    x = np.arange(n_nodes)
    bottom = np.zeros(n_nodes)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_types))
    
    for j, cell_type in enumerate(cell_types):
        ax.bar(x, comp_matrix[:, j], bottom=bottom, label=cell_type,
               color=colors[j], width=0.8)
        bottom += comp_matrix[:, j]
    
    ax.set_xlabel('Node (sorted by filter value)')
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_title('Cell Type Composition per Mapper Node')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return ax


def create_mapper_report(
    result: 'MapperResult',
    data: 'SpatialTissueData',
    output_path: Optional[str] = None,
) -> 'plt.Figure':
    """
    Create a comprehensive Mapper visualization report.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result to visualize.
    data : SpatialTissueData
        Original spatial data.
    output_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure with multiple panels.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Mapper graph
    ax1 = fig.add_subplot(2, 2, 1)
    try:
        plot_mapper_graph(result, ax=ax1, layout='spring')
    except Exception as e:
        ax1.text(0.5, 0.5, f"Graph plot error: {e}", ha='center', va='center')
    
    # Panel 2: Spatial view
    ax2 = fig.add_subplot(2, 2, 2)
    try:
        plot_mapper_spatial(result, data, ax=ax2)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Spatial plot error: {e}", ha='center', va='center')
    
    # Panel 3: Filter distribution
    ax3 = fig.add_subplot(2, 2, 3)
    try:
        plot_filter_distribution(result, ax=ax3)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Filter plot error: {e}", ha='center', va='center')
    
    # Panel 4: Statistics text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    stats = result.statistics
    stats_text = [
        "Mapper Statistics",
        "=" * 30,
        f"Nodes: {stats.get('n_nodes', 'N/A')}",
        f"Edges: {stats.get('n_edges', 'N/A')}",
        f"Connected Components: {stats.get('n_connected_components', 'N/A')}",
        f"Mean Degree: {stats.get('mean_degree', 'N/A'):.2f}" if 'mean_degree' in stats else "",
        f"Mean Node Size: {stats.get('mean_node_size', 'N/A'):.1f}" if 'mean_node_size' in stats else "",
        f"Density: {stats.get('density', 'N/A'):.4f}" if 'density' in stats else "",
        "",
        "Parameters",
        "=" * 30,
        f"Filter: {result.parameters.get('filter_fn', 'N/A')}",
        f"Intervals: {result.parameters.get('n_intervals', 'N/A')}",
        f"Overlap: {result.parameters.get('overlap', 'N/A')}",
        f"Clustering: {result.parameters.get('clustering', 'N/A')}",
    ]
    
    ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
