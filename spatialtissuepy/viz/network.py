"""
Network visualization functions.

This module provides functions for visualizing cell graphs, including
graph overlays on tissue, degree distributions, centrality analysis,
and type mixing matrices.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    get_axes, get_cell_type_colors, get_categorical_palette,
    get_diverging_cmap, despine, _check_matplotlib
)

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    from spatialtissuepy.network import CellGraph
    import matplotlib.pyplot as plt
    import networkx as nx


def plot_cell_graph(
    graph: 'CellGraph',
    layout: str = 'spatial',
    node_color: str = 'cell_type',
    node_size: float = 20,
    edge_alpha: float = 0.3,
    edge_width: float = 0.5,
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot cell graph with various layout options.
    
    Parameters
    ----------
    graph : CellGraph
        Cell graph to visualize.
    layout : str, default 'spatial'
        Layout algorithm: 'spatial', 'spring', 'kamada_kawai', 'circular'.
    node_color : str, default 'cell_type'
        Node coloring: 'cell_type', 'degree', 'component', or a centrality metric.
    node_size : float, default 20
        Base node size.
    edge_alpha : float, default 0.3
        Edge transparency.
    edge_width : float, default 0.5
        Edge line width.
    colors : dict, optional
        Custom color mapping for cell types.
    show_legend : bool, default True
        Show legend for categorical coloring.
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
    
    G = graph.graph
    
    # Determine layout
    if layout == 'spatial':
        pos = {i: graph._coordinates[i, :2] for i in range(len(graph._coordinates))}
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    # Determine node colors
    if node_color == 'cell_type':
        cell_types = graph._cell_types
        unique_types = np.unique(cell_types)
        
        if colors is None:
            colors = get_cell_type_colors(list(unique_types))
        
        node_colors = [colors.get(cell_types[n], '#888888') for n in G.nodes()]
        
        # Draw edges first
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=edge_alpha, width=edge_width
        )
        
        # Draw nodes by type for legend
        for ct in unique_types:
            nodes_of_type = [n for n in G.nodes() if cell_types[n] == ct]
            if nodes_of_type:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes_of_type, ax=ax,
                    node_size=node_size, node_color=colors.get(ct, '#888888'),
                    label=ct, **kwargs
                )
        
        if show_legend:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
            
    elif node_color == 'degree':
        degrees = dict(G.degree())
        node_colors = [degrees[n] for n in G.nodes()]
        
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=edge_alpha, width=edge_width
        )
        
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_size,
            node_color=node_colors, cmap='viridis', **kwargs
        )
        plt.colorbar(nodes, ax=ax, label='Degree')
        
    elif node_color == 'component':
        components = list(nx.connected_components(G))
        node_to_comp = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = i
        
        node_colors = [node_to_comp.get(n, 0) for n in G.nodes()]
        
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=edge_alpha, width=edge_width
        )
        
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_size,
            node_color=node_colors, cmap='tab10', **kwargs
        )
        
    else:
        # Assume it's a centrality metric
        try:
            if node_color == 'betweenness':
                centrality = nx.betweenness_centrality(G)
            elif node_color == 'closeness':
                centrality = nx.closeness_centrality(G)
            elif node_color == 'eigenvector':
                centrality = nx.eigenvector_centrality(G, max_iter=500)
            else:
                centrality = {n: 0 for n in G.nodes()}
            
            node_colors = [centrality.get(n, 0) for n in G.nodes()]
            
            nx.draw_networkx_edges(
                G, pos, ax=ax, alpha=edge_alpha, width=edge_width
            )
            
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax, node_size=node_size,
                node_color=node_colors, cmap='plasma', **kwargs
            )
            plt.colorbar(nodes, ax=ax, label=node_color.capitalize())
            
        except Exception:
            # Fallback to single color
            nx.draw(G, pos, ax=ax, node_size=node_size, **kwargs)
    
    if layout == 'spatial':
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_aspect('equal')
    
    ax.set_title(f'Cell Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    
    return ax


def plot_graph_on_tissue(
    data: 'SpatialTissueData',
    graph: 'CellGraph',
    color_by: str = 'cell_type',
    colors: Optional[Dict[str, str]] = None,
    point_size: float = 10,
    edge_alpha: float = 0.2,
    edge_width: float = 0.3,
    highlight_edges: Optional[List[Tuple[int, int]]] = None,
    highlight_color: str = 'red',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Overlay graph edges on spatial tissue plot.
    
    Parameters
    ----------
    data : SpatialTissueData
        Spatial tissue data.
    graph : CellGraph
        Cell graph.
    color_by : str, default 'cell_type'
        Coloring scheme for points.
    colors : dict, optional
        Custom color mapping.
    point_size : float, default 10
        Point size.
    edge_alpha : float, default 0.2
        Edge transparency.
    edge_width : float, default 0.3
        Edge line width.
    highlight_edges : list of tuples, optional
        Edges to highlight.
    highlight_color : str, default 'red'
        Color for highlighted edges.
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
    from matplotlib.collections import LineCollection
    
    ax = get_axes(ax)
    
    coords = data._coordinates
    G = graph.graph
    
    # Collect edges for LineCollection (more efficient than individual lines)
    edge_segments = []
    for u, v in G.edges():
        edge_segments.append([coords[u, :2], coords[v, :2]])
    
    if edge_segments:
        lc = LineCollection(
            edge_segments,
            colors='gray',
            alpha=edge_alpha,
            linewidths=edge_width,
            zorder=1
        )
        ax.add_collection(lc)
    
    # Highlight specific edges
    if highlight_edges:
        highlight_segments = []
        for u, v in highlight_edges:
            highlight_segments.append([coords[u, :2], coords[v, :2]])
        
        if highlight_segments:
            hl_lc = LineCollection(
                highlight_segments,
                colors=highlight_color,
                alpha=0.8,
                linewidths=edge_width * 3,
                zorder=2
            )
            ax.add_collection(hl_lc)
    
    # Plot points
    if color_by == 'cell_type':
        cell_types = data._cell_types
        unique_types = data.cell_types_unique
        
        if colors is None:
            colors = get_cell_type_colors(list(unique_types))
        
        for ct in unique_types:
            mask = cell_types == ct
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=colors.get(ct, '#888888'), s=point_size,
                label=ct, zorder=3, **kwargs
            )
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=point_size, zorder=3, **kwargs
        )
    
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title('Cell Graph on Tissue')
    ax.set_aspect('equal')
    ax.autoscale_view()
    despine(ax)
    
    return ax


def plot_degree_distribution(
    graph: 'CellGraph',
    by_type: bool = False,
    colors: Optional[Dict[str, str]] = None,
    bins: int = 20,
    log_scale: bool = False,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot degree distribution of cell graph.
    
    Parameters
    ----------
    graph : CellGraph
        Cell graph.
    by_type : bool, default False
        If True, show separate distributions per cell type.
    colors : dict, optional
        Custom color mapping.
    bins : int, default 20
        Number of histogram bins.
    log_scale : bool, default False
        Use log scale for y-axis.
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
    
    G = graph.graph
    degrees = dict(G.degree())
    
    if by_type:
        cell_types = graph._cell_types
        unique_types = np.unique(cell_types)
        
        if colors is None:
            colors = get_cell_type_colors(list(unique_types))
        
        for ct in unique_types:
            type_degrees = [degrees[n] for n in G.nodes() if cell_types[n] == ct]
            if type_degrees:
                ax.hist(
                    type_degrees, bins=bins, alpha=0.6,
                    label=ct, color=colors.get(ct, None), **kwargs
                )
        
        ax.legend(frameon=False)
        
    else:
        all_degrees = list(degrees.values())
        ax.hist(all_degrees, bins=bins, alpha=0.7, edgecolor='black', **kwargs)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('Degree Distribution')
    despine(ax)
    
    return ax


def plot_centrality_by_type(
    graph: 'CellGraph',
    metric: str = 'degree',
    colors: Optional[Dict[str, str]] = None,
    show_points: bool = True,
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot centrality metrics grouped by cell type (violin/box plot).
    
    Parameters
    ----------
    graph : CellGraph
        Cell graph.
    metric : str, default 'degree'
        Centrality metric: 'degree', 'betweenness', 'closeness', 'eigenvector'.
    colors : dict, optional
        Custom color mapping.
    show_points : bool, default True
        Overlay individual data points.
    ax : plt.Axes, optional
        Matplotlib axes.
    **kwargs
        Additional arguments to violinplot().
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the plot.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import networkx as nx
    
    ax = get_axes(ax)
    
    G = graph.graph
    cell_types = graph._cell_types
    unique_types = sorted(np.unique(cell_types))
    
    # Compute centrality
    if metric == 'degree':
        centrality = dict(G.degree())
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif metric == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif metric == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=500)
        except nx.PowerIterationFailedConvergence:
            centrality = {n: 0 for n in G.nodes()}
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Group by cell type
    data_by_type = []
    for ct in unique_types:
        type_values = [centrality[n] for n in G.nodes() if cell_types[n] == ct]
        data_by_type.append(type_values)
    
    if colors is None:
        colors = get_cell_type_colors(unique_types)
    
    # Create violin plot
    positions = range(len(unique_types))
    parts = ax.violinplot(
        data_by_type, positions=positions,
        showmeans=True, showmedians=True, **kwargs
    )
    
    # Color violins
    for i, (pc, ct) in enumerate(zip(parts['bodies'], unique_types)):
        pc.set_facecolor(colors.get(ct, '#888888'))
        pc.set_alpha(0.7)
    
    # Add individual points
    if show_points:
        for i, (values, ct) in enumerate(zip(data_by_type, unique_types)):
            jitter = np.random.uniform(-0.1, 0.1, len(values))
            ax.scatter(
                np.full(len(values), i) + jitter, values,
                c=colors.get(ct, '#888888'), s=3, alpha=0.5, zorder=3
            )
    
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_types, rotation=45, ha='right')
    ax.set_xlabel('Cell Type')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} by Cell Type')
    despine(ax)
    
    return ax


def plot_type_mixing_matrix(
    graph: 'CellGraph',
    normalize: str = 'row',
    cmap: str = 'Blues',
    annot: bool = True,
    fmt: str = '.2f',
    ax: Optional['plt.Axes'] = None,
    **kwargs
) -> 'plt.Axes':
    """
    Plot cell type mixing/interaction matrix.
    
    Parameters
    ----------
    graph : CellGraph
        Cell graph.
    normalize : str, default 'row'
        Normalization: 'row', 'col', 'total', or None.
    cmap : str, default 'Blues'
        Colormap.
    annot : bool, default True
        Annotate cells with values.
    fmt : str, default '.2f'
        Number format for annotations.
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
    
    G = graph.graph
    cell_types = graph._cell_types
    unique_types = sorted(np.unique(cell_types))
    n_types = len(unique_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    
    # Count edges between types
    mixing_matrix = np.zeros((n_types, n_types))
    
    for u, v in G.edges():
        type_u = cell_types[u]
        type_v = cell_types[v]
        idx_u = type_to_idx[type_u]
        idx_v = type_to_idx[type_v]
        mixing_matrix[idx_u, idx_v] += 1
        mixing_matrix[idx_v, idx_u] += 1  # Undirected
    
    # Normalize
    if normalize == 'row':
        row_sums = mixing_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mixing_matrix = mixing_matrix / row_sums
    elif normalize == 'col':
        col_sums = mixing_matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        mixing_matrix = mixing_matrix / col_sums
    elif normalize == 'total':
        total = mixing_matrix.sum()
        if total > 0:
            mixing_matrix = mixing_matrix / total
    
    # Plot heatmap
    im = ax.imshow(mixing_matrix, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add annotations
    if annot:
        for i in range(n_types):
            for j in range(n_types):
                value = mixing_matrix[i, j]
                color = 'white' if value > mixing_matrix.max() / 2 else 'black'
                ax.text(j, i, format(value, fmt), ha='center', va='center', color=color)
    
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(unique_types, rotation=45, ha='right')
    ax.set_yticklabels(unique_types)
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Cell Type')
    ax.set_title('Cell Type Mixing Matrix')
    
    return ax
