"""
Centrality metrics for cell graphs.

Provides functions to compute various centrality measures and aggregate
them by cell type.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
)
import numpy as np

if TYPE_CHECKING:
    from .cell_graph import CellGraph
    import networkx as nx

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# ============================================================================
# Core Centrality Functions
# ============================================================================

def degree_centrality(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute degree centrality for all nodes.
    
    Degree centrality is the fraction of nodes a node is connected to.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.degree_centrality(graph.G)


def betweenness_centrality(
    graph: 'CellGraph',
    k: Optional[int] = None,
    normalized: bool = True,
    seed: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute betweenness centrality for all nodes.
    
    Betweenness centrality measures how often a node lies on shortest
    paths between other nodes. High betweenness indicates "bridge" cells.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    k : int, optional
        Number of source nodes to sample for approximation.
        If None, compute exact betweenness.
    normalized : bool, default True
        Normalize by 2/((n-1)(n-2)) for undirected graphs.
    seed : int, optional
        Random seed for sampling (if k is specified).
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.betweenness_centrality(
        graph.G, k=k, normalized=normalized, seed=seed
    )


def closeness_centrality(
    graph: 'CellGraph',
    wf_improved: bool = True,
) -> Dict[int, float]:
    """
    Compute closeness centrality for all nodes.
    
    Closeness centrality measures how close a node is to all other nodes.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    wf_improved : bool, default True
        Use Wasserman-Faust improved formula for disconnected graphs.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.closeness_centrality(graph.G, wf_improved=wf_improved)


def eigenvector_centrality(
    graph: 'CellGraph',
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Compute eigenvector centrality for all nodes.
    
    A node has high eigenvector centrality if it is connected to other
    nodes that themselves have high centrality.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    max_iter : int, default 100
        Maximum iterations for power method.
    tol : float, default 1e-6
        Convergence tolerance.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    try:
        return nx.eigenvector_centrality(graph.G, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # Fall back to numpy version
        return nx.eigenvector_centrality_numpy(graph.G)


def pagerank(
    graph: 'CellGraph',
    alpha: float = 0.85,
    max_iter: int = 100,
) -> Dict[int, float]:
    """
    Compute PageRank centrality for all nodes.
    
    PageRank is a variant of eigenvector centrality with damping.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    alpha : float, default 0.85
        Damping factor.
    max_iter : int, default 100
        Maximum iterations.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.pagerank(graph.G, alpha=alpha, max_iter=max_iter)


def harmonic_centrality(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute harmonic centrality for all nodes.
    
    Harmonic centrality is the sum of reciprocal distances, which handles
    disconnected components better than closeness centrality.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.harmonic_centrality(graph.G)


def katz_centrality(
    graph: 'CellGraph',
    alpha: float = 0.1,
    beta: float = 1.0,
) -> Dict[int, float]:
    """
    Compute Katz centrality for all nodes.
    
    Katz centrality computes influence based on total walks, with
    attenuation factor alpha.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    alpha : float, default 0.1
        Attenuation factor (should be < 1/lambda_max).
    beta : float, default 1.0
        Weight for immediate neighbors.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.katz_centrality(graph.G, alpha=alpha, beta=beta)


def load_centrality(
    graph: 'CellGraph',
    normalized: bool = True,
) -> Dict[int, float]:
    """
    Compute load centrality for all nodes.
    
    Load centrality counts the fraction of shortest paths that pass
    through a node, weighted by path endpoints.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    normalized : bool, default True
        Normalize values.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.load_centrality(graph.G, normalized=normalized)


def subgraph_centrality(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute subgraph centrality for all nodes.
    
    Subgraph centrality counts closed walks of all lengths starting
    and ending at a node.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to centrality value.
    """
    return nx.subgraph_centrality(graph.G)


# ============================================================================
# Centrality by Cell Type
# ============================================================================

def centrality_by_type(
    graph: 'CellGraph',
    metric: str = 'degree',
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute centrality statistics grouped by cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    metric : str, default 'degree'
        Centrality metric: 'degree', 'betweenness', 'closeness',
        'eigenvector', 'pagerank', 'harmonic', 'katz', 'load'.
    **kwargs
        Additional arguments for the centrality function.
    
    Returns
    -------
    dict
        Dictionary mapping cell type to statistics dict containing
        'mean', 'std', 'median', 'min', 'max'.
    
    Examples
    --------
    >>> stats = centrality_by_type(graph, metric='betweenness')
    >>> print(stats['CD8_T_cell']['mean'])
    0.045
    """
    # Get centrality values
    centrality_funcs = {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality,
        'pagerank': pagerank,
        'harmonic': harmonic_centrality,
        'katz': katz_centrality,
        'load': load_centrality,
        'subgraph': subgraph_centrality,
    }
    
    if metric not in centrality_funcs:
        raise ValueError(
            f"Unknown centrality metric: {metric}. "
            f"Options: {list(centrality_funcs.keys())}"
        )
    
    centrality = centrality_funcs[metric](graph, **kwargs)
    
    # Group by cell type
    result = {}
    cell_types = graph.cell_types
    
    for cell_type in graph.cell_types_unique:
        nodes = graph.get_nodes_by_type(cell_type)
        values = np.array([centrality[n] for n in nodes])
        
        if len(values) > 0:
            result[cell_type] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }
        else:
            result[cell_type] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0,
            }
    
    return result


def mean_centrality_by_type(
    graph: 'CellGraph',
    metric: str = 'degree',
    **kwargs
) -> Dict[str, float]:
    """
    Compute mean centrality for each cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    metric : str, default 'degree'
        Centrality metric.
    **kwargs
        Additional arguments for the centrality function.
    
    Returns
    -------
    dict
        Cell type to mean centrality.
    """
    stats = centrality_by_type(graph, metric=metric, **kwargs)
    return {ct: s['mean'] for ct, s in stats.items()}


def top_central_nodes(
    graph: 'CellGraph',
    metric: str = 'degree',
    n: int = 10,
    cell_type: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Get the top N most central nodes.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    metric : str, default 'degree'
        Centrality metric.
    n : int, default 10
        Number of top nodes to return.
    cell_type : str, optional
        Filter to specific cell type.
    **kwargs
        Additional arguments for the centrality function.
    
    Returns
    -------
    list of dict
        List of dicts with 'node', 'cell_type', 'centrality'.
    """
    centrality_funcs = {
        'degree': degree_centrality,
        'betweenness': betweenness_centrality,
        'closeness': closeness_centrality,
        'eigenvector': eigenvector_centrality,
        'pagerank': pagerank,
        'harmonic': harmonic_centrality,
    }
    
    if metric not in centrality_funcs:
        raise ValueError(f"Unknown metric: {metric}")
    
    centrality = centrality_funcs[metric](graph, **kwargs)
    
    # Filter by cell type if specified
    if cell_type is not None:
        valid_nodes = set(graph.get_nodes_by_type(cell_type))
        centrality = {k: v for k, v in centrality.items() if k in valid_nodes}
    
    # Sort and take top N
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    for node, cent_value in sorted_nodes[:n]:
        result.append({
            'node': node,
            'cell_type': graph.cell_types[node],
            'centrality': cent_value,
            'coordinates': tuple(graph.coordinates[node]),
        })
    
    return result
