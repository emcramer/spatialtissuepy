"""
Clustering and local structure metrics for cell graphs.

Provides functions to compute clustering coefficients and related
local structure measures.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Any
import numpy as np

if TYPE_CHECKING:
    from .cell_graph import CellGraph
    import networkx as nx

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _get_nx_graph(graph: Union['CellGraph', 'nx.Graph']) -> 'nx.Graph':
    """Helper to extract NetworkX graph from CellGraph or return nx.Graph."""
    if hasattr(graph, 'G'):
        return graph.G
    return graph


# ============================================================================
# Clustering Coefficients
# ============================================================================

def clustering_coefficient(graph: Union['CellGraph', 'nx.Graph']) -> Dict[int, float]:
    """
    Compute local clustering coefficient for all nodes.
    
    The clustering coefficient of a node measures the fraction of
    possible triangles through that node that exist.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    dict
        Node index to clustering coefficient.
    """
    return nx.clustering(_get_nx_graph(graph))


def average_clustering(graph: Union['CellGraph', 'nx.Graph']) -> float:
    """
    Compute average clustering coefficient for the graph.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    float
        Average clustering coefficient.
    """
    G = _get_nx_graph(graph)
    if G.number_of_nodes() == 0:
        return 0.0
    return nx.average_clustering(G)


def transitivity(graph: Union['CellGraph', 'nx.Graph']) -> float:
    """
    Compute graph transitivity (global clustering coefficient).
    
    Transitivity is the fraction of all possible triangles that exist.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    float
        Transitivity value between 0 and 1.
    """
    return nx.transitivity(_get_nx_graph(graph))


def square_clustering(graph: Union['CellGraph', 'nx.Graph']) -> Dict[int, float]:
    """
    Compute square clustering coefficient for all nodes.
    
    Square clustering measures the fraction of possible squares
    (4-cycles) through a node that exist.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    dict
        Node index to square clustering coefficient.
    """
    return nx.square_clustering(_get_nx_graph(graph))


def triangles(graph: Union['CellGraph', 'nx.Graph']) -> Dict[int, int]:
    """
    Count triangles for each node.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    dict
        Node index to triangle count.
    """
    return nx.triangles(_get_nx_graph(graph))


# ============================================================================
# Clustering by Cell Type
# ============================================================================

def clustering_by_type(graph: 'CellGraph') -> Dict[str, Dict[str, float]]:
    """
    Compute clustering coefficient statistics grouped by cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Dictionary mapping cell type to statistics dict containing
        'mean', 'std', 'median', 'min', 'max'.
    """
    clustering = clustering_coefficient(graph)
    
    result = {}
    
    for cell_type in graph.cell_types_unique:
        nodes = graph.get_nodes_by_type(cell_type)
        values = np.array([clustering[n] for n in nodes])
        
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


def mean_clustering_by_type(graph: 'CellGraph') -> Dict[str, float]:
    """
    Compute mean clustering coefficient for each cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Cell type to mean clustering coefficient.
    """
    stats = clustering_by_type(graph)
    return {ct: s['mean'] for ct, s in stats.items()}


def triangles_by_type(graph: 'CellGraph') -> Dict[str, Dict[str, float]]:
    """
    Compute triangle count statistics by cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Cell type to triangle statistics.
    """
    tri = triangles(graph)
    
    result = {}
    
    for cell_type in graph.cell_types_unique:
        nodes = graph.get_nodes_by_type(cell_type)
        values = np.array([tri[n] for n in nodes])
        
        if len(values) > 0:
            result[cell_type] = {
                'mean': float(np.mean(values)),
                'total': int(np.sum(values)),
                'max': int(np.max(values)),
                'count': len(values),
            }
        else:
            result[cell_type] = {
                'mean': np.nan,
                'total': 0,
                'max': 0,
                'count': 0,
            }
    
    return result


# ============================================================================
# Graph Structure
# ============================================================================

def connected_components(graph: Union['CellGraph', 'nx.Graph']) -> List[set]:
    """
    Find connected components in the graph.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    list of set
        List of node sets, one per component, sorted by size (largest first).
    """
    components = list(nx.connected_components(_get_nx_graph(graph)))
    return sorted(components, key=len, reverse=True)


def n_connected_components(graph: Union['CellGraph', 'nx.Graph']) -> int:
    """
    Count connected components.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    int
        Number of connected components.
    """
    return nx.number_connected_components(_get_nx_graph(graph))


def largest_component_size(graph: Union['CellGraph', 'nx.Graph']) -> int:
    """
    Get size of the largest connected component.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    int
        Number of nodes in largest component.
    """
    G = _get_nx_graph(graph)
    if G.number_of_nodes() == 0:
        return 0
    components = connected_components(G)
    return len(components[0]) if components else 0


def bridges(graph: Union['CellGraph', 'nx.Graph']) -> List[tuple]:
    """
    Find bridge edges whose removal disconnects the graph.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    list of tuple
        List of (node_i, node_j) edge tuples that are bridges.
    """
    return list(nx.bridges(_get_nx_graph(graph)))


def articulation_points(graph: Union['CellGraph', 'nx.Graph']) -> List[int]:
    """
    Find articulation points (cut vertices).
    
    An articulation point is a node whose removal disconnects the graph.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    list of int
        Node indices that are articulation points.
    """
    return list(nx.articulation_points(_get_nx_graph(graph)))


def articulation_points_by_type(graph: 'CellGraph') -> Dict[str, int]:
    """
    Count articulation points by cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Cell type to count of articulation points.
    """
    ap = articulation_points(graph)
    
    counts = {ct: 0 for ct in graph.cell_types_unique}
    
    for node in ap:
        cell_type = graph.cell_types[node]
        counts[cell_type] += 1
    
    return counts


def bridges_by_type_pair(graph: 'CellGraph') -> Dict[tuple, int]:
    """
    Count bridge edges by cell type pairs.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        (type_a, type_b) to count of bridges.
    """
    bridge_edges = bridges(graph)
    
    counts: Dict[tuple, int] = {}
    
    for i, j in bridge_edges:
        type_i = graph.cell_types[i]
        type_j = graph.cell_types[j]
        key = tuple(sorted([type_i, type_j]))
        counts[key] = counts.get(key, 0) + 1
    
    return counts
