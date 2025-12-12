"""
Communicability and path-based metrics for cell graphs.

Provides functions to compute communicability between nodes and cell types,
as well as shortest path statistics.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
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
# Communicability
# ============================================================================

def communicability(graph: 'CellGraph') -> Dict[int, Dict[int, float]]:
    """
    Compute communicability between all pairs of nodes.
    
    Communicability measures the sum of all walks of different lengths
    between two nodes, weighted by the inverse factorial of the length.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict of dict
        Nested dict: comm[i][j] = communicability between nodes i and j.
    
    Notes
    -----
    This can be memory-intensive for large graphs.
    """
    return nx.communicability(graph.G)


def communicability_exp(graph: 'CellGraph') -> Dict[int, Dict[int, float]]:
    """
    Compute communicability using matrix exponential.
    
    More efficient implementation using spectral decomposition.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict of dict
        Nested dict of communicability values.
    """
    return nx.communicability_exp(graph.G)


def communicability_betweenness(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute communicability betweenness centrality.
    
    This measures how much a node contributes to the communicability
    between all pairs of other nodes.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to communicability betweenness value.
    """
    return nx.communicability_betweenness_centrality(graph.G)


def communicability_between_types(
    graph: 'CellGraph',
    type_a: str,
    type_b: str,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute communicability statistics between two cell types.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    type_a : str
        First cell type.
    type_b : str
        Second cell type.
    sample_size : int, optional
        If specified, sample this many pairs to reduce computation.
    seed : int, optional
        Random seed for sampling.
    
    Returns
    -------
    dict
        Statistics: 'mean', 'std', 'median', 'min', 'max', 'n_pairs'.
    
    Examples
    --------
    >>> comm = communicability_between_types(graph, 'T_cell', 'Tumor')
    >>> print(f"Mean communicability: {comm['mean']:.4f}")
    """
    nodes_a = graph.get_nodes_by_type(type_a)
    nodes_b = graph.get_nodes_by_type(type_b)
    
    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_pairs': 0,
        }
    
    # Get communicability matrix
    comm = communicability_exp(graph)
    
    # Sample pairs if needed
    if sample_size is not None:
        rng = np.random.default_rng(seed)
        
        n_possible = len(nodes_a) * len(nodes_b)
        if sample_size < n_possible:
            # Sample pairs
            pairs_a = rng.choice(nodes_a, size=sample_size, replace=True)
            pairs_b = rng.choice(nodes_b, size=sample_size, replace=True)
            pairs = list(zip(pairs_a, pairs_b))
        else:
            pairs = [(a, b) for a in nodes_a for b in nodes_b]
    else:
        pairs = [(a, b) for a in nodes_a for b in nodes_b]
    
    # Collect communicability values
    values = []
    for a, b in pairs:
        if a in comm and b in comm[a]:
            values.append(comm[a][b])
    
    values = np.array(values)
    
    if len(values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_pairs': 0,
        }
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'n_pairs': len(values),
    }


def communicability_matrix_by_type(
    graph: 'CellGraph',
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Compute mean communicability for all cell type pairs.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    sample_size : int, optional
        Sample size per pair for large graphs.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict
        (type_a, type_b) to mean communicability.
    """
    cell_types = graph.cell_types_unique
    
    result = {}
    
    for i, type_a in enumerate(cell_types):
        for type_b in cell_types[i:]:  # Upper triangle including diagonal
            stats = communicability_between_types(
                graph, type_a, type_b,
                sample_size=sample_size, seed=seed
            )
            result[(type_a, type_b)] = stats['mean']
            if type_a != type_b:
                result[(type_b, type_a)] = stats['mean']  # Symmetric
    
    return result


# ============================================================================
# Shortest Path Metrics
# ============================================================================

def shortest_path_length_between_types(
    graph: 'CellGraph',
    type_a: str,
    type_b: str,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute shortest path length statistics between two cell types.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    type_a : str
        First cell type.
    type_b : str
        Second cell type.
    sample_size : int, optional
        Number of pairs to sample.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict
        Statistics: 'mean', 'std', 'median', 'min', 'max', 'n_pairs', 'n_unreachable'.
    """
    nodes_a = graph.get_nodes_by_type(type_a)
    nodes_b = graph.get_nodes_by_type(type_b)
    
    if len(nodes_a) == 0 or len(nodes_b) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_pairs': 0,
            'n_unreachable': 0,
        }
    
    # Sample pairs if needed
    if sample_size is not None:
        rng = np.random.default_rng(seed)
        
        n_possible = len(nodes_a) * len(nodes_b)
        if sample_size < n_possible:
            pairs_a = rng.choice(nodes_a, size=sample_size, replace=True)
            pairs_b = rng.choice(nodes_b, size=sample_size, replace=True)
            pairs = list(zip(pairs_a, pairs_b))
        else:
            pairs = [(a, b) for a in nodes_a for b in nodes_b]
    else:
        pairs = [(a, b) for a in nodes_a for b in nodes_b]
    
    # Compute shortest paths
    lengths = []
    n_unreachable = 0
    
    for a, b in pairs:
        try:
            length = nx.shortest_path_length(graph.G, source=a, target=b)
            lengths.append(length)
        except nx.NetworkXNoPath:
            n_unreachable += 1
    
    lengths = np.array(lengths)
    
    if len(lengths) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_pairs': len(pairs),
            'n_unreachable': n_unreachable,
        }
    
    return {
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'median': float(np.median(lengths)),
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'n_pairs': len(pairs),
        'n_unreachable': n_unreachable,
    }


def average_shortest_path_length(graph: 'CellGraph') -> float:
    """
    Compute average shortest path length for the graph.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Average shortest path length.
        Returns inf if graph is disconnected.
    """
    if not nx.is_connected(graph.G):
        # Compute for largest component
        largest_cc = max(nx.connected_components(graph.G), key=len)
        subG = graph.G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subG)
    
    return nx.average_shortest_path_length(graph.G)


def diameter(graph: 'CellGraph') -> int:
    """
    Compute graph diameter (maximum eccentricity).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    int
        Diameter of the graph (or largest component if disconnected).
    """
    if not nx.is_connected(graph.G):
        largest_cc = max(nx.connected_components(graph.G), key=len)
        subG = graph.G.subgraph(largest_cc)
        return nx.diameter(subG)
    
    return nx.diameter(graph.G)


def radius(graph: 'CellGraph') -> int:
    """
    Compute graph radius (minimum eccentricity).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    int
        Radius of the graph.
    """
    if not nx.is_connected(graph.G):
        largest_cc = max(nx.connected_components(graph.G), key=len)
        subG = graph.G.subgraph(largest_cc)
        return nx.radius(subG)
    
    return nx.radius(graph.G)


def eccentricity(graph: 'CellGraph') -> Dict[int, int]:
    """
    Compute eccentricity for all nodes.
    
    Eccentricity is the maximum distance from a node to any other.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to eccentricity value.
    """
    if not nx.is_connected(graph.G):
        # Only compute for largest component
        largest_cc = max(nx.connected_components(graph.G), key=len)
        subG = graph.G.subgraph(largest_cc)
        ecc = nx.eccentricity(subG)
        
        # Fill in inf for disconnected nodes
        result = {n: np.inf for n in graph.G.nodes()}
        result.update(ecc)
        return result
    
    return nx.eccentricity(graph.G)


# ============================================================================
# Efficiency Metrics
# ============================================================================

def global_efficiency(graph: 'CellGraph') -> float:
    """
    Compute global efficiency of the graph.
    
    Global efficiency is the average inverse shortest path length.
    Higher efficiency means better "communication" in the network.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Global efficiency between 0 and 1.
    """
    return nx.global_efficiency(graph.G)


def local_efficiency(graph: 'CellGraph') -> float:
    """
    Compute local efficiency of the graph.
    
    Local efficiency is the average efficiency of node neighborhoods.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Local efficiency.
    """
    return nx.local_efficiency(graph.G)


def nodal_efficiency(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute local efficiency for each node.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to local efficiency.
    """
    # NetworkX doesn't have per-node local efficiency, so compute manually
    result = {}
    
    for node in graph.G.nodes():
        neighbors = list(graph.G.neighbors(node))
        
        if len(neighbors) < 2:
            result[node] = 0.0
            continue
        
        # Subgraph of neighbors
        subG = graph.G.subgraph(neighbors)
        result[node] = nx.global_efficiency(subG)
    
    return result
