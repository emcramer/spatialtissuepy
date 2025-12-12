"""
Assortativity and mixing metrics for cell graphs.

Provides functions to compute assortativity coefficients and
cell type mixing patterns.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union
import numpy as np

if TYPE_CHECKING:
    from .cell_graph import CellGraph
    import networkx as nx
    import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# ============================================================================
# Assortativity Coefficients
# ============================================================================

def degree_assortativity(graph: 'CellGraph') -> float:
    """
    Compute degree assortativity coefficient.
    
    Measures whether high-degree nodes connect preferentially to
    other high-degree nodes (positive) or low-degree nodes (negative).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Degree assortativity coefficient in [-1, 1].
    """
    return nx.degree_assortativity_coefficient(graph.G)


def type_assortativity(graph: 'CellGraph') -> float:
    """
    Compute cell type assortativity coefficient.
    
    Measures whether cells of the same type preferentially connect
    to each other (positive) or to different types (negative).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Attribute assortativity coefficient in [-1, 1].
    """
    return nx.attribute_assortativity_coefficient(graph.G, 'cell_type')


def numeric_assortativity(
    graph: 'CellGraph',
    attribute: str
) -> float:
    """
    Compute numeric assortativity for a node attribute.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    attribute : str
        Node attribute name (must be numeric).
    
    Returns
    -------
    float
        Numeric assortativity coefficient.
    """
    return nx.numeric_assortativity_coefficient(graph.G, attribute)


# ============================================================================
# Mixing Patterns
# ============================================================================

def attribute_mixing_matrix(
    graph: 'CellGraph',
    normalized: bool = True,
) -> 'pd.DataFrame':
    """
    Compute the cell type mixing matrix.
    
    The mixing matrix M[i,j] represents the fraction (or count) of edges
    connecting cell type i to cell type j.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    normalized : bool, default True
        If True, normalize so values sum to 1.
    
    Returns
    -------
    pd.DataFrame
        Mixing matrix with cell types as row and column labels.
    
    Examples
    --------
    >>> mixing = attribute_mixing_matrix(graph)
    >>> print(mixing.loc['T_cell', 'Tumor'])
    0.15
    """
    import pandas as pd
    
    cell_types = graph.cell_types_unique
    n_types = len(cell_types)
    type_to_idx = {t: i for i, t in enumerate(cell_types)}
    
    # Count edges by type pair
    matrix = np.zeros((n_types, n_types))
    
    for i, j in graph.G.edges():
        type_i = graph.cell_types[i]
        type_j = graph.cell_types[j]
        
        idx_i = type_to_idx[type_i]
        idx_j = type_to_idx[type_j]
        
        matrix[idx_i, idx_j] += 1
        if idx_i != idx_j:
            matrix[idx_j, idx_i] += 1  # Symmetric for undirected
    
    if normalized and matrix.sum() > 0:
        matrix = matrix / matrix.sum()
    
    return pd.DataFrame(matrix, index=cell_types, columns=cell_types)


def attribute_mixing_dict(
    graph: 'CellGraph',
    normalized: bool = True,
) -> Dict[Tuple[str, str], float]:
    """
    Compute mixing as a dictionary of type pairs.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    normalized : bool, default True
        If True, normalize so values sum to 1.
    
    Returns
    -------
    dict
        (type_a, type_b) to mixing value.
    """
    mixing = nx.attribute_mixing_dict(graph.G, 'cell_type', normalized=normalized)
    
    # Flatten nested dict
    result = {}
    for type_a, inner in mixing.items():
        for type_b, value in inner.items():
            result[(type_a, type_b)] = value
    
    return result


def homophily_ratio(graph: 'CellGraph') -> float:
    """
    Compute homophily ratio (fraction of same-type edges).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Fraction of edges connecting same-type cells.
    """
    same_type_edges = 0
    total_edges = graph.n_edges
    
    if total_edges == 0:
        return np.nan
    
    for i, j in graph.G.edges():
        if graph.cell_types[i] == graph.cell_types[j]:
            same_type_edges += 1
    
    return same_type_edges / total_edges


def heterophily_ratio(graph: 'CellGraph') -> float:
    """
    Compute heterophily ratio (fraction of different-type edges).
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    float
        Fraction of edges connecting different-type cells.
    """
    homo = homophily_ratio(graph)
    return 1 - homo if not np.isnan(homo) else np.nan


def type_pair_edge_fraction(
    graph: 'CellGraph',
    type_a: str,
    type_b: str,
) -> float:
    """
    Compute fraction of edges between two specific cell types.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    type_a : str
        First cell type.
    type_b : str
        Second cell type.
    
    Returns
    -------
    float
        Fraction of total edges connecting type_a and type_b.
    """
    if graph.n_edges == 0:
        return np.nan
    
    count = 0
    for i, j in graph.G.edges():
        ti, tj = graph.cell_types[i], graph.cell_types[j]
        if (ti == type_a and tj == type_b) or (ti == type_b and tj == type_a):
            count += 1
    
    return count / graph.n_edges


# ============================================================================
# Average Neighbor Degree
# ============================================================================

def average_neighbor_degree(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute average neighbor degree for all nodes.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Node index to average neighbor degree.
    """
    return nx.average_neighbor_degree(graph.G)


def average_neighbor_degree_by_type(
    graph: 'CellGraph'
) -> Dict[str, Dict[str, float]]:
    """
    Compute average neighbor degree statistics by cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Cell type to statistics dict.
    """
    and_values = average_neighbor_degree(graph)
    
    result = {}
    
    for cell_type in graph.cell_types_unique:
        nodes = graph.get_nodes_by_type(cell_type)
        values = np.array([and_values[n] for n in nodes])
        
        if len(values) > 0:
            result[cell_type] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
            }
        else:
            result[cell_type] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
            }
    
    return result


def neighbor_type_distribution(
    graph: 'CellGraph',
    cell_type: str,
) -> Dict[str, float]:
    """
    Compute the distribution of neighbor types for a given cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    cell_type : str
        Cell type to analyze.
    
    Returns
    -------
    dict
        Neighbor type to proportion.
    """
    nodes = graph.get_nodes_by_type(cell_type)
    
    neighbor_counts = {t: 0 for t in graph.cell_types_unique}
    total = 0
    
    for node in nodes:
        for neighbor in graph.G.neighbors(node):
            neighbor_type = graph.cell_types[neighbor]
            neighbor_counts[neighbor_type] += 1
            total += 1
    
    if total == 0:
        return {t: np.nan for t in graph.cell_types_unique}
    
    return {t: count / total for t, count in neighbor_counts.items()}


def neighbor_type_matrix(graph: 'CellGraph') -> 'pd.DataFrame':
    """
    Compute neighbor type distribution for all cell types.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    pd.DataFrame
        Matrix where M[i,j] is the proportion of type j among
        neighbors of type i cells.
    """
    import pandas as pd
    
    cell_types = graph.cell_types_unique
    
    data = {}
    for ct in cell_types:
        data[ct] = neighbor_type_distribution(graph, ct)
    
    return pd.DataFrame(data).T


# ============================================================================
# Degree Connectivity
# ============================================================================

def average_degree_connectivity(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute average degree connectivity.
    
    This gives the average neighbor degree for nodes of each degree.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Degree to average neighbor degree.
    """
    return nx.average_degree_connectivity(graph.G)
