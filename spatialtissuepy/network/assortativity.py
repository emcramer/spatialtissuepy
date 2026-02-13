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


def _get_nx_graph(graph: Union['CellGraph', 'nx.Graph']) -> 'nx.Graph':
    """Helper to extract NetworkX graph from CellGraph or return nx.Graph."""
    if hasattr(graph, 'G'):
        return graph.G
    return graph


# ============================================================================
# Assortativity Coefficients
# ============================================================================

def degree_assortativity(graph: Union['CellGraph', 'nx.Graph']) -> float:
    """
    Compute degree assortativity coefficient.
    
    Measures whether high-degree nodes connect preferentially to
    other high-degree nodes (positive) or low-degree nodes (negative).
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    float
        Degree assortativity coefficient in [-1, 1].
    """
    return nx.degree_assortativity_coefficient(_get_nx_graph(graph))


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
    graph: Union['CellGraph', 'nx.Graph'],
    attribute: str
) -> float:
    """
    Compute numeric assortativity for a node attribute.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    attribute : str
        Node attribute name (must be numeric).
    
    Returns
    -------
    float
        Numeric assortativity coefficient.
    """
    return nx.numeric_assortativity_coefficient(_get_nx_graph(graph), attribute)


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

def homophily_ratio_by_cell_type(graph: 'CellGraph') -> dict:
    """
    Vectorized computation of homophily ratio per cell type.
    
    Parameters
    ----------
    graph : CellGraph
        Input cell graph.
    
    Returns
    -------
    dict
        Dictionary mapping cell_type -> ratio (float).
    """
    # 1. Extract Edges
    # Shape: (N_edges, 2)
    edges = np.array(list(graph.G.edges()))
    
    if len(edges) == 0:
        return {}

    # 2. Map Node IDs to Cell Types
    # We extract the source (u) and target (v) columns
    u_nodes = edges[:, 0]
    v_nodes = edges[:, 1]
    
    # Efficient lookup: fast iteration in list comp, then conversion to numpy array
    # This works for integer or string node IDs
    # Assumes graph.cell_types is a dict or supports __getitem__
    u_types = np.array([graph.cell_types[n] for n in u_nodes])
    v_types = np.array([graph.cell_types[n] for n in v_nodes])

    # 3. Calculate Denominators (Total connections per type)
    # Concatenate both sides to get all 'stubs'
    all_stubs = np.concatenate([u_types, v_types])
    # np.unique returns sorted unique elements and their counts
    unique_types, counts_total = np.unique(all_stubs, return_counts=True)
    # Create a quick lookup for totals
    total_map = dict(zip(unique_types, counts_total))

    # 4. Calculate Numerators (Same-type connections)
    # Boolean mask where types match
    mask = (u_types == v_types)
    
    # Filter for types involved in same-type edges
    # We only need one side (u_types) since u == v
    same_stubs = u_types[mask]
    
    unique_same, counts_same = np.unique(same_stubs, return_counts=True)
    # Multiply by 2 because each edge counts for both nodes
    same_map = dict(zip(unique_same, counts_same * 2))

    # 5. Compute Ratios
    # Iterate over total_map to ensure we include types with 0 homophily
    ratios = {}
    for c_type, total in total_map.items():
        same = same_map.get(c_type, 0)
        ratios[c_type] = same / total
        
    return ratios

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

def average_neighbor_degree(graph: Union['CellGraph', 'nx.Graph']) -> Dict[int, float]:
    """
    Compute average neighbor degree for all nodes.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    dict
        Node index to average neighbor degree.
    """
    return nx.average_neighbor_degree(_get_nx_graph(graph))


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

def average_degree_connectivity(graph: Union['CellGraph', 'nx.Graph']) -> Dict[int, float]:
    """
    Compute average degree connectivity.
    
    This gives the average neighbor degree for nodes of each degree.
    
    Parameters
    ----------
    graph : CellGraph or nx.Graph
        Input graph.
    
    Returns
    -------
    dict
        Degree to average neighbor degree.
    """
    return nx.average_degree_connectivity(_get_nx_graph(graph))

def average_node_degree(graph: 'CellGraph') -> Dict[int, float]:
    """
    Compute the average degree across all nodes in the graph. 

    This gives an overall measure of how connected nodes in the graph are.

    Parameters
    ----------
    graph : CellGraph
        Input cell graph.

    Returns
    -------
    float
        Average degree of all nodes in the graph.
    """
    G = graph.G
    node_degrees = np.array([G.degree(node) for node in G.nodes])
    return np.mean(node_degrees)
