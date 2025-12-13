"""
Analysis utilities for Mapper results.

This module provides functions for analyzing and comparing Mapper results,
including multi-sample analysis and feature extraction for downstream ML.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .mapper import MapperResult
    from spatialtissuepy.core.spatial_data import SpatialTissueData


# -----------------------------------------------------------------------------
# Node Analysis
# -----------------------------------------------------------------------------

def node_summary_dataframe(
    result: 'MapperResult',
    data: Optional['SpatialTissueData'] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame of all Mapper nodes.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    data : SpatialTissueData, optional
        Original data for additional statistics.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per node.
    """
    rows = []
    
    for node in result.nodes:
        row = {
            'node_id': node.node_id,
            'size': node.size,
            'cover_element': node.cover_element,
            'cluster_label': node.cluster_label,
            'filter_mean': np.mean(result.filter_values[node.members]),
            'filter_std': np.std(result.filter_values[node.members]),
            'filter_min': np.min(result.filter_values[node.members]),
            'filter_max': np.max(result.filter_values[node.members]),
            'spatial_x': node.spatial_centroid[0],
            'spatial_y': node.spatial_centroid[1],
        }
        
        if len(node.spatial_centroid) > 2:
            row['spatial_z'] = node.spatial_centroid[2]
        
        # Add composition if available
        if result.graph is not None and node.node_id in result.graph.nodes:
            comp = result.graph.nodes[node.node_id].get('composition', {})
            for cell_type, count in comp.items():
                row[f'count_{cell_type}'] = count
                row[f'prop_{cell_type}'] = count / node.size if node.size > 0 else 0
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def edge_summary_dataframe(result: 'MapperResult') -> pd.DataFrame:
    """
    Create a summary DataFrame of all Mapper edges.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per edge.
    """
    rows = []
    
    for edge in result.edges:
        row = {
            'source': edge.source,
            'target': edge.target,
            'weight': edge.weight,
            'n_shared': len(edge.shared_members),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def find_hub_nodes(
    result: 'MapperResult',
    n_hubs: int = 5,
    metric: str = 'degree'
) -> List[Tuple[int, float]]:
    """
    Find hub nodes in the Mapper graph.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    n_hubs : int, default 5
        Number of hub nodes to return.
    metric : str, default 'degree'
        Hub metric: 'degree', 'betweenness', 'closeness', 'size'.
    
    Returns
    -------
    list of (node_id, score)
        Hub nodes sorted by score.
    """
    if result.graph is None:
        raise ValueError("MapperResult has no graph")
    
    import networkx as nx
    
    G = result.graph
    
    if metric == 'degree':
        scores = dict(G.degree())
    elif metric == 'betweenness':
        scores = nx.betweenness_centrality(G)
    elif metric == 'closeness':
        scores = nx.closeness_centrality(G)
    elif metric == 'size':
        scores = {n: G.nodes[n].get('size', 0) for n in G.nodes()}
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_nodes[:n_hubs]


def find_bridge_nodes(
    result: 'MapperResult',
    n_bridges: int = 5
) -> List[Tuple[int, float]]:
    """
    Find bridge nodes connecting different components.
    
    These are nodes whose removal would most increase graph fragmentation.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    n_bridges : int, default 5
        Number of bridge nodes to return.
    
    Returns
    -------
    list of (node_id, bridge_score)
        Bridge nodes sorted by score.
    """
    if result.graph is None:
        raise ValueError("MapperResult has no graph")
    
    import networkx as nx
    
    G = result.graph
    
    # Use betweenness centrality as bridge metric
    betweenness = nx.betweenness_centrality(G)
    
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_nodes[:n_bridges]


# -----------------------------------------------------------------------------
# Component Analysis  
# -----------------------------------------------------------------------------

def component_statistics(result: 'MapperResult') -> pd.DataFrame:
    """
    Compute statistics for each connected component.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per component.
    """
    if result.graph is None:
        raise ValueError("MapperResult has no graph")
    
    import networkx as nx
    
    G = result.graph
    components = list(nx.connected_components(G))
    
    # Sort by size
    components = sorted(components, key=len, reverse=True)
    
    rows = []
    
    for i, comp_nodes in enumerate(components):
        subgraph = G.subgraph(comp_nodes)
        
        # Get all cells in this component
        cells = []
        for node_id in comp_nodes:
            for node in result.nodes:
                if node.node_id == node_id:
                    cells.extend(node.members)
                    break
        cells = np.unique(cells)
        
        row = {
            'component_id': i,
            'n_nodes': len(comp_nodes),
            'n_edges': subgraph.number_of_edges(),
            'n_cells': len(cells),
            'filter_mean': np.mean(result.filter_values[cells]),
            'filter_std': np.std(result.filter_values[cells]),
        }
        
        # Aggregate composition
        composition = {}
        for node_id in comp_nodes:
            if node_id in G.nodes:
                comp = G.nodes[node_id].get('composition', {})
                for ct, count in comp.items():
                    composition[ct] = composition.get(ct, 0) + count
        
        for ct, count in composition.items():
            row[f'count_{ct}'] = count
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_component_cells(
    result: 'MapperResult',
    component_idx: int = 0
) -> np.ndarray:
    """
    Get all cell indices in a specific component.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    component_idx : int, default 0
        Component index (0 = largest).
    
    Returns
    -------
    np.ndarray
        Cell indices in the component.
    """
    return result.get_cells_by_component(component_idx)


# -----------------------------------------------------------------------------
# Multi-Sample Comparison
# -----------------------------------------------------------------------------

def compare_mapper_results(
    results: List['MapperResult'],
    sample_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare Mapper results across multiple samples.
    
    Parameters
    ----------
    results : list of MapperResult
        Mapper results from different samples.
    sample_ids : list of str, optional
        Names for each sample.
    
    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with one row per sample.
    """
    if sample_ids is None:
        sample_ids = [f'sample_{i}' for i in range(len(results))]
    
    rows = []
    
    for sample_id, result in zip(sample_ids, results):
        stats = result.statistics
        
        row = {
            'sample_id': sample_id,
            'n_nodes': result.n_nodes,
            'n_edges': result.n_edges,
            'n_components': result.n_components,
            'mean_degree': stats.get('mean_degree', np.nan),
            'mean_node_size': stats.get('mean_node_size', np.nan),
            'total_cells_in_graph': stats.get('total_cells_in_nodes', np.nan),
            'density': stats.get('density', np.nan),
            'avg_clustering': stats.get('avg_clustering', np.nan),
        }
        
        # Filter statistics
        row['filter_mean'] = np.mean(result.filter_values)
        row['filter_std'] = np.std(result.filter_values)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_mapper_features(
    result: 'MapperResult',
    prefix: str = 'mapper'
) -> Dict[str, float]:
    """
    Extract summary features from Mapper result for ML.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    prefix : str, default 'mapper'
        Prefix for feature names.
    
    Returns
    -------
    dict
        Dictionary of feature name -> value.
    """
    stats = result.statistics
    
    features = {
        f'{prefix}_n_nodes': result.n_nodes,
        f'{prefix}_n_edges': result.n_edges,
        f'{prefix}_n_components': result.n_components,
        f'{prefix}_mean_degree': stats.get('mean_degree', 0),
        f'{prefix}_max_degree': stats.get('max_degree', 0),
        f'{prefix}_mean_node_size': stats.get('mean_node_size', 0),
        f'{prefix}_density': stats.get('density', 0),
        f'{prefix}_avg_clustering': stats.get('avg_clustering', 0),
        f'{prefix}_filter_mean': np.mean(result.filter_values),
        f'{prefix}_filter_std': np.std(result.filter_values),
    }
    
    # Node size distribution
    node_sizes = [node.size for node in result.nodes]
    if node_sizes:
        features[f'{prefix}_node_size_std'] = np.std(node_sizes)
        features[f'{prefix}_node_size_max'] = np.max(node_sizes)
        features[f'{prefix}_node_size_min'] = np.min(node_sizes)
    
    # Edge weight distribution
    if result.edges:
        edge_weights = [e.weight for e in result.edges]
        features[f'{prefix}_edge_weight_mean'] = np.mean(edge_weights)
        features[f'{prefix}_edge_weight_max'] = np.max(edge_weights)
    
    # Component sizes
    if result.n_components > 0 and result.graph is not None:
        import networkx as nx
        comp_sizes = [len(c) for c in nx.connected_components(result.graph)]
        features[f'{prefix}_largest_component_nodes'] = max(comp_sizes) if comp_sizes else 0
        features[f'{prefix}_component_size_ratio'] = (
            max(comp_sizes) / result.n_nodes if result.n_nodes > 0 else 0
        )
    
    return features


# -----------------------------------------------------------------------------
# Cell-Level Analysis
# -----------------------------------------------------------------------------

def cell_mapper_features(
    result: 'MapperResult',
    data: 'SpatialTissueData'
) -> pd.DataFrame:
    """
    Compute per-cell features from Mapper result.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    data : SpatialTissueData
        Original data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per cell.
    """
    n_cells = data.n_cells
    
    # Initialize features
    features = {
        'filter_value': result.filter_values,
        'n_nodes': np.zeros(n_cells, dtype=int),
        'in_graph': np.zeros(n_cells, dtype=bool),
    }
    
    # Count nodes per cell
    for cell_idx, node_ids in result.cell_node_map.items():
        features['n_nodes'][cell_idx] = len(node_ids)
        features['in_graph'][cell_idx] = len(node_ids) > 0
    
    # Component assignment
    if result.graph is not None:
        import networkx as nx
        components = list(nx.connected_components(result.graph))
        components = sorted(components, key=len, reverse=True)
        
        node_to_comp = {}
        for i, comp in enumerate(components):
            for node in comp:
                node_to_comp[node] = i
        
        features['component_id'] = np.full(n_cells, -1, dtype=int)
        
        for cell_idx, node_ids in result.cell_node_map.items():
            if node_ids:
                features['component_id'][cell_idx] = node_to_comp.get(node_ids[0], -1)
    
    return pd.DataFrame(features)


def cells_in_multiple_nodes(result: 'MapperResult') -> np.ndarray:
    """
    Find cells that belong to multiple Mapper nodes.
    
    These cells are in the overlap regions and may represent
    transitional or boundary cells.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    
    Returns
    -------
    np.ndarray
        Indices of cells in multiple nodes.
    """
    multi_node_cells = []
    
    for cell_idx, node_ids in result.cell_node_map.items():
        if len(node_ids) > 1:
            multi_node_cells.append(cell_idx)
    
    return np.array(multi_node_cells)


def uncovered_cells(
    result: 'MapperResult',
    n_cells: int
) -> np.ndarray:
    """
    Find cells not covered by any Mapper node.
    
    Parameters
    ----------
    result : MapperResult
        Mapper result.
    n_cells : int
        Total number of cells in data.
    
    Returns
    -------
    np.ndarray
        Indices of uncovered cells.
    """
    covered = set(result.cell_node_map.keys())
    all_cells = set(range(n_cells))
    uncovered = all_cells - covered
    
    return np.array(sorted(list(uncovered)))


# -----------------------------------------------------------------------------
# Persistence and Stability
# -----------------------------------------------------------------------------

def mapper_stability_score(
    data: 'SpatialTissueData',
    n_runs: int = 10,
    subsample_fraction: float = 0.8,
    **mapper_kwargs
) -> Dict[str, float]:
    """
    Assess Mapper stability through subsampling.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    n_runs : int, default 10
        Number of subsample runs.
    subsample_fraction : float, default 0.8
        Fraction of cells to subsample.
    **mapper_kwargs
        Arguments passed to SpatialMapper.
    
    Returns
    -------
    dict
        Stability metrics.
    """
    from .mapper import SpatialMapper
    
    results = []
    n_cells = data.n_cells
    n_subsample = int(n_cells * subsample_fraction)
    
    for run in range(n_runs):
        # Random subsample
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        
        # Create subsampled data
        from spatialtissuepy.core import SpatialTissueData
        subdata = SpatialTissueData(
            coordinates=data._coordinates[indices],
            cell_types=data._cell_types[indices]
        )
        
        # Run Mapper
        mapper = SpatialMapper(**mapper_kwargs)
        result = mapper.fit(subdata)
        
        results.append({
            'n_nodes': result.n_nodes,
            'n_edges': result.n_edges,
            'n_components': result.n_components,
        })
    
    # Compute stability metrics
    df = pd.DataFrame(results)
    
    stability = {
        'n_nodes_mean': df['n_nodes'].mean(),
        'n_nodes_std': df['n_nodes'].std(),
        'n_nodes_cv': df['n_nodes'].std() / df['n_nodes'].mean() if df['n_nodes'].mean() > 0 else 0,
        'n_edges_mean': df['n_edges'].mean(),
        'n_edges_std': df['n_edges'].std(),
        'n_components_mean': df['n_components'].mean(),
        'n_components_std': df['n_components'].std(),
    }
    
    return stability


def optimal_n_intervals(
    data: 'SpatialTissueData',
    interval_range: List[int] = None,
    metric: str = 'n_components',
    **mapper_kwargs
) -> Tuple[int, pd.DataFrame]:
    """
    Find optimal number of intervals for Mapper.
    
    Parameters
    ----------
    data : SpatialTissueData
        Input data.
    interval_range : list of int, optional
        Interval values to try. Default: [5, 8, 10, 12, 15, 20].
    metric : str, default 'n_components'
        Metric to optimize: 'n_components', 'n_nodes', 'coverage'.
    **mapper_kwargs
        Additional arguments for SpatialMapper.
    
    Returns
    -------
    optimal_n : int
        Optimal number of intervals.
    results_df : pd.DataFrame
        Results for all interval values.
    """
    from .mapper import SpatialMapper
    
    if interval_range is None:
        interval_range = [5, 8, 10, 12, 15, 20]
    
    results = []
    
    for n_int in interval_range:
        mapper = SpatialMapper(n_intervals=n_int, **mapper_kwargs)
        result = mapper.fit(data)
        
        coverage = len(result.cell_node_map) / data.n_cells
        
        results.append({
            'n_intervals': n_int,
            'n_nodes': result.n_nodes,
            'n_edges': result.n_edges,
            'n_components': result.n_components,
            'coverage': coverage,
        })
    
    df = pd.DataFrame(results)
    
    # Find optimal based on metric
    if metric == 'n_components':
        # Prefer fewer components (more connected)
        optimal_idx = df['n_components'].idxmin()
    elif metric == 'coverage':
        # Prefer higher coverage
        optimal_idx = df['coverage'].idxmax()
    else:
        optimal_idx = df[metric].idxmax()
    
    optimal_n = df.loc[optimal_idx, 'n_intervals']
    
    return int(optimal_n), df
