"""
Network metrics for integration with the summary module.

Registers network-based metrics that can be included in StatisticsPanels.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from spatialtissuepy.core.spatial_data import SpatialTissueData

# Import registry for metric registration
try:
    from spatialtissuepy.summary.registry import register_metric
    HAS_SUMMARY = True
except ImportError:
    HAS_SUMMARY = False
    # Create dummy decorator
    def register_metric(**kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Graph Construction Helper
# ============================================================================

def _build_graph_from_data(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
):
    """Build a CellGraph from SpatialTissueData."""
    from .cell_graph import CellGraph
    
    return CellGraph.from_spatial_data(
        data, method=method, radius=radius, k=k
    )


# ============================================================================
# Registered Network Metrics
# ============================================================================

@register_metric(
    name='graph_density',
    category='network',
    description='Density of the cell graph',
    parameters={'method': str, 'radius': float, 'k': int}
)
def graph_density(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute graph density."""
    graph = _build_graph_from_data(data, method, radius, k)
    return {'graph_density': graph.density}


@register_metric(
    name='average_clustering',
    category='network',
    description='Average clustering coefficient of the cell graph',
    parameters={'method': str, 'radius': float, 'k': int}
)
def average_clustering_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute average clustering coefficient."""
    from .clustering import average_clustering
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'average_clustering': average_clustering(graph)}


@register_metric(
    name='transitivity',
    category='network',
    description='Graph transitivity (global clustering coefficient)',
    parameters={'method': str, 'radius': float, 'k': int}
)
def transitivity_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute graph transitivity."""
    from .clustering import transitivity
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'transitivity': transitivity(graph)}


@register_metric(
    name='mean_clustering_by_type',
    category='network',
    description='Mean clustering coefficient for each cell type',
    parameters={'method': str, 'radius': float, 'k': int},
    dynamic_columns=True
)
def mean_clustering_by_type_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute mean clustering coefficient per cell type."""
    from .clustering import mean_clustering_by_type
    
    graph = _build_graph_from_data(data, method, radius, k)
    stats = mean_clustering_by_type(graph)
    
    return {f'clustering_{ct}': val for ct, val in stats.items()}


@register_metric(
    name='type_assortativity',
    category='network',
    description='Cell type assortativity coefficient',
    parameters={'method': str, 'radius': float, 'k': int}
)
def type_assortativity_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute cell type assortativity."""
    from .assortativity import type_assortativity
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'type_assortativity': type_assortativity(graph)}


@register_metric(
    name='degree_assortativity',
    category='network',
    description='Degree assortativity coefficient',
    parameters={'method': str, 'radius': float, 'k': int}
)
def degree_assortativity_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute degree assortativity."""
    from .assortativity import degree_assortativity
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'degree_assortativity': degree_assortativity(graph)}


@register_metric(
    name='homophily_ratio',
    category='network',
    description='Fraction of edges connecting same-type cells',
    parameters={'method': str, 'radius': float, 'k': int}
)
def homophily_ratio_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute homophily ratio."""
    from .assortativity import homophily_ratio
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'homophily_ratio': homophily_ratio(graph)}


@register_metric(
    name='mean_degree_centrality_by_type',
    category='network',
    description='Mean degree centrality for each cell type',
    parameters={'method': str, 'radius': float, 'k': int},
    dynamic_columns=True
)
def mean_degree_centrality_by_type_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute mean degree centrality per cell type."""
    from .centrality import mean_centrality_by_type
    
    graph = _build_graph_from_data(data, method, radius, k)
    stats = mean_centrality_by_type(graph, metric='degree')
    
    return {f'degree_centrality_{ct}': val for ct, val in stats.items()}


@register_metric(
    name='mean_betweenness_centrality_by_type',
    category='network',
    description='Mean betweenness centrality for each cell type',
    parameters={'method': str, 'radius': float, 'k': int},
    dynamic_columns=True
)
def mean_betweenness_centrality_by_type_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute mean betweenness centrality per cell type."""
    from .centrality import mean_centrality_by_type
    
    graph = _build_graph_from_data(data, method, radius, k)
    stats = mean_centrality_by_type(graph, metric='betweenness')
    
    return {f'betweenness_centrality_{ct}': val for ct, val in stats.items()}


@register_metric(
    name='mean_closeness_centrality_by_type',
    category='network',
    description='Mean closeness centrality for each cell type',
    parameters={'method': str, 'radius': float, 'k': int},
    dynamic_columns=True
)
def mean_closeness_centrality_by_type_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute mean closeness centrality per cell type."""
    from .centrality import mean_centrality_by_type
    
    graph = _build_graph_from_data(data, method, radius, k)
    stats = mean_centrality_by_type(graph, metric='closeness')
    
    return {f'closeness_centrality_{ct}': val for ct, val in stats.items()}


@register_metric(
    name='global_efficiency',
    category='network',
    description='Global efficiency of the cell graph',
    parameters={'method': str, 'radius': float, 'k': int}
)
def global_efficiency_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute global efficiency."""
    from .communicability import global_efficiency
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'global_efficiency': global_efficiency(graph)}


@register_metric(
    name='local_efficiency',
    category='network',
    description='Local efficiency of the cell graph',
    parameters={'method': str, 'radius': float, 'k': int}
)
def local_efficiency_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Compute local efficiency."""
    from .communicability import local_efficiency
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'local_efficiency': local_efficiency(graph)}


@register_metric(
    name='n_connected_components',
    category='network',
    description='Number of connected components in the cell graph',
    parameters={'method': str, 'radius': float, 'k': int}
)
def n_connected_components_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Count connected components."""
    from .clustering import n_connected_components
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'n_connected_components': float(n_connected_components(graph))}


@register_metric(
    name='largest_component_fraction',
    category='network',
    description='Fraction of cells in the largest connected component',
    parameters={'method': str, 'radius': float, 'k': int}
)
def largest_component_fraction_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Fraction of cells in largest component."""
    from .clustering import largest_component_size
    
    graph = _build_graph_from_data(data, method, radius, k)
    
    if graph.n_nodes == 0:
        return {'largest_component_fraction': np.nan}
    
    return {
        'largest_component_fraction': largest_component_size(graph) / graph.n_nodes
    }


@register_metric(
    name='n_articulation_points',
    category='network',
    description='Number of articulation points (cut vertices)',
    parameters={'method': str, 'radius': float, 'k': int}
)
def n_articulation_points_metric(
    data: 'SpatialTissueData',
    method: str = 'proximity',
    radius: float = 50.0,
    k: int = 6,
) -> Dict[str, float]:
    """Count articulation points."""
    from .clustering import articulation_points
    
    graph = _build_graph_from_data(data, method, radius, k)
    return {'n_articulation_points': float(len(articulation_points(graph)))}


@register_metric(
    name='communicability_between_types',
    category='network',
    description='Mean communicability between two cell types',
    parameters={'type_a': str, 'type_b': str, 'method': str, 'radius': float}
)
def communicability_between_types_metric(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    method: str = 'proximity',
    radius: float = 50.0,
    sample_size: int = 1000,
) -> Dict[str, float]:
    """Compute communicability between two cell types."""
    from .communicability import communicability_between_types
    
    graph = _build_graph_from_data(data, method, radius, k=6)
    stats = communicability_between_types(
        graph, type_a, type_b, sample_size=sample_size
    )
    
    return {f'communicability_{type_a}_{type_b}': stats['mean']}


@register_metric(
    name='shortest_path_between_types',
    category='network',
    description='Mean shortest path length between two cell types',
    parameters={'type_a': str, 'type_b': str, 'method': str, 'radius': float}
)
def shortest_path_between_types_metric(
    data: 'SpatialTissueData',
    type_a: str,
    type_b: str,
    method: str = 'proximity',
    radius: float = 50.0,
    sample_size: int = 1000,
) -> Dict[str, float]:
    """Compute mean shortest path between two cell types."""
    from .communicability import shortest_path_length_between_types
    
    graph = _build_graph_from_data(data, method, radius, k=6)
    stats = shortest_path_length_between_types(
        graph, type_a, type_b, sample_size=sample_size
    )
    
    return {f'shortest_path_{type_a}_{type_b}': stats['mean']}
