"""
Mapper metrics for integration with summary module.

This module registers Mapper-derived metrics with the StatisticsPanel
for standardized computation across samples.
"""

from typing import Dict, TYPE_CHECKING
import numpy as np

from spatialtissuepy.summary.registry import register_metric

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData


# Store fitted results for reuse within a session
_result_cache: Dict[str, 'MapperResult'] = {}


# -----------------------------------------------------------------------------
# Graph Structure Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mapper_n_nodes',
    category='topology',
    description='Number of nodes in Mapper graph',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_n_nodes(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute number of Mapper nodes."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    return {'mapper_n_nodes': float(result.n_nodes)}


@register_metric(
    name='mapper_n_components',
    category='topology',
    description='Number of connected components in Mapper graph',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_n_components(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute number of connected components."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    return {'mapper_n_components': float(result.n_components)}


@register_metric(
    name='mapper_graph_density',
    category='topology',
    description='Density of Mapper graph',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_density(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute Mapper graph density."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    stats = result.statistics
    
    return {'mapper_density': float(stats.get('density', 0))}


@register_metric(
    name='mapper_summary',
    category='topology',
    description='Summary statistics from Mapper graph',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_summary(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute comprehensive Mapper statistics."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    stats = result.statistics
    
    output = {
        'mapper_n_nodes': float(result.n_nodes),
        'mapper_n_edges': float(result.n_edges),
        'mapper_n_components': float(result.n_components),
        'mapper_mean_degree': float(stats.get('mean_degree', 0)),
        'mapper_mean_node_size': float(stats.get('mean_node_size', 0)),
        'mapper_density': float(stats.get('density', 0)),
    }
    
    # Node coverage
    coverage = len(result.cell_node_map) / data.n_cells if data.n_cells > 0 else 0
    output['mapper_coverage'] = coverage
    
    return output


@register_metric(
    name='mapper_clustering',
    category='topology',
    description='Average clustering coefficient of Mapper graph',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_clustering(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute Mapper average clustering coefficient."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    stats = result.statistics
    
    return {'mapper_avg_clustering': float(stats.get('avg_clustering', 0))}


# -----------------------------------------------------------------------------
# Spatial Filter Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mapper_spatial_filter',
    category='topology',
    description='Mapper using spatial x-coordinate filter',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_spatial_filter(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute Mapper with spatial x-coordinate filter."""
    from .mapper import SpatialMapper
    from .spatial_filters import spatial_coordinate_filter
    
    mapper = SpatialMapper(
        filter_fn=spatial_coordinate_filter('x'),
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    return {
        'mapper_spatial_n_nodes': float(result.n_nodes),
        'mapper_spatial_n_components': float(result.n_components),
    }


@register_metric(
    name='mapper_distance_to_type',
    category='topology',
    description='Mapper using distance to cell type filter',
    parameters={'cell_type': str, 'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_distance_to_type(
    data: 'SpatialTissueData',
    cell_type: str,
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute Mapper with distance-to-type filter."""
    from .mapper import SpatialMapper
    from .spatial_filters import distance_to_type_filter
    
    # Check if cell type exists
    if cell_type not in data.cell_types_unique:
        return {
            f'mapper_dist_{cell_type}_n_nodes': np.nan,
            f'mapper_dist_{cell_type}_n_components': np.nan,
        }
    
    mapper = SpatialMapper(
        filter_fn=distance_to_type_filter(cell_type),
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    return {
        f'mapper_dist_{cell_type}_n_nodes': float(result.n_nodes),
        f'mapper_dist_{cell_type}_n_components': float(result.n_components),
    }


# -----------------------------------------------------------------------------
# Component Analysis Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mapper_largest_component',
    category='topology',
    description='Size of largest connected component',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_largest_component(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute largest component statistics."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    if result.graph is not None and result.n_nodes > 0:
        import networkx as nx
        components = list(nx.connected_components(result.graph))
        if components:
            largest_size = max(len(c) for c in components)
            ratio = largest_size / result.n_nodes
        else:
            largest_size = 0
            ratio = 0
    else:
        largest_size = 0
        ratio = 0
    
    return {
        'mapper_largest_component_nodes': float(largest_size),
        'mapper_component_ratio': float(ratio),
    }


# -----------------------------------------------------------------------------
# Node Distribution Metrics
# -----------------------------------------------------------------------------

@register_metric(
    name='mapper_node_size_stats',
    category='topology',
    description='Node size distribution statistics',
    parameters={'n_intervals': int, 'overlap': float, 'radius': float}
)
def _mapper_node_size_stats(
    data: 'SpatialTissueData',
    n_intervals: int = 10,
    overlap: float = 0.5,
    radius: float = 50.0
) -> Dict[str, float]:
    """Compute node size distribution statistics."""
    from .mapper import SpatialMapper
    
    mapper = SpatialMapper(
        filter_fn='density',
        n_intervals=n_intervals,
        overlap=overlap,
    )
    result = mapper.fit(data, neighborhood_radius=radius)
    
    if result.nodes:
        sizes = [node.size for node in result.nodes]
        return {
            'mapper_node_size_mean': float(np.mean(sizes)),
            'mapper_node_size_std': float(np.std(sizes)),
            'mapper_node_size_max': float(np.max(sizes)),
            'mapper_node_size_min': float(np.min(sizes)),
        }
    else:
        return {
            'mapper_node_size_mean': 0.0,
            'mapper_node_size_std': 0.0,
            'mapper_node_size_max': 0.0,
            'mapper_node_size_min': 0.0,
        }
