"""
Topology module for spatialtissuepy.

Provides topological data analysis (TDA) tools for discovering cell communities,
with a focus on the Mapper algorithm with spatial-aware filter functions.

Key Components
--------------
SpatialMapper : Main Mapper algorithm implementation
MapperResult : Container for Mapper outputs
spatial_mapper : Convenience function for running Mapper

Filter Functions (filters.py)
-----------------------------
density_filter : Local cell density
pca_filter : PCA projection
eccentricity_filter : Distance from centroid
entropy_filter : Neighborhood diversity

Spatial Filter Functions (spatial_filters.py)
---------------------------------------------
spatial_coordinate_filter : Direct x/y/z coordinate projection
distance_to_type_filter : Distance to nearest cell of specified type
radial_filter : Distance from reference point
distance_to_boundary_filter : Distance from tissue edge
gaussian_smoothed_filter : Spatially smoothed version of any filter
composite_filter : Weighted combination of filters

Cover Types (cover.py)
----------------------
UniformCover : Equal-width intervals
AdaptiveCover : Equal-count intervals (quantile-based)
BallCover : For 2D filter outputs

Analysis Functions (analysis.py)
--------------------------------
node_summary_dataframe : Summary of all nodes
component_statistics : Statistics per connected component
compare_mapper_results : Cross-sample comparison
extract_mapper_features : Feature extraction for ML

Visualization (visualization.py)
--------------------------------
plot_mapper_graph : Plot the Mapper graph
plot_mapper_spatial : Spatial embedding of results
plot_filter_distribution : Filter value histogram
create_mapper_report : Multi-panel report

Example
-------
>>> from spatialtissuepy.topology import SpatialMapper, spatial_mapper
>>> from spatialtissuepy.topology.spatial_filters import distance_to_type_filter
>>>
>>> # Using the class
>>> mapper = SpatialMapper(
...     filter_fn=distance_to_type_filter('Tumor'),
...     n_intervals=10,
...     overlap=0.5,
... )
>>> result = mapper.fit(data, neighborhood_radius=50)
>>>
>>> # Using the convenience function
>>> result = spatial_mapper(data, filter_fn='density', n_intervals=10)
>>>
>>> print(f"Found {result.n_nodes} community nodes in {result.n_components} components")
>>>
>>> # Visualization
>>> from spatialtissuepy.topology.visualization import plot_mapper_graph
>>> plot_mapper_graph(result)
"""

from .mapper import SpatialMapper, MapperResult, spatial_mapper
from .cover import Cover, UniformCover, AdaptiveCover, BallCover, create_cover
from .nerve import MapperNode, MapperEdge

# Filter functions
from .filters import (
    density_filter,
    pca_filter,
    eccentricity_filter,
    linfinity_centrality_filter,
    sum_filter,
    entropy_filter,
    constant_filter,
)

# Spatial filters
from .spatial_filters import (
    spatial_coordinate_filter,
    radial_filter,
    distance_to_type_filter,
    distance_to_boundary_filter,
    spatial_density_filter,
    gaussian_smoothed_filter,
    composite_filter,
    multiscale_spatial_filter,
    type_proportion_filter,
)

# Analysis functions
from .analysis import (
    node_summary_dataframe,
    edge_summary_dataframe,
    find_hub_nodes,
    find_bridge_nodes,
    component_statistics,
    get_component_cells,
    compare_mapper_results,
    extract_mapper_features,
    cell_mapper_features,
    cells_in_multiple_nodes,
    uncovered_cells,
    mapper_stability_score,
    optimal_n_intervals,
)

__all__ = [
    # Main classes
    'SpatialMapper',
    'MapperResult',
    'spatial_mapper',
    # Cover classes
    'Cover',
    'UniformCover',
    'AdaptiveCover', 
    'BallCover',
    'create_cover',
    # Nerve classes
    'MapperNode',
    'MapperEdge',
    # Standard filters
    'density_filter',
    'pca_filter',
    'eccentricity_filter',
    'linfinity_centrality_filter',
    'sum_filter',
    'entropy_filter',
    'constant_filter',
    # Spatial filters
    'spatial_coordinate_filter',
    'radial_filter',
    'distance_to_type_filter',
    'distance_to_boundary_filter',
    'spatial_density_filter',
    'gaussian_smoothed_filter',
    'composite_filter',
    'multiscale_spatial_filter',
    'type_proportion_filter',
    # Analysis functions
    'node_summary_dataframe',
    'edge_summary_dataframe',
    'find_hub_nodes',
    'find_bridge_nodes',
    'component_statistics',
    'get_component_cells',
    'compare_mapper_results',
    'extract_mapper_features',
    'cell_mapper_features',
    'cells_in_multiple_nodes',
    'uncovered_cells',
    'mapper_stability_score',
    'optimal_n_intervals',
]

# Import summary metrics to register them (lazy import to avoid circular deps)
def _register_metrics():
    from . import summary_metrics

try:
    _register_metrics()
except ImportError:
    pass  # Summary module may not be available
