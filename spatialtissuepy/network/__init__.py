"""
Network analysis module for spatialtissuepy.

Provides graph-based analysis of spatial tissue data using NetworkX.
Cells are represented as nodes, with edges defined by spatial proximity
or other graph construction methods.

Graph Construction Methods
--------------------------
- proximity : Connect cells within a distance radius
- knn : k-nearest neighbors
- delaunay : Delaunay triangulation
- gabriel : Gabriel graph (subset of Delaunay)

Key Classes
-----------
CellGraph : Main graph container with cell type metadata

Metric Categories
-----------------
- Centrality : degree, betweenness, closeness, eigenvector, PageRank
- Clustering : local/global clustering, transitivity, triangles
- Communicability : path-based communication metrics
- Assortativity : mixing patterns, type assortativity
- Structure : connected components, bridges, articulation points

Example
-------
>>> from spatialtissuepy.network import CellGraph
>>>
>>> # Build graph from spatial data
>>> graph = CellGraph.from_spatial_data(
...     data,
...     method='proximity',  # or 'knn', 'delaunay', 'gabriel'
...     radius=30.0
... )
>>>
>>> # Analyze centrality by cell type
>>> from spatialtissuepy.network import centrality_by_type
>>> stats = centrality_by_type(graph, metric='betweenness')
>>> print(stats['CD8_T_cell']['mean'])
>>>
>>> # Communicability between types
>>> from spatialtissuepy.network import communicability_between_types
>>> comm = communicability_between_types(graph, 'T_cell', 'Tumor')
>>>
>>> # Mixing patterns
>>> from spatialtissuepy.network import attribute_mixing_matrix
>>> mixing = attribute_mixing_matrix(graph)
"""

# Graph construction
# Import metrics to register them with summary module
from . import metrics

# Assortativity and mixing
from .assortativity import (
    attribute_mixing_dict,
    attribute_mixing_matrix,
    average_degree_connectivity,
    average_neighbor_degree,
    average_neighbor_degree_by_type,
    average_node_degree,
    degree_assortativity,
    heterophily_ratio,
    homophily_ratio,
    homophily_ratio_by_cell_type,
    neighbor_type_distribution,
    neighbor_type_matrix,
    numeric_assortativity,
    type_assortativity,
    type_pair_edge_fraction,
)

# CellGraph class
from .cell_graph import CellGraph

# Centrality metrics
from .centrality import (
    betweenness_centrality,
    centrality_by_type,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    harmonic_centrality,
    katz_centrality,
    load_centrality,
    mean_centrality_by_type,
    pagerank,
    subgraph_centrality,
    top_central_nodes,
)

# Clustering metrics
from .clustering import (
    articulation_points,
    articulation_points_by_type,
    average_clustering,
    bridges,
    bridges_by_type_pair,
    clustering_by_type,
    clustering_coefficient,
    connected_components,
    largest_component_size,
    mean_clustering_by_type,
    n_connected_components,
    square_clustering,
    transitivity,
    triangles,
    triangles_by_type,
)

# Communicability and path metrics
from .communicability import (
    average_shortest_path_length,
    communicability,
    communicability_between_types,
    communicability_betweenness,
    communicability_exp,
    communicability_matrix_by_type,
    diameter,
    eccentricity,
    global_efficiency,
    local_efficiency,
    nodal_efficiency,
    radius,
    shortest_path_length_between_types,
)
from .graph_construction import (
    GraphMethod,
    build_delaunay_graph,
    build_gabriel_graph,
    build_graph,
    build_knn_graph,
    build_proximity_graph,
)

__all__ = [
    # Graph construction
    'GraphMethod',
    'build_graph',
    'build_proximity_graph',
    'build_knn_graph',
    'build_delaunay_graph',
    'build_gabriel_graph',
    # CellGraph
    'CellGraph',
    # Centrality
    'degree_centrality',
    'betweenness_centrality',
    'closeness_centrality',
    'eigenvector_centrality',
    'pagerank',
    'harmonic_centrality',
    'katz_centrality',
    'load_centrality',
    'subgraph_centrality',
    'centrality_by_type',
    'mean_centrality_by_type',
    'top_central_nodes',
    # Clustering
    'clustering_coefficient',
    'average_clustering',
    'transitivity',
    'square_clustering',
    'triangles',
    'clustering_by_type',
    'mean_clustering_by_type',
    'triangles_by_type',
    'connected_components',
    'n_connected_components',
    'largest_component_size',
    'bridges',
    'articulation_points',
    'articulation_points_by_type',
    'bridges_by_type_pair',
    # Communicability
    'communicability',
    'communicability_exp',
    'communicability_betweenness',
    'communicability_between_types',
    'communicability_matrix_by_type',
    'shortest_path_length_between_types',
    'average_shortest_path_length',
    'diameter',
    'radius',
    'eccentricity',
    'global_efficiency',
    'local_efficiency',
    'nodal_efficiency',
    # Assortativity
    'degree_assortativity',
    'type_assortativity',
    'numeric_assortativity',
    'attribute_mixing_matrix',
    'attribute_mixing_dict',
    'homophily_ratio',
    'heterophily_ratio',
    'type_pair_edge_fraction',
    'average_neighbor_degree',
    'average_neighbor_degree_by_type',
    'neighbor_type_distribution',
    'neighbor_type_matrix',
    'average_degree_connectivity',
]
