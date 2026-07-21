"""
Spatial operations module for spatialtissuepy.

This module provides core spatial analysis operations including:
- Distance calculations and nearest neighbor queries
- Neighborhood analysis and composition
- Spatial clustering algorithms

Submodules
----------
distance : Distance metrics and nearest neighbor operations
neighborhood : Neighborhood computation and composition analysis
clustering : Spatial clustering methods (DBSCAN, k-means, hierarchical, etc.)

Example
-------
>>> from spatialtissuepy.spatial import (
...     nearest_neighbors,
...     neighborhood_composition,
...     dbscan_clustering,
... )
>>>
>>> # Find k-nearest neighbors
>>> distances, indices = nearest_neighbors(data.coordinates, k=10)
>>>
>>> # Compute neighborhood composition
>>> composition = neighborhood_composition(data, method='radius', radius=50)
>>>
>>> # Cluster cells spatially
>>> labels = dbscan_clustering(data, eps=30, min_samples=5)
"""

# Distance module
# Clustering module
from .clustering import (
    # Methods enum
    ClusteringMethod,
    cluster_purity,
    # Utilities
    cluster_statistics,
    connected_components_spatial,
    cut_dendrogram,
    dbscan_by_type,
    # DBSCAN
    dbscan_clustering,
    # HDBSCAN (optional)
    hdbscan_clustering,
    # Hierarchical
    hierarchical_clustering,
    hierarchical_linkage,
    kmeans_by_type,
    # K-means
    kmeans_spatial,
    # Graph-based
    leiden_clustering,
    louvain_clustering,
    silhouette_spatial,
    # Spatial regions
    spatial_regions,
)
from .distance import (
    # Distance metrics
    DistanceMetric,
    bounding_box,
    # Nearest neighbors
    build_kdtree,
    # Utilities
    centroid,
    centroid_by_type,
    condensed_distances,
    convex_hull_area,
    distance_matrix_by_type,
    distance_to_nearest_different_type,
    # Distance to types
    distance_to_type,
    mean_nearest_neighbor_distance,
    nearest_neighbor_distances,
    nearest_neighbors,
    pairwise_distances,
    pairwise_distances_between,
    point_density,
    radius_neighbors,
)

# Neighborhood module
from .neighborhood import (
    # Neighborhood computation
    NeighborhoodMethod,
    # Adjacency
    adjacency_matrix,
    compute_neighborhoods,
    interface_cells,
    neighborhood_composition,
    neighborhood_counts,
    neighborhood_diversity,
    neighborhood_enrichment,
    # Statistics
    neighborhood_size,
    # DataFrame conversion
    neighborhood_to_dataframe,
    type_adjacency_matrix,
    window_composition,
)

__all__ = [
    # Distance
    'DistanceMetric',
    'pairwise_distances',
    'pairwise_distances_between',
    'condensed_distances',
    'build_kdtree',
    'nearest_neighbors',
    'radius_neighbors',
    'nearest_neighbor_distances',
    'mean_nearest_neighbor_distance',
    'distance_to_type',
    'distance_to_nearest_different_type',
    'distance_matrix_by_type',
    'centroid',
    'centroid_by_type',
    'bounding_box',
    'convex_hull_area',
    'point_density',
    # Neighborhood
    'NeighborhoodMethod',
    'compute_neighborhoods',
    'neighborhood_counts',
    'neighborhood_composition',
    'window_composition',
    'adjacency_matrix',
    'type_adjacency_matrix',
    'neighborhood_size',
    'neighborhood_diversity',
    'neighborhood_enrichment',
    'interface_cells',
    'neighborhood_to_dataframe',
    # Clustering
    'ClusteringMethod',
    'dbscan_clustering',
    'dbscan_by_type',
    'hdbscan_clustering',
    'kmeans_spatial',
    'kmeans_by_type',
    'hierarchical_clustering',
    'hierarchical_linkage',
    'cut_dendrogram',
    'leiden_clustering',
    'louvain_clustering',
    'cluster_statistics',
    'cluster_purity',
    'silhouette_spatial',
    'spatial_regions',
    'connected_components_spatial',
]

# Import metrics to register them with summary module
from . import metrics
