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
"""

from .mapper import SpatialMapper, MapperResult, spatial_mapper
from .cover import Cover, UniformCover, AdaptiveCover, BallCover, create_cover
from .nerve import MapperNode, MapperEdge

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
]
