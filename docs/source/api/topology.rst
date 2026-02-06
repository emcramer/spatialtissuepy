========
Topology
========

.. module:: spatialtissuepy.topology

The topology module provides the Mapper algorithm and related topological
data analysis methods for discovering tissue organization patterns.


Mapper Classes
--------------

.. autoclass:: spatialtissuepy.topology.SpatialMapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: spatialtissuepy.topology.MapperResult
   :members:
   :undoc-members:
   :show-inheritance:


Convenience Functions
---------------------

.. autofunction:: spatialtissuepy.topology.spatial_mapper


Filter Functions
----------------

Built-in filter functions for the Mapper algorithm:

.. autofunction:: spatialtissuepy.topology.density_filter

.. autofunction:: spatialtissuepy.topology.eccentricity_filter

.. autofunction:: spatialtissuepy.topology.pca_filter


Spatial Filters
---------------

Spatial-aware filter functions:

.. autofunction:: spatialtissuepy.topology.radial_filter

.. autofunction:: spatialtissuepy.topology.distance_to_type_filter

.. autofunction:: spatialtissuepy.topology.spatial_density_filter


Analysis Functions
------------------

.. autofunction:: spatialtissuepy.topology.find_hub_nodes

.. autofunction:: spatialtissuepy.topology.find_bridge_nodes

.. autofunction:: spatialtissuepy.topology.component_statistics

.. autofunction:: spatialtissuepy.topology.extract_mapper_features


Module Contents
---------------

.. automodule:: spatialtissuepy.topology
   :members:
   :undoc-members:
   :show-inheritance:
