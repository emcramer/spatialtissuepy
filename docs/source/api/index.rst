=============
API Reference
=============

This section contains the complete API documentation for spatialtissuepy,
automatically generated from the source code docstrings.

.. toctree::
   :maxdepth: 2

   core
   io
   spatial
   statistics
   network
   lda
   topology
   summary
   synthetic
   viz


Quick Reference
---------------

**Core Data Structures**

.. autosummary::
   :nosignatures:

   spatialtissuepy.SpatialTissueData

**I/O Functions**

.. autosummary::
   :nosignatures:

   spatialtissuepy.io.read_csv
   spatialtissuepy.io.write_csv
   spatialtissuepy.io.read_json
   spatialtissuepy.io.write_json

**Spatial Functions**

.. autosummary::
   :nosignatures:

   spatialtissuepy.spatial.pairwise_distances
   spatialtissuepy.spatial.nearest_neighbors
   spatialtissuepy.spatial.compute_neighborhoods
   spatialtissuepy.spatial.neighborhood_composition

**Statistics**

.. autosummary::
   :nosignatures:

   spatialtissuepy.statistics.ripleys_k
   spatialtissuepy.statistics.ripleys_h
   spatialtissuepy.statistics.colocalization_quotient
   spatialtissuepy.statistics.morans_i

**Network**

.. autosummary::
   :nosignatures:

   spatialtissuepy.network.CellGraph

**LDA**

.. autosummary::
   :nosignatures:

   spatialtissuepy.lda.SpatialLDA
   spatialtissuepy.lda.fit_spatial_lda

**Topology**

.. autosummary::
   :nosignatures:

   spatialtissuepy.topology.SpatialMapper
   spatialtissuepy.topology.spatial_mapper

**Summary**

.. autosummary::
   :nosignatures:

   spatialtissuepy.summary.StatisticsPanel
   spatialtissuepy.summary.SpatialSummary
   spatialtissuepy.summary.MultiSampleSummary
