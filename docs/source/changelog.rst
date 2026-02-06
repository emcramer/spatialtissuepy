=========
Changelog
=========

All notable changes to spatialtissuepy will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.


[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation
- 12 tutorial notebooks covering all modules
- Full API reference with examples
- Helper functions ``is_alive``, ``is_dead``, ``get_phase_name``, ``CELL_CYCLE_PHASES`` exported from PhysiCell module

Changed
~~~~~~~
- Improved docstrings throughout all modules
- Tutorial 11 (PhysiCell Integration) updated to use real simulation data from ``examples/sample_data/example_physicell_sim``

Fixed
~~~~~
- PhysiCell file detection now supports both naming conventions (``output*_cells_physicell.mat`` and ``output*_cells.mat``)
- Various bug fixes and improvements


[0.1.0] - 2024-12-01
--------------------

Initial release of spatialtissuepy.

Added
~~~~~

**Core Module**

- ``SpatialTissueData`` class for spatial tissue data handling
- Support for 2D and 3D coordinates
- Multi-sample data support
- Marker expression data integration

**I/O Module**

- CSV and JSON file reading/writing
- Flexible column mapping

**Spatial Module**

- Pairwise distance calculations
- Nearest neighbor queries with KD-trees
- Neighborhood composition analysis
- Spatial density estimation
- Boundary cell detection

**Statistics Module**

- Ripley's K, L, H functions with edge correction
- Colocalization quotient (CLQ)
- Getis-Ord Gi* hotspot detection
- Local Moran's I (LISA)
- Global Moran's I
- Clark-Evans index

**Network Module**

- Cell graph construction (proximity, kNN, Delaunay)
- Centrality measures (degree, betweenness, closeness)
- Type assortativity
- Attribute mixing matrix

**LDA Module**

- Spatial LDA for neighborhood-based topic modeling
- Topic coherence and diversity metrics
- Multi-sample fitting

**Topology Module**

- Mapper algorithm implementation
- Spatial filter functions
- Hub and bridge node detection
- Feature extraction from Mapper graphs

**Summary Module**

- StatisticsPanel for metric collection
- Single and multi-sample summaries
- Metric registry system
- Predefined panels (basic, spatial, comprehensive)

**Synthetic Module**

- PhysiCell MultiCellDS XML parsing
- Simulation and experiment classes
- Temporal analysis utilities

**Visualization Module**

- Publication-style configuration
- Spatial scatter plots
- Density maps
- Statistics plots (Ripley's, colocalization heatmaps)
- Network visualizations
- LDA and Mapper visualizations
