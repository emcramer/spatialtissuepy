"""
spatialtissuepy: Spatial analysis tools for tissue biology
===========================================================

A Python package for analyzing spatial organization of cells in tissue samples,
with support for spatial statistics, neighborhood analysis, Spatial LDA, and
comprehensive visualization tools.

Main Classes
------------
SpatialTissueData : Core data container for spatial cell data

Modules
-------
core : Data structures and validation
spatial : Distance metrics, neighborhoods, clustering
statistics : Spatial statistics (Ripley's K, co-localization, hotspots)
lda : Spatial Latent Dirichlet Allocation for neighborhood discovery
microenvironment : Tumor microenvironment analysis
viz : Visualization suite
io : Input/output utilities
"""

__version__ = "0.1.0"
__author__ = "spatialtissuepy developers"

from spatialtissuepy.core.spatial_data import SpatialTissueData

__all__ = [
    "SpatialTissueData",
    "__version__",
]
