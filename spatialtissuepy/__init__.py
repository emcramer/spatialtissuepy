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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the version declared in pyproject.toml, as
    # recorded in the installed distribution metadata.
    __version__ = _pkg_version("spatialtissuepy")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__author__ = "spatialtissuepy developers"

from spatialtissuepy.core.spatial_data import SpatialTissueData

__all__ = [
    "SpatialTissueData",
    "__version__",
]
