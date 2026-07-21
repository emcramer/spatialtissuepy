"""
Core data structures for spatialtissuepy.

Classes
-------
SpatialTissueData : Main container for spatial cell data
Cell : Lightweight cell representation
"""

from spatialtissuepy.core.cell import Cell
from spatialtissuepy.core.spatial_data import SpatialTissueData

__all__ = ["SpatialTissueData", "Cell"]
