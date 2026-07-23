"""
Tumor microenvironment analysis.

Higher-level, domain-framed analyses of tissue microenvironment structure,
composed from the spatial, statistics, and PhysiCell-substrate machinery:

- **Niche identification** (:func:`identify_niches`): recurring local cell-type
  compositions, found by clustering neighborhood composition vectors.
- **Boundary detection** (:func:`detect_boundaries`): cells at the interface
  between cell types or niches.
- **Gradient analysis** (:func:`spatial_gradient`, :func:`substrate_gradient`,
  :func:`density_gradient`): spatial gradients of substrate concentration or
  cell density.
"""

from .boundaries import BoundaryResult, detect_boundaries
from .gradients import (
    GradientField,
    density_gradient,
    spatial_gradient,
    substrate_gradient,
)
from .niches import NicheResult, identify_niches

__all__ = [
    # Niches
    'identify_niches',
    'NicheResult',
    # Boundaries
    'detect_boundaries',
    'BoundaryResult',
    # Gradients
    'spatial_gradient',
    'substrate_gradient',
    'density_gradient',
    'GradientField',
]
