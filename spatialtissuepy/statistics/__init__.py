"""
Spatial statistics module for spatialtissuepy.

This module provides classical spatial point pattern statistics for analyzing
cell spatial organization in tissue samples.

Submodules
----------
spatial_stats : Ripley's K, L, H functions, G/F/J functions, pair correlation
colocalization : Co-localization analysis, neighborhood enrichment, Moran's I
hotspots : Getis-Ord Gi*, Local Moran's I (LISA), hotspot detection

Key Concepts
------------
- **CSR (Complete Spatial Randomness)**: Null hypothesis of random distribution
- **Clustering**: Cells closer together than expected under CSR
- **Dispersion/Regularity**: Cells more evenly spaced than expected
- **Co-localization**: Two cell types occurring together more than by chance

Example
-------
>>> from spatialtissuepy.statistics import (
...     ripleys_h,
...     colocalization_quotient,
...     detect_hotspots,
... )
>>>
>>> # Test for clustering using Ripley's H
>>> radii = np.linspace(0, 100, 50)
>>> H = ripleys_h(data.coordinates, radii)
>>> # H > 0 indicates clustering at that scale
>>>
>>> # Test co-localization between cell types
>>> clq = colocalization_quotient(data, 'T_cell', 'Tumor', radius=50)
>>> # CLQ > 1 indicates attraction, CLQ < 1 indicates repulsion
>>>
>>> # Find spatial hotspots
>>> result = detect_hotspots(data, values, radius=50)
>>> hotspot_cells = result['hotspot_idx']

References
----------
.. [1] Ripley, B. D. (1977). Modelling spatial patterns. JRSS-B.
.. [2] Baddeley, A. et al. (2015). Spatial Point Patterns. CRC Press.
.. [3] Getis, A., & Ord, J. K. (1992). Distance statistics. Geographical Analysis.
.. [4] Anselin, L. (1995). Local indicators of spatial association. Geographical Analysis.
"""

# Spatial statistics (Ripley's K, etc.)
# Co-localization analysis
from .colocalization import (
    colocalization_matrix,
    # Co-localization quotient
    colocalization_quotient,
    gearys_c,
    morans_i,
    neighborhood_enrichment_matrix,
    # Neighborhood enrichment
    neighborhood_enrichment_score,
    neighborhood_enrichment_test,
    # Spatial autocorrelation
    spatial_cross_correlation,
    # Spatial interaction
    spatial_interaction_matrix,
)

# Hotspot detection
from .hotspots import (
    cell_type_hotspots,
    # Detection functions
    detect_hotspots,
    getis_ord_gi,
    # Getis-Ord statistics
    getis_ord_gi_star,
    hotspot_regions,
    # Statistics and regions
    hotspot_statistics,
    hotspot_summary_by_type,
    # Local Moran's I
    local_morans_i,
    marker_hotspots,
)
from .spatial_stats import (
    cross_h,
    # Cross-type functions
    cross_k,
    cross_l,
    cross_type_statistics,
    # CSR envelope testing
    csr_envelope,
    f_function,
    # Nearest-neighbor functions
    g_function,
    g_function_cross,
    j_function,
    # Pair correlation
    pair_correlation_function,
    ripleys_h,
    # K-function and variants
    ripleys_k,
    ripleys_l,
    # High-level functions
    spatial_statistics,
)

__all__ = [
    # Spatial statistics
    'ripleys_k',
    'ripleys_l',
    'ripleys_h',
    'cross_k',
    'cross_l',
    'cross_h',
    'g_function',
    'g_function_cross',
    'f_function',
    'j_function',
    'pair_correlation_function',
    'csr_envelope',
    'spatial_statistics',
    'cross_type_statistics',
    # Co-localization
    'colocalization_quotient',
    'colocalization_matrix',
    'neighborhood_enrichment_score',
    'neighborhood_enrichment_test',
    'neighborhood_enrichment_matrix',
    'spatial_interaction_matrix',
    'spatial_cross_correlation',
    'morans_i',
    'gearys_c',
    # Hotspots
    'getis_ord_gi_star',
    'getis_ord_gi',
    'local_morans_i',
    'detect_hotspots',
    'cell_type_hotspots',
    'marker_hotspots',
    'hotspot_statistics',
    'hotspot_regions',
    'hotspot_summary_by_type',
]

# Import metrics to register them with summary module
from . import metrics
