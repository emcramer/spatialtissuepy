"""
Visualization utilities for Mapper algorithm results.

DEPRECATED: This module is maintained for backward compatibility.
Please use spatialtissuepy.viz.mapper instead.

Example migration:
    # Old
    from spatialtissuepy.topology.visualization import plot_mapper_graph
    
    # New (recommended)
    from spatialtissuepy.viz import plot_mapper_graph
"""

import warnings

# Re-export from viz module for backward compatibility
from spatialtissuepy.viz.mapper import (
    plot_mapper_graph,
    plot_mapper_spatial,
    plot_filter_distribution,
    plot_node_composition,
    plot_mapper_diagnostics,
    create_mapper_report,
)

__all__ = [
    'plot_mapper_graph',
    'plot_mapper_spatial', 
    'plot_filter_distribution',
    'plot_node_composition',
    'plot_mapper_diagnostics',
    'create_mapper_report',
]

# Issue deprecation warning on import
warnings.warn(
    "spatialtissuepy.topology.visualization is deprecated. "
    "Please use spatialtissuepy.viz.mapper instead.",
    DeprecationWarning,
    stacklevel=2
)
