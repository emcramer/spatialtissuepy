"""
Summary module for spatialtissuepy.

Provides tools for computing spatial statistics summaries that describe
tissue samples as feature vectors for downstream analysis.

Key Components
--------------
StatisticsPanel : Define custom panels of metrics
SpatialSummary : Compute summary for a single sample
MultiSampleSummary : Compute summaries for multiple samples

Predefined Panels
-----------------
'basic' : Essential population and spatial statistics
'spatial' : Comprehensive spatial statistics
'neighborhood' : Neighborhood composition and interactions
'comprehensive' : All available statistics

Example
-------
>>> from spatialtissuepy.summary import (
...     StatisticsPanel, SpatialSummary, MultiSampleSummary,
...     load_panel, compute_summary
... )
>>>
>>> # Quick summary with predefined panel
>>> summary = compute_summary(data, panel='basic')
>>>
>>> # Custom panel
>>> panel = StatisticsPanel(name='my_panel')
>>> panel.add('cell_counts')
>>> panel.add('cell_proportions')
>>> panel.add('ripleys_k', radii=[50, 100, 200])
>>> panel.add('colocalization_score', type_a='T_cell', type_b='Tumor')
>>>
>>> # Single sample
>>> summary = SpatialSummary(data, panel)
>>> vector = summary.to_array()
>>> series = summary.to_series()
>>>
>>> # Multiple samples
>>> multi = MultiSampleSummary(samples, panel, sample_ids=['A', 'B', 'C'])
>>> df = multi.to_dataframe()
>>> df.to_csv('spatial_features.csv')
"""

# Import metrics to register them
from . import population
from . import spatial
from . import neighborhood

# Public API
from .registry import (
    register_metric,
    get_metric,
    list_metrics,
    list_categories,
    get_registry,
    MetricInfo,
)

from .panel import (
    StatisticsPanel,
    PanelMetric,
    load_panel,
    list_panels,
)

from .summary import (
    SpatialSummary,
    MultiSampleSummary,
    compute_summary,
    compute_multi_summary,
)

__all__ = [
    # Registry
    'register_metric',
    'get_metric',
    'list_metrics',
    'list_categories',
    'get_registry',
    'MetricInfo',
    # Panel
    'StatisticsPanel',
    'PanelMetric',
    'load_panel',
    'list_panels',
    # Summary
    'SpatialSummary',
    'MultiSampleSummary',
    'compute_summary',
    'compute_multi_summary',
]
