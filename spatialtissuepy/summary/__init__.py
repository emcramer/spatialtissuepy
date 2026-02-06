"""
Summary module for spatialtissuepy.

Provides tools for computing spatial statistics summaries that describe
tissue samples as feature vectors for downstream analysis.

Key Components
--------------
StatisticsPanel : Define custom panels of metrics
SpatialSummary : Compute summary for a single sample
MultiSampleSummary : Compute summaries for multiple samples

Custom Metrics
--------------
Users can define custom metrics using two approaches:

1. **Global registration** (recommended for reusable metrics):

   >>> @register_custom_metric(
   ...     name='cd8_treg_ratio',
   ...     description='Ratio of CD8+ T cells to Tregs'
   ... )
   ... def cd8_treg_ratio(data: SpatialTissueData) -> Dict[str, float]:
   ...     counts = data.cell_type_counts
   ...     cd8 = counts.get('CD8_T', 0)
   ...     treg = counts.get('Treg', 0)
   ...     return {'cd8_treg_ratio': cd8 / max(treg, 1)}
   >>>
   >>> # Now use in any panel
   >>> panel.add('cd8_treg_ratio')

2. **Inline panel functions** (for quick one-off metrics):

   >>> def my_ratio(data):
   ...     return {'ratio': data.cell_type_counts.get('A', 0) / data.n_cells}
   >>>
   >>> panel.add_custom_function('my_ratio', my_ratio)

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
...     load_panel, compute_summary, register_custom_metric
... )
>>>
>>> # Quick summary with predefined panel
>>> summary = compute_summary(data, panel='basic')
>>>
>>> # Custom panel with registered and inline metrics
>>> panel = StatisticsPanel(name='my_panel')
>>> panel.add('cell_counts')
>>> panel.add('cell_proportions')
>>> panel.add('ripleys_k', radii=[50, 100, 200])
>>>
>>> # Add custom inline function
>>> panel.add_custom_function(
...     'tumor_immune_ratio',
...     lambda data: {'ratio': data.cell_type_counts.get('Tumor', 0) /
...                           max(sum(data.cell_type_counts.get(t, 0)
...                               for t in ['CD8_T', 'CD4_T']), 1)}
... )
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

# Public API - Registry
from .registry import (
    # Core registry functions
    register_metric,
    get_metric,
    list_metrics,
    list_categories,
    get_registry,
    describe_metric,
    # Custom metric support
    register_custom_metric,
    unregister_custom_metric,
    list_custom_metrics,
    clear_custom_metrics,
    # Classes and exceptions
    MetricInfo,
    MetricValidationError,
    MetricRegistrationError,
)

# Public API - Panel
from .panel import (
    StatisticsPanel,
    PanelMetric,
    load_panel,
    list_panels,
)

# Public API - Summary
from .summary import (
    SpatialSummary,
    MultiSampleSummary,
    compute_summary,
    compute_multi_summary,
)

__all__ = [
    # Registry - core
    'register_metric',
    'get_metric',
    'list_metrics',
    'list_categories',
    'get_registry',
    'describe_metric',
    'MetricInfo',
    # Registry - custom metrics
    'register_custom_metric',
    'unregister_custom_metric',
    'list_custom_metrics',
    'clear_custom_metrics',
    'MetricValidationError',
    'MetricRegistrationError',
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
