"""
Tests for spatialtissuepy.summary module.

Tests metric registry, statistics panels, and multi-sample summarization
for creating ML-ready feature vectors from spatial tissue data.
"""

import pytest
import numpy as np
import pandas as pd

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.summary import (
    # Registry
    register_metric,
    get_metric,
    list_metrics,
    list_categories,
    get_registry,
    MetricInfo,
    register_custom_metric,
    unregister_custom_metric,
    # Panel
    StatisticsPanel,
    PanelMetric,
    load_panel,
    list_panels,
    # Summary
    SpatialSummary,
    MultiSampleSummary,
    compute_summary,
    compute_multi_summary,
)


# =============================================================================
# Metric Registry Tests
# =============================================================================

class TestMetricRegistry:
    """Tests for metric registration system."""
    
    def test_list_metrics(self):
        """Test listing all registered metrics."""
        metrics = list_metrics()
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        # Should have basic metrics
        assert 'cell_counts' in metrics
        assert 'cell_proportions' in metrics
    
    def test_list_categories(self):
        """Test listing metric categories."""
        categories = list_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        # Should have standard categories
        assert 'population' in categories
        assert 'spatial' in categories
        assert 'neighborhood' in categories
    
    def test_get_metric(self):
        """Test retrieving a specific metric."""
        metric_info = get_metric('cell_counts')
        
        assert isinstance(metric_info, MetricInfo)
        assert metric_info.name == 'cell_counts'
        assert metric_info.func is not None
        assert metric_info.category == 'population'
    
    def test_get_metric_not_found(self):
        """Test error when metric doesn't exist."""
        with pytest.raises(KeyError):
            get_metric('nonexistent_metric_xyz')
    
    def test_get_registry(self):
        """Test getting full registry."""
        registry = get_registry()
        assert registry is not None
        assert 'cell_counts' in registry
    
    def test_register_custom_metric(self):
        """Test registering a custom metric."""
        # Define custom metric
        def custom_metric(data):
            return {'custom_value': 42.0}
        
        # Register it
        register_custom_metric(
            name='test_custom_metric',
            fn=custom_metric,
            category='test'
        )
        
        # Should be retrievable
        metric = get_metric('test_custom_metric')
        assert metric.name == 'test_custom_metric'
        assert metric.category == 'test'
        
        # Clean up
        unregister_custom_metric('test_custom_metric')


# =============================================================================
# Statistics Panel Tests
# =============================================================================

class TestStatisticsPanel:
    """Tests for StatisticsPanel class."""
    
    def test_panel_initialization(self):
        """Test creating a statistics panel."""
        panel = StatisticsPanel(name='my_panel')
        
        assert isinstance(panel, StatisticsPanel)
        assert panel.name == 'my_panel'
        assert len(panel.metrics) == 0
    
    def test_panel_add_metric(self):
        """Test adding metrics to panel."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        assert len(panel.metrics) == 1
        assert panel.metrics[0].name == 'cell_counts'
    
    def test_panel_add_metric_with_params(self):
        """Test adding metric with parameters."""
        panel = StatisticsPanel()
        panel.add('ripleys_k', radii=[50, 100, 200])
        
        assert len(panel.metrics) == 1
        assert panel.metrics[0].params['radii'] == [50, 100, 200]
    
    def test_panel_add_custom_function(self):
        """Test adding custom inline function to panel."""
        panel = StatisticsPanel()
        panel.add_custom_function('my_metric', lambda d: {'val': 1.0})
        
        assert len(panel.metrics) == 1
        assert panel.metrics[0].name == 'my_metric'
        assert panel.metrics[0].is_inline
    
    def test_panel_remove_metric(self):
        """Test removing metrics from panel."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        panel.add('cell_proportions')
        assert len(panel.metrics) == 2
        
        panel.remove('cell_counts')
        assert len(panel.metrics) == 1
        assert panel.metrics[0].name == 'cell_proportions'
    
    def test_panel_clear(self):
        """Test clearing all metrics."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        panel.clear()
        assert len(panel.metrics) == 0


class TestPanelPresets:
    """Tests for predefined panel presets."""
    
    def test_load_panel_basic(self):
        """Test loading basic panel preset."""
        panel = load_panel('basic')
        assert isinstance(panel, StatisticsPanel)
        assert len(panel.metrics) > 0
        assert any(m.name == 'cell_counts' for m in panel.metrics)
    
    def test_list_panels(self):
        """Test listing available panel presets."""
        panels = list_panels()
        assert isinstance(panels, list)
        assert 'basic' in panels
        assert 'spatial' in panels


# =============================================================================
# Spatial Summary Tests
# =============================================================================

class TestSpatialSummary:
    """Tests for single-sample spatial summary."""
    
    def test_spatial_summary_basic(self, small_tissue):
        """Test computing summary for a sample."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        summary = SpatialSummary(small_tissue, panel)
        assert isinstance(summary, SpatialSummary)
        
        results = summary.to_dict()
        assert 'n_cells' in results
        assert results['n_cells'] == small_tissue.n_cells
    
    def test_spatial_summary_to_series(self, small_tissue):
        """Test converting summary to pandas Series."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        summary = SpatialSummary(small_tissue, panel)
        series = summary.to_series(name='sample1')
        
        assert isinstance(series, pd.Series)
        assert series.name == 'sample1'
        assert 'n_cells' in series.index
    
    def test_spatial_summary_to_array(self, small_tissue):
        """Test converting summary to numpy array."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        summary = SpatialSummary(small_tissue, panel)
        array = summary.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0


# =============================================================================
# Multi-Sample Summary Tests
# =============================================================================

class TestMultiSampleSummary:
    """Tests for multi-sample summary."""
    
    def test_multi_sample_summary_basic(self, multisample_cohort):
        """Test computing summaries for multiple samples."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        summary = MultiSampleSummary.from_multisample(multisample_cohort, panel)
        
        assert isinstance(summary, MultiSampleSummary)
        assert summary.n_samples == multisample_cohort.n_samples
    
    def test_multi_sample_summary_to_dataframe(self, multisample_cohort):
        """Test converting multi-sample summary to DataFrame."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        summary = MultiSampleSummary.from_multisample(multisample_cohort, panel)
        df = summary.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == multisample_cohort.n_samples
        assert 'n_cells' in df.columns
    
    def test_multi_sample_summary_parallel(self, multisample_cohort):
        """Test parallel computation of summaries."""
        panel = StatisticsPanel()
        panel.add('cell_counts')
        
        # Parallel if joblib available
        summary = MultiSampleSummary.from_multisample(
            multisample_cohort, panel, n_jobs=2
        )
        assert summary.n_samples == multisample_cohort.n_samples


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_summary(self, small_tissue):
        """Test compute_summary convenience function."""
        series = compute_summary(small_tissue, panel='basic')
        assert isinstance(series, pd.Series)
        assert 'n_cells' in series.index
    
    def test_compute_multi_summary(self, multisample_cohort):
        """Test compute_multi_summary convenience function."""
        df = compute_multi_summary(
            [s for _, s in multisample_cohort.iter_samples()], 
            panel='basic'
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == multisample_cohort.n_samples


# =============================================================================
# Integration Tests
# =============================================================================

class TestSummaryIntegration:
    """Integration tests for summary module."""
    
    def test_complete_summary_workflow(self, medium_tissue):
        """Test complete summary workflow."""
        panel = StatisticsPanel(name='analysis_panel')
        panel.add('cell_counts')
        panel.add('cell_proportions')
        panel.add('shannon_diversity')
        panel.add('mean_nearest_neighbor_distance')
        
        summary = SpatialSummary(medium_tissue, panel)
        df_row = summary.to_series()
        
        assert 'n_cells' in df_row
        assert 'shannon_diversity' in df_row
        assert 'mean_nnd' in df_row


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestSummaryEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_panel(self, small_tissue):
        """Test summary with empty panel."""
        panel = StatisticsPanel()
        summary = SpatialSummary(small_tissue, panel)
        assert len(summary.to_dict()) == 0
    
    def test_invalid_panel_name(self, small_tissue):
        """Test error with invalid panel name."""
        with pytest.raises(ValueError):
            compute_summary(small_tissue, panel='nonexistent_panel')

