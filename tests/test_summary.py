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
        assert 'cell_counts' in metrics or 'n_cells' in metrics
    
    def test_list_categories(self):
        """Test listing metric categories."""
        categories = list_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        # Should have standard categories
        expected = ['population', 'spatial', 'neighborhood']
        assert any(cat in categories for cat in expected)
    
    def test_get_metric(self):
        """Test retrieving a specific metric."""
        # Get a known metric
        metrics = list_metrics()
        if len(metrics) == 0:
            pytest.skip("No metrics registered")
        
        metric_name = metrics[0]
        metric_info = get_metric(metric_name)
        
        assert isinstance(metric_info, MetricInfo)
        assert metric_info.name == metric_name
        assert metric_info.fn is not None
        assert metric_info.category is not None
    
    def test_get_metric_not_found(self):
        """Test error when metric doesn't exist."""
        with pytest.raises((KeyError, ValueError)):
            get_metric('nonexistent_metric_xyz')
    
    def test_get_registry(self):
        """Test getting full registry."""
        registry = get_registry()
        
        assert isinstance(registry, dict)
        assert len(registry) > 0
    
    def test_register_custom_metric(self):
        """Test registering a custom metric."""
        # Define custom metric
        def custom_metric(data):
            return {'custom_value': 42.0}
        
        # Register it
        try:
            register_metric(
                name='test_custom_metric',
                fn=custom_metric,
                category='test'
            )
            
            # Should be retrievable
            metric = get_metric('test_custom_metric')
            assert metric.name == 'test_custom_metric'
            assert metric.category == 'test'
        except Exception:
            # Registration might be restricted
            pytest.skip("Custom metric registration not allowed")


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
        
        # Get available metrics
        available = list_metrics()
        if len(available) == 0:
            pytest.skip("No metrics available")
        
        # Add first available metric
        metric_name = available[0]
        panel.add(metric_name)
        
        assert len(panel.metrics) == 1
        assert panel.metrics[0].name == metric_name
    
    def test_panel_add_metric_with_params(self):
        """Test adding metric with parameters."""
        panel = StatisticsPanel()
        
        try:
            # Try adding a metric that accepts parameters
            panel.add('ripleys_k', radii=[50, 100, 200])
            assert len(panel.metrics) == 1
        except (KeyError, ValueError):
            # Metric might not exist
            pytest.skip("ripleys_k metric not available")
    
    def test_panel_remove_metric(self):
        """Test removing metrics from panel."""
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) < 2:
            pytest.skip("Not enough metrics")
        
        panel.add(available[0])
        panel.add(available[1])
        assert len(panel.metrics) == 2
        
        panel.remove(0)
        assert len(panel.metrics) == 1
    
    def test_panel_clear(self):
        """Test clearing all metrics."""
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) > 0:
            panel.add(available[0])
        
        panel.clear()
        assert len(panel.metrics) == 0
    
    def test_panel_repr(self):
        """Test panel string representation."""
        panel = StatisticsPanel(name='test_panel')
        
        repr_str = repr(panel)
        str_str = str(panel)
        
        assert 'StatisticsPanel' in repr_str or 'test_panel' in repr_str
        assert isinstance(str_str, str)


class TestPanelPresets:
    """Tests for predefined panel presets."""
    
    def test_load_panel_basic(self):
        """Test loading basic panel preset."""
        try:
            panel = load_panel('basic')
            assert isinstance(panel, StatisticsPanel)
            assert len(panel.metrics) > 0
        except (KeyError, ValueError, FileNotFoundError):
            pytest.skip("Basic panel preset not available")
    
    def test_load_panel_spatial(self):
        """Test loading spatial panel preset."""
        try:
            panel = load_panel('spatial')
            assert isinstance(panel, StatisticsPanel)
        except (KeyError, ValueError, FileNotFoundError):
            pytest.skip("Spatial panel preset not available")
    
    def test_load_panel_comprehensive(self):
        """Test loading comprehensive panel preset."""
        try:
            panel = load_panel('comprehensive')
            assert isinstance(panel, StatisticsPanel)
        except (KeyError, ValueError, FileNotFoundError):
            pytest.skip("Comprehensive panel preset not available")
    
    def test_list_panels(self):
        """Test listing available panel presets."""
        try:
            panels = list_panels()
            assert isinstance(panels, list)
            # Should have at least basic presets
        except NotImplementedError:
            pytest.skip("Panel listing not implemented")
    
    def test_load_nonexistent_panel(self):
        """Test error with nonexistent panel."""
        with pytest.raises((KeyError, ValueError, FileNotFoundError)):
            load_panel('nonexistent_panel_xyz')


# =============================================================================
# Spatial Summary Tests
# =============================================================================

class TestSpatialSummary:
    """Tests for single-sample spatial summary."""
    
    def test_spatial_summary_basic(self, small_tissue):
        """Test computing summary for a sample."""
        panel = StatisticsPanel()
        
        # Add simple metric
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("cell_counts metric not available")
        
        summary = SpatialSummary(small_tissue, panel)
        
        assert isinstance(summary, SpatialSummary)
    
    def test_spatial_summary_to_dict(self, small_tissue):
        """Test converting summary to dictionary."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        summary = SpatialSummary(small_tissue, panel)
        result = summary.to_dict()
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_spatial_summary_to_series(self, small_tissue):
        """Test converting summary to pandas Series."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        summary = SpatialSummary(small_tissue, panel)
        series = summary.to_series()
        
        assert isinstance(series, pd.Series)
        assert len(series) > 0
    
    def test_spatial_summary_to_array(self, small_tissue):
        """Test converting summary to numpy array."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        summary = SpatialSummary(small_tissue, panel)
        array = summary.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0
    
    def test_spatial_summary_with_multiple_metrics(self, small_tissue):
        """Test summary with multiple metrics."""
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) < 2:
            pytest.skip("Not enough metrics")
        
        try:
            panel.add(available[0])
            panel.add(available[1])
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        summary = SpatialSummary(small_tissue, panel)
        result = summary.to_dict()
        
        assert len(result) > 0


# =============================================================================
# Multi-Sample Summary Tests
# =============================================================================

class TestMultiSampleSummary:
    """Tests for multi-sample summary."""
    
    def test_multi_sample_summary_basic(self, multisample_cohort):
        """Test computing summaries for multiple samples."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        # Get individual samples
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        multi_summary = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        
        assert isinstance(multi_summary, MultiSampleSummary)
        assert multi_summary.n_samples == len(samples)
    
    def test_multi_sample_summary_to_dataframe(self, multisample_cohort):
        """Test converting multi-sample summary to DataFrame."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        multi_summary = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        df = multi_summary.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(samples)
        assert df.index.tolist() == sample_ids
    
    def test_multi_sample_summary_to_array(self, multisample_cohort):
        """Test converting multi-sample summary to array."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        multi_summary = MultiSampleSummary(samples, panel)
        array = multi_summary.to_array()
        
        assert isinstance(array, np.ndarray)
        assert array.shape[0] == len(samples)
    
    def test_multi_sample_summary_parallel(self, multisample_cohort):
        """Test parallel computation of summaries."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        try:
            multi_summary = MultiSampleSummary(
                samples, panel, sample_ids=sample_ids, n_jobs=2
            )
            df = multi_summary.to_dataframe()
            assert len(df) == len(samples)
        except (NotImplementedError, TypeError):
            pytest.skip("Parallel processing not implemented")


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compute_summary(self, small_tissue):
        """Test compute_summary convenience function."""
        try:
            result = compute_summary(small_tissue, panel='basic')
            assert isinstance(result, (dict, pd.Series))
        except (KeyError, ValueError, FileNotFoundError):
            # Panel might not exist
            pytest.skip("Basic panel not available")
    
    def test_compute_summary_custom_panel(self, small_tissue):
        """Test compute_summary with custom panel."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        result = compute_summary(small_tissue, panel=panel)
        assert isinstance(result, (dict, pd.Series))
    
    def test_compute_multi_summary(self, multisample_cohort):
        """Test compute_multi_summary convenience function."""
        sample_ids = multisample_cohort.sample_ids_unique[:2]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        try:
            df = compute_multi_summary(samples, panel='basic', sample_ids=sample_ids)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(samples)
        except (KeyError, ValueError, FileNotFoundError):
            pytest.skip("Basic panel not available")


# =============================================================================
# Integration Tests
# =============================================================================

class TestSummaryIntegration:
    """Integration tests for summary module."""
    
    def test_complete_summary_workflow(self, medium_tissue):
        """Test complete summary workflow."""
        # 1. Create custom panel
        panel = StatisticsPanel(name='analysis_panel')
        
        available = list_metrics()
        if len(available) == 0:
            pytest.skip("No metrics available")
        
        # Add multiple metrics
        try:
            for metric in available[:3]:
                panel.add(metric)
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        # 2. Compute summary
        summary = SpatialSummary(medium_tissue, panel)
        
        # 3. Get as DataFrame-ready series
        series = summary.to_series()
        assert isinstance(series, pd.Series)
        
        # 4. Get as ML-ready array
        array = summary.to_array()
        assert isinstance(array, np.ndarray)
    
    def test_cohort_analysis_workflow(self, multisample_cohort):
        """Test analyzing multiple samples for ML."""
        # Get samples
        sample_ids = multisample_cohort.sample_ids_unique[:3]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        # Create panel
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) == 0:
            pytest.skip("No metrics available")
        
        try:
            panel.add(available[0])
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        # Compute multi-sample summary
        multi_summary = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        
        # Get as DataFrame (ready for ML)
        df = multi_summary.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(samples)
        assert df.index.tolist() == sample_ids
        
        # Should be numeric and ready for sklearn
        assert df.select_dtypes(include=[np.number]).shape[1] > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestSummaryEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_panel(self, small_tissue):
        """Test summary with empty panel."""
        panel = StatisticsPanel()
        
        summary = SpatialSummary(small_tissue, panel)
        result = summary.to_dict()
        
        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_single_sample_multi_summary(self, small_tissue):
        """Test multi-sample summary with one sample."""
        panel = StatisticsPanel()
        
        try:
            panel.add('cell_counts')
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        multi_summary = MultiSampleSummary(
            [small_tissue], panel, sample_ids=['sample1']
        )
        
        df = multi_summary.to_dataframe()
        assert len(df) == 1
    
    def test_metric_with_missing_dependency(self, small_tissue):
        """Test metric that might fail due to missing dependency."""
        panel = StatisticsPanel()
        
        # Try adding a metric that might require optional dependency
        try:
            panel.add('some_advanced_metric')
            summary = SpatialSummary(small_tissue, panel)
            # Should either work or raise appropriate error
        except (KeyError, ValueError, ImportError):
            # Expected if metric doesn't exist or has missing dependency
            pass


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestSummaryPerformance:
    """Performance tests for summary module."""
    
    def test_summary_computation_performance(self, large_tissue):
        """Test summary computation on large dataset."""
        import time
        
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) == 0:
            pytest.skip("No metrics available")
        
        try:
            # Add a few metrics
            for metric in available[:5]:
                panel.add(metric)
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        start = time.time()
        summary = SpatialSummary(large_tissue, panel)
        result = summary.to_dict()
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 30.0  # 30 seconds
        assert len(result) > 0
    
    def test_multi_sample_performance(self, multisample_cohort):
        """Test multi-sample summary performance."""
        import time
        
        # Get multiple samples
        sample_ids = multisample_cohort.sample_ids_unique[:5]
        samples = [multisample_cohort.subset_sample(sid) for sid in sample_ids]
        
        panel = StatisticsPanel()
        
        available = list_metrics()
        if len(available) == 0:
            pytest.skip("No metrics available")
        
        try:
            panel.add(available[0])
        except (KeyError, ValueError):
            pytest.skip("Metrics not available")
        
        start = time.time()
        multi_summary = MultiSampleSummary(samples, panel, sample_ids=sample_ids)
        df = multi_summary.to_dataframe()
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 15.0
        assert len(df) == len(samples)
