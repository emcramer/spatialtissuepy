"""
Tests for custom metric registration in spatialtissuepy.summary module.

Tests the user-facing API for defining and registering custom metrics,
including both decorator-based global registration and inline panel functions.
"""

import pytest
import numpy as np
from typing import Dict

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.summary import (
    # Custom metric API
    register_custom_metric,
    unregister_custom_metric,
    list_custom_metrics,
    clear_custom_metrics,
    get_metric,
    list_metrics,
    describe_metric,
    # Exceptions
    MetricValidationError,
    MetricRegistrationError,
    # Panel
    StatisticsPanel,
    SpatialSummary,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_tissue():
    """Small tissue with 100 cells and known cell types."""
    np.random.seed(42)
    coords = np.random.rand(100, 2) * 500
    # Create predictable cell type distribution
    types = np.array(
        ['CD8_T'] * 30 +
        ['Treg'] * 20 +
        ['Tumor'] * 40 +
        ['Stroma'] * 10
    )
    return SpatialTissueData(coords, types)


@pytest.fixture
def cleanup_custom_metrics():
    """Fixture to clean up custom metrics after each test."""
    yield
    # Cleanup after test
    clear_custom_metrics()


@pytest.fixture(autouse=True)
def isolate_custom_metrics(cleanup_custom_metrics):
    """Automatically clean up custom metrics before and after each test."""
    clear_custom_metrics()
    yield


# =============================================================================
# Test: Decorator-Based Registration
# =============================================================================

class TestDecoratorRegistration:
    """Tests for decorator-based custom metric registration."""

    def test_register_simple_metric(self, small_tissue):
        """Test registering a simple custom metric with decorator."""

        @register_custom_metric(
            name='test_simple_metric',
            description='A simple test metric'
        )
        def test_simple_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'simple_value': float(data.n_cells)}

        # Verify registration
        assert 'test_simple_metric' in list_custom_metrics()
        assert 'test_simple_metric' in list_metrics()

        # Verify it works
        metric_info = get_metric('test_simple_metric')
        result = metric_info(small_tissue)
        assert result == {'simple_value': 100.0}

    def test_register_cell_type_ratio(self, small_tissue):
        """Test registering a cell type ratio metric (Example 1 from requirements)."""

        @register_custom_metric(
            name='cd8_treg_ratio',
            category='interaction',
            description='Ratio of CD8+ T cells to regulatory T cells'
        )
        def cd8_treg_ratio(data: SpatialTissueData) -> Dict[str, float]:
            counts = data.cell_type_counts
            cd8 = counts.get('CD8_T', 0)
            treg = counts.get('Treg', 0)
            ratio = cd8 / treg if treg > 0 else float('inf')
            return {'cd8_treg_ratio': ratio}

        # Verify registration
        assert 'cd8_treg_ratio' in list_custom_metrics()

        # Test computation
        result = get_metric('cd8_treg_ratio')(small_tissue)
        expected_ratio = 30 / 20  # 1.5
        assert result['cd8_treg_ratio'] == pytest.approx(expected_ratio)

    def test_register_metric_with_dependencies(self, small_tissue):
        """Test registering a metric with dependencies (Example 2: squared distance)."""

        @register_custom_metric(
            name='mean_squared_distance',
            category='spatial',
            description='Mean squared Euclidean distance between cells',
            required_dependencies=['scipy']
        )
        def mean_squared_distance(data: SpatialTissueData) -> Dict[str, float]:
            from scipy.spatial.distance import pdist
            coords = data.coordinates
            if len(coords) < 2:
                return {'mean_sq_dist': 0.0, 'max_sq_dist': 0.0}
            sq_dists = pdist(coords, metric='sqeuclidean')
            return {
                'mean_sq_dist': float(sq_dists.mean()),
                'max_sq_dist': float(sq_dists.max())
            }

        # Verify registration
        assert 'mean_squared_distance' in list_custom_metrics()

        # Test computation
        result = get_metric('mean_squared_distance')(small_tissue)
        assert 'mean_sq_dist' in result
        assert 'max_sq_dist' in result
        assert result['mean_sq_dist'] > 0
        assert result['max_sq_dist'] > result['mean_sq_dist']

    def test_register_metric_with_parameters(self, small_tissue):
        """Test registering a metric that accepts parameters."""

        @register_custom_metric(
            name='cell_type_fraction',
            category='population',
            description='Fraction of a specific cell type',
            parameters={'cell_type': str}
        )
        def cell_type_fraction(
            data: SpatialTissueData,
            cell_type: str = 'Tumor'
        ) -> Dict[str, float]:
            counts = data.cell_type_counts
            fraction = counts.get(cell_type, 0) / data.n_cells
            return {f'{cell_type}_fraction': fraction}

        # Test with default parameter
        result = get_metric('cell_type_fraction')(small_tissue)
        assert result['Tumor_fraction'] == pytest.approx(0.4)

        # Test with custom parameter
        result = get_metric('cell_type_fraction')(small_tissue, cell_type='CD8_T')
        assert result['CD8_T_fraction'] == pytest.approx(0.3)

    def test_registration_category(self, small_tissue):
        """Test that custom metrics go to the correct category."""

        @register_custom_metric(
            name='test_interaction_metric',
            category='interaction'
        )
        def test_interaction_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'interaction': 1.0}

        # Check category
        metrics_in_interaction = list_metrics(category='interaction')
        assert 'test_interaction_metric' in metrics_in_interaction

        # Check metric info
        info = get_metric('test_interaction_metric')
        assert info.category == 'interaction'
        assert info.is_custom is True

    def test_describe_custom_metric(self, small_tissue):
        """Test describing a custom metric."""

        @register_custom_metric(
            name='described_metric',
            category='test',
            description='A well-documented metric'
        )
        def described_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 42.0}

        description = describe_metric('described_metric')
        assert 'described_metric' in description
        assert 'test' in description
        assert 'Custom: True' in description
        assert 'A well-documented metric' in description


# =============================================================================
# Test: Direct Registration
# =============================================================================

class TestDirectRegistration:
    """Tests for direct function registration (non-decorator)."""

    def test_register_function_directly(self, small_tissue):
        """Test registering a function directly without decorator."""

        def tumor_immune_ratio(data: SpatialTissueData) -> Dict[str, float]:
            counts = data.cell_type_counts
            tumor = counts.get('Tumor', 0)
            immune = counts.get('CD8_T', 0) + counts.get('Treg', 0)
            return {'tumor_immune_ratio': tumor / max(immune, 1)}

        register_custom_metric(
            name='tumor_immune_ratio',
            fn=tumor_immune_ratio,
            description='Ratio of tumor cells to immune cells'
        )

        # Verify registration
        assert 'tumor_immune_ratio' in list_custom_metrics()

        # Test computation
        result = get_metric('tumor_immune_ratio')(small_tissue)
        expected = 40 / 50  # 0.8
        assert result['tumor_immune_ratio'] == pytest.approx(expected)

    def test_register_lambda_function(self, small_tissue):
        """Test registering a lambda function."""

        register_custom_metric(
            name='lambda_metric',
            fn=lambda data: {'cell_count': float(data.n_cells)},
            description='Simple lambda metric'
        )

        result = get_metric('lambda_metric')(small_tissue)
        assert result['cell_count'] == 100.0


# =============================================================================
# Test: Panel Integration
# =============================================================================

class TestPanelIntegration:
    """Tests for using custom metrics in StatisticsPanel."""

    def test_add_registered_custom_metric_to_panel(self, small_tissue):
        """Test adding a globally registered custom metric to a panel."""

        @register_custom_metric(name='panel_test_metric')
        def panel_test_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'panel_value': 123.0}

        # Add to panel by name
        panel = StatisticsPanel()
        panel.add('panel_test_metric')

        # Compute
        result = panel.compute(small_tissue)
        assert result['panel_value'] == 123.0

    def test_add_custom_function_inline(self, small_tissue):
        """Test adding a custom function directly to panel (inline)."""

        def inline_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'inline_value': float(data.n_cells * 2)}

        panel = StatisticsPanel()
        panel.add_custom_function('my_inline', inline_metric)

        result = panel.compute(small_tissue)
        assert result['inline_value'] == 200.0

    def test_inline_function_not_globally_registered(self, small_tissue):
        """Test that inline functions are NOT globally registered."""

        def local_metric(data):
            return {'local': 1.0}

        panel = StatisticsPanel()
        panel.add_custom_function('local_only', local_metric)

        # Should NOT be in global registry
        assert 'local_only' not in list_metrics()
        assert 'local_only' not in list_custom_metrics()

        # But should work in the panel
        result = panel.compute(small_tissue)
        assert result['local'] == 1.0

    def test_mixed_panel_registered_and_inline(self, small_tissue):
        """Test panel with both registered and inline metrics."""

        @register_custom_metric(name='registered_metric')
        def registered_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'registered': 1.0}

        def inline_metric(data):
            return {'inline': 2.0}

        panel = StatisticsPanel()
        panel.add('cell_counts')  # Built-in
        panel.add('registered_metric')  # Custom registered
        panel.add_custom_function('inline', inline_metric)  # Inline

        result = panel.compute(small_tissue)

        assert 'n_cells' in result  # From cell_counts
        assert result['registered'] == 1.0
        assert result['inline'] == 2.0

    def test_inline_function_with_parameters(self, small_tissue):
        """Test inline function with baked-in parameters."""

        def parameterized_metric(data, multiplier=1):
            return {'result': float(data.n_cells * multiplier)}

        panel = StatisticsPanel()
        panel.add_custom_function(
            'doubled',
            parameterized_metric,
            multiplier=2
        )
        panel.add_custom_function(
            'tripled',
            parameterized_metric,
            multiplier=3
        )

        result = panel.compute(small_tissue)
        assert result['result'] == 200.0  # First one wins in dict update

    def test_panel_has_inline_metrics_property(self, small_tissue):
        """Test panel.has_inline_metrics property."""

        panel1 = StatisticsPanel()
        panel1.add('cell_counts')
        assert panel1.has_inline_metrics is False

        panel2 = StatisticsPanel()
        panel2.add_custom_function('inline', lambda d: {'v': 1.0})
        assert panel2.has_inline_metrics is True

    def test_panel_is_serializable_property(self, small_tissue):
        """Test panel.is_serializable property."""

        panel1 = StatisticsPanel()
        panel1.add('cell_counts')
        assert panel1.is_serializable is True

        panel2 = StatisticsPanel()
        panel2.add_custom_function('inline', lambda d: {'v': 1.0})
        assert panel2.is_serializable is False


# =============================================================================
# Test: Validation
# =============================================================================

class TestValidation:
    """Tests for metric function validation."""

    def test_validation_non_callable_raises(self):
        """Test that registering non-callable raises error."""
        with pytest.raises(MetricValidationError):
            register_custom_metric(name='bad', fn="not a function")

    def test_validation_no_parameters_raises(self):
        """Test that function with no parameters raises error."""

        def no_params() -> Dict[str, float]:
            return {'value': 1.0}

        with pytest.raises(MetricValidationError):
            register_custom_metric(name='no_params', fn=no_params)

    def test_validation_bad_return_type(self, small_tissue):
        """Test that non-dict return raises error at compute time."""

        @register_custom_metric(name='bad_return')
        def bad_return(data: SpatialTissueData):
            return [1, 2, 3]  # Wrong type!

        metric = get_metric('bad_return')
        with pytest.raises(MetricValidationError):
            metric(small_tissue)

    def test_validation_non_string_keys(self, small_tissue):
        """Test that non-string dict keys raise error."""

        @register_custom_metric(name='bad_keys')
        def bad_keys(data: SpatialTissueData) -> Dict[str, float]:
            return {1: 1.0, 2: 2.0}  # Int keys!

        metric = get_metric('bad_keys')
        with pytest.raises(MetricValidationError):
            metric(small_tissue)

    def test_validation_non_numeric_values(self, small_tissue):
        """Test that non-numeric values raise error."""

        @register_custom_metric(name='bad_values')
        def bad_values(data: SpatialTissueData) -> Dict[str, float]:
            return {'key': 'not a number'}

        metric = get_metric('bad_values')
        with pytest.raises(MetricValidationError):
            metric(small_tissue)

    def test_validation_missing_dependency(self):
        """Test that missing dependency raises error at registration."""
        with pytest.raises(MetricRegistrationError) as exc_info:
            @register_custom_metric(
                name='needs_nonexistent',
                required_dependencies=['nonexistent_package_xyz']
            )
            def needs_nonexistent(data):
                return {'value': 1.0}

        assert 'nonexistent_package_xyz' in str(exc_info.value)


# =============================================================================
# Test: Registration Errors
# =============================================================================

class TestRegistrationErrors:
    """Tests for registration error handling."""

    def test_duplicate_name_raises(self):
        """Test that duplicate registration raises error."""

        @register_custom_metric(name='duplicate_test')
        def first_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 1.0}

        with pytest.raises(MetricRegistrationError):
            @register_custom_metric(name='duplicate_test')
            def second_metric(data: SpatialTissueData) -> Dict[str, float]:
                return {'value': 2.0}

    def test_overwrite_existing_with_flag(self):
        """Test overwriting custom metric with overwrite=True."""

        @register_custom_metric(name='overwrite_test')
        def first_version(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 1.0}

        @register_custom_metric(name='overwrite_test', overwrite=True)
        def second_version(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 2.0}

        # Should have second version
        np.random.seed(42)
        coords = np.random.rand(10, 2) * 100
        types = np.array(['A'] * 10)
        data = SpatialTissueData(coords, types)

        result = get_metric('overwrite_test')(data)
        assert result['value'] == 2.0

    def test_cannot_overwrite_builtin(self):
        """Test that built-in metrics cannot be overwritten."""
        with pytest.raises(MetricRegistrationError) as exc_info:
            @register_custom_metric(name='cell_counts')
            def fake_cell_counts(data: SpatialTissueData) -> Dict[str, float]:
                return {'fake': 1.0}

        assert 'built-in' in str(exc_info.value).lower()


# =============================================================================
# Test: Unregistration
# =============================================================================

class TestUnregistration:
    """Tests for unregistering custom metrics."""

    def test_unregister_custom_metric(self):
        """Test unregistering a custom metric."""

        @register_custom_metric(name='to_unregister')
        def to_unregister(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 1.0}

        assert 'to_unregister' in list_custom_metrics()

        result = unregister_custom_metric('to_unregister')
        assert result is True
        assert 'to_unregister' not in list_custom_metrics()
        assert 'to_unregister' not in list_metrics()

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering nonexistent metric returns False."""
        result = unregister_custom_metric('nonexistent_xyz')
        assert result is False

    def test_cannot_unregister_builtin(self):
        """Test that built-in metrics cannot be unregistered."""
        with pytest.raises(MetricRegistrationError):
            unregister_custom_metric('cell_counts')

    def test_clear_custom_metrics(self):
        """Test clearing all custom metrics."""

        @register_custom_metric(name='clear_test_1')
        def clear_test_1(data: SpatialTissueData) -> Dict[str, float]:
            return {'v': 1.0}

        @register_custom_metric(name='clear_test_2')
        def clear_test_2(data: SpatialTissueData) -> Dict[str, float]:
            return {'v': 2.0}

        assert len(list_custom_metrics()) >= 2

        count = clear_custom_metrics()
        assert count >= 2
        assert len(list_custom_metrics()) == 0

        # Built-in should still exist
        assert 'cell_counts' in list_metrics()


# =============================================================================
# Test: Panel Serialization
# =============================================================================

class TestPanelSerialization:
    """Tests for panel serialization with custom metrics."""

    def test_serialize_panel_with_registered_custom_metric(self, tmp_path):
        """Test serializing panel with globally registered custom metric."""

        @register_custom_metric(name='serializable_metric')
        def serializable_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'value': 1.0}

        panel = StatisticsPanel()
        panel.add('cell_counts')
        panel.add('serializable_metric')

        # Should serialize successfully
        filepath = tmp_path / 'panel.json'
        panel.to_json(str(filepath))

        # Should load successfully
        loaded = StatisticsPanel.from_json(str(filepath))
        assert len(loaded.metrics) == 2

    def test_serialize_panel_with_inline_raises(self, tmp_path):
        """Test that serializing panel with inline function raises error."""

        panel = StatisticsPanel()
        panel.add('cell_counts')
        panel.add_custom_function('inline', lambda d: {'v': 1.0})

        filepath = tmp_path / 'panel.json'
        with pytest.raises(ValueError) as exc_info:
            panel.to_json(str(filepath))

        assert 'inline' in str(exc_info.value)


# =============================================================================
# Test: SpatialSummary Integration
# =============================================================================

class TestSpatialSummaryIntegration:
    """Tests for custom metrics with SpatialSummary."""

    def test_spatial_summary_with_custom_metrics(self, small_tissue):
        """Test full workflow with SpatialSummary."""

        @register_custom_metric(name='summary_test_metric')
        def summary_test_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'custom_summary': 42.0}

        panel = StatisticsPanel()
        panel.add('cell_counts')
        panel.add('summary_test_metric')

        summary = SpatialSummary(small_tissue, panel)

        # to_dict
        result_dict = summary.to_dict()
        assert 'custom_summary' in result_dict
        assert result_dict['custom_summary'] == 42.0

        # to_series
        series = summary.to_series()
        assert 'custom_summary' in series.index

        # to_array
        array = summary.to_array()
        assert len(array) > 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_tissue(self):
        """Test custom metric on empty tissue."""
        coords = np.array([]).reshape(0, 2)
        types = np.array([])
        empty_data = SpatialTissueData(coords, types)

        @register_custom_metric(name='empty_safe')
        def empty_safe(data: SpatialTissueData) -> Dict[str, float]:
            if data.n_cells == 0:
                return {'value': 0.0}
            return {'value': float(data.n_cells)}

        result = get_metric('empty_safe')(empty_data)
        assert result['value'] == 0.0

    def test_metric_returns_nan(self, small_tissue):
        """Test metric that returns NaN values."""

        @register_custom_metric(name='nan_metric')
        def nan_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'nan_value': float('nan')}

        result = get_metric('nan_metric')(small_tissue)
        assert np.isnan(result['nan_value'])

    def test_metric_returns_inf(self, small_tissue):
        """Test metric that returns infinity."""

        @register_custom_metric(name='inf_metric')
        def inf_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'inf_value': float('inf')}

        result = get_metric('inf_metric')(small_tissue)
        assert result['inf_value'] == float('inf')

    def test_metric_with_many_output_columns(self, small_tissue):
        """Test metric that returns many columns."""

        @register_custom_metric(name='many_columns', dynamic_columns=True)
        def many_columns(data: SpatialTissueData) -> Dict[str, float]:
            return {f'col_{i}': float(i) for i in range(100)}

        result = get_metric('many_columns')(small_tissue)
        assert len(result) == 100
        assert result['col_0'] == 0.0
        assert result['col_99'] == 99.0

    def test_inline_function_error_handling(self, small_tissue):
        """Test that inline function errors are caught in panel compute."""

        def error_metric(data):
            raise ValueError("Intentional error")

        panel = StatisticsPanel()
        panel.add_custom_function('error_fn', error_metric)

        # Should not raise, but return NaN
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = panel.compute(small_tissue)
            assert len(w) == 1
            assert 'failed' in str(w[0].message).lower()

        assert 'error_fn' in result
        assert np.isnan(result['error_fn'])


# =============================================================================
# Test: Use Cases from Requirements
# =============================================================================

class TestRequirementUseCases:
    """Tests for specific use cases mentioned in requirements."""

    def test_usecase_cell_type_ratio(self, small_tissue):
        """
        Use case: Custom cell type ratio.

        User wants to compute CD8/Treg ratio as a biomarker.
        """
        @register_custom_metric(
            name='cd8_to_treg',
            category='biomarker',
            description='CD8+ T cell to Treg ratio (immune checkpoint biomarker)'
        )
        def cd8_to_treg(data: SpatialTissueData) -> Dict[str, float]:
            counts = data.cell_type_counts
            cd8 = counts.get('CD8_T', 0)
            treg = counts.get('Treg', 0)

            if treg == 0:
                ratio = float('inf') if cd8 > 0 else float('nan')
            else:
                ratio = cd8 / treg

            return {
                'cd8_count': float(cd8),
                'treg_count': float(treg),
                'cd8_treg_ratio': ratio
            }

        # Test
        result = get_metric('cd8_to_treg')(small_tissue)
        assert result['cd8_count'] == 30.0
        assert result['treg_count'] == 20.0
        assert result['cd8_treg_ratio'] == 1.5

        # Use in panel
        panel = StatisticsPanel()
        panel.add('cd8_to_treg')

        summary = SpatialSummary(small_tissue, panel)
        df_ready = summary.to_series()
        assert 'cd8_treg_ratio' in df_ready.index

    def test_usecase_squared_euclidean_distance(self, small_tissue):
        """
        Use case: Custom squared Euclidean distance metric.

        User wants squared distances instead of Euclidean for their analysis.
        """
        @register_custom_metric(
            name='squared_distances',
            category='spatial',
            description='Summary statistics of squared Euclidean distances',
            required_dependencies=['scipy']
        )
        def squared_distances(data: SpatialTissueData) -> Dict[str, float]:
            from scipy.spatial.distance import pdist

            if data.n_cells < 2:
                return {
                    'sq_dist_mean': 0.0,
                    'sq_dist_std': 0.0,
                    'sq_dist_min': 0.0,
                    'sq_dist_max': 0.0
                }

            sq_dists = pdist(data.coordinates, metric='sqeuclidean')
            return {
                'sq_dist_mean': float(np.mean(sq_dists)),
                'sq_dist_std': float(np.std(sq_dists)),
                'sq_dist_min': float(np.min(sq_dists)),
                'sq_dist_max': float(np.max(sq_dists))
            }

        # Test
        result = get_metric('squared_distances')(small_tissue)
        assert result['sq_dist_mean'] > 0
        assert result['sq_dist_std'] > 0
        assert result['sq_dist_min'] <= result['sq_dist_mean']
        assert result['sq_dist_max'] >= result['sq_dist_mean']

        # Verify it's squared (compare with regular Euclidean)
        from scipy.spatial.distance import pdist
        regular_dists = pdist(small_tissue.coordinates)
        expected_sq_mean = np.mean(regular_dists ** 2)
        assert result['sq_dist_mean'] == pytest.approx(expected_sq_mean)

    def test_usecase_reuse_across_panels(self, small_tissue):
        """
        Use case: Register once, use in multiple panels.

        User defines a metric once and adds it to multiple analysis panels.
        """
        @register_custom_metric(name='reusable_metric')
        def reusable_metric(data: SpatialTissueData) -> Dict[str, float]:
            return {'reusable': float(data.n_cells)}

        # Panel 1: Basic analysis
        panel1 = StatisticsPanel(name='basic_analysis')
        panel1.add('cell_counts')
        panel1.add('reusable_metric')

        # Panel 2: Detailed analysis
        panel2 = StatisticsPanel(name='detailed_analysis')
        panel2.add('cell_counts')
        panel2.add('cell_proportions')
        panel2.add('reusable_metric')

        # Both should work
        result1 = panel1.compute(small_tissue)
        result2 = panel2.compute(small_tissue)

        assert result1['reusable'] == 100.0
        assert result2['reusable'] == 100.0
        assert 'n_cells' in result1
        assert 'prop_Tumor' in result2


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestCustomMetricPerformance:
    """Performance tests for custom metrics."""

    def test_custom_metric_large_dataset(self):
        """Test custom metric performance on large dataset."""
        import time

        np.random.seed(42)
        coords = np.random.rand(10000, 2) * 2000
        types = np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000)
        large_data = SpatialTissueData(coords, types)

        @register_custom_metric(name='perf_test_metric')
        def perf_test_metric(data: SpatialTissueData) -> Dict[str, float]:
            # Simulate some computation
            return {
                'n': float(data.n_cells),
                'types': float(len(data.cell_types_unique))
            }

        start = time.time()
        for _ in range(100):
            result = get_metric('perf_test_metric')(large_data)
        elapsed = time.time() - start

        # Should be fast (< 1 second for 100 iterations)
        assert elapsed < 1.0
        assert result['n'] == 10000.0
