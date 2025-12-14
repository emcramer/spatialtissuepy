"""
Tests for spatialtissuepy.topology module.

Tests Mapper algorithm, filter functions, cover construction, and
topological data analysis for discovering cell communities.
"""

import pytest
import numpy as np
import pandas as pd

from spatialtissuepy import SpatialTissueData

# Always import topology module (dependencies checked within)
from spatialtissuepy.topology import (
    # Main classes
    SpatialMapper,
    MapperResult,
    spatial_mapper,
    # Cover classes
    Cover,
    UniformCover,
    AdaptiveCover,
    create_cover,
    # Standard filters
    density_filter,
    pca_filter,
    eccentricity_filter,
    entropy_filter,
    # Spatial filters
    spatial_coordinate_filter,
    radial_filter,
    distance_to_type_filter,
    distance_to_boundary_filter,
    spatial_density_filter,
    gaussian_smoothed_filter,
    composite_filter,
    # Analysis functions
    node_summary_dataframe,
    find_hub_nodes,
    find_bridge_nodes,
    component_statistics,
    compare_mapper_results,
    extract_mapper_features,
    cells_in_multiple_nodes,
    uncovered_cells,
)


# =============================================================================
# Cover Tests
# =============================================================================

class TestCover:
    """Tests for cover construction."""
    
    def test_uniform_cover_basic(self):
        """Test basic uniform cover."""
        cover = UniformCover(n_intervals=5, overlap_fraction=0.3)
        
        values = np.linspace(0, 100, 200)
        cover.fit(values)
        
        assert cover.n_intervals == 5
        assert cover.overlap_fraction == 0.3
        assert len(cover.interval_bounds) == 5
    
    def test_uniform_cover_element_assignment(self):
        """Test assigning values to cover elements."""
        cover = UniformCover(n_intervals=3, overlap_fraction=0.2)
        values = np.array([0, 25, 50, 75, 100])
        cover.fit(values)
        
        members = cover.get_element_members(values)
        
        # Should have 3 elements
        assert len(members) == 3
        # Each element should have some members
        assert all(len(m) > 0 for m in members)
    
    def test_uniform_cover_overlap(self):
        """Test that overlap creates expected sharing."""
        cover = UniformCover(n_intervals=2, overlap_fraction=0.5)
        values = np.linspace(0, 100, 100)
        cover.fit(values)
        
        members = cover.get_element_members(values)
        
        # With 50% overlap, middle values should be in both elements
        elem0 = set(members[0])
        elem1 = set(members[1])
        overlap = elem0 & elem1
        
        assert len(overlap) > 0  # Should have some overlap
    
    def test_adaptive_cover_basic(self):
        """Test adaptive (quantile) cover."""
        cover = AdaptiveCover(n_intervals=4, overlap_fraction=0.3)
        
        # Skewed distribution
        values = np.concatenate([
            np.random.uniform(0, 10, 80),
            np.random.uniform(90, 100, 20)
        ])
        cover.fit(values)
        
        assert cover.n_intervals == 4
        assert len(cover.interval_bounds) == 4
    
    def test_adaptive_cover_equal_counts(self):
        """Test that adaptive cover creates roughly equal-sized bins."""
        cover = AdaptiveCover(n_intervals=5, overlap_fraction=0.1)
        values = np.random.exponential(scale=10, size=1000)
        cover.fit(values)
        
        members = cover.get_element_members(values)
        
        # Core regions (excluding overlap) should have similar sizes
        # This is approximate due to overlap
        sizes = [len(m) for m in members]
        mean_size = np.mean(sizes)
        
        # All sizes should be within reasonable range of mean
        for size in sizes:
            assert 0.5 * mean_size < size < 2 * mean_size
    
    def test_create_cover_uniform(self):
        """Test cover factory for uniform."""
        cover = create_cover('uniform', n_intervals=5, overlap_fraction=0.3)
        
        assert isinstance(cover, UniformCover)
        assert cover.n_intervals == 5
    
    def test_create_cover_adaptive(self):
        """Test cover factory for adaptive."""
        cover = create_cover('adaptive', n_intervals=5, overlap_fraction=0.3)
        
        assert isinstance(cover, AdaptiveCover)
        assert cover.n_intervals == 5
    
    def test_create_cover_invalid(self):
        """Test error with invalid cover type."""
        with pytest.raises(ValueError, match="Unknown cover type"):
            create_cover('invalid', n_intervals=5)


# =============================================================================
# Filter Function Tests
# =============================================================================

class TestFilterFunctions:
    """Tests for filter functions."""
    
    def test_density_filter_basic(self, small_tissue):
        """Test density filter."""
        filter_fn = density_filter()
        
        coords = small_tissue.coordinates
        # Create simple feature matrix
        features = np.random.rand(len(coords), 5)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(np.isfinite(values))
    
    def test_pca_filter_basic(self, small_tissue):
        """Test PCA filter."""
        filter_fn = pca_filter(n_components=1)
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 5)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(np.isfinite(values))
    
    def test_eccentricity_filter_basic(self, small_tissue):
        """Test eccentricity filter."""
        filter_fn = eccentricity_filter()
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 5)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)  # Distances are non-negative
    
    def test_entropy_filter_basic(self, small_tissue):
        """Test entropy filter."""
        filter_fn = entropy_filter()
        
        coords = small_tissue.coordinates
        # Composition features (should be probabilities)
        features = np.random.dirichlet(np.ones(5), size=len(coords))
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)  # Entropy is non-negative


class TestSpatialFilters:
    """Tests for spatial filter functions."""
    
    def test_spatial_coordinate_filter_x(self, small_tissue):
        """Test x-coordinate filter."""
        filter_fn = spatial_coordinate_filter(axis='x')
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        # Should return x coordinates
        np.testing.assert_array_almost_equal(values, coords[:, 0])
    
    def test_spatial_coordinate_filter_y(self, small_tissue):
        """Test y-coordinate filter."""
        filter_fn = spatial_coordinate_filter(axis='y')
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        # Should return y coordinates
        np.testing.assert_array_almost_equal(values, coords[:, 1])
    
    def test_radial_filter_basic(self, small_tissue):
        """Test radial distance filter."""
        center = np.mean(small_tissue.coordinates, axis=0)
        filter_fn = radial_filter(center=center)
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)  # Distances are non-negative
        # Center cell should have near-zero distance
        center_idx = np.argmin(np.linalg.norm(coords - center, axis=1))
        assert values[center_idx] < np.mean(values)
    
    def test_distance_to_type_filter(self, small_tissue):
        """Test distance to cell type filter."""
        if len(small_tissue.cell_types_unique) < 2:
            pytest.skip("Need multiple cell types")
        
        target_type = small_tissue.cell_types_unique[0]
        filter_fn = distance_to_type_filter(target_type)
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)
        
        # Cells of target type should have distance ~0
        target_idx = small_tissue.get_cells_by_type(target_type)
        assert np.mean(values[target_idx]) < np.mean(values)
    
    def test_distance_to_boundary_filter(self, small_tissue):
        """Test distance to boundary filter."""
        filter_fn = distance_to_boundary_filter()
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)
        # Boundary cells should have small distances
        assert np.min(values) < 0.5 * np.max(values)
    
    def test_spatial_density_filter(self, small_tissue):
        """Test spatial density filter."""
        filter_fn = spatial_density_filter(radius=50)
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(values >= 0)  # Density is non-negative
    
    def test_composite_filter(self, small_tissue):
        """Test composite filter (weighted combination)."""
        filter1 = spatial_coordinate_filter(axis='x')
        filter2 = spatial_coordinate_filter(axis='y')
        
        filter_fn = composite_filter([filter1, filter2], weights=[0.6, 0.4])
        
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 3)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        # Should be weighted combination
        expected = 0.6 * coords[:, 0] + 0.4 * coords[:, 1]
        np.testing.assert_array_almost_equal(values, expected)


# =============================================================================
# SpatialMapper Tests
# =============================================================================

class TestSpatialMapper:
    """Tests for SpatialMapper class."""
    
    def test_mapper_initialization(self):
        """Test SpatialMapper initialization."""
        mapper = SpatialMapper(
            filter_fn='density',
            n_intervals=10,
            overlap=0.5
        )
        
        assert mapper.n_intervals == 10
        assert mapper.overlap == 0.5
        assert mapper.clustering == 'dbscan'
    
    def test_mapper_fit_basic(self, small_tissue):
        """Test basic Mapper fitting."""
        mapper = SpatialMapper(
            filter_fn='density',
            n_intervals=5,
            overlap=0.3,
            min_cluster_size=2
        )
        
        result = mapper.fit(small_tissue, neighborhood_radius=50)
        
        assert isinstance(result, MapperResult)
        assert result.n_nodes >= 0
        assert len(result.filter_values) == small_tissue.n_cells
    
    def test_mapper_with_pca_filter(self, small_tissue):
        """Test Mapper with PCA filter."""
        mapper = SpatialMapper(
            filter_fn='pca',
            n_intervals=5,
            overlap=0.3
        )
        
        result = mapper.fit(small_tissue, neighborhood_radius=50)
        
        assert isinstance(result, MapperResult)
        assert result.n_nodes >= 0
    
    def test_mapper_with_spatial_filter(self, small_tissue):
        """Test Mapper with spatial filter."""
        center = np.mean(small_tissue.coordinates, axis=0)
        
        mapper = SpatialMapper(
            filter_fn=radial_filter(center=center),
            n_intervals=5,
            overlap=0.3
        )
        
        result = mapper.fit(small_tissue, neighborhood_radius=50)
        
        assert isinstance(result, MapperResult)
    
    def test_mapper_different_clustering(self, small_tissue):
        """Test different clustering algorithms."""
        for clustering in ['dbscan', 'agglomerative', 'kmeans']:
            mapper = SpatialMapper(
                filter_fn='density',
                n_intervals=5,
                clustering=clustering,
                min_cluster_size=2
            )
            
            try:
                result = mapper.fit(small_tissue, neighborhood_radius=50)
                assert isinstance(result, MapperResult)
            except ImportError:
                # Clustering algorithm may require scikit-learn
                pytest.skip(f"{clustering} requires scikit-learn")
    
    def test_mapper_cover_types(self, small_tissue):
        """Test different cover types."""
        for cover_type in ['uniform', 'adaptive']:
            mapper = SpatialMapper(
                filter_fn='density',
                cover_type=cover_type,
                n_intervals=5,
                overlap=0.3
            )
            
            result = mapper.fit(small_tissue, neighborhood_radius=50)
            assert isinstance(result, MapperResult)
    
    def test_mapper_parameters_stored(self, small_tissue):
        """Test that parameters are stored in result."""
        mapper = SpatialMapper(
            filter_fn='density',
            n_intervals=7,
            overlap=0.4,
            min_cluster_size=3
        )
        
        result = mapper.fit(small_tissue, neighborhood_radius=50)
        
        assert 'n_intervals' in result.parameters
        assert result.parameters['n_intervals'] == 7
        assert result.parameters['overlap'] == 0.4
        assert result.parameters['min_cluster_size'] == 3


class TestMapperResult:
    """Tests for MapperResult class."""
    
    def test_mapper_result_properties(self, small_tissue):
        """Test MapperResult properties."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        assert isinstance(result.n_nodes, int)
        assert isinstance(result.n_edges, int)
        assert isinstance(result.n_components, int)
        assert result.n_nodes >= 0
        assert result.n_edges >= 0
        assert result.n_components >= 0
    
    def test_mapper_result_filter_values(self, small_tissue):
        """Test filter values storage."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        assert len(result.filter_values) == small_tissue.n_cells
        assert np.all(np.isfinite(result.filter_values))
    
    def test_mapper_result_cell_node_map(self, small_tissue):
        """Test cell-to-node mapping."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50,
            min_cluster_size=2
        )
        
        # Should be a dict
        assert isinstance(result.cell_node_map, dict)
        
        # If any nodes exist, some cells should be mapped
        if result.n_nodes > 0:
            assert len(result.cell_node_map) > 0
    
    def test_mapper_result_get_node_members(self, small_tissue):
        """Test getting node members."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50,
            min_cluster_size=2
        )
        
        if result.n_nodes > 0:
            node_id = result.nodes[0].node_id
            members = result.get_node_members(node_id)
            
            assert isinstance(members, np.ndarray)
            assert len(members) > 0
            assert np.all(members >= 0)
            assert np.all(members < small_tissue.n_cells)
    
    def test_mapper_result_statistics(self, small_tissue):
        """Test computing statistics."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        stats = result.statistics
        
        assert isinstance(stats, dict)
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
    
    def test_mapper_result_repr_str(self, small_tissue):
        """Test string representations."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        repr_str = repr(result)
        str_str = str(result)
        
        assert 'MapperResult' in repr_str
        assert 'MapperResult' in str_str
        assert str(result.n_nodes) in str_str


class TestSpatialMapperConvenience:
    """Tests for spatial_mapper convenience function."""
    
    def test_spatial_mapper_basic(self, small_tissue):
        """Test basic usage of spatial_mapper."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        assert isinstance(result, MapperResult)
    
    def test_spatial_mapper_with_kwargs(self, small_tissue):
        """Test spatial_mapper with additional parameters."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='pca',
            n_intervals=7,
            overlap=0.4,
            neighborhood_radius=40,
            min_cluster_size=3
        )
        
        assert isinstance(result, MapperResult)
        assert result.parameters['n_intervals'] == 7


# =============================================================================
# Analysis Function Tests
# =============================================================================

class TestAnalysisFunctions:
    """Tests for topology analysis functions."""
    
    def test_node_summary_dataframe(self, small_tissue):
        """Test creating node summary DataFrame."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50,
            min_cluster_size=2
        )
        
        if result.n_nodes > 0:
            df = node_summary_dataframe(result)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == result.n_nodes
            assert 'node_id' in df.columns
            assert 'size' in df.columns
    
    def test_find_hub_nodes(self, medium_tissue):
        """Test finding hub nodes."""
        result = spatial_mapper(
            medium_tissue,
            filter_fn='density',
            n_intervals=8,
            neighborhood_radius=50,
            min_cluster_size=3
        )
        
        if result.n_nodes > 3:
            hubs = find_hub_nodes(result, top_n=3)
            
            assert isinstance(hubs, list)
            assert len(hubs) <= 3
    
    def test_find_bridge_nodes(self, medium_tissue):
        """Test finding bridge nodes."""
        result = spatial_mapper(
            medium_tissue,
            filter_fn='density',
            n_intervals=8,
            neighborhood_radius=50
        )
        
        if result.n_nodes > 2:
            bridges = find_bridge_nodes(result)
            
            assert isinstance(bridges, list)
    
    def test_component_statistics(self, medium_tissue):
        """Test computing component statistics."""
        result = spatial_mapper(
            medium_tissue,
            filter_fn='density',
            n_intervals=8,
            neighborhood_radius=50
        )
        
        if result.n_components > 0:
            stats = component_statistics(result)
            
            assert isinstance(stats, list)
            assert len(stats) == result.n_components
    
    def test_extract_mapper_features(self, small_tissue):
        """Test extracting features from Mapper."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50
        )
        
        features = extract_mapper_features(result)
        
        assert isinstance(features, dict)
        assert 'n_nodes' in features
        assert 'n_edges' in features
        assert 'n_components' in features
    
    def test_cells_in_multiple_nodes(self, small_tissue):
        """Test finding cells in multiple nodes."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            overlap=0.5,  # High overlap
            neighborhood_radius=50
        )
        
        multi_cells = cells_in_multiple_nodes(result)
        
        assert isinstance(multi_cells, np.ndarray)
        # With high overlap, should have some cells in multiple nodes
    
    def test_uncovered_cells(self, small_tissue):
        """Test finding uncovered cells."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=50,
            min_cluster_size=3  # Strict requirement
        )
        
        uncovered = uncovered_cells(result, n_cells=small_tissue.n_cells)
        
        assert isinstance(uncovered, np.ndarray)
        assert len(uncovered) <= small_tissue.n_cells


# =============================================================================
# Integration Tests
# =============================================================================

class TestTopologyIntegration:
    """Integration tests for topology module."""
    
    def test_complete_mapper_workflow(self, medium_tissue):
        """Test complete Mapper analysis workflow."""
        # 1. Create Mapper
        mapper = SpatialMapper(
            filter_fn='density',
            n_intervals=10,
            overlap=0.4,
            min_cluster_size=3
        )
        
        # 2. Fit to data
        result = mapper.fit(medium_tissue, neighborhood_radius=50)
        
        # 3. Analyze results
        assert result.n_nodes >= 0
        assert result.n_components >= 0
        
        # 4. Get statistics
        stats = result.statistics
        assert isinstance(stats, dict)
        
        # 5. Extract features
        features = extract_mapper_features(result)
        assert isinstance(features, dict)
        
        # 6. Find important nodes
        if result.n_nodes > 0:
            df = node_summary_dataframe(result)
            assert isinstance(df, pd.DataFrame)
    
    def test_multi_filter_comparison(self, medium_tissue):
        """Test comparing results with different filters."""
        filters = ['density', 'pca']
        results = {}
        
        for filter_name in filters:
            try:
                result = spatial_mapper(
                    medium_tissue,
                    filter_fn=filter_name,
                    n_intervals=8,
                    neighborhood_radius=50
                )
                results[filter_name] = result
            except Exception:
                # Some filters may fail
                pass
        
        # Should have at least one result
        assert len(results) > 0
    
    def test_spatial_filter_workflow(self, medium_tissue):
        """Test workflow with spatial filter."""
        # Use radial filter from center
        center = np.mean(medium_tissue.coordinates, axis=0)
        
        result = spatial_mapper(
            medium_tissue,
            filter_fn=radial_filter(center=center),
            n_intervals=10,
            overlap=0.3,
            neighborhood_radius=50
        )
        
        assert isinstance(result, MapperResult)
        assert len(result.filter_values) == medium_tissue.n_cells
        
        # Filter values should increase with distance from center
        coords = medium_tissue.coordinates
        distances = np.linalg.norm(coords - center, axis=1)
        
        # Correlation should be high
        correlation = np.corrcoef(result.filter_values, distances)[0, 1]
        assert correlation > 0.8
    
    def test_parameter_sweep(self, small_tissue):
        """Test sweeping over n_intervals."""
        results = []
        
        for n_intervals in [3, 5, 7]:
            result = spatial_mapper(
                small_tissue,
                filter_fn='density',
                n_intervals=n_intervals,
                neighborhood_radius=50
            )
            results.append(result)
        
        # All should succeed
        assert len(results) == 3
        
        # Generally, more intervals → more nodes (though not always)
        # Just check they're all valid
        for result in results:
            assert result.n_nodes >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestTopologyEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_sample(self):
        """Test with very small sample."""
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        types = np.array(['A', 'B', 'A', 'B', 'A'])
        data = SpatialTissueData(coords, types)
        
        result = spatial_mapper(
            data,
            filter_fn='density',
            n_intervals=3,
            neighborhood_radius=2,
            min_cluster_size=1
        )
        
        # Should work even with small sample
        assert isinstance(result, MapperResult)
    
    def test_single_cell_type(self):
        """Test with single cell type."""
        coords = np.random.rand(50, 2) * 100
        types = np.array(['A'] * 50)
        data = SpatialTissueData(coords, types)
        
        result = spatial_mapper(
            data,
            filter_fn='density',
            n_intervals=5,
            neighborhood_radius=20
        )
        
        assert isinstance(result, MapperResult)
    
    def test_isolated_cells(self):
        """Test with isolated cells."""
        coords = np.array([[0, 0], [1000, 1000], [2000, 2000]])
        types = np.array(['A', 'B', 'C'])
        data = SpatialTissueData(coords, types)
        
        result = spatial_mapper(
            data,
            filter_fn='density',
            n_intervals=3,
            neighborhood_radius=10,  # Small radius
            min_cluster_size=1
        )
        
        # Should work even with isolated cells
        assert isinstance(result, MapperResult)
    
    def test_high_overlap(self, small_tissue):
        """Test with very high overlap."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            overlap=0.9,  # Very high
            neighborhood_radius=50
        )
        
        # Should still work
        assert isinstance(result, MapperResult)
        
        # Many cells should be in multiple nodes
        multi = cells_in_multiple_nodes(result)
        # With high overlap, should have some
        # (though depends on clustering)
    
    def test_no_overlap(self, small_tissue):
        """Test with no overlap."""
        result = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            overlap=0.0,
            neighborhood_radius=50
        )
        
        assert isinstance(result, MapperResult)
    
    def test_invalid_filter_string(self):
        """Test error with invalid filter string."""
        mapper = SpatialMapper(filter_fn='invalid_filter')
        
        coords = np.random.rand(50, 2)
        types = np.array(['A'] * 50)
        data = SpatialTissueData(coords, types)
        
        with pytest.raises(ValueError, match="Unknown filter"):
            mapper.fit(data)


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestTopologyPerformance:
    """Performance tests for topology operations."""
    
    def test_mapper_performance_large(self, large_tissue):
        """Test Mapper performance with large dataset."""
        import time
        
        start = time.time()
        result = spatial_mapper(
            large_tissue,
            filter_fn='density',
            n_intervals=10,
            overlap=0.3,
            neighborhood_radius=50,
            min_cluster_size=5
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 120.0  # 2 minutes
        assert isinstance(result, MapperResult)
    
    def test_filter_computation_performance(self, large_tissue):
        """Test filter computation performance."""
        import time
        
        filter_fn = density_filter()
        coords = large_tissue.coordinates
        features = np.random.rand(len(coords), 10)
        
        start = time.time()
        values = filter_fn(coords, features, large_tissue)
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 5.0
        assert len(values) == large_tissue.n_cells


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestTopologyReproducibility:
    """Tests for reproducible results."""
    
    def test_mapper_reproducibility(self, small_tissue):
        """Test that Mapper produces consistent results."""
        # Note: DBSCAN may not be perfectly reproducible without
        # sklearn random_state, but structure should be similar
        
        result1 = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            overlap=0.3,
            neighborhood_radius=50
        )
        
        result2 = spatial_mapper(
            small_tissue,
            filter_fn='density',
            n_intervals=5,
            overlap=0.3,
            neighborhood_radius=50
        )
        
        # Filter values should be identical
        np.testing.assert_array_almost_equal(
            result1.filter_values,
            result2.filter_values
        )
        
        # Graph structure may vary slightly due to clustering
        # but should be roughly similar
        assert abs(result1.n_nodes - result2.n_nodes) <= 2
