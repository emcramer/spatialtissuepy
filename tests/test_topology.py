"""
Tests for spatialtissuepy.topology module.

Tests Mapper algorithm, filter functions, cover construction, and
topological data analysis for discovering cell communities.
"""

import pytest
import numpy as np
import pandas as pd

from spatialtissuepy import SpatialTissueData

# Import topology module
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

# Check for optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
        assert len(cover.elements) == 5
    
    def test_uniform_cover_element_assignment(self):
        """Test assigning values to cover elements."""
        cover = UniformCover(n_intervals=3, overlap_fraction=0.2)
        values = np.array([0, 25, 50, 75, 100])
        cover.fit(values)
        
        members = cover.get_element_members(values)
        
        assert len(members) == 3
        assert all(len(m) > 0 for m in members)
    
    def test_adaptive_cover_basic(self):
        """Test adaptive (quantile) cover."""
        cover = AdaptiveCover(n_intervals=4, overlap_fraction=0.3)
        
        values = np.concatenate([
            np.random.uniform(0, 10, 80),
            np.random.uniform(90, 100, 20)
        ])
        cover.fit(values)
        
        assert cover.n_intervals == 4
        assert len(cover.elements) == 4
    
    def test_create_cover_factory(self):
        """Test cover factory."""
        assert isinstance(create_cover('uniform'), UniformCover)
        assert isinstance(create_cover('adaptive'), AdaptiveCover)


# =============================================================================
# Filter Function Tests
# =============================================================================

class TestFilterFunctions:
    """Tests for filter functions."""
    
    def test_density_filter_basic(self, small_tissue):
        """Test density filter."""
        filter_fn = density_filter()
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 5)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
        assert np.all(np.isfinite(values))
    
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required for PCA")
    def test_pca_filter_basic(self, small_tissue):
        """Test PCA filter."""
        filter_fn = pca_filter(n_components=1)
        coords = small_tissue.coordinates
        features = np.random.rand(len(coords), 5)
        
        values = filter_fn(coords, features, small_tissue)
        
        assert len(values) == len(coords)
    
    def test_spatial_coordinate_filter(self, small_tissue):
        """Test coordinate filter."""
        filter_fn = spatial_coordinate_filter(axis='x', normalize=False)
        coords = small_tissue.coordinates
        values = filter_fn(coords, None, small_tissue)
        
        np.testing.assert_array_almost_equal(values, coords[:, 0])
    
    def test_radial_filter(self, small_tissue):
        """Test radial filter."""
        center = np.array([250, 250])
        filter_fn = radial_filter(center=center, normalize=False)
        coords = small_tissue.coordinates
        values = filter_fn(coords, None, small_tissue)
        
        expected = np.linalg.norm(coords - center, axis=1)
        np.testing.assert_array_almost_equal(values, expected)


# =============================================================================
# Mapper Tests
# =============================================================================

class TestSpatialMapper:
    """Tests for SpatialMapper class and results."""
    
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required for clustering")
    def test_mapper_fit_basic(self, small_tissue):
        """Test basic Mapper fitting."""
        mapper = SpatialMapper(
            filter_fn='density',
            n_intervals=5,
            overlap=0.3,
            min_cluster_size=2
        )
        
        result = mapper.fit(small_tissue, neighborhood_radius=100)
        
        assert isinstance(result, MapperResult)
        assert result.n_nodes >= 0
        assert len(result.filter_values) == small_tissue.n_cells
    
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn required")
    def test_spatial_mapper_convenience(self, small_tissue):
        """Test spatial_mapper convenience function."""
        result = spatial_mapper(small_tissue, n_intervals=5)
        assert isinstance(result, MapperResult)
    
    @pytest.mark.skipif(not (HAS_SKLEARN and HAS_NETWORKX), reason="sklearn and networkx required")
    def test_mapper_analysis_functions(self, small_tissue):
        """Test analysis functions on Mapper results."""
        result = spatial_mapper(small_tissue, n_intervals=5, min_cluster_size=2)
        
        if result.n_nodes > 0:
            # Node summary
            df = node_summary_dataframe(result)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == result.n_nodes
            
            # Hub nodes
            hubs = find_hub_nodes(result, n_hubs=2)
            assert len(hubs) <= 2
            
            # Features
            features = extract_mapper_features(result)
            assert isinstance(features, dict)
            assert 'mapper_n_nodes' in features


# =============================================================================
# Edge Cases
# =============================================================================

class TestTopologyEdgeCases:
    """Tests for topology edge cases."""
    
    def test_invalid_filter(self):
        """Test error with invalid filter name."""
        with pytest.raises(ValueError):
            SpatialMapper(filter_fn='invalid_filter_name')
    
    def test_invalid_cover(self):
        """Test error with invalid cover type."""
        with pytest.raises(ValueError):
            create_cover('invalid_cover_type')

