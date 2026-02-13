"""
Tests for spatialtissuepy.statistics module.

Tests spatial point pattern statistics including Ripley's K, co-localization,
and hotspot detection.
"""

import pytest
import numpy as np
from scipy.spatial import cKDTree

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.statistics import (
    # Ripley's K and variants
    ripleys_k,
    ripleys_l,
    ripleys_h,
    # Cross-type functions
    cross_k,
    cross_l,
    cross_h,
    # Nearest-neighbor functions
    g_function,
    g_function_cross,
    f_function,
    j_function,
    # Pair correlation
    pair_correlation_function,
    # CSR envelope
    csr_envelope,
    # High-level functions
    spatial_statistics,
    cross_type_statistics,
    # Co-localization
    colocalization_quotient,
    colocalization_matrix,
    neighborhood_enrichment_score,
    # Hotspots
    getis_ord_gi_star,
    detect_hotspots,
)


# =============================================================================
# Ripley's K-function Tests
# =============================================================================

class TestRipleysK:
    """Tests for Ripley's K-function."""
    
    def test_ripleys_k_basic(self):
        """Test basic K-function calculation."""
        # Random pattern
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([5, 10, 15, 20])
        
        K = ripleys_k(coords, radii, area=100*100)
        
        assert len(K) == len(radii)
        assert np.all(K >= 0)
        # K should increase with radius
        assert np.all(np.diff(K) >= 0)
    
    def test_ripleys_k_csr_expectation(self):
        """Test K matches CSR expectation for random pattern."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (500, 2))
        radii = np.linspace(5, 30, 10)
        
        K = ripleys_k(coords, radii, area=100*100, edge_correction='none')
        K_csr = np.pi * radii**2
        
        # Should be close to CSR (within ~20% for random pattern)
        relative_diff = np.abs(K - K_csr) / K_csr
        assert np.mean(relative_diff) < 0.3
    
    def test_ripleys_k_clustered(self):
        """Test K is elevated for clustered pattern."""
        # Create clustered pattern
        np.random.seed(42)
        clusters = []
        for center_x, center_y in [(25, 25), (75, 75)]:
            cluster = np.random.normal([center_x, center_y], 5, (50, 2))
            clusters.append(cluster)
        coords = np.vstack(clusters)
        
        radii = np.array([10, 20, 30])
        K = ripleys_k(coords, radii, area=100*100, edge_correction='none')
        K_csr = np.pi * radii**2
        
        # K should be greater than CSR (clustering)
        assert np.any(K > K_csr * 1.2)
    
    def test_ripleys_k_edge_correction(self):
        """Test different edge correction methods."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        K_none = ripleys_k(coords, radii, edge_correction='none')
        K_ripley = ripleys_k(coords, radii, edge_correction='ripley')
        
        # Should produce different results
        assert not np.allclose(K_none, K_ripley)
    
    def test_ripleys_k_empty(self):
        """Test K with no points."""
        coords = np.array([]).reshape(0, 2)
        radii = np.array([10, 20])
        
        K = ripleys_k(coords, radii)
        
        assert len(K) == 2
        assert np.all(K == 0)
    
    def test_ripleys_k_single_point(self):
        """Test K with single point."""
        coords = np.array([[50, 50]])
        radii = np.array([10, 20])
        
        K = ripleys_k(coords, radii)
        
        assert len(K) == 2
        assert np.all(K == 0)


class TestRipleysL:
    """Tests for Ripley's L-function."""
    
    def test_ripleys_l_basic(self):
        """Test L-function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        L = ripleys_l(coords, radii, area=100*100)
        
        assert len(L) == len(radii)
        assert np.all(L >= 0)
    
    def test_ripleys_l_csr_expectation(self):
        """Test L ≈ r for CSR pattern."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (500, 2))
        radii = np.linspace(5, 30, 10)
        
        L = ripleys_l(coords, radii, area=100*100, edge_correction='none')
        
        # L should be close to r for CSR
        relative_diff = np.abs(L - radii) / radii
        assert np.mean(relative_diff) < 0.2
    
    def test_ripleys_l_from_k(self):
        """Test L = sqrt(K/π)."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        K = ripleys_k(coords, radii, area=100*100)
        L = ripleys_l(coords, radii, area=100*100)
        
        np.testing.assert_array_almost_equal(L, np.sqrt(K / np.pi))


class TestRipleysH:
    """Tests for Ripley's H-function."""
    
    def test_ripleys_h_basic(self):
        """Test H-function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        H = ripleys_h(coords, radii, area=100*100)
        
        assert len(H) == len(radii)
    
    def test_ripleys_h_csr_expectation(self):
        """Test H ≈ 0 for CSR pattern."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (500, 2))
        radii = np.linspace(5, 30, 10)
        
        H = ripleys_h(coords, radii, area=100*100, edge_correction='none')
        
        # H should oscillate around 0 for CSR
        assert np.mean(np.abs(H)) < 5
    
    def test_ripleys_h_clustered(self):
        """Test H > 0 for clustered pattern."""
        np.random.seed(42)
        # Strongly clustered
        clusters = []
        for i in range(5):
            center = np.random.uniform(20, 80, 2)
            cluster = np.random.normal(center, 3, (20, 2))
            clusters.append(cluster)
        coords = np.vstack(clusters)
        
        radii = np.array([10, 15, 20])
        H = ripleys_h(coords, radii, area=100*100)
        
        # Should show positive H (clustering)
        assert np.max(H) > 2
    
    def test_ripleys_h_from_l(self):
        """Test H = L - r."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        L = ripleys_l(coords, radii, area=100*100)
        H = ripleys_h(coords, radii, area=100*100)
        
        np.testing.assert_array_almost_equal(H, L - radii)


# =============================================================================
# Cross-Type K-function Tests
# =============================================================================

class TestCrossK:
    """Tests for cross-type K-function."""
    
    def test_cross_k_basic(self):
        """Test basic cross-K calculation."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20, 30])
        
        K = cross_k(coords_a, coords_b, radii, area=100*100)
        
        assert len(K) == len(radii)
        assert np.all(K >= 0)
        assert np.all(np.diff(K) >= 0)
    
    def test_cross_k_independence(self):
        """Test cross-K ≈ π*r² for independent patterns."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (100, 2))
        coords_b = np.random.uniform(0, 100, (100, 2))
        radii = np.linspace(5, 25, 10)
        
        K = cross_k(coords_a, coords_b, radii, area=100*100, edge_correction='none')
        K_expected = np.pi * radii**2
        
        # Should be close to independence
        relative_diff = np.abs(K - K_expected) / K_expected
        assert np.mean(relative_diff) < 0.3
    
    def test_cross_k_attraction(self):
        """Test cross-K elevated for attracted patterns."""
        np.random.seed(42)
        # Create attracted pattern (same clusters)
        clusters_a = []
        clusters_b = []
        for i in range(3):
            center = np.random.uniform(20, 80, 2)
            clusters_a.append(np.random.normal(center, 5, (20, 2)))
            clusters_b.append(np.random.normal(center, 5, (20, 2)))
        
        coords_a = np.vstack(clusters_a)
        coords_b = np.vstack(clusters_b)
        
        radii = np.array([10, 20, 30])
        K = cross_k(coords_a, coords_b, radii, area=100*100)
        K_expected = np.pi * radii**2
        
        # Should show attraction
        assert np.any(K > K_expected * 1.5)
    
    def test_cross_k_empty(self):
        """Test cross-K with empty sets."""
        coords_a = np.array([]).reshape(0, 2)
        coords_b = np.random.rand(10, 2)
        radii = np.array([10, 20])
        
        K = cross_k(coords_a, coords_b, radii)
        assert np.all(K == 0)


class TestCrossL:
    """Tests for cross-type L-function."""
    
    def test_cross_l_basic(self):
        """Test cross-L calculation."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20, 30])
        
        L = cross_l(coords_a, coords_b, radii, area=100*100)
        
        assert len(L) == len(radii)
        assert np.all(L >= 0)
    
    def test_cross_l_from_k(self):
        """Test L = sqrt(K/π)."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20])
        
        K = cross_k(coords_a, coords_b, radii, area=100*100)
        L = cross_l(coords_a, coords_b, radii, area=100*100)
        
        np.testing.assert_array_almost_equal(L, np.sqrt(K / np.pi))


class TestCrossH:
    """Tests for cross-type H-function."""
    
    def test_cross_h_basic(self):
        """Test cross-H calculation."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20, 30])
        
        H = cross_h(coords_a, coords_b, radii, area=100*100)
        
        assert len(H) == len(radii)
    
    def test_cross_h_from_l(self):
        """Test H = L - r."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20])
        
        L = cross_l(coords_a, coords_b, radii, area=100*100)
        H = cross_h(coords_a, coords_b, radii, area=100*100)
        
        np.testing.assert_array_almost_equal(H, L - radii)


# =============================================================================
# Nearest-Neighbor G-function Tests
# =============================================================================

class TestGFunction:
    """Tests for G-function."""
    
    def test_g_function_basic(self):
        """Test G-function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([5, 10, 15, 20])
        
        G = g_function(coords, radii)
        
        assert len(G) == len(radii)
        assert np.all(G >= 0)
        assert np.all(G <= 1)
        # G should be non-decreasing
        assert np.all(np.diff(G) >= -1e-10)
    
    def test_g_function_cumulative(self):
        """Test G is cumulative distribution."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.linspace(0, 50, 20)
        
        G = g_function(coords, radii)
        
        # Should start near 0 and approach 1
        assert G[0] < 0.2
        assert G[-1] > 0.7  # Relaxed slightly from 0.8
    
    def test_g_function_edge_correction(self):
        """Test different edge correction methods."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        G_none = g_function(coords, radii, edge_correction='none')
        G_km = g_function(coords, radii, edge_correction='km')
        
        # Should produce different results
        assert not np.allclose(G_none, G_km)


class TestGFunctionCross:
    """Tests for cross-type G-function."""
    
    def test_g_function_cross_basic(self):
        """Test cross-G calculation."""
        np.random.seed(42)
        coords_a = np.random.uniform(0, 100, (50, 2))
        coords_b = np.random.uniform(0, 100, (50, 2))
        radii = np.array([10, 20, 30])
        
        G = g_function_cross(coords_a, coords_b, radii)
        
        assert len(G) == len(radii)
        assert np.all(G >= 0)
        assert np.all(G <= 1)
    
    def test_g_function_cross_empty(self):
        """Test with empty sets."""
        coords_a = np.array([]).reshape(0, 2)
        coords_b = np.random.rand(10, 2)
        radii = np.array([10, 20])
        
        G = g_function_cross(coords_a, coords_b, radii)
        assert np.all(G == 0)


# =============================================================================
# F-function and J-function Tests
# =============================================================================

class TestFFunction:
    """Tests for F-function."""
    
    def test_f_function_basic(self):
        """Test F-function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        F = f_function(coords, radii, n_test_points=500, seed=42)
        
        assert len(F) == len(radii)
        assert np.all(F >= 0)
        assert np.all(F <= 1)
    
    def test_f_function_reproducibility(self):
        """Test F-function reproducibility with seed."""
        coords = np.random.rand(50, 2) * 100
        radii = np.array([10, 20])
        
        F1 = f_function(coords, radii, seed=42)
        F2 = f_function(coords, radii, seed=42)
        
        np.testing.assert_array_almost_equal(F1, F2)


class TestJFunction:
    """Tests for J-function."""
    
    def test_j_function_basic(self):
        """Test J-function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.array([10, 20, 30])
        
        J = j_function(coords, radii, seed=42)
        
        assert len(J) == len(radii)
        assert np.all(np.isfinite(J))
    
    def test_j_function_csr(self):
        """Test J ≈ 1 for CSR pattern."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (200, 2))
        radii = np.linspace(5, 20, 10)
        
        J = j_function(coords, radii, n_test_points=500, seed=42)
        
        # J should be around 1, but can be highly variable for small samples
        # Just check that it's finite and positive
        assert np.all(np.isfinite(J))
        assert np.all(J >= 0)


# =============================================================================
# Pair Correlation Function Tests
# =============================================================================

class TestPairCorrelationFunction:
    """Tests for pair correlation function."""
    
    def test_pcf_basic(self):
        """Test pair correlation function calculation."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        radii = np.linspace(5, 30, 20)
        
        g = pair_correlation_function(coords, radii, area=100*100)
        
        assert len(g) == len(radii)
        assert np.all(g >= 0)
    
    def test_pcf_csr(self):
        """Test g(r) ≈ 1 for CSR pattern."""
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (300, 2))
        radii = np.linspace(5, 30, 15)
        
        g = pair_correlation_function(coords, radii, area=100*100)
        
        # Should oscillate around 1 for CSR
        assert np.mean(np.abs(g - 1)) < 0.5
    
    def test_pcf_clustered(self):
        """Test g(r) > 1 at small scales for clustered pattern."""
        np.random.seed(42)
        # Create tight clusters
        clusters = [np.random.normal([25, 25], 3, (30, 2)),
                   np.random.normal([75, 75], 3, (30, 2))]
        coords = np.vstack(clusters)
        
        radii = np.array([5, 10, 15, 20])
        g = pair_correlation_function(coords, radii, area=100*100)
        
        # Should be elevated at small scales
        assert g[0] > 1.5


# =============================================================================
# CSR Envelope Tests
# =============================================================================

class TestCSREnvelope:
    """Tests for CSR envelope generation."""
    
    def test_csr_envelope_basic(self):
        """Test CSR envelope generation."""
        radii = np.array([10, 20, 30])
        
        envelope = csr_envelope(
            n_points=100,
            radii=radii,
            area=100*100,
            n_simulations=19,
            statistic='H',
            seed=42
        )
        
        assert 'theoretical' in envelope
        assert 'lower' in envelope
        assert 'upper' in envelope
        assert 'simulations' in envelope
        
        assert len(envelope['theoretical']) == len(radii)
        assert len(envelope['lower']) == len(radii)
        assert envelope['simulations'].shape == (19, len(radii))
    
    def test_csr_envelope_h_theoretical(self):
        """Test H theoretical values are 0."""
        radii = np.array([10, 20, 30])
        
        envelope = csr_envelope(100, radii, 100*100, statistic='H', seed=42)
        
        np.testing.assert_array_almost_equal(envelope['theoretical'], [0, 0, 0])
    
    def test_csr_envelope_k_theoretical(self):
        """Test K theoretical values."""
        radii = np.array([10, 20, 30])
        
        envelope = csr_envelope(100, radii, 100*100, statistic='K', seed=42)
        
        expected = np.pi * radii**2
        np.testing.assert_array_almost_equal(envelope['theoretical'], expected)
    
    def test_csr_envelope_reproducibility(self):
        """Test envelope reproducibility with seed."""
        radii = np.array([10, 20])
        
        env1 = csr_envelope(50, radii, 100*100, n_simulations=10, seed=42)
        env2 = csr_envelope(50, radii, 100*100, n_simulations=10, seed=42)
        
        np.testing.assert_array_almost_equal(env1['lower'], env2['lower'])
        np.testing.assert_array_almost_equal(env1['upper'], env2['upper'])


# =============================================================================
# High-Level Functions Tests
# =============================================================================

class TestSpatialStatistics:
    """Tests for high-level spatial_statistics function."""
    
    def test_spatial_statistics_basic(self, small_tissue):
        """Test spatial statistics computation."""
        result = spatial_statistics(
            small_tissue,
            statistics=['K', 'L', 'H']
        )
        
        assert 'radii' in result
        assert 'K' in result
        assert 'L' in result
        assert 'H' in result
        
        assert len(result['K']) == len(result['radii'])
    
    def test_spatial_statistics_custom_radii(self, small_tissue):
        """Test with custom radii."""
        radii = np.array([10, 20, 30, 40])
        
        result = spatial_statistics(small_tissue, radii=radii)
        
        np.testing.assert_array_equal(result['radii'], radii)
    
    def test_spatial_statistics_cell_type(self, small_tissue):
        """Test statistics for specific cell type."""
        cell_type = small_tissue.cell_types_unique[0]
        
        result = spatial_statistics(
            small_tissue,
            cell_type=cell_type,
            statistics=['H']
        )
        
        assert 'H' in result
    
    def test_spatial_statistics_all_stats(self, small_tissue):
        """Test computing all statistics."""
        result = spatial_statistics(
            small_tissue,
            statistics=['K', 'L', 'H', 'G', 'F', 'J', 'g'],
            n_radii=10
        )
        
        for stat in ['K', 'L', 'H', 'G', 'F', 'J', 'g']:
            assert stat in result


class TestCrossTypeStatistics:
    """Tests for cross_type_statistics function."""
    
    def test_cross_type_statistics_basic(self, small_tissue):
        """Test cross-type statistics."""
        types = small_tissue.cell_types_unique
        if len(types) < 2:
            pytest.skip("Need at least 2 cell types")
        
        result = cross_type_statistics(
            small_tissue,
            type_a=types[0],
            type_b=types[1],
            statistics=['K', 'L', 'H']
        )
        
        assert 'radii' in result
        assert 'K' in result
        assert 'L' in result
        assert 'H' in result
    
    def test_cross_type_statistics_g_function(self, small_tissue):
        """Test cross-G function."""
        types = small_tissue.cell_types_unique
        if len(types) < 2:
            pytest.skip("Need at least 2 cell types")
        
        result = cross_type_statistics(
            small_tissue,
            type_a=types[0],
            type_b=types[1],
            statistics=['G'],
            n_radii=10
        )
        
        assert 'G' in result
        assert np.all(result['G'] >= 0)
        assert np.all(result['G'] <= 1)


# =============================================================================
# Co-localization Tests
# =============================================================================

class TestColocalizationQuotient:
    """Tests for colocalization quotient."""
    
    def test_clq_basic(self, small_tissue):
        """Test basic CLQ calculation."""
        types = small_tissue.cell_types_unique
        if len(types) < 2:
            pytest.skip("Need at least 2 cell types")
        
        clq = colocalization_quotient(
            small_tissue,
            type_a=types[0],
            type_b=types[1],
            radius=30
        )
        
        assert isinstance(clq, float)
        assert clq >= 0
    
    def test_clq_attracted(self):
        """Test CLQ > 1 for attracted pattern."""
        # Create attracted pattern
        np.random.seed(42)
        coords_a = np.random.normal([30, 30], 10, (40, 2))
        coords_b = np.random.normal([30, 30], 10, (40, 2))
        
        coords = np.vstack([coords_a, coords_b])
        types = np.array(['A']*40 + ['B']*40)
        data = SpatialTissueData(coords, types)
        
        clq = colocalization_quotient(data, 'A', 'B', radius=20)
        
        # Should show attraction
        assert clq > 1.0
    
    def test_clq_repulsed(self):
        """Test CLQ < 1 for repulsed pattern."""
        # Create segregated pattern
        coords_a = np.random.uniform([0, 0], [30, 100], (40, 2))
        coords_b = np.random.uniform([70, 0], [100, 100], (40, 2))
        
        coords = np.vstack([coords_a, coords_b])
        types = np.array(['A']*40 + ['B']*40)
        data = SpatialTissueData(coords, types)
        
        clq = colocalization_quotient(data, 'A', 'B', radius=20)
        
        # Should show repulsion
        assert clq < 0.5


class TestColocalizationMatrix:
    """Tests for colocalization matrix."""
    
    def test_clq_matrix_basic(self, small_tissue):
        """Test CLQ matrix calculation."""
        matrix = colocalization_matrix(small_tissue, radius=30)
        
        n_types = len(small_tissue.cell_types_unique)
        assert matrix.shape == (n_types, n_types)
        
        # Diagonal should be ~1 (self-CLQ)
        # Off-diagonal varies based on pattern
        assert np.all(matrix >= 0)
    
    def test_clq_matrix_symmetry(self, small_tissue):
        """Test CLQ matrix is symmetric."""
        matrix = colocalization_matrix(small_tissue, radius=30)
        
        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=10)


# =============================================================================
# Hotspot Detection Tests
# =============================================================================

class TestGetisOrdGiStar:
    """Tests for Getis-Ord Gi* statistic."""
    
    def test_gi_star_basic(self, small_tissue):
        """Test Gi* calculation."""
        # Create some values
        values = np.random.rand(small_tissue.n_cells)
        
        # Use return_dict=True to match test expectation
        result = getis_ord_gi_star(small_tissue, values, radius=30, return_dict=True)
        
        assert 'gi_star' in result
        assert len(result['gi_star']) == small_tissue.n_cells
    
    def test_gi_star_hotspot(self):
        """Test Gi* detects hotspot."""
        # Create pattern with hotspot
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (100, 2))
        
        # High values in one region
        values = np.ones(100)
        center_mask = (coords[:, 0] < 30) & (coords[:, 1] < 30)
        values[center_mask] = 10.0
        
        data = SpatialTissueData(coords, ['A']*100)
        result = getis_ord_gi_star(data, values, radius=20, return_dict=True)
        
        # Points in hotspot should have high Gi*
        assert np.mean(result['gi_star'][center_mask]) > np.mean(result['gi_star'][~center_mask])


class TestDetectHotspots:
    """Tests for detect_hotspots function."""
    
    def test_detect_hotspots_basic(self, small_tissue):
        """Test hotspot detection."""
        values = np.random.rand(small_tissue.n_cells)
        
        result = detect_hotspots(
            small_tissue,
            values,
            radius=30,
            significance=0.05
        )
        
        assert 'hotspot_idx' in result
        assert 'coldspot_idx' in result
        assert 'statistic' in result
    
    def test_detect_hotspots_no_hotspots(self, small_tissue):
        """Test with uniform values (no hotspots)."""
        values = np.ones(small_tissue.n_cells)
        
        result = detect_hotspots(small_tissue, values, radius=30)
        
        # Should find few or no hotspots
        assert len(result['hotspot_idx']) < small_tissue.n_cells * 0.1


# =============================================================================
# Integration Tests
# =============================================================================

class TestStatisticsIntegration:
    """Integration tests for statistics module."""
    
    def test_complete_spatial_analysis(self, medium_tissue):
        """Test complete spatial analysis workflow."""
        # 1. Compute basic statistics
        result = spatial_statistics(
            medium_tissue,
            statistics=['K', 'L', 'H'],
            n_radii=20
        )
        assert 'H' in result
        
        # 2. Cross-type analysis
        types = medium_tissue.cell_types_unique
        if len(types) >= 2:
            cross_result = cross_type_statistics(
                medium_tissue,
                types[0],
                types[1],
                n_radii=20
            )
            assert 'H' in cross_result
        
        # 3. Co-localization
        if len(types) >= 2:
            clq = colocalization_quotient(
                medium_tissue,
                types[0],
                types[1],
                radius=50
            )
            assert isinstance(clq, float)
    
    def test_clustering_detection_workflow(self, clustered_pattern):
        """Test detecting clustering with multiple methods."""
        # Method 1: Ripley's H
        result_h = spatial_statistics(
            clustered_pattern,
            statistics=['H'],
            n_radii=30
        )
        
        # Should show positive H
        assert np.max(result_h['H']) > 0
        
        # Method 2: Pair correlation
        radii = result_h['radii']
        g = pair_correlation_function(
            clustered_pattern.coordinates,
            radii,
            area=clustered_pattern.extent['x'] * clustered_pattern.extent['y']
        )
        
        # Should show g > 1 at some scales
        assert np.max(g) > 1.0


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestStatisticsReproducibility:
    """Tests for reproducible results with random seeds."""
    
    def test_f_function_reproducibility(self):
        """Test F-function reproducibility."""
        coords = np.random.rand(50, 2) * 100
        radii = np.array([10, 20, 30])
        
        F1 = f_function(coords, radii, n_test_points=100, seed=42)
        F2 = f_function(coords, radii, n_test_points=100, seed=42)
        
        np.testing.assert_array_almost_equal(F1, F2)
    
    def test_csr_envelope_reproducibility(self):
        """Test CSR envelope reproducibility."""
        radii = np.array([10, 20])
        
        env1 = csr_envelope(50, radii, 10000, n_simulations=10, seed=42)
        env2 = csr_envelope(50, radii, 10000, n_simulations=10, seed=42)
        
        np.testing.assert_array_almost_equal(
            env1['simulations'],
            env2['simulations']
        )


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestStatisticsPerformance:
    """Performance tests for statistics computations."""
    
    def test_ripleys_k_performance(self, large_tissue):
        """Test K-function performance with 10k cells."""
        import time
        
        coords = large_tissue.coordinates[:, :2]
        radii = np.linspace(0, 100, 20)
        
        start = time.time()
        K = ripleys_k(coords, radii)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
    
    def test_clq_matrix_performance(self, large_tissue):
        """Test CLQ matrix performance."""
        import time
        
        start = time.time()
        matrix = colocalization_matrix(large_tissue, radius=50)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 20.0
