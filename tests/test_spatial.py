"""
Tests for spatialtissuepy.spatial module.

Tests distance calculations, nearest neighbor queries, and spatial metrics.
"""

import pytest
import numpy as np
from scipy.spatial import cKDTree

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.spatial import (
    # Distance matrices
    pairwise_distances,
    pairwise_distances_between,
    condensed_distances,
    # KD-tree
    build_kdtree,
    # Nearest neighbors
    nearest_neighbors,
    radius_neighbors,
    nearest_neighbor_distances,
    mean_nearest_neighbor_distance,
    # Distance to types
    distance_to_type,
    distance_to_nearest_different_type,
    distance_matrix_by_type,
    # Utilities
    centroid,
    centroid_by_type,
    bounding_box,
    convex_hull_area,
    point_density,
)


# =============================================================================
# Distance Matrix Tests
# =============================================================================

class TestPairwiseDistances:
    """Tests for pairwise distance calculations."""
    
    def test_pairwise_distances_basic(self):
        """Test basic pairwise distance matrix."""
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        
        D = pairwise_distances(coords)
        
        # Check shape
        assert D.shape == (3, 3)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(D), [0, 0, 0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(D, D.T)
        
        # Check specific distances
        assert D[0, 1] == 1.0  # Euclidean distance
        assert D[0, 2] == 1.0
        np.testing.assert_almost_equal(D[1, 2], np.sqrt(2))
    
    def test_pairwise_distances_euclidean(self):
        """Test Euclidean distance metric."""
        coords = np.array([[0, 0], [3, 4]])
        D = pairwise_distances(coords, metric='euclidean')
        
        assert D[0, 1] == 5.0  # 3-4-5 triangle
    
    def test_pairwise_distances_manhattan(self):
        """Test Manhattan (L1) distance metric."""
        coords = np.array([[0, 0], [3, 4]])
        D = pairwise_distances(coords, metric='manhattan')
        
        assert D[0, 1] == 7.0  # |3| + |4|
    
    def test_pairwise_distances_chebyshev(self):
        """Test Chebyshev (L∞) distance metric."""
        coords = np.array([[0, 0], [3, 4]])
        D = pairwise_distances(coords, metric='chebyshev')
        
        assert D[0, 1] == 4.0  # max(3, 4)
    
    def test_pairwise_distances_single_point(self):
        """Test with single point."""
        coords = np.array([[0, 0]])
        D = pairwise_distances(coords)
        
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0


class TestPairwiseDistancesBetween:
    """Tests for pairwise distances between two sets."""
    
    def test_distances_between_basic(self):
        """Test distances between two point sets."""
        a = np.array([[0, 0], [1, 1]])
        b = np.array([[2, 0], [0, 2]])
        
        D = pairwise_distances_between(a, b)
        
        assert D.shape == (2, 2)
        assert D[0, 0] == 2.0  # (0,0) to (2,0)
        assert D[0, 1] == 2.0  # (0,0) to (0,2)
    
    def test_distances_between_different_sizes(self):
        """Test with different sized sets."""
        a = np.array([[0, 0]])
        b = np.array([[1, 0], [0, 1], [1, 1]])
        
        D = pairwise_distances_between(a, b)
        
        assert D.shape == (1, 3)


class TestCondensedDistances:
    """Tests for condensed distance vector."""
    
    def test_condensed_distances_basic(self):
        """Test condensed distance format."""
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        
        condensed = condensed_distances(coords)
        
        # 3 points → 3 distances in upper triangle
        assert len(condensed) == 3
        
        # Check values
        assert condensed[0] == 1.0  # d(0,1)
        assert condensed[1] == 1.0  # d(0,2)
        np.testing.assert_almost_equal(condensed[2], np.sqrt(2))  # d(1,2)
    
    def test_condensed_to_square(self):
        """Test conversion from condensed to square format."""
        from scipy.spatial.distance import squareform
        
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        
        condensed = condensed_distances(coords)
        square = squareform(condensed)
        
        # Should match pairwise_distances
        expected = pairwise_distances(coords)
        np.testing.assert_array_almost_equal(square, expected)


# =============================================================================
# KD-Tree Tests
# =============================================================================

class TestKDTree:
    """Tests for KD-tree construction."""
    
    def test_build_kdtree(self):
        """Test KD-tree construction."""
        coords = np.random.rand(100, 2)
        tree = build_kdtree(coords)
        
        assert isinstance(tree, cKDTree)
    
    def test_kdtree_query(self):
        """Test querying KD-tree."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = build_kdtree(coords)
        
        # Query nearest to (0.5, 0.5)
        dist, idx = tree.query([0.5, 0.5], k=1)
        
        # Could be any of the 4 corners (equidistant)
        assert idx in [0, 1, 2, 3]
        np.testing.assert_almost_equal(dist, np.sqrt(0.5))


# =============================================================================
# Nearest Neighbor Tests
# =============================================================================

class TestNearestNeighbors:
    """Tests for nearest neighbor queries."""
    
    def test_nearest_neighbors_basic(self):
        """Test basic k-NN query."""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        
        distances, indices = nearest_neighbors(coords, k=2)
        
        assert distances.shape == (4, 2)
        assert indices.shape == (4, 2)
    
    def test_nearest_neighbors_exclude_self(self):
        """Test that self is excluded by default."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        distances, indices = nearest_neighbors(coords, k=1, include_self=False)
        
        # No point should be its own neighbor
        for i in range(len(coords)):
            assert i not in indices[i]
    
    def test_nearest_neighbors_include_self(self):
        """Test including self as neighbor."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        distances, indices = nearest_neighbors(coords, k=1, include_self=True)
        
        # Each point should be its own nearest neighbor
        for i in range(len(coords)):
            assert indices[i, 0] == i
            assert distances[i, 0] == 0.0
    
    def test_nearest_neighbors_no_distances(self):
        """Test returning only indices."""
        coords = np.random.rand(50, 2)
        
        indices = nearest_neighbors(coords, k=3, return_distances=False)
        
        assert isinstance(indices, np.ndarray)
        assert indices.shape == (50, 3)
    
    def test_nearest_neighbors_order(self):
        """Test that neighbors are ordered by distance."""
        coords = np.array([[0, 0], [1, 0], [3, 0], [2, 0]])
        
        distances, indices = nearest_neighbors(coords, k=3)
        
        # For point 0, neighbors should be ordered: 1 (dist=1), 3 (dist=2), 2 (dist=3)
        assert distances[0, 0] < distances[0, 1] < distances[0, 2]
    
    def test_nearest_neighbors_k_larger_than_n(self):
        """Test k larger than number of points."""
        coords = np.array([[0, 0], [1, 0]])
        
        distances, indices = nearest_neighbors(coords, k=5, include_self=False)
        
        # Should return only available neighbors
        assert indices.shape[1] <= 1


class TestRadiusNeighbors:
    """Tests for radius-based neighbor queries."""
    
    def test_radius_neighbors_basic(self):
        """Test basic radius query."""
        coords = np.array([[0, 0], [0.5, 0], [2, 0]])
        
        indices = radius_neighbors(coords, radius=1.0)
        
        assert isinstance(indices, list)
        assert len(indices) == 3
        
        # Point 0 should have point 1 as neighbor (distance 0.5)
        assert 1 in indices[0]
        # Point 0 should NOT have point 2 (distance 2.0)
        assert 2 not in indices[0]
    
    def test_radius_neighbors_exclude_self(self):
        """Test that self is excluded."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        indices = radius_neighbors(coords, radius=1.5)
        
        # No point should include itself
        for i in range(len(coords)):
            assert i not in indices[i]
    
    def test_radius_neighbors_with_distances(self):
        """Test returning distances."""
        coords = np.array([[0, 0], [0.5, 0], [1.5, 0]])
        
        distances, indices = radius_neighbors(coords, radius=1.0, return_distances=True)
        
        assert isinstance(distances, list)
        assert len(distances) == len(indices)
        
        # Point 0 has one neighbor within radius 1
        assert len(indices[0]) == 1
        np.testing.assert_almost_equal(distances[0][0], 0.5)
    
    def test_radius_neighbors_sorted(self):
        """Test sorting neighbors by distance."""
        coords = np.array([[0, 0], [0.5, 0], [0.3, 0], [0.8, 0]])
        
        indices = radius_neighbors(coords, radius=1.0, sort_results=True)
        
        # For point 0, neighbors should be sorted by distance
        # Nearest: point 2 (0.3), then 1 (0.5), then 3 (0.8)
        assert indices[0][0] == 2
        assert indices[0][1] == 1
        assert indices[0][2] == 3
    
    def test_radius_neighbors_no_neighbors(self):
        """Test with isolated points."""
        coords = np.array([[0, 0], [10, 0], [20, 0]])
        
        indices = radius_neighbors(coords, radius=1.0)
        
        # All points should have empty neighbor lists
        for idx_list in indices:
            assert len(idx_list) == 0


class TestNearestNeighborDistances:
    """Tests for nearest neighbor distance calculations."""
    
    def test_nearest_neighbor_distances_k1(self):
        """Test 1st nearest neighbor distances."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        distances = nearest_neighbor_distances(coords, k=1)
        
        assert len(distances) == 3
        assert distances[0] == 1.0  # Nearest to point 0 is point 1
        assert distances[1] == 1.0  # Point 1 has neighbors on both sides
        assert distances[2] == 1.0  # Nearest to point 2 is point 1
    
    def test_nearest_neighbor_distances_k2(self):
        """Test 2nd nearest neighbor distances."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        distances = nearest_neighbor_distances(coords, k=2)
        
        assert len(distances) == 3
        assert distances[0] == 2.0  # 2nd nearest to point 0 is point 2
    
    def test_mean_nearest_neighbor_distance(self):
        """Test mean nearest neighbor distance."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        mean_dist = mean_nearest_neighbor_distance(coords, k=1)
        
        assert mean_dist == 1.0


# =============================================================================
# Distance to Cell Types Tests
# =============================================================================

class TestDistanceToType:
    """Tests for distance to specific cell types."""
    
    def test_distance_to_type_basic(self):
        """Test distance to specific cell type."""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        types = np.array(['A', 'B', 'A', 'B'])
        data = SpatialTissueData(coords, types)
        
        distances = distance_to_type(data, 'B')
        
        assert len(distances) == 4
        assert distances[1] == 0.0  # Point 1 is type B
        assert distances[0] == 1.0  # Nearest B to point 0 is at distance 1
    
    def test_distance_to_type_from_subset(self):
        """Test distance from specific indices."""
        coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        types = np.array(['A', 'B', 'A', 'B'])
        data = SpatialTissueData(coords, types)
        
        # Distance from A cells to B cells
        a_indices = data.get_cells_by_type('A')
        distances = distance_to_type(data, 'B', from_indices=a_indices)
        
        assert len(distances) == 2  # Only 2 A cells
        assert distances[0] == 1.0  # A at position 0, nearest B at 1
        assert distances[1] == 1.0  # A at position 2, nearest B at 3
    
    def test_distance_to_type_nonexistent(self):
        """Test error when target type doesn't exist."""
        coords = np.array([[0, 0], [1, 0]])
        types = np.array(['A', 'A'])
        data = SpatialTissueData(coords, types)
        
        with pytest.raises(ValueError, match="No cells of type"):
            distance_to_type(data, 'B')


class TestDistanceToNearestDifferentType:
    """Tests for distance to nearest different cell type."""
    
    def test_distance_to_nearest_different_type(self):
        """Test distance to nearest different type."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        types = np.array(['A', 'B', 'A'])
        data = SpatialTissueData(coords, types)
        
        distances = distance_to_nearest_different_type(data)
        
        assert len(distances) == 3
        assert distances[0] == 1.0  # A to B
        assert distances[1] == 1.0  # B to A
        assert distances[2] == 1.0  # A to B
    
    def test_distance_to_nearest_different_type_single_type(self):
        """Test with only one cell type."""
        coords = np.array([[0, 0], [1, 0]])
        types = np.array(['A', 'A'])
        data = SpatialTissueData(coords, types)
        
        distances = distance_to_nearest_different_type(data)
        
        # All distances should be inf (no different type)
        assert np.all(np.isinf(distances))


class TestDistanceMatrixByType:
    """Tests for distance matrix between cell types."""
    
    def test_distance_matrix_by_type_mean(self):
        """Test mean distance matrix."""
        coords = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])
        types = np.array(['A', 'A', 'B', 'B'])
        data = SpatialTissueData(coords, types)
        
        dist_matrix = distance_matrix_by_type(data, metric='mean')
        
        # Within type A: distance is 1
        assert dist_matrix[('A', 'A')] == 1.0
        # Within type B: distance is 1
        assert dist_matrix[('B', 'B')] == 1.0
        # Between A and B: mean of [10, 11, 9, 10] = 10
        assert dist_matrix[('A', 'B')] == 10.0
    
    def test_distance_matrix_by_type_min(self):
        """Test minimum distance matrix."""
        coords = np.array([[0, 0], [1, 0], [10, 0]])
        types = np.array(['A', 'A', 'B'])
        data = SpatialTissueData(coords, types)
        
        dist_matrix = distance_matrix_by_type(data, metric='min')
        
        # Minimum distance from A to B
        assert dist_matrix[('A', 'B')] == 9.0  # min(10, 9)
    
    def test_distance_matrix_invalid_metric(self):
        """Test error with invalid metric."""
        coords = np.array([[0, 0], [1, 0]])
        types = np.array(['A', 'B'])
        data = SpatialTissueData(coords, types)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            distance_matrix_by_type(data, metric='invalid')


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestCentroid:
    """Tests for centroid calculations."""
    
    def test_centroid_basic(self):
        """Test basic centroid calculation."""
        coords = np.array([[0, 0], [2, 0], [1, 2]])
        
        c = centroid(coords)
        
        np.testing.assert_array_almost_equal(c, [1, 2/3])
    
    def test_centroid_single_point(self):
        """Test centroid of single point."""
        coords = np.array([[5, 3]])
        
        c = centroid(coords)
        
        np.testing.assert_array_almost_equal(c, [5, 3])


class TestCentroidByType:
    """Tests for centroid by cell type."""
    
    def test_centroid_by_type(self):
        """Test centroid calculation per type."""
        coords = np.array([[0, 0], [2, 0], [10, 10], [12, 10]])
        types = np.array(['A', 'A', 'B', 'B'])
        data = SpatialTissueData(coords, types)
        
        centroids = centroid_by_type(data)
        
        assert 'A' in centroids
        assert 'B' in centroids
        np.testing.assert_array_almost_equal(centroids['A'], [1, 0])
        np.testing.assert_array_almost_equal(centroids['B'], [11, 10])


class TestBoundingBox:
    """Tests for bounding box calculation."""
    
    def test_bounding_box_2d(self):
        """Test 2D bounding box."""
        coords = np.array([[0, 0], [5, 3], [2, 7]])
        
        min_coords, max_coords = bounding_box(coords)
        
        np.testing.assert_array_equal(min_coords, [0, 0])
        np.testing.assert_array_equal(max_coords, [5, 7])
    
    def test_bounding_box_3d(self):
        """Test 3D bounding box."""
        coords = np.array([[0, 0, 0], [1, 2, 3], [2, 1, 1]])
        
        min_coords, max_coords = bounding_box(coords)
        
        assert len(min_coords) == 3
        assert len(max_coords) == 3


class TestConvexHullArea:
    """Tests for convex hull area calculation."""
    
    def test_convex_hull_area_triangle(self):
        """Test area of triangle."""
        coords = np.array([[0, 0], [2, 0], [0, 2]])
        
        area = convex_hull_area(coords)
        
        # Triangle with base=2, height=2: area = 2
        assert area == 2.0
    
    def test_convex_hull_area_square(self):
        """Test area of square."""
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        
        area = convex_hull_area(coords)
        
        assert area == 1.0
    
    def test_convex_hull_area_insufficient_points(self):
        """Test with < 3 points."""
        coords = np.array([[0, 0], [1, 1]])
        
        area = convex_hull_area(coords)
        
        assert area == 0.0
    
    def test_convex_hull_area_3d_raises_error(self):
        """Test that 3D coordinates raise error."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        
        with pytest.raises(ValueError, match="2D coordinates"):
            convex_hull_area(coords)


class TestPointDensity:
    """Tests for point density calculation."""
    
    def test_point_density_bounding_box(self):
        """Test density using bounding box."""
        coords = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        density = point_density(coords, method='bounding_box')
        
        # 4 points in 10x10 box = 0.04 points per unit²
        assert density == 0.04
    
    def test_point_density_convex_hull(self):
        """Test density using convex hull."""
        coords = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        density = point_density(coords, method='convex_hull')
        
        # 4 points in square of area 100
        assert density == 0.04
    
    def test_point_density_invalid_method(self):
        """Test error with invalid method."""
        coords = np.array([[0, 0], [1, 1]])
        
        with pytest.raises(ValueError, match="Unknown method"):
            point_density(coords, method='invalid')


# =============================================================================
# Integration Tests
# =============================================================================

class TestSpatialIntegration:
    """Integration tests combining multiple spatial functions."""
    
    def test_complete_spatial_workflow(self, small_tissue):
        """Test complete spatial analysis workflow."""
        # 1. Build KD-tree
        tree = build_kdtree(small_tissue.coordinates)
        assert tree is not None
        
        # 2. Find nearest neighbors
        distances, indices = nearest_neighbors(small_tissue.coordinates, k=5)
        assert distances.shape == (small_tissue.n_cells, 5)
        
        # 3. Compute pairwise distances
        D = pairwise_distances(small_tissue.coordinates)
        assert D.shape == (small_tissue.n_cells, small_tissue.n_cells)
        
        # 4. Compute centroids
        centroids = centroid_by_type(small_tissue)
        assert len(centroids) == len(small_tissue.cell_types_unique)
        
        # 5. Distance to types
        for cell_type in small_tissue.cell_types_unique:
            dist = distance_to_type(small_tissue, cell_type)
            assert len(dist) == small_tissue.n_cells
    
    def test_neighborhood_analysis_workflow(self, simple_tissue_2d):
        """Test neighborhood-based analysis."""
        # Find neighbors within radius
        radius = 50.0
        neighbors = radius_neighbors(simple_tissue_2d.coordinates, radius=radius)
        
        # Compute neighborhood sizes
        neighborhood_sizes = np.array([len(n) for n in neighbors])
        
        assert len(neighborhood_sizes) == simple_tissue_2d.n_cells
        assert neighborhood_sizes.min() >= 0


# =============================================================================
# Performance Tests (optional, marked as slow)
# =============================================================================

@pytest.mark.slow
class TestSpatialPerformance:
    """Performance tests for spatial operations."""
    
    def test_kdtree_performance_large(self, large_tissue):
        """Test KD-tree performance with 10k cells."""
        import time
        
        start = time.time()
        tree = build_kdtree(large_tissue.coordinates)
        elapsed = time.time() - start
        
        # Should build in <0.1 seconds
        assert elapsed < 0.1
    
    def test_nearest_neighbors_performance(self, large_tissue):
        """Test k-NN performance with 10k cells."""
        import time
        
        start = time.time()
        distances, indices = nearest_neighbors(large_tissue.coordinates, k=10)
        elapsed = time.time() - start
        
        # Should complete in <1 second
        assert elapsed < 1.0
