"""
Tests for spatialtissuepy.core module (continued).

Additional tests for iteration, spatial queries, I/O, and neighborhoods.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path

from spatialtissuepy.core.spatial_data import SpatialTissueData
from spatialtissuepy.core.cell import Cell
from spatialtissuepy.core.validators import ValidationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Simple SpatialTissueData object."""
    np.random.seed(42)
    coords = np.random.rand(100, 2) * 1000
    types = np.random.choice(['T_cell', 'Tumor', 'Stromal', 'Macrophage'], 100)
    return SpatialTissueData(coords, types)


@pytest.fixture
def multisample_data():
    """Multi-sample SpatialTissueData object."""
    np.random.seed(42)
    coords = np.random.rand(200, 2) * 1000
    types = np.random.choice(['T_cell', 'Tumor', 'Stromal'], 200)
    samples = np.array(['sample_A'] * 100 + ['sample_B'] * 100)
    return SpatialTissueData(coords, types, sample_ids=samples)


@pytest.fixture
def data_with_markers():
    """SpatialTissueData with marker expression."""
    np.random.seed(42)
    coords = np.random.rand(50, 2) * 500
    types = np.random.choice(['T_cell', 'Tumor'], 50)
    markers = pd.DataFrame({
        'CD3': np.random.rand(50),
        'CD8': np.random.rand(50),
        'PD1': np.random.rand(50)
    })
    return SpatialTissueData(coords, types, markers=markers)


# =============================================================================
# Iteration Tests
# =============================================================================

class TestSpatialTissueDataIteration:
    """Tests for iteration methods."""

    def test_iter_cells(self, simple_data):
        cells = list(simple_data.iter_cells())
        assert len(cells) == 100
        assert all(isinstance(c, Cell) for c in cells)

    def test_iter_samples_single(self, simple_data):
        samples = list(simple_data.iter_samples())
        assert len(samples) == 1
        assert samples[0][0] == 'default'

    def test_iter_samples_multi(self, multisample_data):
        samples = list(multisample_data.iter_samples())
        assert len(samples) == 2
        sample_ids = [s[0] for s in samples]
        assert 'sample_A' in sample_ids
        assert 'sample_B' in sample_ids


# =============================================================================
# Spatial Query Tests
# =============================================================================

class TestSpatialTissueDataSpatialQueries:
    """Tests for spatial query methods."""

    def test_kdtree_lazy_creation(self, simple_data):
        assert simple_data._kdtree is None
        _ = simple_data.kdtree
        assert simple_data._kdtree is not None

    def test_query_radius(self, simple_data):
        point = simple_data.coordinates[0]
        indices = simple_data.query_radius(point, radius=100)
        assert len(indices) >= 1  # At least the point itself
        assert 0 in indices

    def test_query_radius_large(self, simple_data):
        # Large radius should include many cells
        point = np.array([500, 500])
        indices = simple_data.query_radius(point, radius=1000)
        assert len(indices) > 10

    def test_query_knn(self, simple_data):
        point = simple_data.coordinates[0]
        distances, indices = simple_data.query_knn(point, k=5)
        assert len(indices) == 5
        assert indices[0] == 0  # Nearest is itself
        assert distances[0] == 0.0

    def test_query_knn_sorted(self, simple_data):
        point = simple_data.coordinates[0]
        distances, indices = simple_data.query_knn(point, k=10)
        # Distances should be sorted
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))


# =============================================================================
# I/O Tests
# =============================================================================

class TestSpatialTissueDataIO:
    """Tests for I/O methods."""

    def test_to_dataframe(self, simple_data):
        df = simple_data.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'cell_type' in df.columns
        assert len(df) == 100

    def test_to_dataframe_with_markers(self, data_with_markers):
        df = data_with_markers.to_dataframe()
        assert 'CD3' in df.columns
        assert 'CD8' in df.columns
        assert 'PD1' in df.columns

    def test_to_dataframe_multisample(self, multisample_data):
        df = multisample_data.to_dataframe()
        assert 'sample_id' in df.columns
        assert set(df['sample_id'].unique()) == {'sample_A', 'sample_B'}

    def test_to_csv_and_from_csv(self, simple_data):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            simple_data.to_csv(filepath)
            loaded = SpatialTissueData.from_csv(filepath)
            
            assert loaded.n_cells == 100
            assert loaded.n_cell_types == simple_data.n_cell_types
            np.testing.assert_array_almost_equal(
                loaded.coordinates, simple_data.coordinates
            )
        finally:
            filepath.unlink()

    def test_from_dataframe(self, simple_data):
        df = simple_data.to_dataframe()
        loaded = SpatialTissueData.from_dataframe(df)
        assert loaded.n_cells == simple_data.n_cells
        assert loaded.n_cell_types == simple_data.n_cell_types

    def test_from_csv_missing_column(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write("x,y\n1,2\n3,4\n")
            filepath = Path(f.name)
        
        try:
            with pytest.raises(ValidationError, match="cell_type"):
                SpatialTissueData.from_csv(filepath)
        finally:
            filepath.unlink()


# =============================================================================
# Neighborhood Tests
# =============================================================================

class TestSpatialTissueDataNeighborhoods:
    """Tests for neighborhood management."""

    def test_add_neighborhoods(self, simple_data):
        # Fake neighborhood matrix
        n_types = simple_data.n_cell_types
        neigh = np.random.rand(100, n_types)
        data_with_neigh = simple_data.add_neighborhoods(neigh)
        
        assert data_with_neigh.has_neighborhoods
        assert data_with_neigh.neighborhoods.shape == (100, n_types)
        # Original should be unchanged (immutability)
        assert not simple_data.has_neighborhoods

    def test_add_neighborhoods_wrong_size(self, simple_data):
        neigh = np.random.rand(50, 4)  # Wrong number of rows
        with pytest.raises(ValidationError):
            simple_data.add_neighborhoods(neigh)

    def test_neighborhoods_with_params(self, simple_data):
        neigh = np.random.rand(100, 4)
        params = {'method': 'knn', 'k': 30}
        data_with_neigh = simple_data.add_neighborhoods(neigh, params=params)
        
        assert data_with_neigh._neighborhood_params == params


# =============================================================================
# Dunder Method Tests
# =============================================================================

class TestSpatialTissueDataDunder:
    """Tests for special methods."""

    def test_len(self, simple_data):
        assert len(simple_data) == 100

    def test_repr(self, simple_data):
        repr_str = repr(simple_data)
        assert '100 cells' in repr_str
        assert '4 cell types' in repr_str

    def test_repr_multisample(self, multisample_data):
        repr_str = repr(multisample_data)
        assert '2 samples' in repr_str

    def test_repr_with_markers(self, data_with_markers):
        repr_str = repr(data_with_markers)
        assert '3 markers' in repr_str

    def test_str(self, simple_data):
        str_repr = str(simple_data)
        assert 'SpatialTissueData' in str_repr
        assert 'Cells: 100' in str_repr
        assert 'Dimensions: 2D' in str_repr


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_cell(self):
        coords = np.array([[100.0, 200.0]])
        types = ['T_cell']
        data = SpatialTissueData(coords, types)
        
        assert data.n_cells == 1
        assert data.n_cell_types == 1

    def test_single_cell_type(self):
        coords = np.random.rand(100, 2)
        types = ['T_cell'] * 100
        data = SpatialTissueData(coords, types)
        
        assert data.n_cell_types == 1
        assert data.cell_type_counts['T_cell'] == 100

    def test_3d_data(self):
        coords = np.random.rand(50, 3) * 100
        types = ['A', 'B'] * 25
        data = SpatialTissueData(coords, types)
        
        assert data.n_dims == 3
        assert 'z' in data.bounds
        
        cell = data.get_cell(0)
        assert cell.z is not None
        assert cell.ndim == 3

    def test_unicode_cell_types(self):
        coords = np.random.rand(10, 2)
        types = ['T细胞', 'Célula', 'κύτταρο'] * 3 + ['cell']
        data = SpatialTissueData(coords, types)
        
        assert data.n_cell_types == 4

    def test_very_close_cells(self):
        # All cells at nearly the same location
        coords = np.random.rand(100, 2) * 0.001
        types = ['A'] * 100
        data = SpatialTissueData(coords, types)
        
        # All cells should be in a very small neighborhood
        indices = data.query_radius(coords[0], radius=0.1)
        assert len(indices) == 100
