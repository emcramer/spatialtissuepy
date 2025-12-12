"""
Tests for spatialtissuepy.core module.

Tests SpatialTissueData, Cell, and validators.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from spatialtissuepy.core.spatial_data import SpatialTissueData
from spatialtissuepy.core.cell import Cell
from spatialtissuepy.core.validators import (
    ValidationError,
    validate_coordinates,
    validate_cell_types,
    validate_sample_ids,
    validate_marker_data,
    validate_positive_number,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_coords():
    """Simple 2D coordinates for 100 cells."""
    np.random.seed(42)
    return np.random.rand(100, 2) * 1000


@pytest.fixture
def simple_cell_types():
    """Cell types for 100 cells."""
    np.random.seed(42)
    types = ['T_cell', 'Tumor', 'Stromal', 'Macrophage']
    return np.random.choice(types, 100)


@pytest.fixture
def simple_data(simple_coords, simple_cell_types):
    """Simple SpatialTissueData object."""
    return SpatialTissueData(simple_coords, simple_cell_types)


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
# Validator Tests
# =============================================================================

class TestValidators:
    """Tests for validation functions."""

    def test_validate_coordinates_valid_2d(self):
        coords = np.random.rand(100, 2)
        result = validate_coordinates(coords)
        assert result.shape == (100, 2)
        assert result.dtype == np.float64

    def test_validate_coordinates_valid_3d(self):
        coords = np.random.rand(100, 3)
        result = validate_coordinates(coords)
        assert result.shape == (100, 3)

    def test_validate_coordinates_wrong_dims(self):
        coords = np.random.rand(100, 4)  # 4D not allowed
        with pytest.raises(ValidationError, match="2D or 3D"):
            validate_coordinates(coords)

    def test_validate_coordinates_nan(self):
        coords = np.random.rand(100, 2)
        coords[5, 0] = np.nan
        with pytest.raises(ValidationError, match="NaN"):
            validate_coordinates(coords)

    def test_validate_coordinates_nan_allowed(self):
        coords = np.random.rand(100, 2)
        coords[5, 0] = np.nan
        result = validate_coordinates(coords, allow_nan=True)
        assert np.isnan(result[5, 0])

    def test_validate_coordinates_empty(self):
        coords = np.array([]).reshape(0, 2)
        with pytest.raises(ValidationError, match="empty"):
            validate_coordinates(coords)

    def test_validate_cell_types_valid(self):
        types = ['A', 'B', 'A', 'C']
        result = validate_cell_types(types, 4)
        assert len(result) == 4
        assert result.dtype.kind == 'U'  # Unicode string

    def test_validate_cell_types_wrong_length(self):
        types = ['A', 'B', 'C']
        with pytest.raises(ValidationError, match="does not match"):
            validate_cell_types(types, 5)

    def test_validate_cell_types_empty_string(self):
        types = ['A', '', 'C']
        with pytest.raises(ValidationError, match="empty strings"):
            validate_cell_types(types, 3)

    def test_validate_sample_ids_none(self):
        result = validate_sample_ids(None, 10)
        assert result is None

    def test_validate_sample_ids_valid(self):
        ids = ['s1', 's1', 's2', 's2']
        result = validate_sample_ids(ids, 4)
        assert len(result) == 4

    def test_validate_marker_data_dataframe(self):
        markers = pd.DataFrame({'m1': [1.0, 2.0], 'm2': [3.0, 4.0]})
        result = validate_marker_data(markers, 2)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['m1', 'm2']

    def test_validate_marker_data_array(self):
        markers = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_marker_data(markers, 2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_validate_marker_data_non_numeric(self):
        markers = pd.DataFrame({'m1': [1.0, 2.0], 'm2': ['a', 'b']})
        with pytest.raises(ValidationError, match="non-numeric"):
            validate_marker_data(markers, 2)

    def test_validate_positive_number(self):
        assert validate_positive_number(5, "test") == 5
        assert validate_positive_number(0.5, "test") == 0.5

    def test_validate_positive_number_zero(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(0, "test")
        assert validate_positive_number(0, "test", allow_zero=True) == 0

    def test_validate_positive_number_negative(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_number(-1, "test")


# =============================================================================
# Cell Tests
# =============================================================================

class TestCell:
    """Tests for Cell class."""

    def test_cell_creation_2d(self):
        cell = Cell(0, x=100.5, y=200.3, cell_type='T_cell')
        assert cell.cell_id == 0
        assert cell.x == 100.5
        assert cell.y == 200.3
        assert cell.z is None
        assert cell.cell_type == 'T_cell'
        assert cell.ndim == 2

    def test_cell_creation_3d(self):
        cell = Cell(0, x=100, y=200, z=50, cell_type='Tumor')
        assert cell.z == 50
        assert cell.ndim == 3
        assert len(cell.coordinates) == 3

    def test_cell_coordinates(self):
        cell = Cell(0, x=10, y=20, cell_type='test')
        np.testing.assert_array_equal(cell.coordinates, [10, 20])

    def test_cell_distance_to(self):
        cell1 = Cell(0, x=0, y=0, cell_type='A')
        cell2 = Cell(1, x=3, y=4, cell_type='B')
        assert cell1.distance_to(cell2) == 5.0

    def test_cell_distance_different_dims(self):
        cell1 = Cell(0, x=0, y=0, cell_type='A')
        cell2 = Cell(1, x=0, y=0, z=0, cell_type='B')
        with pytest.raises(ValueError, match="different dimensionality"):
            cell1.distance_to(cell2)

    def test_cell_markers(self):
        cell = Cell(0, x=0, y=0, cell_type='T_cell', 
                    markers={'CD3': 0.8, 'CD8': 0.5})
        assert cell.get_marker('CD3') == 0.8
        assert np.isnan(cell.get_marker('CD4'))
        assert cell.get_marker('CD4', default=0.0) == 0.0

    def test_cell_to_dict(self):
        cell = Cell(0, x=10, y=20, cell_type='T_cell', sample_id='s1')
        d = cell.to_dict()
        assert d['x'] == 10
        assert d['y'] == 20
        assert d['cell_type'] == 'T_cell'
        assert d['sample_id'] == 's1'

    def test_cell_repr(self):
        cell = Cell(5, x=100.123, y=200.456, cell_type='Tumor')
        repr_str = repr(cell)
        assert '5' in repr_str
        assert 'Tumor' in repr_str
        assert '100.1' in repr_str

    def test_cell_equality(self):
        cell1 = Cell(5, x=0, y=0, cell_type='A')
        cell2 = Cell(5, x=100, y=100, cell_type='B')  # Same ID
        cell3 = Cell(6, x=0, y=0, cell_type='A')
        assert cell1 == cell2
        assert cell1 != cell3

    def test_cell_hash(self):
        cell1 = Cell(5, x=0, y=0, cell_type='A')
        cell2 = Cell(5, x=100, y=100, cell_type='B')
        assert hash(cell1) == hash(cell2)


# =============================================================================
# SpatialTissueData Tests
# =============================================================================

class TestSpatialTissueDataCreation:
    """Tests for SpatialTissueData creation."""

    def test_basic_creation(self, simple_coords, simple_cell_types):
        data = SpatialTissueData(simple_coords, simple_cell_types)
        assert data.n_cells == 100
        assert data.n_dims == 2

    def test_creation_with_markers(self):
        coords = np.random.rand(50, 2)
        types = ['A'] * 50
        markers = pd.DataFrame({'m1': np.random.rand(50)})
        data = SpatialTissueData(coords, types, markers=markers)
        assert data.marker_names == ['m1']

    def test_creation_with_sample_ids(self):
        coords = np.random.rand(100, 2)
        types = ['A'] * 100
        samples = ['s1'] * 50 + ['s2'] * 50
        data = SpatialTissueData(coords, types, sample_ids=samples)
        assert data.n_samples == 2
        assert data.is_multisample

    def test_creation_3d(self):
        coords = np.random.rand(50, 3)
        types = ['A'] * 50
        data = SpatialTissueData(coords, types)
        assert data.n_dims == 3


class TestSpatialTissueDataProperties:
    """Tests for SpatialTissueData properties."""

    def test_cell_type_counts(self, simple_data):
        counts = simple_data.cell_type_counts
        assert isinstance(counts, pd.Series)
        assert counts.sum() == 100

    def test_cell_types_unique(self, simple_data):
        unique = simple_data.cell_types_unique
        assert len(unique) == 4

    def test_bounds(self, simple_data):
        bounds = simple_data.bounds
        assert 'x' in bounds
        assert 'y' in bounds
        assert bounds['x'][0] < bounds['x'][1]

    def test_extent(self, simple_data):
        extent = simple_data.extent
        assert extent['x'] > 0
        assert extent['y'] > 0

    def test_coordinates_immutable(self, simple_data):
        coords = simple_data.coordinates
        original = coords.copy()
        coords[0, 0] = 99999
        # Original data should be unchanged
        assert simple_data.coordinates[0, 0] == original[0, 0]

    def test_multisample_properties(self, multisample_data):
        assert multisample_data.is_multisample
        assert multisample_data.n_samples == 2
        assert set(multisample_data.sample_ids_unique) == {'sample_A', 'sample_B'}

    def test_single_sample_properties(self, simple_data):
        assert not simple_data.is_multisample
        assert simple_data.n_samples == 1
        assert simple_data.sample_ids is None


class TestSpatialTissueDataAccess:
    """Tests for SpatialTissueData data access methods."""

    def test_get_cell(self, simple_data):
        cell = simple_data.get_cell(0)
        assert isinstance(cell, Cell)
        assert cell.cell_id == 0

    def test_get_cell_out_of_range(self, simple_data):
        with pytest.raises(IndexError):
            simple_data.get_cell(1000)

    def test_get_cells_by_type(self, simple_data):
        indices = simple_data.get_cells_by_type('T_cell')
        assert len(indices) > 0
        for idx in indices:
            assert simple_data.cell_types[idx] == 'T_cell'

    def test_get_cells_by_sample(self, multisample_data):
        indices = multisample_data.get_cells_by_sample('sample_A')
        assert len(indices) == 100

    def test_get_cells_by_sample_no_samples(self, simple_data):
        with pytest.raises(ValueError, match="No sample IDs"):
            simple_data.get_cells_by_sample('sample_A')

    def test_subset_by_indices(self, simple_data):
        indices = np.array([0, 1, 2, 3, 4])
        subset = simple_data.subset(indices=indices)
        assert subset.n_cells == 5

    def test_subset_by_cell_types(self, simple_data):
        subset = simple_data.subset(cell_types=['T_cell', 'Tumor'])
        assert set(subset.cell_types_unique) <= {'T_cell', 'Tumor'}

    def test_subset_by_sample_ids(self, multisample_data):
        subset = multisample_data.subset(sample_ids=['sample_A'])
        assert subset.n_cells == 100
