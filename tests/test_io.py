"""
Tests for spatialtissuepy.io module.

Tests reading and writing spatial data in various formats (CSV, JSON, AnnData).
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.io import read_csv, read_json, write_csv, write_json
from spatialtissuepy.core.validators import ValidationError
from tests.conftest import assert_tissues_equal


# =============================================================================
# CSV Reader Tests
# =============================================================================

class TestReadCSV:
    """Tests for read_csv function."""
    
    def test_read_csv_basic(self, temp_csv_file):
        """Test basic CSV reading."""
        data = read_csv(temp_csv_file)
        
        assert data.n_cells == 100
        assert data.n_dims == 2
        assert len(data.cell_types_unique) == 4
    
    def test_read_csv_custom_columns(self, temp_dir):
        """Test CSV reading with custom column names."""
        # Create CSV with custom column names
        filepath = temp_dir / "custom_cols.csv"
        df = pd.DataFrame({
            'X_centroid': np.random.rand(50),
            'Y_centroid': np.random.rand(50),
            'phenotype': ['A'] * 25 + ['B'] * 25,
        })
        df.to_csv(filepath, index=False)
        
        # Read with custom column mapping
        data = read_csv(
            filepath,
            x_col='X_centroid',
            y_col='Y_centroid',
            celltype_col='phenotype'
        )
        
        assert data.n_cells == 50
        assert set(data.cell_types_unique) == {'A', 'B'}
    
    def test_read_csv_3d(self, temp_dir):
        """Test reading 3D coordinates from CSV."""
        filepath = temp_dir / "data_3d.csv"
        df = pd.DataFrame({
            'x': np.random.rand(50),
            'y': np.random.rand(50),
            'z': np.random.rand(50),
            'cell_type': ['A'] * 50,
        })
        df.to_csv(filepath, index=False)
        
        data = read_csv(filepath, z_col='z')
        
        assert data.n_dims == 3
        assert data.coordinates.shape == (50, 3)
    
    def test_read_csv_with_markers(self, temp_csv_with_markers):
        """Test reading CSV with marker expression data."""
        data = read_csv(temp_csv_with_markers)
        
        assert data.markers is not None
        assert len(data.marker_names) == 4
        assert 'CD3' in data.marker_names
        assert 'Ki67' in data.marker_names
    
    def test_read_csv_with_sample_ids(self, temp_dir):
        """Test reading CSV with sample IDs."""
        filepath = temp_dir / "multisample.csv"
        df = pd.DataFrame({
            'x': np.random.rand(100),
            'y': np.random.rand(100),
            'cell_type': ['A'] * 100,
            'patient_id': ['patient_1'] * 50 + ['patient_2'] * 50,
        })
        df.to_csv(filepath, index=False)
        
        data = read_csv(filepath, sample_col='patient_id')
        
        assert data.is_multisample
        assert data.n_samples == 2
        assert set(data.sample_ids_unique) == {'patient_1', 'patient_2'}
    
    def test_read_csv_missing_required_columns(self, temp_dir):
        """Test that missing required columns raises error."""
        filepath = temp_dir / "bad_csv.csv"
        df = pd.DataFrame({
            'x': [1, 2, 3],
            # Missing 'y' and 'cell_type'
        })
        df.to_csv(filepath, index=False)
        
        with pytest.raises((ValidationError, KeyError)):
            read_csv(filepath)
    
    def test_read_csv_empty_file(self, temp_dir):
        """Test that empty CSV raises error."""
        filepath = temp_dir / "empty.csv"
        df = pd.DataFrame(columns=['x', 'y', 'cell_type'])
        df.to_csv(filepath, index=False)
        
        with pytest.raises(ValidationError):
            read_csv(filepath)
    
    def test_read_csv_with_nan_coordinates(self, temp_dir):
        """Test that NaN coordinates raise error."""
        filepath = temp_dir / "nan_coords.csv"
        df = pd.DataFrame({
            'x': [1.0, np.nan, 3.0],
            'y': [1.0, 2.0, 3.0],
            'cell_type': ['A', 'A', 'A'],
        })
        df.to_csv(filepath, index=False)
        
        with pytest.raises(ValidationError, match="NaN"):
            read_csv(filepath)
    
    def test_read_csv_explicit_marker_cols(self, temp_dir):
        """Test specifying marker columns explicitly."""
        filepath = temp_dir / "markers.csv"
        df = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20),
            'cell_type': ['A'] * 20,
            'CD3': np.random.rand(20),
            'CD8': np.random.rand(20),
            'noise_column': ['text'] * 20,  # Non-numeric, should be excluded
        })
        df.to_csv(filepath, index=False)
        
        # Explicitly specify markers
        data = read_csv(filepath, marker_cols=['CD3', 'CD8'])
        
        assert set(data.marker_names) == {'CD3', 'CD8'}
        assert 'noise_column' not in data.marker_names


# =============================================================================
# CSV Writer Tests
# =============================================================================

class TestWriteCSV:
    """Tests for write_csv function."""
    
    def test_write_csv_basic(self, simple_tissue_2d, temp_dir):
        """Test basic CSV writing."""
        filepath = temp_dir / "output.csv"
        write_csv(simple_tissue_2d, filepath)
        
        assert filepath.exists()
        
        # Read back and verify
        data = read_csv(filepath)
        assert_tissues_equal(data, simple_tissue_2d, check_markers=False)
    
    def test_write_csv_with_markers(self, tissue_with_markers, temp_dir):
        """Test writing CSV with markers."""
        filepath = temp_dir / "output_markers.csv"
        write_csv(tissue_with_markers, filepath)
        
        # Read back
        data = read_csv(filepath)
        assert_tissues_equal(data, tissue_with_markers, check_markers=True)
    
    def test_write_csv_multisample(self, multisample_tissue, temp_dir):
        """Test writing multi-sample CSV."""
        filepath = temp_dir / "output_multisample.csv"
        write_csv(multisample_tissue, filepath)
        
        # Read back
        data = read_csv(filepath, sample_col='sample_id')
        assert_tissues_equal(data, multisample_tissue, check_markers=False)
    
    def test_write_csv_3d(self, simple_tissue_3d, temp_dir):
        """Test writing 3D coordinates."""
        filepath = temp_dir / "output_3d.csv"
        write_csv(simple_tissue_3d, filepath)
        
        # Read back
        data = read_csv(filepath, z_col='z')
        assert data.n_dims == 3
        np.testing.assert_array_almost_equal(
            data.coordinates,
            simple_tissue_3d.coordinates
        )
    
    def test_roundtrip_csv(self, tissue_with_markers, temp_dir):
        """Test complete roundtrip: write → read → verify equality."""
        filepath = temp_dir / "roundtrip.csv"
        
        # Write
        write_csv(tissue_with_markers, filepath)
        
        # Read
        data = read_csv(filepath)
        
        # Verify
        assert_tissues_equal(data, tissue_with_markers, check_markers=True)


# =============================================================================
# JSON Reader Tests
# =============================================================================

class TestReadJSON:
    """Tests for read_json function."""
    
    def test_read_json_basic(self, temp_json_file):
        """Test basic JSON reading."""
        data = read_json(temp_json_file)
        
        assert data.n_cells == 100
        assert data.n_dims == 2
    
    def test_read_json_custom_keys(self, temp_dir):
        """Test JSON reading with custom key names."""
        filepath = temp_dir / "custom_keys.json"
        
        cells = [
            {'X': 10, 'Y': 20, 'phenotype': 'A'},
            {'X': 30, 'Y': 40, 'phenotype': 'B'},
        ]
        
        with open(filepath, 'w') as f:
            json.dump({'cells': cells}, f)
        
        data = read_json(
            filepath,
            x_key='X',
            y_key='Y',
            celltype_key='phenotype'
        )
        
        assert data.n_cells == 2
        assert data.coordinates[0, 0] == 10
    
    def test_read_json_flat_array(self, temp_dir):
        """Test reading JSON with flat array (no 'cells' wrapper)."""
        filepath = temp_dir / "flat.json"
        
        cells = [
            {'x': 10, 'y': 20, 'cell_type': 'A'},
            {'x': 30, 'y': 40, 'cell_type': 'B'},
        ]
        
        with open(filepath, 'w') as f:
            json.dump(cells, f)
        
        data = read_json(filepath)
        assert data.n_cells == 2
    
    def test_read_json_3d(self, temp_dir):
        """Test reading 3D coordinates from JSON."""
        filepath = temp_dir / "3d.json"
        
        cells = [
            {'x': 10, 'y': 20, 'z': 5, 'cell_type': 'A'},
            {'x': 30, 'y': 40, 'z': 15, 'cell_type': 'B'},
        ]
        
        with open(filepath, 'w') as f:
            json.dump({'cells': cells}, f)
        
        data = read_json(filepath, z_key='z')
        
        assert data.n_dims == 3
        assert data.coordinates.shape == (2, 3)
    
    def test_read_json_with_markers(self, temp_dir):
        """Test reading JSON with marker data."""
        filepath = temp_dir / "markers.json"
        
        cells = [
            {'x': 10, 'y': 20, 'cell_type': 'A', 'CD3': 0.8, 'CD8': 0.5},
            {'x': 30, 'y': 40, 'cell_type': 'B', 'CD3': 0.2, 'CD8': 0.9},
        ]
        
        with open(filepath, 'w') as f:
            json.dump({'cells': cells}, f)
        
        data = read_json(filepath)
        
        assert data.markers is not None
        assert set(data.marker_names) == {'CD3', 'CD8'}
    
    def test_read_json_with_sample_ids(self, temp_dir):
        """Test reading JSON with sample IDs."""
        filepath = temp_dir / "samples.json"
        
        cells = [
            {'x': 10, 'y': 20, 'cell_type': 'A', 'sample': 's1'},
            {'x': 30, 'y': 40, 'cell_type': 'B', 'sample': 's2'},
        ]
        
        with open(filepath, 'w') as f:
            json.dump({'cells': cells}, f)
        
        data = read_json(filepath, sample_key='sample')
        
        assert data.is_multisample
        assert data.n_samples == 2
    
    def test_read_json_missing_coordinates(self, temp_dir):
        """Test that missing coordinates raise error."""
        filepath = temp_dir / "bad.json"
        
        cells = [
            {'x': 10},  # Missing y
            {'y': 20},  # Missing x
        ]
        
        with open(filepath, 'w') as f:
            json.dump({'cells': cells}, f)
        
        with pytest.raises(ValidationError, match="missing coordinates"):
            read_json(filepath)
    
    def test_read_json_empty(self, temp_dir):
        """Test that empty JSON raises error."""
        filepath = temp_dir / "empty.json"
        
        with open(filepath, 'w') as f:
            json.dump({'cells': []}, f)
        
        with pytest.raises(ValidationError, match="No cells"):
            read_json(filepath)
    
    def test_read_json_metadata(self, temp_dir):
        """Test that metadata is preserved."""
        filepath = temp_dir / "meta.json"
        
        data_dict = {
            'cells': [
                {'x': 10, 'y': 20, 'cell_type': 'A'},
            ],
            'metadata': {
                'tissue': 'lung',
                'patient': 'P001',
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_dict, f)
        
        data = read_json(filepath)
        
        assert 'tissue' in data.metadata
        assert data.metadata['tissue'] == 'lung'


# =============================================================================
# JSON Writer Tests
# =============================================================================

class TestWriteJSON:
    """Tests for write_json function."""
    
    def test_write_json_basic(self, simple_tissue_2d, temp_dir):
        """Test basic JSON writing."""
        filepath = temp_dir / "output.json"
        write_json(simple_tissue_2d, filepath)
        
        assert filepath.exists()
        
        # Read back and verify
        data = read_json(filepath)
        assert_tissues_equal(data, simple_tissue_2d, check_markers=False)
    
    def test_write_json_with_markers(self, tissue_with_markers, temp_dir):
        """Test writing JSON with markers."""
        filepath = temp_dir / "output_markers.json"
        write_json(tissue_with_markers, filepath)
        
        # Read back
        data = read_json(filepath)
        assert_tissues_equal(data, tissue_with_markers, check_markers=True)
    
    def test_write_json_multisample(self, multisample_tissue, temp_dir):
        """Test writing multi-sample JSON."""
        filepath = temp_dir / "output_multisample.json"
        write_json(multisample_tissue, filepath)
        
        # Read back
        data = read_json(filepath, sample_key='sample_id')
        assert_tissues_equal(data, multisample_tissue, check_markers=False)
    
    def test_roundtrip_json(self, tissue_with_markers, temp_dir):
        """Test complete roundtrip: write → read → verify equality."""
        filepath = temp_dir / "roundtrip.json"
        
        # Write
        write_json(tissue_with_markers, filepath)
        
        # Read
        data = read_json(filepath)
        
        # Verify
        assert_tissues_equal(data, tissue_with_markers, check_markers=True)


# =============================================================================
# Real Data Tests
# =============================================================================

class TestRealDataLoading:
    """Tests with real sample data from examples/sample_data/."""
    
    def test_load_sample_csv(self, sample_data_dir):
        """Test loading the included sample CSV data."""
        filepath = sample_data_dir / 'random_sample_data.csv'
        
        if not filepath.exists():
            pytest.skip("Sample data file not found")
        
        data = read_csv(filepath)
        
        assert data.n_cells > 0
        assert data.n_dims == 2
        assert len(data.cell_types_unique) > 0


# =============================================================================
# Cross-Format Tests
# =============================================================================

class TestCrossFormat:
    """Test data consistency across different file formats."""
    
    def test_csv_json_equivalence(self, simple_tissue_2d, temp_dir):
        """Test that CSV and JSON produce equivalent data."""
        csv_path = temp_dir / "data.csv"
        json_path = temp_dir / "data.json"
        
        # Write in both formats
        write_csv(simple_tissue_2d, csv_path)
        write_json(simple_tissue_2d, json_path)
        
        # Read back
        data_csv = read_csv(csv_path)
        data_json = read_json(json_path)
        
        # Compare
        assert_tissues_equal(data_csv, data_json, check_markers=False)
