"""
Pytest configuration and shared fixtures for spatialtissuepy tests.

This module provides reusable test data, fixtures, and utilities
for all test modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from spatialtissuepy import SpatialTissueData


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def sample_data_dir():
    """Path to example sample data."""
    return Path(__file__).parent.parent / 'examples' / 'sample_data'


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


# =============================================================================
# Random Seed Fixture
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)


# =============================================================================
# Simple Data Fixtures
# =============================================================================

@pytest.fixture
def simple_coords_2d():
    """Simple 2D coordinates for 100 cells."""
    np.random.seed(42)
    return np.random.rand(100, 2) * 1000


@pytest.fixture
def simple_coords_3d():
    """Simple 3D coordinates for 100 cells."""
    np.random.seed(42)
    return np.random.rand(100, 3) * 1000


@pytest.fixture
def simple_cell_types():
    """Cell types for 100 cells (4 types)."""
    np.random.seed(42)
    types = ['T_cell', 'Tumor', 'Stromal', 'Macrophage']
    return np.random.choice(types, 100)


@pytest.fixture
def simple_tissue_2d(simple_coords_2d, simple_cell_types):
    """Simple 2D SpatialTissueData with 100 cells."""
    return SpatialTissueData(simple_coords_2d, simple_cell_types)


@pytest.fixture
def simple_tissue_3d(simple_coords_3d, simple_cell_types):
    """Simple 3D SpatialTissueData with 100 cells."""
    return SpatialTissueData(simple_coords_3d, simple_cell_types)


# =============================================================================
# Size Variants
# =============================================================================

@pytest.fixture
def tiny_tissue():
    """Tiny tissue with 10 cells for quick tests."""
    np.random.seed(42)
    coords = np.random.rand(10, 2) * 100
    types = np.random.choice(['A', 'B'], 10)
    return SpatialTissueData(coords, types)


@pytest.fixture
def small_tissue():
    """Small tissue with 100 cells."""
    np.random.seed(42)
    coords = np.random.rand(100, 2) * 500
    types = np.random.choice(['A', 'B', 'C'], 100)
    return SpatialTissueData(coords, types)


@pytest.fixture
def medium_tissue():
    """Medium tissue with 1000 cells."""
    np.random.seed(42)
    coords = np.random.rand(1000, 2) * 1000
    types = np.random.choice(['A', 'B', 'C', 'D'], 1000)
    return SpatialTissueData(coords, types)


@pytest.fixture
def large_tissue():
    """Large tissue with 10,000 cells for performance tests."""
    np.random.seed(42)
    coords = np.random.rand(10000, 2) * 2000
    types = np.random.choice(['A', 'B', 'C', 'D', 'E'], 10000)
    return SpatialTissueData(coords, types)


# =============================================================================
# Multi-Sample Fixtures
# =============================================================================

@pytest.fixture
def multisample_tissue():
    """Multi-sample tissue with 2 samples, 100 cells each."""
    np.random.seed(42)
    coords = np.random.rand(200, 2) * 1000
    types = np.random.choice(['T_cell', 'Tumor', 'Stromal'], 200)
    samples = np.array(['sample_A'] * 100 + ['sample_B'] * 100)
    return SpatialTissueData(coords, types, sample_ids=samples)


@pytest.fixture
def multisample_cohort():
    """Multi-sample cohort with 5 samples."""
    np.random.seed(42)
    samples = []
    for i in range(5):
        n = np.random.randint(80, 120)
        coords = np.random.rand(n, 2) * 1000
        types = np.random.choice(['A', 'B', 'C'], n)
        sample_id = f'sample_{i}'
        sample = SpatialTissueData(coords, types, sample_ids=[sample_id] * n)
        samples.append(sample)
    
    # Combine all samples
    all_coords = np.vstack([s.coordinates for s in samples])
    all_types = np.concatenate([s.cell_types for s in samples])
    all_samples = np.concatenate([s.sample_ids for s in samples])
    
    return SpatialTissueData(all_coords, all_types, sample_ids=all_samples)


# =============================================================================
# Marker Data Fixtures
# =============================================================================

@pytest.fixture
def tissue_with_markers():
    """Tissue with marker expression data."""
    np.random.seed(42)
    coords = np.random.rand(100, 2) * 500
    types = np.random.choice(['T_cell', 'Tumor', 'Stromal'], 100)
    
    markers = pd.DataFrame({
        'CD3': np.random.rand(100),
        'CD8': np.random.rand(100),
        'PD1': np.random.rand(100),
        'Ki67': np.random.rand(100),
    })
    
    return SpatialTissueData(coords, types, markers=markers)


# =============================================================================
# Spatial Pattern Fixtures
# =============================================================================

@pytest.fixture
def clustered_pattern():
    """Tissue with clustered spatial pattern."""
    np.random.seed(42)
    
    # Create 3 clusters
    clusters = []
    for center_x, center_y in [(200, 200), (600, 200), (400, 600)]:
        n = 50
        x = np.random.normal(center_x, 30, n)
        y = np.random.normal(center_y, 30, n)
        clusters.append(np.column_stack([x, y]))
    
    coords = np.vstack(clusters)
    types = np.array(['Cluster_A'] * 50 + ['Cluster_B'] * 50 + ['Cluster_C'] * 50)
    
    return SpatialTissueData(coords, types)


@pytest.fixture
def random_pattern():
    """Tissue with random (CSR) spatial pattern."""
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, (500, 2))
    types = np.random.choice(['Type_A', 'Type_B'], 500)
    return SpatialTissueData(coords, types)


@pytest.fixture
def regular_grid():
    """Tissue with regular grid pattern."""
    x = np.arange(0, 100, 10)
    y = np.arange(0, 100, 10)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    types = np.array(['Grid'] * len(coords))
    return SpatialTissueData(coords, types)


# =============================================================================
# File Fixtures (CSV, JSON)
# =============================================================================

@pytest.fixture
def temp_csv_file(temp_dir, simple_tissue_2d):
    """Temporary CSV file with sample data."""
    filepath = temp_dir / "test_data.csv"
    
    df = pd.DataFrame({
        'x': simple_tissue_2d.coordinates[:, 0],
        'y': simple_tissue_2d.coordinates[:, 1],
        'cell_type': simple_tissue_2d.cell_types,
    })
    df.to_csv(filepath, index=False)
    
    return filepath


@pytest.fixture
def temp_csv_with_markers(temp_dir, tissue_with_markers):
    """Temporary CSV file with marker data."""
    filepath = temp_dir / "test_data_markers.csv"
    
    df = pd.DataFrame({
        'x': tissue_with_markers.coordinates[:, 0],
        'y': tissue_with_markers.coordinates[:, 1],
        'cell_type': tissue_with_markers.cell_types,
    })
    
    # Add marker columns
    for col in tissue_with_markers.marker_names:
        df[col] = tissue_with_markers.markers[col]
    
    df.to_csv(filepath, index=False)
    
    return filepath


@pytest.fixture
def temp_json_file(temp_dir, simple_tissue_2d):
    """Temporary JSON file with sample data."""
    filepath = temp_dir / "test_data.json"
    
    cells = []
    for i in range(simple_tissue_2d.n_cells):
        cell = {
            'x': float(simple_tissue_2d.coordinates[i, 0]),
            'y': float(simple_tissue_2d.coordinates[i, 1]),
            'cell_type': str(simple_tissue_2d.cell_types[i]),
        }
        cells.append(cell)
    
    data = {
        'cells': cells,
        'metadata': {
            'source': 'test',
            'n_cells': simple_tissue_2d.n_cells,
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return filepath


# =============================================================================
# Utility Functions
# =============================================================================

def assert_tissues_equal(tissue1, tissue2, check_markers=True):
    """
    Assert that two SpatialTissueData objects are equal.
    
    Parameters
    ----------
    tissue1, tissue2 : SpatialTissueData
        Tissues to compare
    check_markers : bool
        Whether to check marker data
    """
    assert tissue1.n_cells == tissue2.n_cells
    assert tissue1.n_dims == tissue2.n_dims
    
    np.testing.assert_array_almost_equal(
        tissue1.coordinates, 
        tissue2.coordinates
    )
    
    np.testing.assert_array_equal(
        tissue1.cell_types,
        tissue2.cell_types
    )
    
    if tissue1.is_multisample or tissue2.is_multisample:
        assert tissue1.is_multisample == tissue2.is_multisample
        np.testing.assert_array_equal(
            tissue1.sample_ids,
            tissue2.sample_ids
        )
    
    if check_markers:
        if tissue1.markers is not None and tissue2.markers is not None:
            pd.testing.assert_frame_equal(
                tissue1.markers,
                tissue2.markers
            )
        else:
            assert tissue1.markers is None and tissue2.markers is None


# Export utility for use in tests
__all__ = ['assert_tissues_equal']
