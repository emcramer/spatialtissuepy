"""
Tests for PhysiCell file I/O with real simulation data.

This module tests the PhysiCell integration module using the example
simulation data in examples/sample_data/example_physicell_sim.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.synthetic.physicell import (
    PhysiCellTimeStep,
    PhysiCellSimulation,
    read_physicell_timestep,
    read_physicell_simulation,
    discover_physicell_timesteps,
    parse_physicell_xml,
    parse_cells_mat,
    get_cell_type_mapping,
    is_alive,
    is_dead,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def example_physicell_dir():
    """Path to example PhysiCell simulation output folder."""
    path = Path(__file__).parent.parent / 'examples' / 'sample_data' / 'example_physicell_sim'
    if not path.exists():
        pytest.skip(f"Example PhysiCell data not found at {path}")
    return path


@pytest.fixture
def first_timestep_xml(example_physicell_dir):
    """Path to first timestep XML file."""
    xml_path = example_physicell_dir / 'output00000000.xml'
    if not xml_path.exists():
        pytest.skip(f"First timestep XML not found at {xml_path}")
    return xml_path


@pytest.fixture
def final_timestep_xml(example_physicell_dir):
    """Path to final timestep XML file."""
    # Find the highest numbered output file
    xml_files = sorted(example_physicell_dir.glob('output*.xml'))
    # Filter to numbered output files only
    numbered_files = [f for f in xml_files if f.stem.startswith('output') 
                      and f.stem[6:].isdigit()]
    if not numbered_files:
        pytest.skip("No numbered output XML files found")
    return numbered_files[-1]


@pytest.fixture
def loaded_simulation(example_physicell_dir):
    """Pre-loaded PhysiCell simulation."""
    return read_physicell_simulation(example_physicell_dir)


# =============================================================================
# Discovery Tests
# =============================================================================

class TestPhysiCellDiscovery:
    """Tests for discovering PhysiCell output files."""
    
    def test_discover_timesteps_finds_files(self, example_physicell_dir):
        """Test that discover_physicell_timesteps finds output files."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        
        assert len(timesteps) > 0, "Should find at least one timestep"
        
        # Each entry should be (index, xml_path, mat_path)
        for index, xml_path, mat_path in timesteps:
            assert isinstance(index, int)
            assert xml_path.exists(), f"XML file should exist: {xml_path}"
            assert mat_path.exists(), f"MAT file should exist: {mat_path}"
    
    def test_discover_timesteps_sorted(self, example_physicell_dir):
        """Test that discovered timesteps are sorted by index."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        
        indices = [idx for idx, _, _ in timesteps]
        assert indices == sorted(indices), "Timesteps should be sorted by index"
    
    def test_discover_timesteps_mat_file_naming(self, example_physicell_dir):
        """Test that both MAT file naming conventions are supported."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        
        assert len(timesteps) > 0, "Should find timesteps"
        
        # Check that MAT files have expected naming
        for index, xml_path, mat_path in timesteps:
            mat_name = mat_path.name
            # Should be either _cells.mat or _cells_physicell.mat
            assert ('_cells.mat' in mat_name or '_cells_physicell.mat' in mat_name), \
                f"MAT file has unexpected naming: {mat_name}"


# =============================================================================
# XML Parsing Tests
# =============================================================================

class TestPhysiCellXMLParsing:
    """Tests for parsing PhysiCell XML files."""
    
    def test_parse_xml_returns_metadata(self, first_timestep_xml):
        """Test that parse_physicell_xml returns metadata object."""
        metadata = parse_physicell_xml(first_timestep_xml)
        
        assert metadata is not None
        assert hasattr(metadata, 'time')
        assert hasattr(metadata, 'time_units')
        assert hasattr(metadata, 'space_units')
        assert hasattr(metadata, 'domain_min')
        assert hasattr(metadata, 'domain_max')
    
    def test_parse_xml_time_values(self, first_timestep_xml, final_timestep_xml):
        """Test that time values are parsed correctly."""
        meta_first = parse_physicell_xml(first_timestep_xml)
        meta_final = parse_physicell_xml(final_timestep_xml)
        
        # First timestep should have time close to 0
        assert meta_first.time >= 0
        
        # Final timestep should have larger time
        assert meta_final.time > meta_first.time
    
    def test_parse_xml_domain_bounds(self, first_timestep_xml):
        """Test that domain bounds are parsed."""
        metadata = parse_physicell_xml(first_timestep_xml)
        
        # Domain should have valid bounds
        assert len(metadata.domain_min) == 3
        assert len(metadata.domain_max) == 3
        
        # Max should be greater than min
        for i in range(3):
            assert metadata.domain_max[i] >= metadata.domain_min[i]
    
    def test_get_cell_type_mapping(self, first_timestep_xml):
        """Test getting cell type ID to name mapping."""
        mapping = get_cell_type_mapping(first_timestep_xml)
        
        assert isinstance(mapping, dict)
        # Should have at least default mapping
        assert len(mapping) > 0


# =============================================================================
# MAT File Parsing Tests
# =============================================================================

class TestPhysiCellMATParsing:
    """Tests for parsing PhysiCell MAT files."""
    
    def test_parse_cells_mat_structure(self, example_physicell_dir):
        """Test that parse_cells_mat returns expected structure."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        _, _, mat_path = timesteps[0]
        
        data = parse_cells_mat(mat_path)
        
        # Check required keys
        assert 'positions' in data
        assert 'cell_types' in data
        assert 'cell_type_ids' in data
        assert 'volumes' in data
        assert 'radii' in data
        assert 'phases' in data
        assert 'ids' in data
    
    def test_parse_cells_mat_positions_shape(self, example_physicell_dir):
        """Test that positions have correct shape."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        _, _, mat_path = timesteps[0]
        
        data = parse_cells_mat(mat_path)
        
        positions = data['positions']
        assert positions.ndim == 2
        assert positions.shape[1] == 3, "Should have x, y, z coordinates"
    
    def test_parse_cells_mat_consistent_lengths(self, example_physicell_dir):
        """Test that all arrays have consistent length."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        _, _, mat_path = timesteps[0]
        
        data = parse_cells_mat(mat_path)
        
        n_cells = len(data['positions'])
        assert len(data['cell_types']) == n_cells
        assert len(data['cell_type_ids']) == n_cells
        assert len(data['volumes']) == n_cells
        assert len(data['radii']) == n_cells
        assert len(data['phases']) == n_cells
        assert len(data['ids']) == n_cells
    
    def test_parse_cells_mat_positive_volumes(self, example_physicell_dir):
        """Test that cell volumes are positive."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        _, _, mat_path = timesteps[0]
        
        data = parse_cells_mat(mat_path)
        
        if len(data['volumes']) > 0:
            assert np.all(data['volumes'] > 0), "All volumes should be positive"


# =============================================================================
# Single Timestep Reading Tests
# =============================================================================

class TestReadPhysiCellTimestep:
    """Tests for reading individual PhysiCell timesteps."""
    
    def test_read_timestep_returns_object(self, first_timestep_xml):
        """Test that read_physicell_timestep returns PhysiCellTimeStep."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        assert isinstance(timestep, PhysiCellTimeStep)
    
    def test_read_timestep_has_cells(self, first_timestep_xml):
        """Test that timestep has cells."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        assert timestep.n_cells >= 0
    
    def test_read_timestep_time_index(self, first_timestep_xml):
        """Test that time index is extracted from filename."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        assert timestep.time_index == 0
    
    def test_read_timestep_to_spatial_data(self, first_timestep_xml):
        """Test converting timestep to SpatialTissueData."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        spatial_data = timestep.to_spatial_data()
        
        assert isinstance(spatial_data, SpatialTissueData)
        assert spatial_data.n_cells == timestep.n_cells
    
    def test_read_timestep_to_dataframe(self, first_timestep_xml):
        """Test converting timestep to DataFrame."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        df = timestep.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns
        assert 'cell_type' in df.columns
        assert len(df) == timestep.n_cells_total
    
    def test_read_timestep_exclude_dead_cells(self, example_physicell_dir):
        """Test that dead cells can be excluded."""
        timesteps = discover_physicell_timesteps(example_physicell_dir)
        _, xml_path, _ = timesteps[-1]  # Use later timestep more likely to have dead cells
        
        timestep_with_dead = read_physicell_timestep(xml_path, include_dead_cells=True)
        timestep_without_dead = read_physicell_timestep(xml_path, include_dead_cells=False)
        
        # Without dead should have <= cells as with dead
        assert timestep_without_dead.n_cells <= timestep_with_dead.n_cells


# =============================================================================
# Simulation Reading Tests
# =============================================================================

class TestReadPhysiCellSimulation:
    """Tests for reading complete PhysiCell simulations."""
    
    def test_read_simulation_returns_object(self, example_physicell_dir):
        """Test that read_physicell_simulation returns PhysiCellSimulation."""
        sim = read_physicell_simulation(example_physicell_dir)
        
        assert isinstance(sim, PhysiCellSimulation)
    
    def test_read_simulation_has_timesteps(self, example_physicell_dir):
        """Test that simulation has multiple timesteps."""
        sim = read_physicell_simulation(example_physicell_dir)
        
        assert sim.n_timesteps > 0
    
    def test_read_simulation_times_array(self, loaded_simulation):
        """Test that simulation has times array."""
        times = loaded_simulation.times
        
        assert len(times) == loaded_simulation.n_timesteps
        # Times should be monotonically increasing
        assert np.all(np.diff(times) >= 0)
    
    def test_read_simulation_get_timestep(self, loaded_simulation):
        """Test getting specific timestep by index."""
        timestep = loaded_simulation.get_timestep(0)
        
        assert isinstance(timestep, PhysiCellTimeStep)
        assert timestep.time_index == 0
    
    def test_read_simulation_get_timestep_by_time(self, loaded_simulation):
        """Test getting timestep by time value."""
        # Get time of middle timestep
        mid_idx = loaded_simulation.n_timesteps // 2
        target_time = loaded_simulation.times[mid_idx]
        
        timestep = loaded_simulation.get_timestep_by_time(target_time)
        
        # Should return timestep close to requested time
        assert abs(timestep.time - target_time) < 1.0  # Within 1 time unit
    
    def test_read_simulation_iteration(self, loaded_simulation):
        """Test iterating over simulation timesteps."""
        count = 0
        for timestep in loaded_simulation:
            assert isinstance(timestep, PhysiCellTimeStep)
            count += 1
            if count > 5:  # Don't iterate through all for speed
                break
        
        assert count > 0


# =============================================================================
# Statistics and Analysis Tests
# =============================================================================

class TestPhysiCellStatistics:
    """Tests for computing statistics on PhysiCell data."""
    
    def test_cell_counts_over_time(self, loaded_simulation):
        """Test computing cell counts over time."""
        # Sample a few timesteps to check
        indices = [0, loaded_simulation.n_timesteps // 2, loaded_simulation.n_timesteps - 1]
        
        for idx in indices:
            timestep = loaded_simulation.get_timestep(idx)
            assert timestep.n_cells >= 0
            assert timestep.n_cells_total >= timestep.n_cells
    
    def test_cell_counts_by_type(self, first_timestep_xml):
        """Test getting cell counts by type."""
        timestep = read_physicell_timestep(first_timestep_xml)
        
        counts = timestep.cell_counts_by_type()
        
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == timestep.n_cells
    
    def test_spatial_statistics_on_timestep(self, first_timestep_xml):
        """Test computing spatial statistics on PhysiCell timestep."""
        from spatialtissuepy.spatial import pairwise_distances
        
        timestep = read_physicell_timestep(first_timestep_xml)
        spatial_data = timestep.to_spatial_data()
        
        if spatial_data.n_cells > 1:
            # pairwise_distances expects coordinates array, not SpatialTissueData
            dists = pairwise_distances(spatial_data.coordinates)
            
            assert dists.shape == (spatial_data.n_cells, spatial_data.n_cells)
            # Diagonal should be 0
            np.testing.assert_array_almost_equal(np.diag(dists), 0)
    
    def test_ripley_statistics_on_timestep(self, first_timestep_xml):
        """Test computing Ripley's K on PhysiCell timestep."""
        from spatialtissuepy.statistics import ripleys_k
        
        timestep = read_physicell_timestep(first_timestep_xml)
        spatial_data = timestep.to_spatial_data()
        
        if spatial_data.n_cells > 10:
            # Use 2D projection for Ripley's K
            coords_2d = spatial_data.coordinates[:, :2]
            radii = np.linspace(10, 100, 5)
            
            K = ripleys_k(coords_2d, radii)
            
            assert len(K) == len(radii)
            assert np.all(K >= 0)  # K should be non-negative


# =============================================================================
# Visualization Tests
# =============================================================================

class TestPhysiCellVisualization:
    """Tests for visualizing PhysiCell data."""
    
    def test_plot_cell_positions_2d(self, first_timestep_xml, tmp_path):
        """Test plotting 2D cell positions from PhysiCell timestep."""
        timestep = read_physicell_timestep(first_timestep_xml)
        spatial_data = timestep.to_spatial_data()
        
        if spatial_data.n_cells == 0:
            pytest.skip("No cells in timestep")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot x, y positions
        coords = spatial_data.coordinates[:, :2]
        cell_types = spatial_data.cell_types
        unique_types = np.unique(cell_types)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            ax.scatter(coords[mask, 0], coords[mask, 1], 
                      c=[colors[i]], label=ct, alpha=0.6, s=10)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'PhysiCell Timestep (t={timestep.time:.1f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = tmp_path / 'physicell_positions_2d.png'
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        assert fig_path.exists()
    
    def test_plot_cell_positions_3d(self, first_timestep_xml, tmp_path):
        """Test plotting 3D cell positions from PhysiCell timestep."""
        timestep = read_physicell_timestep(first_timestep_xml)
        spatial_data = timestep.to_spatial_data()
        
        if spatial_data.n_cells == 0:
            pytest.skip("No cells in timestep")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        coords = spatial_data.coordinates
        cell_types = spatial_data.cell_types
        unique_types = np.unique(cell_types)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                      c=[colors[i]], label=ct, alpha=0.6, s=10)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(f'PhysiCell Timestep 3D (t={timestep.time:.1f})')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = tmp_path / 'physicell_positions_3d.png'
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        assert fig_path.exists()
    
    def test_plot_cell_growth_trajectory(self, loaded_simulation, tmp_path):
        """Test plotting cell count trajectory over time."""
        # Sample timesteps for faster test
        n_samples = min(20, loaded_simulation.n_timesteps)
        sample_indices = np.linspace(0, loaded_simulation.n_timesteps - 1, 
                                      n_samples, dtype=int)
        
        times = []
        cell_counts = []
        
        for idx in sample_indices:
            timestep = loaded_simulation.get_timestep(idx)
            times.append(timestep.time)
            cell_counts.append(timestep.n_cells)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(times, cell_counts, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Number of Cells')
        ax.set_title('Cell Population Growth Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = tmp_path / 'physicell_growth_trajectory.png'
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        
        assert fig_path.exists()


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestPhysiCellHelpers:
    """Tests for helper functions."""
    
    def test_is_alive_function(self):
        """Test is_alive helper function."""
        # Live phases (< 100)
        assert is_alive(0) == True
        assert is_alive(14) == True  # 'live' phase
        assert is_alive(5) == True   # G0 phase
        
        # Dead phases (>= 100)
        assert is_alive(100) == False  # apoptotic
        assert is_alive(103) == False  # necrotic
    
    def test_is_dead_function(self):
        """Test is_dead helper function."""
        # Live phases
        assert is_dead(0) == False
        assert is_dead(14) == False
        
        # Dead phases
        assert is_dead(100) == True
        assert is_dead(101) == True
        assert is_dead(104) == True  # debris


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestPhysiCellErrorHandling:
    """Tests for error handling in PhysiCell module."""
    
    def test_read_nonexistent_file(self, tmp_path):
        """Test error when reading non-existent file."""
        fake_path = tmp_path / 'nonexistent.xml'
        
        with pytest.raises(FileNotFoundError):
            read_physicell_timestep(fake_path)
    
    def test_read_nonexistent_folder(self, tmp_path):
        """Test error when reading non-existent folder."""
        fake_path = tmp_path / 'nonexistent_folder'
        
        with pytest.raises(FileNotFoundError):
            read_physicell_simulation(fake_path)
    
    def test_read_empty_folder(self, tmp_path):
        """Test error when reading folder with no PhysiCell files."""
        empty_dir = tmp_path / 'empty_output'
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No PhysiCell output files"):
            read_physicell_simulation(empty_dir)
    
    def test_get_timestep_out_of_range(self, loaded_simulation):
        """Test error when requesting out-of-range timestep."""
        with pytest.raises(IndexError):
            loaded_simulation.get_timestep(loaded_simulation.n_timesteps + 100)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhysiCellIntegration:
    """Integration tests for PhysiCell workflow."""
    
    def test_full_analysis_workflow(self, example_physicell_dir):
        """Test complete analysis workflow from loading to statistics."""
        # 1. Load simulation
        sim = read_physicell_simulation(example_physicell_dir)
        assert sim.n_timesteps > 0
        
        # 2. Get first and last timesteps
        first_ts = sim.get_timestep(0)
        last_ts = sim.get_timestep(sim.n_timesteps - 1)
        
        # 3. Convert to SpatialTissueData
        first_spatial = first_ts.to_spatial_data()
        last_spatial = last_ts.to_spatial_data()
        
        # 4. Compare basic properties
        print(f"First timestep: {first_ts.n_cells} cells at t={first_ts.time}")
        print(f"Last timestep: {last_ts.n_cells} cells at t={last_ts.time}")
        
        # 5. Cell types should be consistent (same types present)
        assert len(first_spatial.cell_types_unique) > 0
    
    def test_trajectory_dataframe_workflow(self, loaded_simulation):
        """Test creating trajectory DataFrame from simulation."""
        # Sample a few timesteps
        n_samples = min(5, loaded_simulation.n_timesteps)
        sample_indices = np.linspace(0, loaded_simulation.n_timesteps - 1, 
                                      n_samples, dtype=int)
        
        dfs = []
        for idx in sample_indices:
            ts = loaded_simulation.get_timestep(idx)
            df = ts.to_dataframe()
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        assert 'time' in combined_df.columns
        assert 'x' in combined_df.columns
        assert 'cell_type' in combined_df.columns
        assert len(combined_df['time'].unique()) == n_samples


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPhysiCellPerformance:
    """Performance tests for PhysiCell module."""
    
    def test_load_simulation_speed(self, example_physicell_dir):
        """Test that simulation loading is reasonably fast."""
        import time
        
        start = time.time()
        sim = read_physicell_simulation(example_physicell_dir)
        load_time = time.time() - start
        
        # Loading should complete within reasonable time
        # (adjust threshold based on expected data size)
        assert load_time < 60, f"Loading took too long: {load_time:.1f}s"
        print(f"Loaded {sim.n_timesteps} timesteps in {load_time:.2f}s")
    
    def test_timestep_access_speed(self, loaded_simulation):
        """Test that accessing timesteps is fast."""
        import time
        
        n_accesses = min(50, loaded_simulation.n_timesteps)
        
        start = time.time()
        for i in range(n_accesses):
            idx = i % loaded_simulation.n_timesteps
            ts = loaded_simulation.get_timestep(idx)
            _ = ts.n_cells  # Access a property
        access_time = time.time() - start
        
        per_access = access_time / n_accesses
        assert per_access < 1.0, f"Timestep access too slow: {per_access:.3f}s each"
        print(f"Average timestep access time: {per_access*1000:.1f}ms")
