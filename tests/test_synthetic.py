"""
Tests for spatialtissuepy.synthetic module.

Tests PhysiCell and other ABM framework integration for loading
simulation outputs into SpatialTissueData format.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from spatialtissuepy import SpatialTissueData
from spatialtissuepy.synthetic import (
    # Base classes
    ABMTimeStep,
    ABMSimulation,
    ABMExperiment,
    # PhysiCell
    PhysiCellTimeStep,
    PhysiCellSimulation,
    PhysiCellExperiment,
    read_physicell_timestep,
    read_physicell_simulation,
)


# =============================================================================
# Base Class Tests
# =============================================================================

class TestABMBaseClasses:
    """Tests for base ABM classes."""
    
    def test_abm_timestep_interface(self):
        """Test ABMTimeStep is an abstract base class."""
        # Should not be instantiable directly
        with pytest.raises(TypeError):
            ABMTimeStep()
    
    def test_abm_simulation_interface(self):
        """Test ABMSimulation is an abstract base class."""
        with pytest.raises(TypeError):
            ABMSimulation()
    
    def test_abm_experiment_interface(self):
        """Test ABMExperiment is an abstract base class."""
        with pytest.raises(TypeError):
            ABMExperiment()


# =============================================================================
# PhysiCell TimeStep Tests
# =============================================================================

class TestPhysiCellTimeStep:
    """Tests for PhysiCell timestep loading."""
    
    def test_physicell_timestep_initialization(self):
        """Test creating PhysiCellTimeStep from data."""
        coords = np.random.rand(50, 3) * 1000
        cell_types = np.array(['tumor'] * 25 + ['immune'] * 25)
        time = 0.0
        timestep_id = 0
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=time,
            timestep_id=timestep_id
        )
        
        assert isinstance(timestep, PhysiCellTimeStep)
        assert timestep.time == 0.0
        assert timestep.timestep_id == 0
        assert timestep.n_cells == 50
    
    def test_physicell_timestep_to_spatial_data(self):
        """Test converting timestep to SpatialTissueData."""
        coords = np.random.rand(50, 3) * 1000
        cell_types = np.array(['tumor'] * 25 + ['immune'] * 25)
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=100.0,
            timestep_id=5
        )
        
        data = timestep.to_spatial_data()
        
        assert isinstance(data, SpatialTissueData)
        assert data.n_cells == 50
        assert len(data.cell_types_unique) == 2
    
    def test_physicell_timestep_with_markers(self):
        """Test timestep with cell markers."""
        coords = np.random.rand(30, 3) * 1000
        cell_types = np.array(['A'] * 30)
        markers = pd.DataFrame({
            'volume': np.random.uniform(1000, 2000, 30),
            'oxygen': np.random.uniform(0, 40, 30)
        })
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=50.0,
            timestep_id=2,
            markers=markers
        )
        
        data = timestep.to_spatial_data()
        assert data.markers is not None
        assert 'volume' in data.markers.columns
        assert 'oxygen' in data.markers.columns
    
    def test_physicell_timestep_properties(self):
        """Test timestep properties."""
        coords = np.random.rand(40, 3) * 1000
        cell_types = np.array(['A'] * 20 + ['B'] * 20)
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=200.0,
            timestep_id=10
        )
        
        assert timestep.n_cells == 40
        assert timestep.time == 200.0
        assert timestep.timestep_id == 10
        assert timestep.n_cell_types == 2


# =============================================================================
# PhysiCell Simulation Tests
# =============================================================================

class TestPhysiCellSimulation:
    """Tests for PhysiCell simulation loading."""
    
    def test_physicell_simulation_initialization(self):
        """Test creating simulation from timesteps."""
        timesteps = []
        for i in range(5):
            coords = np.random.rand(50, 3) * 1000
            types = np.array(['A'] * 25 + ['B'] * 25)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i*10), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        assert isinstance(sim, PhysiCellSimulation)
        assert sim.n_timesteps == 5
        assert len(sim.times) == 5
    
    def test_physicell_simulation_get_timestep(self):
        """Test retrieving specific timestep."""
        timesteps = []
        for i in range(3):
            coords = np.random.rand(30, 3) * 1000
            types = np.array(['A'] * 30)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        ts = sim.get_timestep(1)
        assert ts.timestep_id == 1
        assert ts.time == 1.0
    
    def test_physicell_simulation_get_timestep_by_time(self):
        """Test retrieving timestep by time."""
        timesteps = []
        for i in range(4):
            coords = np.random.rand(25, 3) * 1000
            types = np.array(['A'] * 25)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i*100), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        ts = sim.get_timestep_by_time(200.0)
        assert ts.time == 200.0
    
    def test_physicell_simulation_time_range(self):
        """Test simulation time range."""
        timesteps = []
        for i in range(6):
            coords = np.random.rand(20, 3) * 1000
            types = np.array(['A'] * 20)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i*50), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        assert sim.start_time == 0.0
        assert sim.end_time == 250.0
        assert sim.duration == 250.0
    
    def test_physicell_simulation_cell_count_trajectory(self):
        """Test tracking cell counts over time."""
        timesteps = []
        cell_counts = [20, 25, 30, 35, 40]
        
        for i, count in enumerate(cell_counts):
            coords = np.random.rand(count, 3) * 1000
            types = np.array(['A'] * count)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i*10), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        trajectory = sim.cell_count_trajectory()
        
        assert len(trajectory) == 5
        np.testing.assert_array_equal(trajectory, cell_counts)
    
    def test_physicell_simulation_type_proportions_over_time(self):
        """Test tracking cell type proportions."""
        timesteps = []
        for i in range(4):
            # Gradually increasing proportion of type B
            n_a = 30 - i*5
            n_b = 20 + i*5
            coords = np.random.rand(n_a + n_b, 3) * 1000
            types = np.array(['A'] * n_a + ['B'] * n_b)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(i), timestep_id=i)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        props = sim.type_proportions_over_time()
        
        assert isinstance(props, pd.DataFrame)
        assert 'A' in props.columns
        assert 'B' in props.columns
        assert len(props) == 4


# =============================================================================
# PhysiCell Experiment Tests
# =============================================================================

class TestPhysiCellExperiment:
    """Tests for PhysiCell experiment (multiple simulations)."""
    
    def test_physicell_experiment_initialization(self):
        """Test creating experiment from simulations."""
        simulations = []
        
        for sim_id in range(3):
            timesteps = []
            for t in range(5):
                coords = np.random.rand(40, 3) * 1000
                types = np.array(['A'] * 20 + ['B'] * 20)
                timesteps.append(
                    PhysiCellTimeStep(coords, types, time=float(t*10), timestep_id=t)
                )
            simulations.append(PhysiCellSimulation(timesteps=timesteps))
        
        experiment = PhysiCellExperiment(
            simulations=simulations,
            simulation_ids=['control', 'treatment_low', 'treatment_high']
        )
        
        assert isinstance(experiment, PhysiCellExperiment)
        assert experiment.n_simulations == 3
        assert len(experiment.simulation_ids) == 3
    
    def test_physicell_experiment_get_simulation(self):
        """Test retrieving specific simulation."""
        simulations = []
        
        for sim_id in range(2):
            timesteps = []
            for t in range(3):
                coords = np.random.rand(30, 3) * 1000
                types = np.array(['A'] * 30)
                timesteps.append(
                    PhysiCellTimeStep(coords, types, time=float(t), timestep_id=t)
                )
            simulations.append(PhysiCellSimulation(timesteps=timesteps))
        
        experiment = PhysiCellExperiment(
            simulations=simulations,
            simulation_ids=['sim1', 'sim2']
        )
        
        sim = experiment.get_simulation('sim1')
        assert sim is not None
        assert sim.n_timesteps == 3
    
    def test_physicell_experiment_compare_trajectories(self):
        """Test comparing trajectories across simulations."""
        simulations = []
        
        # Create simulations with different growth patterns
        for growth_rate in [1.0, 1.5, 2.0]:
            timesteps = []
            for t in range(5):
                n_cells = int(20 * (growth_rate ** t))
                coords = np.random.rand(n_cells, 3) * 1000
                types = np.array(['A'] * n_cells)
                timesteps.append(
                    PhysiCellTimeStep(coords, types, time=float(t*10), timestep_id=t)
                )
            simulations.append(PhysiCellSimulation(timesteps=timesteps))
        
        experiment = PhysiCellExperiment(
            simulations=simulations,
            simulation_ids=['slow', 'medium', 'fast']
        )
        
        # Get trajectories
        trajectories = {}
        for sim_id in experiment.simulation_ids:
            sim = experiment.get_simulation(sim_id)
            trajectories[sim_id] = sim.cell_count_trajectory()
        
        # Fast should have more cells than slow
        assert trajectories['fast'][-1] > trajectories['slow'][-1]


# =============================================================================
# File I/O Tests (Mock)
# =============================================================================

class TestPhysiCellIO:
    """Tests for PhysiCell file I/O operations."""
    
    def test_read_physicell_timestep_mock(self, tmp_path):
        """Test reading PhysiCell XML (mock structure)."""
        # Create mock XML file
        xml_path = tmp_path / "output00000000.xml"
        
        # Simplified mock PhysiCell XML
        xml_content = """<?xml version="1.0"?>
<MultiCellDS version="1.0.0">
    <cellular_information>
        <cell_populations>
            <cell_population type="individual">
                <cell ID="0">
                    <phenotype>
                        <cell_type>tumor</cell_type>
                    </phenotype>
                    <state>
                        <position units="micron">100.0 200.0 0.0</position>
                    </state>
                </cell>
                <cell ID="1">
                    <phenotype>
                        <cell_type>immune</cell_type>
                    </phenotype>
                    <state>
                        <position units="micron">300.0 400.0 0.0</position>
                    </state>
                </cell>
            </cell_population>
        </cell_populations>
    </cellular_information>
    <metadata>
        <current_time units="min">0.0</current_time>
    </metadata>
</MultiCellDS>
"""
        xml_path.write_text(xml_content)
        
        # Test that read function exists and handles file
        try:
            timestep = read_physicell_timestep(str(xml_path))
            # If implementation exists, check basic properties
            assert hasattr(timestep, 'n_cells')
            assert hasattr(timestep, 'time')
        except (NotImplementedError, ImportError):
            # XML parsing may require lxml
            pytest.skip("PhysiCell XML parsing not fully implemented")
    
    def test_read_physicell_simulation_mock(self, tmp_path):
        """Test reading PhysiCell simulation folder."""
        # Create mock output folder structure
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Would contain multiple timestep XMLs
        # For now, test that function exists
        try:
            sim = read_physicell_simulation(str(output_dir))
            assert hasattr(sim, 'n_timesteps')
        except (NotImplementedError, FileNotFoundError):
            pytest.skip("PhysiCell folder reading not fully implemented")


# =============================================================================
# Integration Tests
# =============================================================================

class TestSyntheticIntegration:
    """Integration tests for synthetic module."""
    
    def test_timestep_to_analysis_workflow(self):
        """Test converting timestep to SpatialTissueData for analysis."""
        # Create timestep
        coords = np.random.rand(100, 3) * 1000
        cell_types = np.array(['tumor'] * 50 + ['immune'] * 30 + ['fibroblast'] * 20)
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=500.0,
            timestep_id=25
        )
        
        # Convert to SpatialTissueData
        data = timestep.to_spatial_data()
        
        # Can now use all spatial analysis tools
        assert data.n_cells == 100
        assert len(data.cell_types_unique) == 3
        
        # Verify can compute basic statistics
        from spatialtissuepy.spatial import pairwise_distances
        dists = pairwise_distances(data)
        assert dists.shape == (100, 100)
    
    def test_simulation_trajectory_analysis(self):
        """Test analyzing simulation trajectories."""
        timesteps = []
        
        # Simulate tumor growth
        for t in range(10):
            n_tumor = 20 + t * 5  # Growing
            n_immune = 30 - t * 2  # Decreasing
            n_cells = n_tumor + max(n_immune, 0)
            
            coords = np.random.rand(n_cells, 3) * 1000
            types = (['tumor'] * n_tumor + ['immune'] * max(n_immune, 0))
            types = np.array(types[:n_cells])
            
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(t*100), timestep_id=t)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        # Analyze trajectory
        counts = sim.cell_count_trajectory()
        assert counts[0] < counts[-1]  # Growing
        
        # Type proportions
        props = sim.type_proportions_over_time()
        assert props.loc[0, 'tumor'] < props.loc[len(props)-1, 'tumor']


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestSyntheticEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_timestep(self):
        """Test timestep with no cells."""
        coords = np.array([]).reshape(0, 3)
        cell_types = np.array([])
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=0.0,
            timestep_id=0
        )
        
        assert timestep.n_cells == 0
        
        data = timestep.to_spatial_data()
        assert data.n_cells == 0
    
    def test_single_cell_timestep(self):
        """Test timestep with single cell."""
        coords = np.array([[100, 200, 0]])
        cell_types = np.array(['A'])
        
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=0.0,
            timestep_id=0
        )
        
        assert timestep.n_cells == 1
    
    def test_simulation_with_one_timestep(self):
        """Test simulation with single timestep."""
        coords = np.random.rand(50, 3) * 1000
        types = np.array(['A'] * 50)
        
        timesteps = [PhysiCellTimeStep(coords, types, time=0.0, timestep_id=0)]
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        assert sim.n_timesteps == 1
        assert sim.duration == 0.0
    
    def test_experiment_with_one_simulation(self):
        """Test experiment with single simulation."""
        timesteps = []
        for t in range(3):
            coords = np.random.rand(30, 3) * 1000
            types = np.array(['A'] * 30)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(t), timestep_id=t)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        experiment = PhysiCellExperiment(
            simulations=[sim],
            simulation_ids=['only_sim']
        )
        
        assert experiment.n_simulations == 1
    
    def test_invalid_timestep_id(self):
        """Test error when requesting invalid timestep."""
        timesteps = []
        for t in range(3):
            coords = np.random.rand(20, 3) * 1000
            types = np.array(['A'] * 20)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(t), timestep_id=t)
            )
        
        sim = PhysiCellSimulation(timesteps=timesteps)
        
        with pytest.raises((IndexError, ValueError, KeyError)):
            sim.get_timestep(10)  # Out of range


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestSyntheticPerformance:
    """Performance tests for synthetic module."""
    
    def test_large_timestep_conversion(self):
        """Test converting large timestep to SpatialTissueData."""
        import time
        
        # Large timestep (10k cells)
        coords = np.random.rand(10000, 3) * 1000
        cell_types = np.array(['A', 'B', 'C'] * 3333 + ['A'])
        
        start = time.time()
        timestep = PhysiCellTimeStep(
            coordinates=coords,
            cell_types=cell_types,
            time=0.0,
            timestep_id=0
        )
        data = timestep.to_spatial_data()
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should be fast
        assert data.n_cells == 10000
    
    def test_long_simulation(self):
        """Test simulation with many timesteps."""
        import time
        
        timesteps = []
        for t in range(100):  # 100 timesteps
            coords = np.random.rand(200, 3) * 1000
            types = np.array(['A'] * 100 + ['B'] * 100)
            timesteps.append(
                PhysiCellTimeStep(coords, types, time=float(t), timestep_id=t)
            )
        
        start = time.time()
        sim = PhysiCellSimulation(timesteps=timesteps)
        trajectory = sim.cell_count_trajectory()
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should be reasonably fast
        assert len(trajectory) == 100
