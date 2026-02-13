"""
Tests for spatialtissuepy.synthetic module.

Tests PhysiCell and other ABM framework integration for loading
simulation outputs into SpatialTissueData format.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

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
            ABMTimeStep(time=0.0, time_index=0, source_path=Path("test.xml"))


# =============================================================================
# PhysiCell TimeStep Tests
# =============================================================================

class TestPhysiCellTimeStep:
    """Tests for PhysiCell timestep class logic."""
    
    @pytest.fixture
    def mock_timestep(self):
        """Create a mock timestep with pre-loaded data."""
        ts = PhysiCellTimeStep(
            time=100.0,
            time_index=10,
            source_path=Path("output00000010.xml"),
            cells_mat_path=Path("output00000010_cells.mat")
        )
        
        # Inject mock pre-loaded data to avoid actual file parsing
        ts._cell_data = {
            'positions': np.random.rand(50, 3) * 1000,
            'cell_types': np.array(['tumor'] * 25 + ['immune'] * 25),
            'dead_flags': np.zeros(50),
            'volumes': np.ones(50) * 2000,
            'radii': np.ones(50) * 8,
            'ids': np.arange(50),
            'phases': np.ones(50) * 14
        }
        return ts

    def test_physicell_timestep_properties(self, mock_timestep):
        """Test timestep properties."""
        assert mock_timestep.time == 100.0
        assert mock_timestep.time_index == 10
        assert mock_timestep.n_cells == 50
        assert len(mock_timestep.cell_types) == 2

    def test_physicell_timestep_to_spatial_data(self, mock_timestep):
        """Test converting timestep to SpatialTissueData."""
        data = mock_timestep.to_spatial_data()
        
        assert isinstance(data, SpatialTissueData)
        assert data.n_cells == 50
        assert data.n_dims == 3
        assert 'tumor' in data.cell_types_unique


# =============================================================================
# PhysiCell Simulation Tests
# =============================================================================

class TestPhysiCellSimulation:
    """Tests for PhysiCell simulation class logic."""
    
    def test_simulation_iteration(self):
        """Test that simulation can be iterated over."""
        # Mock class since we can't easily create valid PhysiCell folders here
        class MockSim(PhysiCellSimulation):
            def __init__(self):
                self.output_folder = Path("test")
                self._timestep_files = [(0, Path("0.xml"), Path("0.mat")), 
                                       (1, Path("1.xml"), Path("1.mat"))]
                self.cell_type_mapping = {0: 'A'}
                self.include_dead_cells = False
                self.metadata = {}
            
            @property
            def n_timesteps(self): return 2
            
            def get_timestep(self, idx):
                ts = PhysiCellTimeStep(time=float(idx*10), time_index=idx, 
                                      source_path=self._timestep_files[idx][1])
                ts._cell_data = {'positions': np.zeros((10,3)), 'cell_types': np.array(['A']*10),
                                'dead_flags': np.zeros(10), 'volumes': np.ones(10), 
                                'radii': np.ones(10), 'ids': np.arange(10), 'phases': np.ones(10)}
                return ts

        sim = MockSim()
        assert len(list(sim)) == 2
        assert sim[0].time == 0.0
        assert sim[1].time == 10.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestSyntheticIntegration:
    """Integration tests for synthetic module."""
    
    def test_timestep_to_analysis_workflow(self):
        """Test converting timestep to SpatialTissueData for analysis."""
        ts = PhysiCellTimeStep(time=0.0, time_index=0, source_path=Path("test.xml"))
        ts._cell_data = {
            'positions': np.random.rand(10, 3),
            'cell_types': np.array(['A'] * 10),
            'dead_flags': np.zeros(10),
            'volumes': np.ones(10),
            'radii': np.ones(10),
            'ids': np.arange(10),
            'phases': np.ones(10)
        }
        
        data = ts.to_spatial_data()
        assert data.n_cells == 10
        
        # Verify can compute basic statistics
        from spatialtissuepy.summary import StatisticsPanel
        panel = StatisticsPanel().add('cell_counts')
        stats = ts.summarize(panel)
        assert stats['n_cells'] == 10

