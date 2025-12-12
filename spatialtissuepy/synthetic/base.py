"""
Base classes for agent-based modeling interfaces.

These abstract base classes define the interface for loading and analyzing
outputs from various ABM frameworks. Subclasses implement framework-specific
parsing logic while maintaining a consistent API.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Optional, Dict, List, Any, Iterator, Tuple, Union,
    TYPE_CHECKING
)
from pathlib import Path
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatialtissuepy.core import SpatialTissueData
    from spatialtissuepy.summary import StatisticsPanel


# -----------------------------------------------------------------------------
# ABM TimeStep Base Class
# -----------------------------------------------------------------------------

@dataclass
class ABMTimeStep(ABC):
    """
    Base class representing a single time step from an ABM simulation.
    
    Attributes
    ----------
    time : float
        Simulation time (in simulation units, typically minutes).
    time_index : int
        Integer index of this time step in the simulation.
    source_path : Path
        Path to the source file(s) for this time step.
    metadata : dict
        Additional metadata (units, software version, etc.).
    """
    time: float
    time_index: int
    source_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def to_spatial_data(self) -> 'SpatialTissueData':
        """
        Convert this time step to a SpatialTissueData object.
        
        Returns
        -------
        SpatialTissueData
            Spatial tissue data with cell coordinates and types.
        """
        pass
    
    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Number of cells at this time step."""
        pass
    
    @property
    @abstractmethod
    def cell_types(self) -> List[str]:
        """List of unique cell type names."""
        pass
    
    def summarize(
        self,
        panel: 'StatisticsPanel'
    ) -> Dict[str, float]:
        """
        Compute summary statistics for this time step.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
            
        Returns
        -------
        dict
            Dictionary mapping statistic names to values.
        """
        data = self.to_spatial_data()
        return panel.compute(data)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"time={self.time}, "
            f"time_index={self.time_index}, "
            f"n_cells={self.n_cells})"
        )


# -----------------------------------------------------------------------------
# ABM Simulation Base Class
# -----------------------------------------------------------------------------

@dataclass
class ABMSimulation(ABC):
    """
    Base class representing a complete ABM simulation (time series).
    
    A simulation consists of multiple time steps that can be iterated over
    or accessed by index/time.
    
    Attributes
    ----------
    output_folder : Path
        Path to the simulation output folder.
    simulation_id : str
        Unique identifier for this simulation.
    metadata : dict
        Simulation-level metadata (parameters, config, etc.).
    """
    output_folder: Path
    simulation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def n_timesteps(self) -> int:
        """Number of time steps in the simulation."""
        pass
    
    @property
    @abstractmethod
    def times(self) -> np.ndarray:
        """Array of simulation times for all time steps."""
        pass
    
    @property
    @abstractmethod
    def time_indices(self) -> np.ndarray:
        """Array of time step indices."""
        pass
    
    @abstractmethod
    def get_timestep(self, index: int) -> ABMTimeStep:
        """
        Get a specific time step by index.
        
        Parameters
        ----------
        index : int
            Time step index (0-based).
            
        Returns
        -------
        ABMTimeStep
            The requested time step.
        """
        pass
    
    @abstractmethod
    def get_timestep_by_time(
        self,
        time: float,
        tolerance: float = 1e-6
    ) -> ABMTimeStep:
        """
        Get a time step closest to the specified time.
        
        Parameters
        ----------
        time : float
            Target simulation time.
        tolerance : float, default 1e-6
            Tolerance for time matching.
            
        Returns
        -------
        ABMTimeStep
            The closest time step.
        """
        pass
    
    def __iter__(self) -> Iterator[ABMTimeStep]:
        """Iterate over all time steps."""
        for i in range(self.n_timesteps):
            yield self.get_timestep(i)
    
    def __len__(self) -> int:
        return self.n_timesteps
    
    def __getitem__(self, index: int) -> ABMTimeStep:
        return self.get_timestep(index)
    
    def summarize(
        self,
        panel: 'StatisticsPanel',
        progress: bool = False
    ) -> pd.DataFrame:
        """
        Compute summary statistics for all time steps.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
        progress : bool, default False
            Show progress bar (requires tqdm).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with time steps as rows and statistics as columns.
            Includes 'time' and 'time_index' columns.
        """
        results = []
        
        iterator = range(self.n_timesteps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Summarizing timesteps")
            except ImportError:
                pass
        
        for i in iterator:
            timestep = self.get_timestep(i)
            row = {
                'time': timestep.time,
                'time_index': timestep.time_index,
                'n_cells': timestep.n_cells,
            }
            row.update(timestep.summarize(panel))
            results.append(row)
        
        df = pd.DataFrame(results)
        df = df.set_index('time_index')
        
        return df
    
    def summarize_timesteps(
        self,
        panel: 'StatisticsPanel',
        indices: Optional[List[int]] = None,
        times: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute summary statistics for selected time steps.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
        indices : list of int, optional
            Specific time step indices to summarize.
        times : list of float, optional
            Specific simulation times to summarize.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics for selected time steps.
        """
        if indices is None and times is None:
            return self.summarize(panel)
        
        results = []
        
        if indices is not None:
            for i in indices:
                timestep = self.get_timestep(i)
                row = {
                    'time': timestep.time,
                    'time_index': timestep.time_index,
                    'n_cells': timestep.n_cells,
                }
                row.update(timestep.summarize(panel))
                results.append(row)
        
        elif times is not None:
            for t in times:
                timestep = self.get_timestep_by_time(t)
                row = {
                    'time': timestep.time,
                    'time_index': timestep.time_index,
                    'n_cells': timestep.n_cells,
                }
                row.update(timestep.summarize(panel))
                results.append(row)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.set_index('time_index')
        
        return df
    
    def to_spatial_data_series(self) -> List['SpatialTissueData']:
        """
        Convert all time steps to SpatialTissueData objects.
        
        Returns
        -------
        list of SpatialTissueData
            List of spatial data for each time step.
        """
        return [ts.to_spatial_data() for ts in self]
    
    def cell_counts_over_time(self) -> pd.DataFrame:
        """
        Get cell counts by type over simulation time.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with time as index and cell type counts as columns.
        """
        results = []
        
        for timestep in self:
            data = timestep.to_spatial_data()
            row = {'time': timestep.time, 'time_index': timestep.time_index}
            row['total_cells'] = data.n_cells
            
            for cell_type in data.cell_types_unique:
                row[f'n_{cell_type}'] = len(data.get_cells_by_type(cell_type))
            
            results.append(row)
        
        return pd.DataFrame(results).set_index('time_index')


# -----------------------------------------------------------------------------
# ABM Experiment Base Class
# -----------------------------------------------------------------------------

@dataclass
class ABMExperiment(ABC):
    """
    Base class representing a collection of ABM simulations.
    
    An experiment typically contains multiple simulations with different
    parameter settings (e.g., control vs treatment, parameter sweeps).
    
    Attributes
    ----------
    simulations : list of ABMSimulation
        List of simulations in this experiment.
    experiment_id : str
        Unique identifier for this experiment.
    metadata : dict
        Experiment-level metadata.
    """
    simulations: List[ABMSimulation] = field(default_factory=list)
    experiment_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_simulations(self) -> int:
        """Number of simulations in the experiment."""
        return len(self.simulations)
    
    def __iter__(self) -> Iterator[ABMSimulation]:
        """Iterate over simulations."""
        return iter(self.simulations)
    
    def __len__(self) -> int:
        return self.n_simulations
    
    def __getitem__(self, index: int) -> ABMSimulation:
        return self.simulations[index]
    
    def add_simulation(self, simulation: ABMSimulation) -> None:
        """Add a simulation to the experiment."""
        self.simulations.append(simulation)
    
    def summarize(
        self,
        panel: 'StatisticsPanel',
        progress: bool = False,
        include_simulation_id: bool = True
    ) -> pd.DataFrame:
        """
        Compute summary statistics for all simulations and time steps.
        
        Creates a master DataFrame with all time steps from all simulations,
        suitable for training ML models on simulation dynamics.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
        progress : bool, default False
            Show progress bar.
        include_simulation_id : bool, default True
            Include simulation ID column.
            
        Returns
        -------
        pd.DataFrame
            Stacked DataFrame with all time steps from all simulations.
            Includes 'simulation_id', 'time', 'time_index' columns.
        """
        all_results = []
        
        iterator = enumerate(self.simulations)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    list(iterator), 
                    desc="Summarizing simulations"
                )
            except ImportError:
                pass
        
        for i, sim in iterator:
            sim_df = sim.summarize(panel, progress=False)
            
            if include_simulation_id:
                sim_df['simulation_id'] = sim.simulation_id or f"sim_{i}"
                sim_df['simulation_index'] = i
            
            all_results.append(sim_df.reset_index())
        
        if len(all_results) == 0:
            return pd.DataFrame()
        
        master_df = pd.concat(all_results, ignore_index=True)
        
        return master_df
    
    def summarize_by_simulation(
        self,
        panel: 'StatisticsPanel'
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute summaries separately for each simulation.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
            
        Returns
        -------
        dict
            Dictionary mapping simulation IDs to DataFrames.
        """
        return {
            sim.simulation_id or f"sim_{i}": sim.summarize(panel)
            for i, sim in enumerate(self.simulations)
        }
    
    def get_simulation_by_id(self, simulation_id: str) -> ABMSimulation:
        """
        Get a simulation by its ID.
        
        Parameters
        ----------
        simulation_id : str
            Simulation identifier.
            
        Returns
        -------
        ABMSimulation
            The requested simulation.
            
        Raises
        ------
        KeyError
            If simulation ID not found.
        """
        for sim in self.simulations:
            if sim.simulation_id == simulation_id:
                return sim
        raise KeyError(f"Simulation not found: {simulation_id}")
    
    def final_timesteps(self) -> List[ABMTimeStep]:
        """
        Get the final time step from each simulation.
        
        Returns
        -------
        list of ABMTimeStep
            Final time steps from all simulations.
        """
        return [sim.get_timestep(sim.n_timesteps - 1) for sim in self.simulations]
    
    def summarize_final_timesteps(
        self,
        panel: 'StatisticsPanel'
    ) -> pd.DataFrame:
        """
        Summarize only the final time step of each simulation.
        
        Useful for comparing end states of different simulations.
        
        Parameters
        ----------
        panel : StatisticsPanel
            Panel of statistics to compute.
            
        Returns
        -------
        pd.DataFrame
            One row per simulation with final time step statistics.
        """
        results = []
        
        for i, sim in enumerate(self.simulations):
            final_ts = sim.get_timestep(sim.n_timesteps - 1)
            row = {
                'simulation_id': sim.simulation_id or f"sim_{i}",
                'simulation_index': i,
                'final_time': final_ts.time,
                'final_n_cells': final_ts.n_cells,
            }
            row.update(final_ts.summarize(panel))
            results.append(row)
        
        return pd.DataFrame(results)
