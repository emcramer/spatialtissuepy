"""
High-level readers for PhysiCell simulation output.

This module provides the main classes and functions for loading PhysiCell
simulation data and converting it to SpatialTissueData for analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from pathlib import Path
import re
import numpy as np
import pandas as pd

from ..base import ABMTimeStep, ABMSimulation, ABMExperiment
from .parser import (
    parse_physicell_xml,
    parse_cells_mat,
    get_cell_type_mapping,
    is_alive,
    PhysiCellMetadata,
)


# -----------------------------------------------------------------------------
# PhysiCell TimeStep
# -----------------------------------------------------------------------------

@dataclass
class PhysiCellTimeStep(ABMTimeStep):
    """
    A single time step from a PhysiCell simulation.
    
    This class lazily loads data from PhysiCell output files and converts
    them to SpatialTissueData for spatial analysis.
    
    Attributes
    ----------
    time : float
        Simulation time in minutes.
    time_index : int
        Time step index (from filename).
    source_path : Path
        Path to the output XML file.
    cells_mat_path : Path
        Path to the cells_physicell.mat file.
    metadata : dict
        Simulation metadata from XML.
    cell_type_mapping : dict
        Mapping from cell type IDs to names.
    include_dead_cells : bool
        Whether to include dead cells in analysis.
    
    Examples
    --------
    >>> timestep = read_physicell_timestep('./output/output00000010.xml')
    >>> print(f"Time: {timestep.time} min, Cells: {timestep.n_cells}")
    >>> 
    >>> # Convert to SpatialTissueData for analysis
    >>> data = timestep.to_spatial_data()
    >>> from spatialtissuepy.statistics import ripleys_h
    >>> H = ripleys_h(data.coordinates, radii)
    """
    cells_mat_path: Path = None
    cell_type_mapping: Dict[int, str] = field(default_factory=dict)
    include_dead_cells: bool = False
    _cell_data: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    _physicell_metadata: Optional[PhysiCellMetadata] = field(default=None, repr=False)
    
    def _load_cell_data(self) -> Dict[str, np.ndarray]:
        """Load cell data from MAT file (cached)."""
        if self._cell_data is None:
            self._cell_data = parse_cells_mat(
                self.cells_mat_path,
                self.cell_type_mapping
            )
        return self._cell_data
    
    def _load_metadata(self) -> PhysiCellMetadata:
        """Load metadata from XML file (cached)."""
        if self._physicell_metadata is None:
            self._physicell_metadata = parse_physicell_xml(self.source_path)
        return self._physicell_metadata
    
    @property
    def n_cells(self) -> int:
        """Number of cells at this time step."""
        data = self._load_cell_data()
        if self.include_dead_cells:
            return len(data['cell_types'])
        else:
            # Filter out dead cells using dead_flags
            alive_mask = data['dead_flags'] == 0
            return np.sum(alive_mask)
    
    @property
    def n_cells_total(self) -> int:
        """Total number of cells including dead."""
        return len(self._load_cell_data()['cell_types'])
    
    @property
    def n_dead_cells(self) -> int:
        """Number of dead cells."""
        data = self._load_cell_data()
        return np.sum(data['dead_flags'] == 1)
    
    @property
    def cell_types(self) -> List[str]:
        """List of unique cell type names."""
        data = self._load_cell_data()
        return list(np.unique(data['cell_types']))
    
    @property
    def positions(self) -> np.ndarray:
        """Cell positions as (n_cells, 3) array."""
        data = self._load_cell_data()
        if self.include_dead_cells:
            return data['positions']
        else:
            alive_mask = data['dead_flags'] == 0
            return data['positions'][alive_mask]
    
    @property
    def domain_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Simulation domain bounds."""
        meta = self._load_metadata()
        return {
            'x': (meta.domain_min[0], meta.domain_max[0]),
            'y': (meta.domain_min[1], meta.domain_max[1]),
            'z': (meta.domain_min[2], meta.domain_max[2]),
        }
    
    def to_spatial_data(self) -> 'SpatialTissueData':
        """
        Convert to SpatialTissueData for spatial analysis.
        
        Returns
        -------
        SpatialTissueData
            Spatial tissue data object.
        """
        from spatialtissuepy.core import SpatialTissueData
        
        data = self._load_cell_data()
        
        if self.include_dead_cells:
            positions = data['positions']
            cell_types = data['cell_types']
            volumes = data['volumes']
            radii = data['radii']
        else:
            alive_mask = data['dead_flags'] == 0
            positions = data['positions'][alive_mask]
            cell_types = data['cell_types'][alive_mask]
            volumes = data['volumes'][alive_mask]
            radii = data['radii'][alive_mask]
        
        # Create markers DataFrame with cell properties
        markers = pd.DataFrame({
            'volume': volumes,
            'radius': radii,
        })
        
        # Add metadata
        extra_metadata = {
            'source': 'PhysiCell',
            'time': self.time,
            'time_index': self.time_index,
            'source_path': str(self.source_path),
        }
        
        return SpatialTissueData(
            coordinates=positions,
            cell_types=cell_types,
            markers=markers,
            metadata=extra_metadata,
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert cell data to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cell properties.
        """
        data = self._load_cell_data()
        
        df = pd.DataFrame({
            'cell_id': data['ids'],
            'x': data['positions'][:, 0],
            'y': data['positions'][:, 1],
            'z': data['positions'][:, 2],
            'cell_type': data['cell_types'],
            'cell_type_id': data['cell_type_ids'],
            'volume': data['volumes'],
            'radius': data['radii'],
            'phase': data['phases'],
            'is_dead': data['dead_flags'].astype(bool),
            'is_alive': ~data['dead_flags'].astype(bool),
        })
        
        df['time'] = self.time
        df['time_index'] = self.time_index
        
        return df
    
    def cell_counts_by_type(self) -> Dict[str, int]:
        """Get cell counts by type."""
        data = self.to_spatial_data()
        return {
            ct: len(data.get_cells_by_type(ct))
            for ct in data.cell_types_unique
        }


# -----------------------------------------------------------------------------
# PhysiCell Simulation
# -----------------------------------------------------------------------------

@dataclass
class PhysiCellSimulation(ABMSimulation):
    """
    A complete PhysiCell simulation (time series).
    
    This class manages all time steps from a PhysiCell simulation output folder
    and provides methods for analyzing the full time series.
    
    Attributes
    ----------
    output_folder : Path
        Path to the simulation output folder.
    simulation_id : str
        Unique identifier for this simulation.
    cell_type_mapping : dict
        Mapping from cell type IDs to names.
    include_dead_cells : bool
        Whether to include dead cells.
    
    Examples
    --------
    >>> sim = PhysiCellSimulation.from_output_folder('./output')
    >>> print(f"Found {sim.n_timesteps} time steps")
    >>> 
    >>> # Iterate over time steps
    >>> for timestep in sim:
    ...     print(f"t={timestep.time}: {timestep.n_cells} cells")
    >>> 
    >>> # Summarize with statistics panel
    >>> panel = StatisticsPanel()
    >>> panel.add('cell_counts')
    >>> panel.add('ripleys_h_max')
    >>> df = sim.summarize(panel)
    """
    cell_type_mapping: Dict[int, str] = field(default_factory=dict)
    include_dead_cells: bool = False
    _timestep_files: List[Tuple[int, Path, Path]] = field(
        default_factory=list, repr=False
    )
    _times: Optional[np.ndarray] = field(default=None, repr=False)
    
    @classmethod
    def from_output_folder(
        cls,
        output_folder: Union[str, Path],
        simulation_id: Optional[str] = None,
        settings_xml: Optional[Union[str, Path]] = None,
        include_dead_cells: bool = False
    ) -> 'PhysiCellSimulation':
        """
        Create a PhysiCellSimulation from an output folder.
        
        Parameters
        ----------
        output_folder : str or Path
            Path to PhysiCell output folder.
        simulation_id : str, optional
            Identifier for this simulation. Defaults to folder name.
        settings_xml : str or Path, optional
            Path to PhysiCell_settings.xml for cell type names.
        include_dead_cells : bool, default False
            Whether to include dead cells in analysis.
            
        Returns
        -------
        PhysiCellSimulation
            Loaded simulation.
        """
        output_folder = Path(output_folder)
        
        if not output_folder.exists():
            raise FileNotFoundError(f"Output folder not found: {output_folder}")
        
        # Auto-generate simulation ID from folder name
        if simulation_id is None:
            simulation_id = output_folder.name
        
        # Discover time step files
        timestep_files = discover_physicell_timesteps(output_folder)
        
        if len(timestep_files) == 0:
            raise ValueError(f"No PhysiCell output files found in {output_folder}")
        
        # Get cell type mapping
        first_xml = timestep_files[0][1]
        if settings_xml is not None:
            settings_path = Path(settings_xml)
        else:
            # Try to find settings XML in common locations
            settings_path = None
            for candidate in [
                output_folder / 'config.xml',  # PhysiCell often copies config here
                output_folder / 'PhysiCell_settings.xml',
                output_folder.parent / 'config' / 'PhysiCell_settings.xml',
                output_folder.parent / 'PhysiCell_settings.xml',
            ]:
                if candidate.exists():
                    settings_path = candidate
                    break
        
        cell_type_mapping = get_cell_type_mapping(first_xml, settings_path)
        
        sim = cls(
            output_folder=output_folder,
            simulation_id=simulation_id,
            cell_type_mapping=cell_type_mapping,
            include_dead_cells=include_dead_cells,
            _timestep_files=timestep_files,
        )
        
        return sim
    
    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return len(self._timestep_files)
    
    @property
    def times(self) -> np.ndarray:
        """Array of simulation times."""
        if self._times is None:
            self._times = np.array([
                parse_physicell_xml(xml_path).time
                for _, xml_path, _ in self._timestep_files
            ])
        return self._times
    
    @property
    def time_indices(self) -> np.ndarray:
        """Array of time step indices."""
        return np.array([idx for idx, _, _ in self._timestep_files])
    
    def get_timestep(self, index: int) -> PhysiCellTimeStep:
        """
        Get a specific time step by index.
        
        Parameters
        ----------
        index : int
            Time step index (0-based position in sorted list).
            Supports negative indexing (e.g., -1 for last timestep).
            
        Returns
        -------
        PhysiCellTimeStep
            The requested time step.
        """
        # Support negative indexing
        if index < 0:
            index = self.n_timesteps + index
        
        if index < 0 or index >= self.n_timesteps:
            raise IndexError(f"Time step index out of range: {index}")
        
        time_idx, xml_path, mat_path = self._timestep_files[index]
        
        # Get time from XML
        metadata = parse_physicell_xml(xml_path)
        
        return PhysiCellTimeStep(
            time=metadata.time,
            time_index=time_idx,
            source_path=xml_path,
            cells_mat_path=mat_path,
            metadata=self.metadata,
            cell_type_mapping=self.cell_type_mapping,
            include_dead_cells=self.include_dead_cells,
        )
    
    def get_timestep_by_time(
        self,
        time: float,
        tolerance: float = 1e-6
    ) -> PhysiCellTimeStep:
        """
        Get time step closest to specified time.
        
        Parameters
        ----------
        time : float
            Target simulation time.
        tolerance : float, default 1e-6
            Tolerance for exact matching.
            
        Returns
        -------
        PhysiCellTimeStep
            Closest time step.
        """
        times = self.times
        idx = np.argmin(np.abs(times - time))
        return self.get_timestep(idx)
    
    def get_timestep_by_original_index(
        self,
        original_index: int
    ) -> PhysiCellTimeStep:
        """
        Get time step by its original PhysiCell index.
        
        Parameters
        ----------
        original_index : int
            Original index from filename (e.g., 87 for output00000087.xml).
            
        Returns
        -------
        PhysiCellTimeStep
            The requested time step.
        """
        for i, (idx, _, _) in enumerate(self._timestep_files):
            if idx == original_index:
                return self.get_timestep(i)
        
        raise KeyError(f"Time step with index {original_index} not found")
    
    def cell_counts_over_time(self) -> pd.DataFrame:
        """
        Get cell counts over simulation time.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with time, total cells, and per-type counts.
        """
        results = []
        
        for timestep in self:
            row = {
                'time': timestep.time,
                'time_index': timestep.time_index,
                'total_cells': timestep.n_cells,
                'dead_cells': timestep.n_dead_cells,
            }
            row.update({
                f'n_{ct}': count 
                for ct, count in timestep.cell_counts_by_type().items()
            })
            results.append(row)
        
        return pd.DataFrame(results)
    
    def to_trajectory_dataframe(self) -> pd.DataFrame:
        """
        Create a full trajectory DataFrame with all cells at all times.
        
        Warning: This can be memory-intensive for large simulations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all cell data at all time steps.
        """
        dfs = []
        for timestep in self:
            dfs.append(timestep.to_dataframe())
        
        return pd.concat(dfs, ignore_index=True)


# -----------------------------------------------------------------------------
# PhysiCell Experiment
# -----------------------------------------------------------------------------

@dataclass
class PhysiCellExperiment(ABMExperiment):
    """
    Collection of PhysiCell simulations for comparative analysis.
    
    An experiment contains multiple simulations, typically with different
    parameter settings (control, treatments, parameter sweeps).
    
    Examples
    --------
    >>> experiment = PhysiCellExperiment.from_folders([
    ...     './sim_control/output',
    ...     './sim_treatment_low/output',
    ...     './sim_treatment_high/output',
    ... ])
    >>> 
    >>> # Summarize all simulations
    >>> panel = StatisticsPanel()
    >>> panel.add('cell_counts')
    >>> panel.add('ripleys_h_max')
    >>> master_df = experiment.summarize(panel)
    >>> 
    >>> # Use for ML training
    >>> X = master_df[panel.get_statistic_names()]
    >>> y = master_df['simulation_id']
    """
    
    @classmethod
    def from_folders(
        cls,
        folders: List[Union[str, Path]],
        simulation_ids: Optional[List[str]] = None,
        experiment_id: str = "",
        include_dead_cells: bool = False,
        **kwargs
    ) -> 'PhysiCellExperiment':
        """
        Create experiment from multiple output folders.
        
        Parameters
        ----------
        folders : list of str or Path
            Paths to simulation output folders.
        simulation_ids : list of str, optional
            IDs for each simulation. Defaults to folder names.
        experiment_id : str, default ""
            Identifier for this experiment.
        include_dead_cells : bool, default False
            Whether to include dead cells.
            
        Returns
        -------
        PhysiCellExperiment
            Loaded experiment.
        """
        if simulation_ids is None:
            simulation_ids = [Path(f).name for f in folders]
        
        simulations = []
        for folder, sim_id in zip(folders, simulation_ids):
            try:
                sim = PhysiCellSimulation.from_output_folder(
                    folder,
                    simulation_id=sim_id,
                    include_dead_cells=include_dead_cells,
                )
                simulations.append(sim)
            except Exception as e:
                print(f"Warning: Could not load {folder}: {e}")
        
        return cls(
            simulations=simulations,
            experiment_id=experiment_id,
            metadata=kwargs,
        )
    
    @classmethod
    def from_parent_folder(
        cls,
        parent_folder: Union[str, Path],
        experiment_id: str = "",
        include_dead_cells: bool = False,
        output_subfolder: str = "output"
    ) -> 'PhysiCellExperiment':
        """
        Create experiment from a parent folder containing simulation folders.
        
        Assumes structure:
            parent_folder/
                sim1/output/
                sim2/output/
                sim3/output/
        
        Parameters
        ----------
        parent_folder : str or Path
            Parent folder containing simulation folders.
        experiment_id : str, default ""
            Identifier for this experiment.
        include_dead_cells : bool, default False
            Whether to include dead cells.
        output_subfolder : str, default "output"
            Name of output subfolder in each simulation.
            
        Returns
        -------
        PhysiCellExperiment
            Loaded experiment.
        """
        parent = Path(parent_folder)
        
        folders = []
        sim_ids = []
        
        for child in sorted(parent.iterdir()):
            if child.is_dir():
                output_path = child / output_subfolder
                if output_path.exists():
                    folders.append(output_path)
                    sim_ids.append(child.name)
        
        return cls.from_folders(
            folders,
            simulation_ids=sim_ids,
            experiment_id=experiment_id or parent.name,
            include_dead_cells=include_dead_cells,
        )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _find_cells_mat_file(output_folder: Path, index: int) -> Optional[Path]:
    """
    Find the cells MAT file for a given output index.
    
    PhysiCell versions use different naming conventions:
      - Newer versions (1.10+): output{index}_cells_physicell.mat
      - Older versions: output{index}_cells.mat
    
    Parameters
    ----------
    output_folder : Path
        Path to output folder.
    index : int
        Output file index.
        
    Returns
    -------
    Path or None
        Path to the MAT file if found, None otherwise.
    """
    # Try newer naming convention first
    mat_file_new = output_folder / f'output{index:08d}_cells_physicell.mat'
    if mat_file_new.exists():
        return mat_file_new
    
    # Fall back to older naming convention
    mat_file_old = output_folder / f'output{index:08d}_cells.mat'
    if mat_file_old.exists():
        return mat_file_old
    
    return None


def discover_physicell_timesteps(
    output_folder: Path
) -> List[Tuple[int, Path, Path]]:
    """
    Discover PhysiCell output files in a folder.
    
    Parameters
    ----------
    output_folder : Path
        Path to output folder.
        
    Returns
    -------
    list of (int, Path, Path)
        List of (time_index, xml_path, mat_path) tuples, sorted by index.
    """
    output_folder = Path(output_folder)
    
    # Pattern for PhysiCell output files
    xml_pattern = re.compile(r'output(\d{8})\.xml$')
    
    timesteps = []
    
    for xml_file in output_folder.glob('output*.xml'):
        match = xml_pattern.match(xml_file.name)
        if match:
            index = int(match.group(1))
            
            # Find corresponding MAT file (handles both naming conventions)
            mat_file = _find_cells_mat_file(output_folder, index)
            
            if mat_file is not None:
                timesteps.append((index, xml_file, mat_file))
    
    # Sort by index
    timesteps.sort(key=lambda x: x[0])
    
    return timesteps


def read_physicell_timestep(
    xml_path: Union[str, Path],
    cell_type_mapping: Optional[Dict[int, str]] = None,
    include_dead_cells: bool = False
) -> PhysiCellTimeStep:
    """
    Read a single PhysiCell time step.
    
    Parameters
    ----------
    xml_path : str or Path
        Path to the output*.xml file.
    cell_type_mapping : dict, optional
        Mapping from cell type IDs to names.
    include_dead_cells : bool, default False
        Whether to include dead cells.
        
    Returns
    -------
    PhysiCellTimeStep
        Loaded time step.
    """
    xml_path = Path(xml_path)
    
    # Parse XML for metadata
    metadata = parse_physicell_xml(xml_path)
    
    # Extract index from filename
    base_name = xml_path.stem  # e.g., "output00000087"
    match = re.search(r'output(\d+)', base_name)
    time_index = int(match.group(1)) if match else 0
    
    # Find corresponding MAT file (handles both naming conventions)
    mat_path = _find_cells_mat_file(xml_path.parent, time_index)
    
    if mat_path is None:
        # Provide helpful error message listing what was checked
        mat_path_new = xml_path.parent / f'{base_name}_cells_physicell.mat'
        mat_path_old = xml_path.parent / f'{base_name}_cells.mat'
        raise FileNotFoundError(
            f"Cell MAT file not found. Checked:\n"
            f"  - {mat_path_new}\n"
            f"  - {mat_path_old}"
        )
    
    # Get cell type mapping if not provided
    if cell_type_mapping is None:
        cell_type_mapping = get_cell_type_mapping(xml_path)
    
    return PhysiCellTimeStep(
        time=metadata.time,
        time_index=time_index,
        source_path=xml_path,
        cells_mat_path=mat_path,
        metadata={
            'program': metadata.program_name,
            'version': metadata.program_version,
            'space_units': metadata.space_units,
            'time_units': metadata.time_units,
        },
        cell_type_mapping=cell_type_mapping,
        include_dead_cells=include_dead_cells,
    )


def read_physicell_simulation(
    output_folder: Union[str, Path],
    **kwargs
) -> PhysiCellSimulation:
    """
    Read a complete PhysiCell simulation.
    
    Parameters
    ----------
    output_folder : str or Path
        Path to output folder.
    **kwargs
        Additional arguments passed to PhysiCellSimulation.from_output_folder.
        
    Returns
    -------
    PhysiCellSimulation
        Loaded simulation.
    """
    return PhysiCellSimulation.from_output_folder(output_folder, **kwargs)


def read_physicell_experiment(
    folders: List[Union[str, Path]],
    **kwargs
) -> PhysiCellExperiment:
    """
    Read multiple PhysiCell simulations as an experiment.
    
    Parameters
    ----------
    folders : list of str or Path
        Paths to simulation output folders.
    **kwargs
        Additional arguments passed to PhysiCellExperiment.from_folders.
        
    Returns
    -------
    PhysiCellExperiment
        Loaded experiment.
    """
    return PhysiCellExperiment.from_folders(folders, **kwargs)
