"""
Synthetic/ABM tools for MCP server.

PhysiCell and agent-based modeling integration.

Tools (9 total):
- synthetic_load_physicell_simulation: Load simulation folder
- synthetic_load_physicell_timestep: Load single timestep
- synthetic_list_physicell_timesteps: List available timesteps
- synthetic_get_timestep: Get specific timestep
- synthetic_timestep_to_spatial_data: Convert to SpatialTissueData
- synthetic_cell_count_trajectory: Cell counts over time
- synthetic_type_proportions_trajectory: Proportions over time
- synthetic_summarize_simulation: Summarize with panel
- synthetic_load_physicell_experiment: Load multiple simulations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastmcp import FastMCP


class SimulationInfo(BaseModel):
    """Information about a loaded PhysiCell simulation."""

    session_id: str
    simulation_key: str
    output_folder: str
    n_timesteps: int
    timestep_indices: List[int]
    cell_types: List[str]
    has_substrate: bool


class TimestepInfo(BaseModel):
    """Information about a simulation timestep."""

    session_id: str
    simulation_key: str
    timestep_index: int
    time: float
    n_cells: int
    cell_type_counts: Dict[str, int]


class TimestepList(BaseModel):
    """List of available timesteps."""

    session_id: str
    simulation_key: str
    n_timesteps: int
    timesteps: List[Dict[str, Any]]


class TrajectoryResult(BaseModel):
    """Cell count trajectory over time."""

    session_id: str
    simulation_key: str
    times: List[float]
    total_counts: List[int]
    by_type: Dict[str, List[int]]


class ProportionsResult(BaseModel):
    """Type proportions over time."""

    session_id: str
    simulation_key: str
    times: List[float]
    by_type: Dict[str, List[float]]


class SummarizeResult(BaseModel):
    """Simulation summary result."""

    session_id: str
    simulation_key: str
    n_timesteps: int
    n_features: int
    feature_names: List[str]


class ExperimentInfo(BaseModel):
    """Information about a PhysiCell experiment."""

    session_id: str
    experiment_key: str
    n_simulations: int
    simulation_names: List[str]


def register_tools(mcp: "FastMCP") -> None:
    """Register synthetic tools with the MCP server."""

    @mcp.tool()
    def synthetic_load_physicell_simulation(
        session_id: str,
        output_folder: str,
        simulation_key: str = "simulation",
    ) -> SimulationInfo:
        """
        Load a PhysiCell simulation from an output folder.

        Parameters
        ----------
        session_id : str
            Session to store the simulation.
        output_folder : str
            Path to PhysiCell output folder.
        simulation_key : str
            Key to store the simulation.

        Returns
        -------
        SimulationInfo
            Information about the loaded simulation.
        """
        from spatialtissuepy.synthetic import PhysiCellSimulation
        from ..server import get_session_manager, resolve_data_path
        import pickle

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        path = resolve_data_path(output_folder)
        sim = PhysiCellSimulation.from_output_folder(str(path))

        # Store simulation
        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sim_path, "wb") as f:
            pickle.dump(sim, f)

        return SimulationInfo(
            session_id=session_id,
            simulation_key=simulation_key,
            output_folder=str(path),
            n_timesteps=len(sim.timesteps),
            timestep_indices=list(sim.timesteps.keys()),
            cell_types=list(sim.cell_types) if hasattr(sim, "cell_types") else [],
            has_substrate=hasattr(sim, "substrates") and sim.substrates is not None,
        )

    @mcp.tool()
    def synthetic_load_physicell_timestep(
        session_id: str,
        file_path: str,
        data_key: str = "timestep",
    ) -> TimestepInfo:
        """
        Load a single PhysiCell timestep file.

        Parameters
        ----------
        session_id : str
            Session to store the data.
        file_path : str
            Path to .mat or .xml file.
        data_key : str
            Key to store the converted data.

        Returns
        -------
        TimestepInfo
            Information about the timestep.
        """
        from spatialtissuepy.synthetic import read_physicell_timestep
        from ..server import get_session_manager, resolve_data_path

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        path = resolve_data_path(file_path)
        timestep = read_physicell_timestep(str(path))

        # Convert to SpatialTissueData and store
        data = timestep.to_spatial_data()
        session_mgr.store_data(session_id, data_key, data)

        return TimestepInfo(
            session_id=session_id,
            simulation_key="single_timestep",
            timestep_index=timestep.index if hasattr(timestep, "index") else 0,
            time=timestep.time if hasattr(timestep, "time") else 0.0,
            n_cells=data.n_cells,
            cell_type_counts=dict(data.cell_type_counts),
        )

    @mcp.tool()
    def synthetic_list_physicell_timesteps(
        session_id: str,
        simulation_key: str = "simulation",
    ) -> TimestepList:
        """
        List all timesteps in a loaded simulation.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        simulation_key : str
            Key of the simulation.

        Returns
        -------
        TimestepList
            Available timesteps with basic info.
        """
        from ..server import get_session_manager
        import pickle

        session_mgr = get_session_manager()

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        timesteps_info = []
        for idx, ts in sim.timesteps.items():
            timesteps_info.append({
                "index": idx,
                "time": ts.time if hasattr(ts, "time") else 0.0,
                "n_cells": ts.n_cells if hasattr(ts, "n_cells") else 0,
            })

        return TimestepList(
            session_id=session_id,
            simulation_key=simulation_key,
            n_timesteps=len(timesteps_info),
            timesteps=timesteps_info,
        )

    @mcp.tool()
    def synthetic_get_timestep(
        session_id: str,
        timestep_index: int,
        simulation_key: str = "simulation",
        data_key: Optional[str] = None,
    ) -> TimestepInfo:
        """
        Get a specific timestep from a simulation.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        timestep_index : int
            Index of the timestep to retrieve.
        simulation_key : str
            Key of the simulation.
        data_key : str, optional
            Key to store converted data. Default: timestep_{index}

        Returns
        -------
        TimestepInfo
            Timestep information.
        """
        from ..server import get_session_manager
        import pickle

        session_mgr = get_session_manager()

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        timestep = sim.get_timestep(timestep_index)
        data = timestep.to_spatial_data()

        out_key = data_key or f"timestep_{timestep_index}"
        session_mgr.store_data(session_id, out_key, data)

        return TimestepInfo(
            session_id=session_id,
            simulation_key=simulation_key,
            timestep_index=timestep_index,
            time=timestep.time if hasattr(timestep, "time") else 0.0,
            n_cells=data.n_cells,
            cell_type_counts=dict(data.cell_type_counts),
        )

    @mcp.tool()
    def synthetic_timestep_to_spatial_data(
        session_id: str,
        timestep_index: int,
        simulation_key: str = "simulation",
        data_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert a simulation timestep to SpatialTissueData.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        timestep_index : int
            Timestep to convert.
        simulation_key : str
            Key of the simulation.
        data_key : str, optional
            Key for the converted data.

        Returns
        -------
        dict
            Data info after conversion.
        """
        result = synthetic_get_timestep(
            session_id=session_id,
            timestep_index=timestep_index,
            simulation_key=simulation_key,
            data_key=data_key,
        )
        return {
            "session_id": result.session_id,
            "data_key": data_key or f"timestep_{timestep_index}",
            "n_cells": result.n_cells,
            "cell_type_counts": result.cell_type_counts,
        }

    @mcp.tool()
    def synthetic_cell_count_trajectory(
        session_id: str,
        simulation_key: str = "simulation",
    ) -> TrajectoryResult:
        """
        Get cell count trajectory over simulation time.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        simulation_key : str
            Key of the simulation.

        Returns
        -------
        TrajectoryResult
            Cell counts over time, total and by type.
        """
        from ..server import get_session_manager
        import pickle

        session_mgr = get_session_manager()

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        trajectory = sim.cell_count_trajectory()

        times = list(trajectory.index)
        total_counts = trajectory.sum(axis=1).tolist()

        by_type = {}
        for col in trajectory.columns:
            by_type[str(col)] = trajectory[col].tolist()

        return TrajectoryResult(
            session_id=session_id,
            simulation_key=simulation_key,
            times=times,
            total_counts=total_counts,
            by_type=by_type,
        )

    @mcp.tool()
    def synthetic_type_proportions_trajectory(
        session_id: str,
        simulation_key: str = "simulation",
    ) -> ProportionsResult:
        """
        Get cell type proportions over simulation time.

        Parameters
        ----------
        session_id : str
            Session containing the simulation.
        simulation_key : str
            Key of the simulation.

        Returns
        -------
        ProportionsResult
            Proportions over time by type.
        """
        from ..server import get_session_manager
        import pickle

        session_mgr = get_session_manager()

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        proportions = sim.type_proportions_over_time()

        times = list(proportions.index)
        by_type = {}
        for col in proportions.columns:
            by_type[str(col)] = proportions[col].tolist()

        return ProportionsResult(
            session_id=session_id,
            simulation_key=simulation_key,
            times=times,
            by_type=by_type,
        )

    @mcp.tool()
    def synthetic_summarize_simulation(
        session_id: str,
        simulation_key: str = "simulation",
        panel_key: str = "custom_panel",
        timestep_indices: Optional[List[int]] = None,
    ) -> SummarizeResult:
        """
        Compute summary statistics for simulation timesteps.

        Parameters
        ----------
        session_id : str
            Session containing the simulation and panel.
        simulation_key : str
            Key of the simulation.
        panel_key : str
            Key of the statistics panel.
        timestep_indices : list of int, optional
            Specific timesteps to summarize. Default: all.

        Returns
        -------
        SummarizeResult
            Summary statistics over time.
        """
        from ..server import get_session_manager
        import pickle

        session_mgr = get_session_manager()
        panel = session_mgr.load_panel(session_id, panel_key)

        sim_path = session_mgr.base_dir / session_id / "models" / f"{simulation_key}.pkl"
        if not sim_path.exists():
            raise ValueError(f"No simulation found with key '{simulation_key}'")
        if panel is None:
            raise ValueError(f"No panel found with key '{panel_key}'")

        with open(sim_path, "rb") as f:
            sim = pickle.load(f)

        df = sim.summarize(panel, timestep_indices=timestep_indices)

        return SummarizeResult(
            session_id=session_id,
            simulation_key=simulation_key,
            n_timesteps=len(df),
            n_features=len(df.columns),
            feature_names=list(df.columns),
        )

    @mcp.tool()
    def synthetic_load_physicell_experiment(
        session_id: str,
        output_folders: List[str],
        experiment_key: str = "experiment",
        simulation_names: Optional[List[str]] = None,
    ) -> ExperimentInfo:
        """
        Load multiple PhysiCell simulations as an experiment.

        Parameters
        ----------
        session_id : str
            Session to store the experiment.
        output_folders : list of str
            Paths to simulation output folders.
        experiment_key : str
            Key to store the experiment.
        simulation_names : list of str, optional
            Names for each simulation. Default: folder names.

        Returns
        -------
        ExperimentInfo
            Information about the experiment.
        """
        from spatialtissuepy.synthetic import PhysiCellExperiment
        from ..server import get_session_manager, resolve_data_path
        import pickle

        session_mgr = get_session_manager()
        session_id = session_mgr.get_or_create_session(session_id)

        paths = [str(resolve_data_path(f)) for f in output_folders]
        experiment = PhysiCellExperiment.from_folders(paths, names=simulation_names)

        # Store experiment
        exp_path = session_mgr.base_dir / session_id / "models" / f"{experiment_key}.pkl"
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_path, "wb") as f:
            pickle.dump(experiment, f)

        names = simulation_names or [p.split("/")[-1] for p in paths]

        return ExperimentInfo(
            session_id=session_id,
            experiment_key=experiment_key,
            n_simulations=len(paths),
            simulation_names=names,
        )
