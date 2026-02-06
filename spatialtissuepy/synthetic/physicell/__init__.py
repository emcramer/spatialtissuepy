"""
PhysiCell I/O module for spatialtissuepy.

This module provides readers for PhysiCell/MultiCellDS output files,
allowing spatial analysis of agent-based simulation results.

PhysiCell Output Format
-----------------------
PhysiCell saves simulation snapshots as MultiCellDS digital snapshots:
- output{NNNNNNNN}.xml: Main metadata file (time, units, substrate info)
- output{NNNNNNNN}_cells_physicell.mat: Cell data (positions, phenotypes)
- output{NNNNNNNN}_microenvironment0.mat: Substrate concentrations
- initial_mesh0.mat: Computational mesh (saved only at t=0)

Data Structure
--------------
The cells_physicell.mat file contains a matrix where:
- Each column represents one cell
- Rows contain: ID, position (x,y,z), volumes, cycle info, death info, etc.

The exact row structure depends on PhysiCell version, but core fields include:
- Position: x, y, z coordinates
- Cell type: cell_type ID or cell_definition index
- Volumes: total, nuclear, cytoplasmic, fluid, solid
- Cycle: current_phase, elapsed_time_in_phase
- Death: dead flag, death model

References
----------
.. [1] Ghaffarizadeh, A. et al. (2018). PhysiCell: An open source physics-based
       cell simulator for 3-D multicellular systems. PLoS Comput Biol.
"""

from .reader import (
    PhysiCellTimeStep,
    PhysiCellSimulation,
    PhysiCellExperiment,
    read_physicell_timestep,
    read_physicell_simulation,
    read_physicell_experiment,
    discover_physicell_timesteps,
)

from .parser import (
    parse_physicell_xml,
    parse_cells_mat,
    parse_microenvironment_mat,
    get_cell_type_mapping,
    is_alive,
    is_dead,
    get_phase_name,
    CELL_CYCLE_PHASES,
)

__all__ = [
    # Main classes
    'PhysiCellTimeStep',
    'PhysiCellSimulation',
    'PhysiCellExperiment',
    # Reader functions
    'read_physicell_timestep',
    'read_physicell_simulation',
    'read_physicell_experiment',
    'discover_physicell_timesteps',
    # Parser functions
    'parse_physicell_xml',
    'parse_cells_mat',
    'parse_microenvironment_mat',
    'get_cell_type_mapping',
    # Cell state helpers
    'is_alive',
    'is_dead',
    'get_phase_name',
    'CELL_CYCLE_PHASES',
]
