"""
Synthetic module for processing agent-based modeling outputs.

This module provides interfaces for loading and analyzing outputs from various
agent-based modeling (ABM) frameworks used in computational biology and cancer
research. The design is modular to support multiple ABM platforms.

Supported Frameworks
--------------------
- PhysiCell: Physics-based cell simulator (https://physicell.org/)
- [Planned] HAL: Hybrid Automata Library (https://halloworld.org/)
- [Planned] Chaste: Cancer, Heart and Soft Tissue Environment

Key Concepts
------------
- **TimeStep**: Single time point from a simulation (converted to SpatialTissueData)
- **Simulation**: Complete time series from a single simulation run
- **Experiment**: Collection of simulations with different parameters

The module enables:
1. Loading ABM outputs into SpatialTissueData for spatial analysis
2. Summarizing each time step as feature vectors using StatisticsPanel
3. Tracking spatial dynamics over simulation time
4. Comparing multiple simulations/parameter sets for ML training

Example
-------
>>> from spatialtissuepy.synthetic import PhysiCellSimulation
>>> from spatialtissuepy.summary import StatisticsPanel
>>>
>>> # Load a PhysiCell simulation
>>> sim = PhysiCellSimulation.from_output_folder('./output')
>>> 
>>> # Create analysis panel
>>> panel = StatisticsPanel()
>>> panel.add('cell_counts')
>>> panel.add('ripleys_h_max', max_radius=100)
>>> panel.add('mean_nearest_neighbor_distance')
>>>
>>> # Summarize all time steps
>>> df = sim.summarize(panel)
>>> # DataFrame with time as rows, statistics as columns
>>>
>>> # For multiple simulations (experiment)
>>> from spatialtissuepy.synthetic import PhysiCellExperiment
>>> experiment = PhysiCellExperiment.from_folders([
...     './output_control',
...     './output_treatment_low',
...     './output_treatment_high',
... ])
>>> master_df = experiment.summarize(panel)

References
----------
.. [1] Ghaffarizadeh, A. et al. (2018). PhysiCell: An open source physics-based
       cell simulator for 3-D multicellular systems. PLoS Comput Biol.
.. [2] Bravo, R. R. et al. (2020). Hybrid Automata Library: A flexible platform
       for hybrid modeling with real-time visualization. PLoS Comput Biol.
.. [3] Mirams, G. R. et al. (2013). Chaste: An open source C++ library for
       computational physiology and biology. PLoS Comput Biol.
"""

# Base classes for ABM interface
from .base import (
    ABMTimeStep,
    ABMSimulation,
    ABMExperiment,
)

# PhysiCell support
from .physicell import (
    PhysiCellTimeStep,
    PhysiCellSimulation,
    PhysiCellExperiment,
    read_physicell_timestep,
    read_physicell_simulation,
)

__all__ = [
    # Base classes
    'ABMTimeStep',
    'ABMSimulation',
    'ABMExperiment',
    # PhysiCell
    'PhysiCellTimeStep',
    'PhysiCellSimulation',
    'PhysiCellExperiment',
    'read_physicell_timestep',
    'read_physicell_simulation',
]
