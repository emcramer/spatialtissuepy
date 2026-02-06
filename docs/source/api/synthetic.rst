=========
Synthetic
=========

.. module:: spatialtissuepy.synthetic

The synthetic module provides interfaces for loading and analyzing outputs
from agent-based modeling (ABM) frameworks, particularly PhysiCell.


PhysiCell Classes
-----------------

.. autoclass:: spatialtissuepy.synthetic.PhysiCellSimulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spatialtissuepy.synthetic.PhysiCellTimeStep
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spatialtissuepy.synthetic.PhysiCellExperiment
   :members:
   :undoc-members:
   :show-inheritance:


Convenience Functions
---------------------

.. autofunction:: spatialtissuepy.synthetic.read_physicell_timestep

.. autofunction:: spatialtissuepy.synthetic.read_physicell_simulation

.. autofunction:: spatialtissuepy.synthetic.read_physicell_experiment

.. autofunction:: spatialtissuepy.synthetic.discover_physicell_timesteps


Cell State Helpers
------------------

Utility functions for working with PhysiCell cell cycle phases:

.. autofunction:: spatialtissuepy.synthetic.is_alive

.. autofunction:: spatialtissuepy.synthetic.is_dead

.. autofunction:: spatialtissuepy.synthetic.get_phase_name

.. autodata:: spatialtissuepy.synthetic.CELL_CYCLE_PHASES


Base Classes
------------

Abstract base classes for implementing new ABM interfaces:

.. autoclass:: spatialtissuepy.synthetic.ABMTimeStep
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spatialtissuepy.synthetic.ABMSimulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: spatialtissuepy.synthetic.ABMExperiment
   :members:
   :undoc-members:
   :show-inheritance:


Module Contents
---------------

.. automodule:: spatialtissuepy.synthetic
   :members:
   :undoc-members:
   :show-inheritance:
