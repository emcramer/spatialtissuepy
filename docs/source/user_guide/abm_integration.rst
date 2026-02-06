===============
ABM Integration
===============

This guide covers analyzing agent-based model (ABM) outputs.


Overview
--------

spatialtissuepy provides interfaces for loading and analyzing outputs from
agent-based modeling frameworks, particularly PhysiCell.

**Supported frameworks:**

- PhysiCell (MultiCellDS XML format)
- Custom ABM outputs (via base classes)


PhysiCell Integration
---------------------

Loading a Simulation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.synthetic import PhysiCellSimulation

    # Load from output folder
    sim = PhysiCellSimulation.from_output_folder('./output')

    print(f"Timesteps: {sim.n_timesteps}")
    print(f"Times: {sim.times}")

Accessing Timesteps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get specific timestep
    timestep = sim.get_timestep(100)

    print(f"Time: {timestep.time}")
    print(f"Cells: {timestep.n_cells}")
    print(f"Cell types: {timestep.cell_types}")

    # Convert to SpatialTissueData
    data = timestep.to_spatial_data()

Iterating Over Time
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Iterate over all timesteps
    for timestep in sim:
        data = timestep.to_spatial_data()
        # Analyze...

    # Or specific indices
    for i in [0, 50, 100]:
        timestep = sim.get_timestep(i)


Temporal Analysis
-----------------

Cell Count Trajectories
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Built-in method
    counts_df = sim.cell_counts_over_time()
    print(counts_df.head())

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for col in counts_df.columns:
        if col.startswith('n_'):
            plt.plot(counts_df['time'], counts_df[col], label=col[2:])

    plt.xlabel('Time (min)')
    plt.ylabel('Cell count')
    plt.legend()
    plt.show()

Spatial Metrics Over Time
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.summary import StatisticsPanel

    # Define metrics
    panel = StatisticsPanel()
    panel.add('cell_counts')
    panel.add('ripleys_h_max', max_radius=100)
    panel.add('mean_neighborhood_entropy', radius=50)

    # Summarize all timesteps
    df = sim.summarize(panel, progress=True)

    print(df.head())

    # Plot temporal dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['ripleys_h_max'], 'b-')
    plt.xlabel('Time (min)')
    plt.ylabel("Ripley's H max")
    plt.title('Spatial Clustering Over Time')
    plt.show()


Experiment Analysis
-------------------

Loading Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.synthetic import PhysiCellExperiment

    # Load from multiple folders
    exp = PhysiCellExperiment.from_folders([
        './output_control',
        './output_drug_low',
        './output_drug_high'
    ])

    print(f"Simulations: {exp.n_simulations}")

Comparing Conditions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Summarize all simulations
    master_df = exp.summarize(panel, progress=True)

    # Group by simulation
    for sim_id in master_df['simulation_id'].unique():
        sim_data = master_df[master_df['simulation_id'] == sim_id]
        plt.plot(sim_data['time'], sim_data['n_Tumor'], label=sim_id)

    plt.xlabel('Time')
    plt.ylabel('Tumor cell count')
    plt.legend()
    plt.show()

Endpoint Comparison
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare final timesteps
    final_df = exp.summarize_final_timesteps(panel)

    print(final_df[['simulation_id', 'final_n_cells', 'ripleys_h_max']])


Visualization
-------------

Timestep Snapshots
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.viz import plot_spatial_scatter
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    timesteps_to_show = [0, 20, 40, 60, 80, 100]

    for ax, t in zip(axes.flat, timesteps_to_show):
        data = sim.get_timestep(t).to_spatial_data()
        plot_spatial_scatter(data, ax=ax)
        ax.set_title(f'Time = {sim.get_timestep(t).time:.0f} min')

    plt.tight_layout()
    plt.show()

Animation
~~~~~~~~~

.. code-block:: python

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        data = sim.get_timestep(frame).to_spatial_data()
        plot_spatial_scatter(data, ax=ax)
        ax.set_title(f'Time = {sim.get_timestep(frame).time:.0f}')
        return ax,

    anim = FuncAnimation(
        fig, update,
        frames=range(0, sim.n_timesteps, 5),
        interval=200
    )

    anim.save('simulation.gif', writer='pillow')


Custom ABM Integration
----------------------

For other ABM frameworks, extend the base classes:

.. code-block:: python

    from spatialtissuepy.synthetic import ABMTimeStep, ABMSimulation
    import numpy as np

    class MyABMTimeStep(ABMTimeStep):
        """Custom timestep implementation."""

        def __init__(self, time, time_index, source_path, cell_data):
            super().__init__(time, time_index, source_path)
            self._cell_data = cell_data

        def to_spatial_data(self):
            from spatialtissuepy import SpatialTissueData
            return SpatialTissueData(
                coordinates=self._cell_data['coords'],
                cell_types=self._cell_data['types']
            )

        @property
        def n_cells(self):
            return len(self._cell_data['types'])

        @property
        def cell_types(self):
            return list(np.unique(self._cell_data['types']))


    class MyABMSimulation(ABMSimulation):
        """Custom simulation implementation."""

        def __init__(self, output_folder):
            super().__init__(output_folder)
            self._timesteps = self._load_timesteps()

        def _load_timesteps(self):
            # Load your data here
            pass

        @property
        def n_timesteps(self):
            return len(self._timesteps)

        @property
        def times(self):
            return np.array([ts.time for ts in self._timesteps])

        @property
        def time_indices(self):
            return np.arange(self.n_timesteps)

        def get_timestep(self, index):
            return self._timesteps[index]

        def get_timestep_by_time(self, time, tolerance=1e-6):
            idx = np.argmin(np.abs(self.times - time))
            return self._timesteps[idx]


Best Practices
--------------

1. **Sample timesteps**: For long simulations, analyze every Nth timestep
2. **Cache summaries**: Save computed DataFrames to avoid recomputation
3. **Memory management**: Load timesteps one at a time for large simulations
4. **Parallel processing**: Use multiprocessing for parameter sweeps
5. **Validation**: Compare simulation outputs to experimental data
