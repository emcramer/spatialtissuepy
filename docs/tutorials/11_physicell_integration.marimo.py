import marimo

__generated_with = "0.16.0"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Tutorial 11: PhysiCell Integration

    **Duration:** 25-30 minutes

    This tutorial covers how to load and analyze agent-based model (ABM) outputs from PhysiCell using spatialtissuepy. We'll work with a real tumor-immune microenvironment simulation featuring tumor cells, macrophages (M0, M1, M2 polarization), and T cells (effector and exhausted states).

    ## Learning Objectives

    By the end of this tutorial, you will be able to:
    - Load PhysiCell simulation outputs using the `PhysiCellSimulation` class
    - Extract spatial data from individual timesteps
    - Track spatial metrics over simulation time
    - Analyze cell population dynamics and spatial organization
    - Use spatial analysis to understand tumor-immune interactions

    ## Prerequisites

    - Tutorials 1-10 completed
    - Basic understanding of agent-based modeling
    - (Optional) PhysiCell installation for running your own simulations

    ## Biological Context

    **Why analyze ABM outputs?**
    - Validate simulation predictions against experimental data
    - Track how spatial organization evolves over time
    - Identify emergent spatial patterns in tumor-immune dynamics
    - Generate hypotheses for experimental testing

    **PhysiCell:**
    - Open-source physics-based cell simulator
    - Outputs MultiCellDS XML format
    - Commonly used for tumor growth, immune interactions, drug response

    **Our Example Simulation:**
    - Tumor-immune microenvironment model
    - 6 cell types: Malignant epithelial, M0/M1/M2 macrophages, Effector/Exhausted T cells
    - ~696 timesteps (saved every 10 minutes, ~116 hours total)
    - 2mm × 2mm domain

    ## Setup
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    from spatialtissuepy import SpatialTissueData
    from spatialtissuepy.synthetic.physicell import (
        PhysiCellSimulation,
        PhysiCellTimeStep,
        read_physicell_timestep,
        discover_physicell_timesteps,
        is_alive,
        is_dead,
    )
    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary
    from spatialtissuepy.viz import plot_spatial_scatter

    np.random.seed(42)
    return (
        Path,
        PhysiCellSimulation,
        SpatialSummary,
        StatisticsPanel,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 1: Understanding PhysiCell Output Structure

    PhysiCell outputs data in the MultiCellDS format:
    ```
    output/
    ├── initial.xml           # Initial conditions
    ├── output00000000.xml    # Timestep 0 metadata
    ├── output00000000_cells_physicell.mat  # Timestep 0 cell data
    ├── output00000001.xml    # Timestep 1 metadata
    ├── output00000001_cells_physicell.mat  # Timestep 1 cell data
    ├── ...
    ├── config.xml            # Simulation parameters
    └── final.xml             # Final state
    ```

    Each XML file contains simulation metadata (time, units, substrate info), while the corresponding `.mat` file stores the actual cell data:
    - Cell positions (x, y, z)
    - Cell types and phenotypes
    - Cell states (live, dead, cycling phases)
    - Volumes, radii, and other properties
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 2: Loading PhysiCell Data

    Let's load the example tumor-immune simulation included with spatialtissuepy.
    """
    )
    return


@app.cell
def _(Path, PhysiCellSimulation):
    # Path to example PhysiCell simulation output
    # Adjust this path based on your spatialtissuepy installation
    example_data_path = Path("../examples/sample_data/example_physicell_sim")

    # If running from a different location, try the package location
    if not example_data_path.exists():
        import spatialtissuepy
        pkg_path = Path(spatialtissuepy.__file__).parent.parent
        example_data_path = pkg_path / "examples" / "sample_data" / "example_physicell_sim"

    # Load the simulation
    sim = PhysiCellSimulation.from_output_folder(example_data_path)

    print(f"Simulation ID: {sim.simulation_id}")
    print(f"Number of timesteps: {sim.n_timesteps}")
    print(f"Time range: {sim.times[0]:.0f} - {sim.times[-1]:.0f} minutes")
    print(f"Time range: {sim.times[0]/60:.1f} - {sim.times[-1]/60:.1f} hours")
    print(f"\nCell type mapping:")
    for cell_id, cell_name in sim.cell_type_mapping.items():
        print(f"  {cell_id}: {cell_name}")
    return (sim,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### 2.1 Loading Individual Timesteps

    We can load specific timesteps by index or by simulation time.
    """
    )
    return


@app.cell
def _(sim):
    sim.get_timestep
    return


@app.cell
def _(sim):
    # Load the first timestep
    ts_initial = sim.get_timestep(0)
    print(f"Initial timestep:")
    print(f"  Time: {ts_initial.time:.0f} min ({ts_initial.time/60:.1f} hours)")
    print(f"  Live cells: {ts_initial.n_cells}")
    print(f"  Dead cells: {ts_initial.n_dead_cells}")
    print(f"  Cell types: {ts_initial.cell_types}")

    # Load a middle timestep
    ts_middle = sim.get_timestep(sim.n_timesteps // 2)
    print(f"\nMiddle timestep (index {sim.n_timesteps // 2}):")
    print(f"  Time: {ts_middle.time:.0f} min ({ts_middle.time/60:.1f} hours)")
    print(f"  Live cells: {ts_middle.n_cells}")
    print(f"  Dead cells: {ts_middle.n_dead_cells}")

    # Load the final timestep
    ts_final = sim.get_timestep(sim.n_timesteps-1)
    print(f"\nFinal timestep:")
    print(f"  Time: {ts_final.time:.0f} min ({ts_final.time/60:.1f} hours)")
    print(f"  Live cells: {ts_final.n_cells}")
    print(f"  Dead cells: {ts_final.n_dead_cells}")
    return (ts_final,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### 2.2 Converting to SpatialTissueData

    PhysiCell timesteps can be converted to `SpatialTissueData` objects for spatial analysis with spatialtissuepy.
    """
    )
    return


@app.cell
def _(ts_final):
    # Convert to SpatialTissueData
    spatial_data = ts_final.to_spatial_data()

    print(f"SpatialTissueData object:")
    print(f"  Coordinates shape: {spatial_data.coordinates.shape}")
    print(f"  Cell types: {spatial_data.cell_types_unique}")
    print(f"  Has markers: {spatial_data.markers is not None}")
    if spatial_data.markers is not None:
        print(f"  Marker columns: {list(spatial_data.markers.columns)}")
    return (spatial_data,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 3: Visualizing Spatial Organization

    Let's visualize how the tumor-immune microenvironment evolves over time.
    """
    )
    return


@app.cell
def _():
    # Define colors for each cell type
    cell_colors = {
        'malignant_epithelial_cell': '#E41A1C',  # Red
        'M0_macrophage': '#377EB8',               # Blue
        'M1_macrophage': '#4DAF4A',               # Green
        'M2_macrophage': '#984EA3',               # Purple
        'effector_T_cell': '#FF7F00',             # Orange
        'exhausted_T_cell': '#FFFF33',            # Yellow
    }
    return (cell_colors,)


@app.cell
def _(cell_colors, np, plt, sim, spatial_data):
    def _():
        # Select timesteps to visualize (every ~20 hours)
        hours_to_show = [0, 20, 40, 60, 80, 100]
        indices_to_show = []

        for target_hours in hours_to_show:
            target_min = target_hours * 60
            # Find closest timestep
            idx = np.argmin(np.abs(sim.times - target_min))
            indices_to_show.append(idx)

        with plt.style.context('seaborn-v0_8'):

            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            for ax, idx in zip(axes.flat, indices_to_show):
                ts = sim.get_timestep(idx)
                data = ts.to_spatial_data()

                # Plot each cell type
                for cell_type in spatial_data.cell_types_unique:
                    mask = data.cell_types == cell_type
                    coords = data.coordinates[mask]
                    color = cell_colors.get(cell_type, '#999999')
                    # Shorten labels for legend
                    label = cell_type.replace('_cell', '').replace('_macrophage', '_mac').replace('malignant_epithelial', 'tumor')
                    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=5, alpha=0.6, label=label)

                ax.set_xlim(-1000, 1000)
                ax.set_ylim(-1000, 1000)
                ax.set_aspect('equal')
                ax.set_title(f"t = {ts.time/60:.0f} hours\n({ts.n_cells} cells)")
                ax.set_xlabel("x (μm)")
                ax.set_ylabel("y (μm)")

            # Add legend to last subplot
            axes.flat[-1].legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 4: Tracking Cell Population Dynamics

    Let's analyze how cell populations change over the course of the simulation.
    """
    )
    return


@app.cell
def _(sim):
    # Get cell counts over time (built-in method)
    counts_df = sim.cell_counts_over_time()
    print("Cell counts DataFrame columns:")
    print(counts_df.columns.tolist())
    print(f"\nDataFrame shape: {counts_df.shape}")
    counts_df.head()
    return (counts_df,)


@app.cell
def _(cell_colors, counts_df, plt):
    def _():
        with plt.style.context('seaborn-v0_8'):
            # Plot cell population dynamics
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Convert time to hours
            time_hours = counts_df['time'] / 60

            # Plot tumor cells
            ax1.plot(time_hours, counts_df['n_malignant_epithelial_cell'], 
                     color=cell_colors['malignant_epithelial_cell'], linewidth=2, label='Tumor')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Cell Count')
            ax1.set_title('Tumor Cell Population')
            ax1.grid(alpha=0.3)
            ax1.legend()

            # Plot immune cell populations
            immune_types = [
                ('n_M0_macrophage', 'M0_macrophage', 'M0 Mac'),
                ('n_M1_macrophage', 'M1_macrophage', 'M1 Mac'),
                ('n_M2_macrophage', 'M2_macrophage', 'M2 Mac'),
                ('n_effector_T_cell', 'effector_T_cell', 'Effector T'),
                ('n_exhausted_T_cell', 'exhausted_T_cell', 'Exhausted T'),
            ]

            for col, ct, label in immune_types:
                if col in counts_df.columns:
                    ax2.plot(time_hours, counts_df[col], 
                             color=cell_colors[ct], linewidth=2, label=label)

            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Cell Count')
            ax2.set_title('Immune Cell Populations')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell
def _(counts_df, plt):
    def _():
        # Plot macrophage polarization ratio (M1 vs M2)
        with plt.style.context('seaborn-v0_8'):
            fig2, ax3 = plt.subplots(figsize=(10, 5))
        
            time_h = counts_df['time'] / 60
            m1_counts = counts_df['n_M1_macrophage'].values
            m2_counts = counts_df['n_M2_macrophage'].values
        
            # Avoid division by zero
            total_polarized = m1_counts + m2_counts
            m1_ratio = m1_counts / (total_polarized + 1e-6)  # M1 / (M1 + M2)
        
            ax3.plot(time_h, m1_ratio, 'g-', linewidth=2, label='M1 Ratio')
            ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal M1/M2')
            ax3.fill_between(time_h, m1_ratio, 0.5, 
                             where=(m1_ratio > 0.5), alpha=0.3, color='green', 
                             label='Pro-inflammatory')
            ax3.fill_between(time_h, m1_ratio, 0.5, 
                             where=(m1_ratio < 0.5), alpha=0.3, color='purple',
                             label='Anti-inflammatory')
        
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('M1 / (M1 + M2) Ratio')
            ax3.set_title('Macrophage Polarization Over Time')
            ax3.set_ylim(0, 1)
            ax3.legend(loc='upper right')
            ax3.grid(alpha=0.3)
        
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 5: Tracking Spatial Metrics Over Time

    Now let's compute spatial statistics at multiple timepoints to understand how spatial organization evolves.
    """
    )
    return


@app.cell
def _(SpatialSummary, StatisticsPanel, np, pd, sim):
    # Create a statistics panel for tracking
    panel = StatisticsPanel(name='tumor_immune_tracking')
    panel.add('cell_counts')
    panel.add('mean_nearest_neighbor_distance')

    # Sample timesteps for spatial analysis (every ~10 hours)
    sample_hours = np.arange(0, 120, 10)
    sample_indices = []

    def _():    
        for target_hours in sample_hours:
            target_min = target_hours * 60
            idx = np.argmin(np.abs(sim.times - target_min))
            if idx not in sample_indices:  # Avoid duplicates
                sample_indices.append(idx)
    
        print(f"Computing spatial statistics for {len(sample_indices)} timesteps...")
    
        # Compute statistics over time
        metrics_over_time = []
    
        for i, idx in enumerate(sample_indices):
            ts = sim.get_timestep(idx)
            tissue = ts.to_spatial_data()
    
            summary = SpatialSummary(tissue, panel)
            results = summary.to_dict()
            results['time'] = ts.time
            results['time_hours'] = ts.time / 60
            results['time_index'] = ts.time_index
            metrics_over_time.append(results)
    
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(sample_indices)} timesteps")
    
        metrics_df = pd.DataFrame(metrics_over_time)
        print("\nTracked metrics:")
        print(metrics_df.columns.tolist())

        return metrics_df
    metrics_df = _()
    mo.ui.dataframe(metrics_df)
    return metrics_df, sample_indices


@app.cell
def _(metrics_df, plt):
    def _():
        with plt.style.context('seaborn-v0_8'):
            # Plot spatial metrics over time
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        
            # Mean nearest neighbor distance
            ax_nn = axes3[0]
            ax_nn.plot(metrics_df['time_hours'], metrics_df['mean_nnd'], 
                       'b-o', linewidth=2, markersize=4)
            ax_nn.set_xlabel('Time (hours)')
            ax_nn.set_ylabel('Mean NN Distance (μm)')
            ax_nn.set_title('Cell Density Over Time\n(Lower = More Dense)')
            ax_nn.grid(alpha=0.3)
        
            # Total cell count trend
            ax_total = axes3[1]
            ax_total.plot(metrics_df['time_hours'], metrics_df['n_cells'], 
                          'g-o', linewidth=2, markersize=4)
            ax_total.set_xlabel('Time (hours)')
            ax_total.set_ylabel('Total Live Cells')
            ax_total.set_title('Total Cell Population')
            ax_total.grid(alpha=0.3)
        
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 6: Type-Specific Spatial Analysis

    Let's analyze the spatial relationships between different cell types, particularly tumor-immune interactions.
    """
    )
    return


@app.cell
def _(np, sample_indices, sim):
    from scipy.spatial.distance import cdist

    def compute_mean_distance_to_tumor(tissue):
        """Compute mean distance from immune cells to nearest tumor cell."""
        tumor_mask = tissue.cell_types == 'malignant_epithelial_cell'
        tumor_coords = tissue.coordinates[tumor_mask]

        if len(tumor_coords) == 0:
            return {}

        results = {}
        immune_types = [
            'M0_macrophage', 'M1_macrophage', 'M2_macrophage',
            'effector_T_cell', 'exhausted_T_cell'
        ]

        for cell_type in immune_types:
            mask = tissue.cell_types == cell_type
            if np.sum(mask) > 0:
                immune_coords = tissue.coordinates[mask]
                # Compute distance to nearest tumor cell
                distances = cdist(immune_coords, tumor_coords)
                min_distances = distances.min(axis=1)
                results[f'{cell_type}_to_tumor'] = np.mean(min_distances)

        return results

    def _():
        # Compute tumor proximity over time
        tumor_proximity = []
    
        for idx in sample_indices:
            ts = sim.get_timestep(idx)
            tissue = ts.to_spatial_data()
    
            row = {
                'time_hours': ts.time / 60,
                'n_tumor': np.sum(tissue.cell_types == 'malignant_epithelial_cell')
            }
            row.update(compute_mean_distance_to_tumor(tissue))
            tumor_proximity.append(row)
        return tumor_proximity
    tumor_proximity = _()
    return (tumor_proximity,)


@app.cell
def _(cell_colors, pd, plt, tumor_proximity):
    proximity_df = pd.DataFrame(tumor_proximity)

    def _():
        with plt.style.context('seaborn-v0_8'):
            fig4, ax4 = plt.subplots(figsize=(12, 6))
        
            immune_proximity_cols = [c for c in proximity_df.columns if '_to_tumor' in c]
        
            for col in immune_proximity_cols:
                cell_type = col.replace('_to_tumor', '')
                color = cell_colors.get(cell_type, '#999999')
                label = cell_type.replace('_macrophage', ' Mac').replace('_cell', '')
                ax4.plot(proximity_df['time_hours'], proximity_df[col], 
                         color=color, linewidth=2, marker='o', markersize=4, label=label)
        
            ax4.set_xlabel('Time (hours)')
            ax4.set_ylabel('Mean Distance to Nearest Tumor Cell (μm)')
            ax4.set_title('Immune Cell Proximity to Tumor Over Time')
            ax4.legend(loc='upper right')
            ax4.grid(alpha=0.3)
        
            plt.tight_layout()
            plt.show()
            plt.close()
    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 7: Analyzing Spatial Snapshots with Full Statistics

    Let's perform a comprehensive spatial analysis at key timepoints.
    """
    )
    return


@app.cell
def _(SpatialSummary, StatisticsPanel, np, pd, sim):
    # Create a comprehensive statistics panel
    full_panel = StatisticsPanel(name='comprehensive')
    full_panel.add('cell_counts')
    full_panel.add('mean_nearest_neighbor_distance')
    #full_panel.add('shannon_entropy')

    # Analyze at three key timepoints
    key_times = [0, 50, 100]  # hours
    key_results = []

    def _():
        for target_hours in key_times:
            target_min = target_hours * 60
            idx = np.argmin(np.abs(sim.times - target_min))
            ts = sim.get_timestep(idx)
            tissue = ts.to_spatial_data()
    
            summary = SpatialSummary(tissue, full_panel)
            results = summary.to_dict()
            results['time_hours'] = ts.time / 60
            results['timepoint'] = f"t={target_hours}h"
            key_results.append(results)
    
        key_df = pd.DataFrame(key_results)
        key_df = key_df.set_index('timepoint')
    
        # Display key metrics
        display_cols = ['time_hours', 'n_cells', 'mean_nn_distance']#, 'shannon_entropy']
        display_cols = [c for c in display_cols if c in key_df.columns]
        key_df[display_cols]

        return display_cols, key_df
    display_cols, key_df = _()
    mo.ui.dataframe(key_df)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Section 8: Working with Raw Cell Data

    Sometimes you need access to the raw cell data for custom analyses.
    """
    )
    return


@app.cell
def _(sim):
    # Get raw cell data as DataFrame
    ts_final_raw = sim.get_timestep(-1)
    cell_df = ts_final_raw.to_dataframe()

    print("Raw cell data columns:")
    print(cell_df.columns.tolist())
    print(f"\nDataFrame shape: {cell_df.shape}")

    # Show summary by cell type
    cell_summary = cell_df.groupby('cell_type').agg({
        'cell_id': 'count',
        'volume': ['mean', 'std'],
        'radius': ['mean', 'std'],
        'is_alive': 'mean',
    }).round(2)

    cell_summary.columns = ['_'.join(col).strip() for col in cell_summary.columns.values]
    cell_summary
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Exercise: Analyze Tumor-Immune Dynamics

    Using the tools from this tutorial, complete the following exercises:

    1. **Identify the peak macrophage polarization:** At what timepoint is the M1/M2 ratio highest? What might cause this?

    2. **Track T cell exhaustion:** Plot the ratio of exhausted to effector T cells over time. Does exhaustion increase as the tumor grows?

    3. **Spatial clustering analysis:** Using the techniques from earlier tutorials, analyze whether immune cells cluster near the tumor core or periphery.
    """
    )
    return


@app.cell
def _():
    # Exercise 1: Find peak M1/M2 ratio
    # Your code here

    # Exercise 2: T cell exhaustion ratio
    # Your code here

    # Exercise 3: Spatial clustering analysis
    # Your code here
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Summary

    In this tutorial, you learned:

    - **Loading PhysiCell data:** Using `PhysiCellSimulation.from_output_folder()` to load complete simulations
    - **Timestep access:** Getting individual timesteps with `get_timestep()` and converting to `SpatialTissueData`
    - **Population dynamics:** Tracking cell counts and polarization states over time with `cell_counts_over_time()`
    - **Temporal spatial analysis:** Computing spatial statistics at multiple timepoints
    - **Cell type interactions:** Analyzing distances between cell types to understand tumor-immune proximity

    **Key Insights from This Simulation:**
    - Tumor cell population dynamics are influenced by immune cell activity
    - Macrophage polarization shifts over time as the microenvironment evolves
    - Spatial proximity between immune and tumor cells reveals infiltration patterns
    - The same spatialtissuepy tools work seamlessly with simulation data

    **PhysiCell Integration Benefits:**
    - ABM outputs are naturally suited for spatial analysis
    - Temporal tracking reveals dynamics not captured in static experimental snapshots
    - Same analysis pipeline for simulations and experimental data enables validation

    ## Next Steps

    - **Tutorial 12: Advanced Workflows** - Complete analysis pipelines for publication-ready results
    - Explore the PhysiCell website (physicell.org) for more simulation examples
    - Try generating your own simulations and analyzing them with spatialtissuepy
    """
    )
    return


if __name__ == "__main__":
    app.run()
