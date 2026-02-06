==========
Quickstart
==========

This guide will walk you through the basics of spatialtissuepy in 5 minutes.


Creating Spatial Data
---------------------

The central object in spatialtissuepy is :class:`~spatialtissuepy.SpatialTissueData`:

.. code-block:: python

    import numpy as np
    from spatialtissuepy import SpatialTissueData

    # Create sample data
    n_cells = 500
    coordinates = np.random.uniform(0, 1000, (n_cells, 2))
    cell_types = np.random.choice(
        ['Tumor', 'CD8_T_cell', 'Macrophage', 'Stromal'],
        n_cells,
        p=[0.4, 0.2, 0.15, 0.25]
    )

    # Create SpatialTissueData object
    data = SpatialTissueData(
        coordinates=coordinates,
        cell_types=cell_types
    )

    print(data)


Loading Data from Files
-----------------------

Load data from CSV files:

.. code-block:: python

    from spatialtissuepy.io import read_csv

    data = read_csv(
        'tissue_data.csv',
        x_col='X',
        y_col='Y',
        cell_type_col='CellType'
    )


Basic Visualization
-------------------

Visualize the spatial distribution:

.. code-block:: python

    from spatialtissuepy.viz import plot_spatial_scatter

    plot_spatial_scatter(data, color_by='cell_type')


Computing Spatial Statistics
----------------------------

Calculate Ripley's H function to detect clustering:

.. code-block:: python

    from spatialtissuepy.statistics import ripleys_h

    radii = np.linspace(10, 200, 20)
    H = ripleys_h(data, radii=radii)

    # H > 0 indicates clustering
    # H < 0 indicates dispersion
    # H ≈ 0 indicates random distribution

Calculate colocalization between cell types:

.. code-block:: python

    from spatialtissuepy.statistics import colocalization_quotient

    clq = colocalization_quotient(
        data,
        type_a='CD8_T_cell',
        type_b='Tumor',
        radius=50
    )

    # CLQ > 1: Types colocalize (more neighbors than expected)
    # CLQ < 1: Types avoid each other
    # CLQ ≈ 1: Random mixing


Building Cell Networks
----------------------

Create a spatial graph of cell relationships:

.. code-block:: python

    from spatialtissuepy.network import CellGraph

    graph = CellGraph.from_spatial_data(
        data,
        method='proximity',
        radius=50.0
    )

    print(f"Nodes: {graph.n_nodes}")
    print(f"Edges: {graph.n_edges}")
    print(f"Density: {graph.density:.4f}")


Extracting Features for ML
--------------------------

Create a feature vector from spatial data:

.. code-block:: python

    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary

    # Define which metrics to compute
    panel = StatisticsPanel()
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('mean_nearest_neighbor_distance')
    panel.add('ripleys_h_max', max_radius=100)

    # Compute summary
    summary = SpatialSummary(data, panel)
    features = summary.to_dict()

    print(features)


Next Steps
----------

- :doc:`concepts` - Learn the key concepts in depth
- :doc:`../tutorials/index` - Follow step-by-step tutorials
- :doc:`../api/index` - Explore the full API reference
