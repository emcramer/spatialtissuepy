==========
User Guide
==========

This section provides in-depth guides on using spatialtissuepy for
spatial tissue analysis.

.. toctree::
   :maxdepth: 2

   data_handling
   spatial_statistics
   network_analysis
   topic_modeling
   mapper_analysis
   feature_extraction
   visualization
   abm_integration


Overview
--------

spatialtissuepy provides a comprehensive toolkit for analyzing the spatial
organization of cells in tissue samples. This user guide covers:

1. **Data Handling**: Loading, manipulating, and saving spatial tissue data
2. **Spatial Statistics**: Quantifying spatial patterns and relationships
3. **Network Analysis**: Graph-based analysis of cell relationships
4. **Topic Modeling**: Discovering cellular microenvironments with Spatial LDA
5. **Mapper Analysis**: Topological data analysis for community discovery
6. **Feature Extraction**: Building feature vectors for machine learning
7. **Visualization**: Creating publication-quality figures
8. **ABM Integration**: Analyzing agent-based model outputs


Workflow Overview
-----------------

A typical analysis workflow in spatialtissuepy:

.. code-block:: python

    # 1. Load data
    from spatialtissuepy.io import read_csv
    data = read_csv('tissue.csv', x_col='X', y_col='Y', cell_type_col='CellType')

    # 2. Explore and visualize
    from spatialtissuepy.viz import plot_spatial_scatter
    plot_spatial_scatter(data)

    # 3. Compute statistics
    from spatialtissuepy.statistics import ripleys_h, colocalization_quotient
    H = ripleys_h(data, radii=[25, 50, 100, 200])
    clq = colocalization_quotient(data, 'TypeA', 'TypeB', radius=50)

    # 4. Extract features
    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary
    panel = StatisticsPanel()
    panel.add('cell_counts')
    panel.add('ripleys_h_max', max_radius=100)
    summary = SpatialSummary(data, panel)
    features = summary.to_dict()

    # 5. (Optional) Train ML model
    # Use features as input to sklearn, etc.


Choosing the Right Analysis
---------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Question
     - Method
   * - Are cells clustered or dispersed?
     - Ripley's K/L/H functions
   * - Do two cell types colocalize?
     - Colocalization quotient
   * - Where are hotspots of a cell type?
     - Getis-Ord Gi*, Local Moran's I
   * - What is the local neighborhood diversity?
     - Neighborhood entropy
   * - How connected are cells spatially?
     - Cell graph analysis
   * - What microenvironment patterns exist?
     - Spatial LDA
   * - What is the overall tissue structure?
     - Mapper algorithm
   * - How do samples differ?
     - Multi-sample feature comparison
