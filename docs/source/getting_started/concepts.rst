============
Key Concepts
============

This page explains the core concepts in spatialtissuepy.


SpatialTissueData
-----------------

The :class:`~spatialtissuepy.SpatialTissueData` class is the central data container:

.. code-block:: python

    from spatialtissuepy import SpatialTissueData

    data = SpatialTissueData(
        coordinates=coords,      # (n_cells, 2) or (n_cells, 3) array
        cell_types=types,        # (n_cells,) array of strings
        markers=markers_df,      # Optional: pandas DataFrame
        sample_ids=sample_ids,   # Optional: for multi-sample data
        metadata={'roi': 'A1'}   # Optional: custom metadata
    )

Key properties:

- ``data.n_cells``: Number of cells
- ``data.cell_types_unique``: List of unique cell types
- ``data.coordinates``: Cell coordinates array
- ``data.markers``: Marker expression DataFrame
- ``data.bounds``: Spatial extent dictionary


Spatial Statistics
------------------

Spatial statistics quantify how cells are organized in space.

**Ripley's K/L/H Functions**

These functions measure spatial clustering:

- **K(r)**: Expected number of cells within distance r of a typical cell
- **L(r)**: Variance-stabilized K, where L(r) = sqrt(K(r)/π)
- **H(r)**: L(r) - r, centered so H=0 for complete spatial randomness

.. code-block:: python

    from spatialtissuepy.statistics import ripleys_h

    H = ripleys_h(data, radii=[25, 50, 100, 200])
    # H > 0: clustering
    # H < 0: dispersion

**Colocalization**

Measures whether cell types are found together more than expected:

.. code-block:: python

    from spatialtissuepy.statistics import colocalization_quotient

    clq = colocalization_quotient(data, 'TypeA', 'TypeB', radius=50)
    # CLQ > 1: colocalization
    # CLQ < 1: avoidance

**Hotspot Detection**

Identifies statistically significant clusters:

.. code-block:: python

    from spatialtissuepy.statistics import cell_type_hotspots

    result = cell_type_hotspots(data, 'Tumor', radius=50)
    hotspot_cells = result['hotspot_idx']


Cell Neighborhoods
------------------

A **neighborhood** is the set of cells within a specified distance of a focal cell.

.. code-block:: python

    from spatialtissuepy.spatial import compute_neighborhoods

    # Get neighbors within 50 units
    neighborhoods = compute_neighborhoods(data, method='radius', radius=50)

**Neighborhood Composition**

The cell type makeup of each cell's neighborhood:

.. code-block:: python

    from spatialtissuepy.spatial import neighborhood_composition

    composition = neighborhood_composition(data, radius=50)
    # Returns (n_cells, n_cell_types) matrix

**Neighborhood Entropy**

Measures local diversity:

.. code-block:: python

    from spatialtissuepy.spatial import neighborhood_entropy

    entropy = neighborhood_entropy(data, radius=50)
    # High entropy = diverse neighborhood
    # Low entropy = homogeneous neighborhood


Cell Networks
-------------

Cell networks represent tissue as a graph where cells are nodes and
spatial relationships are edges.

**Graph Construction Methods**

.. code-block:: python

    from spatialtissuepy.network import CellGraph

    # Proximity graph: connect cells within radius
    graph = CellGraph.from_spatial_data(data, method='proximity', radius=50)

    # k-Nearest neighbors graph
    graph = CellGraph.from_spatial_data(data, method='knn', k=6)

    # Delaunay triangulation
    graph = CellGraph.from_spatial_data(data, method='delaunay')

**Network Metrics**

.. code-block:: python

    # Degree centrality
    centrality = graph.degree_centrality()

    # Cell type assortativity
    assort = graph.type_assortativity()

    # Edge type counts
    edge_counts = graph.edge_type_counts()


Spatial LDA
-----------

Spatial LDA discovers recurrent **cellular microenvironment patterns**
(topics) from neighborhood compositions.

.. code-block:: python

    from spatialtissuepy.lda import SpatialLDA

    model = SpatialLDA(n_topics=5, neighborhood_radius=50)
    model.fit(data)

    # Get topic weights for each cell
    topic_weights = model.transform(data)

    # Get dominant topic
    dominant_topic = model.predict(data)

Topics represent patterns like "tumor core", "immune infiltrate",
"stromal border", etc.


Mapper Algorithm
----------------

The **Mapper** algorithm creates a simplified graph representation that
reveals tissue structure.

.. code-block:: python

    from spatialtissuepy.topology import spatial_mapper

    result = spatial_mapper(
        data,
        filter_fn='density',
        n_intervals=10,
        overlap=0.5
    )

    print(f"Found {result.n_nodes} community nodes")
    print(f"Connected components: {result.n_components}")


Statistics Panels
-----------------

A **StatisticsPanel** defines which metrics to compute:

.. code-block:: python

    from spatialtissuepy.summary import StatisticsPanel, SpatialSummary

    panel = StatisticsPanel(name='my_panel')
    panel.add('cell_counts')
    panel.add('cell_proportions')
    panel.add('ripleys_h_max', max_radius=100)
    panel.add('colocalization_quotient', type_a='A', type_b='B', radius=50)

    summary = SpatialSummary(data, panel)
    features = summary.to_array()  # For ML

Predefined panels are available:

.. code-block:: python

    from spatialtissuepy.summary import load_panel

    panel = load_panel('basic')      # Essential metrics
    panel = load_panel('spatial')    # Spatial statistics
    panel = load_panel('comprehensive')  # All categories


Multi-Sample Analysis
---------------------

Analyze cohorts of tissue samples:

.. code-block:: python

    from spatialtissuepy.summary import MultiSampleSummary

    samples = [data1, data2, data3]
    summary = MultiSampleSummary(samples, panel, sample_ids=['S1', 'S2', 'S3'])

    # Get feature matrix for ML
    df = summary.to_dataframe()


ABM Integration
---------------

Load agent-based model outputs:

.. code-block:: python

    from spatialtissuepy.synthetic import PhysiCellSimulation

    sim = PhysiCellSimulation.from_output_folder('./output')

    # Iterate over timesteps
    for timestep in sim:
        data = timestep.to_spatial_data()
        # Analyze...

    # Compute metrics over time
    df = sim.summarize(panel)
