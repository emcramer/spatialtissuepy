===============
Mapper Analysis
===============

This guide covers the Mapper algorithm for topological data analysis.


What is Mapper?
---------------

Mapper is an algorithm from topological data analysis (TDA) that creates
a simplified graph representation of high-dimensional data. It reveals
the "shape" of the data, identifying clusters, branches, and loops.

**How it works:**

1. Project data through a filter function
2. Cover the filter range with overlapping intervals
3. Cluster points within each interval
4. Connect clusters that share points

The result is a graph where nodes are clusters and edges represent overlap.


Basic Usage
-----------

.. code-block:: python

    from spatialtissuepy.topology import spatial_mapper

    result = spatial_mapper(
        data,
        filter_fn='density',
        n_intervals=10,
        overlap=0.5,
        neighborhood_radius=50
    )

    print(f"Nodes: {result.n_nodes}")
    print(f"Edges: {result.n_edges}")
    print(f"Components: {result.n_components}")


Filter Functions
----------------

The filter function determines what aspect of the data is revealed.

Built-in Filters
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Density filter: local cell density
    result = spatial_mapper(data, filter_fn='density', n_intervals=10)

    # PCA filter: first principal component
    result = spatial_mapper(data, filter_fn='pca', n_intervals=10)

Spatial Filters
~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.topology import (
        radial_filter,
        distance_to_type_filter,
        spatial_density_filter
    )

    # Radial: distance from a center point
    result = spatial_mapper(
        data,
        filter_fn=radial_filter(center=[500, 500]),
        n_intervals=12
    )

    # Distance to cell type: gradients from tumor
    result = spatial_mapper(
        data,
        filter_fn=distance_to_type_filter('Tumor'),
        n_intervals=10
    )

Choosing a Filter
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Filter
     - Use case
   * - density
     - Find dense vs sparse regions
   * - eccentricity
     - Find central vs peripheral cells
   * - distance_to_type
     - Discover gradients from a cell population
   * - radial
     - Analyze radial organization around a point
   * - x_coordinate
     - Horizontal spatial gradients


Parameters
----------

n_intervals
~~~~~~~~~~~

Number of overlapping bins covering the filter range.

- **Low (5-8)**: Coarse view, fewer nodes
- **Medium (10-15)**: Standard resolution
- **High (15-25)**: Fine detail, more nodes

overlap
~~~~~~~

Fraction of overlap between adjacent intervals (0 to 1).

- **Low (0.2-0.3)**: Sparser graph, fewer edges
- **Medium (0.4-0.5)**: Standard connectivity
- **High (0.6-0.8)**: Highly connected graph

.. code-block:: python

    # Compare parameter effects
    for n_int in [5, 10, 20]:
        result = spatial_mapper(data, n_intervals=n_int, overlap=0.5)
        print(f"n_intervals={n_int}: {result.n_nodes} nodes, {result.n_edges} edges")


Class Interface
---------------

For more control:

.. code-block:: python

    from spatialtissuepy.topology import SpatialMapper

    mapper = SpatialMapper(
        filter_fn=distance_to_type_filter('Tumor'),
        n_intervals=12,
        overlap=0.4,
        clustering='dbscan',
        clustering_params={'eps': 30, 'min_samples': 3},
        min_cluster_size=5
    )

    result = mapper.fit(data, neighborhood_radius=50)


Analyzing Results
-----------------

Node Information
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get cells in a node
    node_cells = result.get_node_members(node_id=0)
    print(f"Node 0 has {len(node_cells)} cells")

    # Node spatial centroids
    centroids = result.node_spatial_centroids
    print(f"Node 0 centroid: {centroids[0]}")

    # Node cell type compositions
    for node_id, composition in result.node_compositions.items():
        print(f"Node {node_id}: {composition}")

Hub and Bridge Nodes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.topology import find_hub_nodes, find_bridge_nodes

    # Hub nodes: highly connected
    hubs = find_hub_nodes(result, top_n=5)
    for node_id, degree in hubs:
        print(f"Hub {node_id}: degree {degree}")

    # Bridge nodes: connect separate regions
    bridges = find_bridge_nodes(result)
    print(f"Bridge nodes: {bridges}")

Component Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.topology import component_statistics

    stats = component_statistics(result, data)

    for comp_id, comp_stats in stats.items():
        print(f"\nComponent {comp_id}:")
        print(f"  Nodes: {comp_stats['n_nodes']}")
        print(f"  Cells: {comp_stats['n_cells']}")
        print(f"  Dominant type: {comp_stats['dominant_type']}")


Visualization
-------------

.. code-block:: python

    from spatialtissuepy.viz import plot_mapper_graph, plot_mapper_spatial

    # Mapper graph
    plot_mapper_graph(result, color_by='size')

    # Mapper nodes on tissue
    plot_mapper_spatial(result, data)


Feature Extraction
------------------

Extract features from Mapper for ML:

.. code-block:: python

    from spatialtissuepy.topology import extract_mapper_features

    features = extract_mapper_features(result)

    print("Mapper features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

Features include:

- Number of nodes, edges, components
- Graph density, mean degree
- Node size statistics
- Connectivity metrics


Biological Interpretation
-------------------------

Interpreting the Graph
~~~~~~~~~~~~~~~~~~~~~~

- **Nodes**: Represent cell communities with similar neighborhoods
- **Edges**: Represent spatial overlap between communities
- **Components**: Separate tissue regions
- **Hubs**: Transition zones connecting different microenvironments
- **Bridges**: Critical cells connecting otherwise separate regions

Example Patterns
~~~~~~~~~~~~~~~~

1. **Single connected component**: Continuous tissue organization
2. **Multiple components**: Distinct tissue regions
3. **Hub nodes**: Tumor-immune interfaces
4. **Linear structure**: Spatial gradient (e.g., from tumor core to periphery)
5. **Branching**: Multiple distinct microenvironment transitions


Advanced Usage
--------------

Custom Filter
~~~~~~~~~~~~~

.. code-block:: python

    def my_filter(coordinates, features, data):
        # Combine density and distance to tumor
        from spatialtissuepy.topology import spatial_density_filter
        density = spatial_density_filter(data, radius=50)
        dist_tumor = distance_to_type_filter('Tumor')(data)

        # Normalize and combine
        d_norm = (density - density.min()) / (density.max() - density.min())
        t_norm = (dist_tumor - dist_tumor.min()) / (dist_tumor.max() - dist_tumor.min())

        return 0.5 * d_norm + 0.5 * t_norm

    result = spatial_mapper(data, filter_fn=my_filter, n_intervals=10)

2D Filter (Multi-Filter)
~~~~~~~~~~~~~~~~~~~~~~~~

For 2D filters, use two filter values:

.. code-block:: python

    # This creates a 2D cover (more nodes, captures more structure)
    from spatialtissuepy.topology import SpatialMapper

    def filter_2d(coords, features, data):
        x = coords[:, 0]  # X position
        density = spatial_density_filter(data, radius=50)
        return np.column_stack([x, density])

    # Note: 2D Mapper requires additional configuration
