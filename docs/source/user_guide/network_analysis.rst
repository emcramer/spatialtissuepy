================
Network Analysis
================

This guide covers graph-based analysis of spatial tissue data.


Building Cell Graphs
--------------------

Graph Construction Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.network import CellGraph

    # Proximity graph: connect cells within radius
    graph = CellGraph.from_spatial_data(
        data,
        method='proximity',
        radius=50.0
    )

    # k-Nearest neighbors graph
    graph = CellGraph.from_spatial_data(
        data,
        method='knn',
        k=6
    )

    # Delaunay triangulation
    graph = CellGraph.from_spatial_data(
        data,
        method='delaunay',
        max_edge_length=100  # Optional: prune long edges
    )

Choosing a Method
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Method
     - Pros
     - Cons
   * - Proximity
     - Biologically intuitive, adjustable radius
     - Requires choosing radius
   * - k-NN
     - Consistent degree per cell
     - May create non-local edges
   * - Delaunay
     - No parameters needed
     - May include long edges


Graph Properties
----------------

Basic Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

    print(f"Nodes: {graph.n_nodes}")
    print(f"Edges: {graph.n_edges}")
    print(f"Density: {graph.density:.4f}")
    print(f"Cell types: {graph.cell_types_unique}")

Node Access
~~~~~~~~~~~

.. code-block:: python

    # Get nodes of a specific type
    tumor_nodes = graph.get_nodes_by_type('Tumor')

    # Get neighbors of a node
    neighbors = graph.neighbors_of_type(node=0)

    # Get neighbors of specific type
    tumor_neighbors = graph.neighbors_of_type(node=0, cell_type='Tumor')


Centrality Measures
-------------------

Degree Centrality
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.network import degree_centrality

    centrality = degree_centrality(graph)

    # Find most connected cells
    top_cells = np.argsort(centrality)[::-1][:10]

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~

Identifies cells that bridge different regions:

.. code-block:: python

    from spatialtissuepy.network import betweenness_centrality

    betweenness = betweenness_centrality(graph)

    # High betweenness = bridge between communities

Closeness Centrality
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.network import closeness_centrality

    closeness = closeness_centrality(graph)


Assortativity
-------------

Type Assortativity
~~~~~~~~~~~~~~~~~~

Measures whether cells connect preferentially to same-type cells:

.. code-block:: python

    from spatialtissuepy.network import type_assortativity

    assort = type_assortativity(graph)

    # Interpretation:
    # assort > 0: Cells prefer same-type neighbors
    # assort < 0: Cells prefer different-type neighbors
    # assort ≈ 0: Random mixing

Attribute Mixing Matrix
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from spatialtissuepy.network import attribute_mixing_matrix

    mixing = attribute_mixing_matrix(graph)
    print(mixing)

    # Rows/columns are cell types
    # Values are fraction of edges between types


Edge Analysis
-------------

Edge Type Counts
~~~~~~~~~~~~~~~~

.. code-block:: python

    edge_counts = graph.edge_type_counts()

    for (type_a, type_b), count in edge_counts.items():
        print(f"{type_a} - {type_b}: {count} edges")


Subgraph Analysis
-----------------

Extract subgraphs for specific analysis:

.. code-block:: python

    # Subgraph with only Tumor and T_cell
    subgraph = graph.subgraph_by_type(['Tumor', 'CD8_T_cell'])

    print(f"Subgraph nodes: {subgraph.n_nodes}")
    print(f"Subgraph edges: {subgraph.n_edges}")


Visualization
-------------

.. code-block:: python

    from spatialtissuepy.viz import plot_cell_graph, plot_degree_distribution

    # Plot graph on tissue
    plot_cell_graph(graph, data, edge_alpha=0.2)

    # Degree distribution
    plot_degree_distribution(graph)


NetworkX Integration
--------------------

Access the underlying NetworkX graph:

.. code-block:: python

    # Get NetworkX graph
    G = graph.to_networkx()

    # Use any NetworkX function
    import networkx as nx

    # Connected components
    components = list(nx.connected_components(G))

    # Clustering coefficient
    clustering = nx.clustering(G)

    # Shortest path
    path = nx.shortest_path(G, source=0, target=100)


Community Detection
-------------------

.. code-block:: python

    import networkx as nx
    from networkx.algorithms import community

    G = graph.to_networkx()

    # Louvain community detection
    communities = community.louvain_communities(G)

    print(f"Found {len(communities)} communities")


Performance Tips
----------------

1. **Large graphs**: Use proximity with larger radius for sparser graphs
2. **Memory**: Consider subsetting data for very large samples
3. **Caching**: Results are cached in CellGraph; call ``graph.clear_cache()``
   if needed
