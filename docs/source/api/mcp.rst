===========
MCP Server
===========

The ``spatialtissuepy.mcp`` module provides a `Model Context Protocol (MCP)
<https://modelcontextprotocol.io/>`_ server that exposes spatialtissuepy
functionality to LLMs and coding agents (such as Claude) through a standardized
protocol. It enables AI-driven spatial tissue analysis: an agent can load data,
compute spatial statistics, build cell graphs, fit topic models, run topological
analyses, and render visualizations — all by calling tools.

For a task-oriented walkthrough, see :doc:`/user_guide/mcp_integration`.


Architecture
============

The MCP server is a **thin wrapper layer** over the existing package API. It
adds protocol handling, session management, and serialization, but does not
reimplement any analytical functionality::

    ┌─────────────────────────────────────────────────────────┐
    │                    MCP Server Layer                       │
    │   (protocol handling, session management, serialization)  │
    └───────────────────────────┬───────────────────────────────┘
                                │ calls
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    MCP Tools Layer                        │
    │      (thin wrappers: input validation, formatting)        │
    └───────────────────────────┬───────────────────────────────┘
                                │ calls
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │               Existing spatialtissuepy API                │
    │   (core, spatial, statistics, lda, topology, network …)   │
    └─────────────────────────────────────────────────────────┘


Installation and Launch
=======================

The MCP server ships as an optional extra:

.. code-block:: bash

    pip install spatialtissuepy[mcp]

This installs the additional dependencies (``mcp``, ``pydantic``, ``uvicorn``)
and registers the ``spatialtissuepy-mcp`` command-line entry point:

.. code-block:: bash

    spatialtissuepy-mcp [--data-dir PATH] [--session-dir PATH] [--debug]

The server can also be created programmatically:

.. code-block:: python

    from spatialtissuepy.mcp import create_server

    server = create_server(
        data_dir="/path/to/data",
        session_dir="~/.spatialtissuepy/mcp_sessions",
    )
    server.run()


Claude Desktop Configuration
============================

Add the server to your Claude Desktop configuration file
(``claude_desktop_config.json``):

.. code-block:: json

    {
      "mcpServers": {
        "spatialtissuepy": {
          "command": "spatialtissuepy-mcp",
          "args": ["--data-dir", "/path/to/tissue/data"]
        }
      }
    }


Session Persistence
===================

Analysis state is persisted to disk so that long-running, multi-step analyses
survive server restarts. Sessions are stored under
``~/.spatialtissuepy/mcp_sessions/`` (configurable via ``--session-dir``).

Each session is a directory containing JSON metadata plus pickled data objects::

    ~/.spatialtissuepy/mcp_sessions/
    └── {session_id}/
        ├── metadata.json        # Session info, timestamps, results cache
        ├── primary.pkl          # Main SpatialTissueData object
        ├── {other_key}.pkl      # Additional data objects
        ├── graphs/              # Serialized NetworkX graphs (reconstructable)
        │   └── {graph_key}.json
        └── models/              # Model parameters + random seeds
            └── {model_key}.json

Graphs are serialized as node/edge lists with attributes so that the NetworkX
graph can be reconstructed exactly. Models (LDA, Mapper) store their constructor
parameters and random seeds so that re-fitting yields identical results.


Tool Categories
===============

The server exposes **97 tools** across nine categories. Tools are
module-prefixed (e.g. ``statistics_ripleys_k``) to make organization clear and
avoid name collisions.

Data tools (``data_*``) — 14 tools
    ``data_load_csv``, ``data_load_json``, ``data_save_csv``, ``data_save_json``,
    ``data_get_info``, ``data_get_cell_types``, ``data_get_cell_counts``,
    ``data_get_bounds``, ``data_get_markers``, ``data_subset_by_type``,
    ``data_subset_by_region``, ``data_subset_by_sample``, ``data_list_sessions``,
    ``data_delete_session``.

Spatial tools (``spatial_*``) — 7 tools
    ``spatial_pairwise_distances``, ``spatial_nearest_neighbors``,
    ``spatial_radius_neighbors``, ``spatial_density``, ``spatial_boundary_cells``,
    ``spatial_convex_hull``, ``spatial_voronoi_areas``.

Statistics tools (``statistics_*``) — 10 tools
    ``statistics_ripleys_k``, ``statistics_ripleys_l``, ``statistics_ripleys_h``,
    ``statistics_pair_correlation``, ``statistics_nearest_neighbor_g``,
    ``statistics_colocalization_quotient``, ``statistics_cross_k``,
    ``statistics_getis_ord_gi_star``, ``statistics_morans_i``,
    ``statistics_mark_correlation``.

Network tools (``network_*``) — 14 tools
    ``network_build_proximity_graph``, ``network_build_knn_graph``,
    ``network_build_delaunay_graph``, ``network_build_gabriel_graph``,
    ``network_degree_centrality``, ``network_betweenness_centrality``,
    ``network_closeness_centrality``, ``network_eigenvector_centrality``,
    ``network_clustering_coefficient``, ``network_average_clustering``,
    ``network_type_assortativity``, ``network_degree_assortativity``,
    ``network_attribute_mixing_matrix``, ``network_connected_components``.

LDA tools (``lda_*``) — 8 tools
    ``lda_fit``, ``lda_transform``, ``lda_get_topic_composition``,
    ``lda_get_dominant_topics``, ``lda_topic_coherence``, ``lda_topic_diversity``,
    ``lda_topic_spatial_consistency``, ``lda_select_n_topics``.

Topology tools (``topology_*``) — 10 tools
    ``topology_run_mapper``, ``topology_density_filter``,
    ``topology_eccentricity_filter``, ``topology_pca_filter``,
    ``topology_distance_to_type_filter``, ``topology_radial_filter``,
    ``topology_get_nodes``, ``topology_get_edges``, ``topology_get_components``,
    ``topology_hub_nodes``.

Summary tools (``summary_*``) — 8 tools
    ``summary_create_panel``, ``summary_add_metric``,
    ``summary_list_available_metrics``, ``summary_compute``, ``summary_to_dict``,
    ``summary_to_array``, ``summary_multi_sample``,
    ``summary_multi_sample_to_dataframe``.

Synthetic/ABM tools (``synthetic_*``) — 9 tools
    ``synthetic_load_physicell_simulation``, ``synthetic_load_physicell_timestep``,
    ``synthetic_list_physicell_timesteps``, ``synthetic_get_timestep``,
    ``synthetic_timestep_to_spatial_data``, ``synthetic_cell_count_trajectory``,
    ``synthetic_type_proportions_trajectory``, ``synthetic_summarize_simulation``,
    ``synthetic_load_physicell_experiment``.

Visualization tools (``viz_*``) — 17 tools
    All ``viz_plot_*`` functions return a base64-encoded PNG so the agent can
    decide when to render: ``viz_plot_spatial_scatter``, ``viz_plot_cell_types``,
    ``viz_plot_density_map``, ``viz_plot_marker_expression``, ``viz_plot_voronoi``,
    ``viz_plot_ripleys_curve``, ``viz_plot_colocalization_heatmap``,
    ``viz_plot_hotspot_map``, ``viz_plot_neighborhood_enrichment``,
    ``viz_plot_network``, ``viz_plot_degree_distribution``,
    ``viz_plot_mixing_matrix``, ``viz_plot_topic_composition``,
    ``viz_plot_topic_spatial``, ``viz_plot_mapper_graph``, ``viz_plot_trajectory``,
    and ``viz_save_figure`` (returns a filepath).


API Reference
=============

.. automodule:: spatialtissuepy.mcp
   :members:
   :undoc-members:
   :show-inheritance:
