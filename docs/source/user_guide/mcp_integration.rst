===============================
MCP Integration (AI Agents)
===============================

spatialtissuepy ships with a `Model Context Protocol (MCP)
<https://modelcontextprotocol.io/>`_ server that lets AI agents — such as
Claude Desktop — drive a complete spatial analysis workflow on your behalf.
Rather than writing Python yourself, you can ask an agent in natural language to
load a tissue sample, quantify clustering, build a cell graph, and produce
figures; the agent calls the corresponding spatialtissuepy tools and reports the
results.

This chapter walks through enabling and using the server. For the full tool
inventory and the API reference, see :doc:`/api/mcp`.


What the MCP Server Enables
===========================

- **AI agent access** to the entire package: data I/O, spatial operations,
  spatial statistics, network analysis, spatial LDA, topological (Mapper)
  analysis, feature extraction, PhysiCell ABM integration, and visualization.
- **Natural-language workflows**: describe the analysis you want; the agent
  selects and sequences the appropriate tools.
- **Persistent sessions**: state is stored on disk, so multi-step analyses
  survive restarts and can be resumed later.
- **97 tools across 9 categories**, all thin wrappers over the same functions
  documented elsewhere in this guide — so results are identical to calling the
  Python API directly.


Installation
============

Install spatialtissuepy with the ``mcp`` extra:

.. code-block:: bash

    pip install spatialtissuepy[mcp]

This pulls in the MCP runtime dependencies and registers the
``spatialtissuepy-mcp`` command:

.. code-block:: bash

    spatialtissuepy-mcp --help


Configuring Claude Desktop
==========================

Add the server to your Claude Desktop configuration file
(``claude_desktop_config.json``), pointing ``--data-dir`` at the folder that
contains your tissue data:

.. code-block:: json

    {
      "mcpServers": {
        "spatialtissuepy": {
          "command": "spatialtissuepy-mcp",
          "args": ["--data-dir", "/path/to/tissue/data"]
        }
      }
    }

Restart Claude Desktop. The spatialtissuepy tools will then be available to the
assistant.


Example Workflow
================

Once the server is connected, you can drive an analysis conversationally. A
typical exchange might be:

    **You:** Load ``tissue.csv`` from my data directory, then tell me whether
    Tumor cells are spatially clustered.

    **Agent:** *(calls* ``data_load_csv`` *, then* ``statistics_ripleys_h`` *)*
    The Ripley's H curve rises well above zero across all radii up to 200 µm,
    indicating Tumor cells are significantly clustered relative to complete
    spatial randomness.

    **You:** Do Tumor and Immune cells colocalize? Show me a scatter plot.

    **Agent:** *(calls* ``statistics_colocalization_quotient`` *and*
    ``viz_plot_spatial_scatter`` *)*
    The colocalization quotient at 50 µm is 0.42 (< 1), suggesting Immune cells
    are under-represented near Tumor cells. Here is the spatial scatter: …

Behind the scenes the agent is calling the same functions you would use in code:

.. code-block:: python

    from spatialtissuepy.io import read_csv
    from spatialtissuepy.statistics import ripleys_h, colocalization_quotient
    from spatialtissuepy.viz import plot_spatial_scatter

    data = read_csv("tissue.csv", x_col="x", y_col="y", cell_type_col="cell_type")
    H = ripleys_h(data, radii=[25, 50, 100, 200])
    clq = colocalization_quotient(data, "Tumor", "Immune", radius=50)
    plot_spatial_scatter(data)


Session Management
==================

The server persists each analysis as a session under
``~/.spatialtissuepy/mcp_sessions/`` (override with ``--session-dir``). A session
stores the primary :class:`~spatialtissuepy.SpatialTissueData` object, any
derived data, serialized cell graphs, and fitted model parameters (with random
seeds, so models can be reconstructed exactly).

Use the data tools to manage sessions:

- ``data_list_sessions`` — list active sessions.
- ``data_delete_session`` — remove a session and its stored artifacts.

Because state lives on disk, you can close and reopen the agent and pick up an
analysis where you left off.
