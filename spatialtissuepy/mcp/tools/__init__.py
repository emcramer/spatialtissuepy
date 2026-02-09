"""
MCP tools for spatialtissuepy.

Collects and registers all tools from category modules.

Tool Categories
---------------
- data_* : Load, save, and inspect spatial tissue data (14 tools)
- spatial_* : Distance and neighborhood operations (7 tools)
- statistics_* : Spatial statistics (10 tools)
- network_* : Graph-based analysis (14 tools)
- lda_* : Spatial topic modeling (8 tools)
- topology_* : Mapper/TDA analysis (10 tools)
- summary_* : Feature extraction for ML (8 tools)
- synthetic_* : PhysiCell ABM integration (9 tools)
- viz_* : Visualization (17 tools)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_all_tools(mcp: "FastMCP") -> None:
    """
    Register all spatialtissuepy tools with the MCP server.

    Parameters
    ----------
    mcp : FastMCP
        Server instance to register tools with.
    """
    # Import and register each tool category
    # Each module has a register_tools(mcp) function

    from . import data
    data.register_tools(mcp)
    logger.debug("Registered data tools")

    from . import spatial
    spatial.register_tools(mcp)
    logger.debug("Registered spatial tools")

    from . import statistics
    statistics.register_tools(mcp)
    logger.debug("Registered statistics tools")

    from . import network
    network.register_tools(mcp)
    logger.debug("Registered network tools")

    from . import lda
    lda.register_tools(mcp)
    logger.debug("Registered LDA tools")

    from . import topology
    topology.register_tools(mcp)
    logger.debug("Registered topology tools")

    from . import summary
    summary.register_tools(mcp)
    logger.debug("Registered summary tools")

    from . import synthetic
    synthetic.register_tools(mcp)
    logger.debug("Registered synthetic tools")

    from . import viz
    viz.register_tools(mcp)
    logger.debug("Registered visualization tools")

    logger.info("All MCP tools registered successfully")
