"""
MCP server setup for spatialtissuepy.

Provides the FastMCP server instance and tool registration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from fastmcp import FastMCP

from .session import SessionManager
from .serialization import MCPSerializer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global server instance for context access
_server_instance: Optional[FastMCP] = None
_session_manager: Optional[SessionManager] = None
_serializer: Optional[MCPSerializer] = None
_data_dir: Optional[Path] = None


def create_server(
    name: str = "spatialtissuepy",
    session_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> FastMCP:
    """
    Create and configure the spatialtissuepy MCP server.

    Parameters
    ----------
    name : str
        Server name for identification.
    session_dir : Path or str, optional
        Directory for session persistence.
        Default: ~/.spatialtissuepy/mcp_sessions/
    data_dir : Path or str, optional
        Default directory for data file access.

    Returns
    -------
    FastMCP
        Configured server instance ready to run.

    Examples
    --------
    >>> server = create_server(data_dir="/path/to/data")
    >>> server.run()
    """
    global _server_instance, _session_manager, _serializer, _data_dir

    # Initialize FastMCP server
    mcp = FastMCP(
        name,
        instructions="""
spatialtissuepy MCP Server - Spatial Tissue Analysis Tools

This server provides tools for analyzing spatial organization of cells in
tissue samples from multiplexed imaging experiments and agent-based simulations.

WORKFLOW:
1. Load data: Use data_load_csv or data_load_json to load spatial data
2. Store session_id returned - it's needed for all subsequent operations
3. Analyze: Use spatial_*, statistics_*, network_*, lda_*, topology_* tools
4. Visualize: Use viz_* tools to generate plots (returns base64 PNG)
5. Extract features: Use summary_* tools for ML-ready feature vectors

KEY CONCEPTS:
- session_id: Unique identifier for your analysis session (persists across restarts)
- data_key: Name for stored datasets (default: "primary")
- Cell types: Categorical labels for each cell
- Markers: Quantitative expression data per cell

COMMON ANALYSIS PATTERNS:
- Clustering detection: statistics_ripleys_h
- Cell-cell interactions: statistics_colocalization_quotient
- Neighborhood analysis: spatial_nearest_neighbors, network_build_proximity_graph
- Topic modeling: lda_fit, lda_transform
- Topological analysis: topology_run_mapper

TIP: Start by loading data and calling data_get_info to understand your dataset.
""",
    )

    # Set up session manager
    if session_dir is None:
        session_dir = Path.home() / ".spatialtissuepy" / "mcp_sessions"
    else:
        session_dir = Path(session_dir).expanduser()

    _session_manager = SessionManager(session_dir)
    _serializer = MCPSerializer()
    _data_dir = Path(data_dir).expanduser() if data_dir else None
    _server_instance = mcp

    logger.info(f"Initialized MCP server '{name}'")
    logger.info(f"Session directory: {session_dir}")
    if _data_dir:
        logger.info(f"Data directory: {_data_dir}")

    # Register all tools
    from .tools import register_all_tools
    register_all_tools(mcp)

    return mcp


def get_server() -> FastMCP:
    """Get the current server instance."""
    if _server_instance is None:
        raise RuntimeError("Server not initialized. Call create_server() first.")
    return _server_instance


def get_session_manager() -> SessionManager:
    """Get the session manager from the current server context."""
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized. Call create_server() first.")
    return _session_manager


def get_serializer() -> MCPSerializer:
    """Get the serializer from the current server context."""
    if _serializer is None:
        raise RuntimeError("Serializer not initialized. Call create_server() first.")
    return _serializer


def get_data_dir() -> Optional[Path]:
    """Get the configured data directory."""
    return _data_dir


def resolve_data_path(file_path: str) -> Path:
    """
    Resolve a file path, using data_dir as base if path is relative.

    Parameters
    ----------
    file_path : str
        File path (absolute or relative).

    Returns
    -------
    Path
        Resolved absolute path.
    """
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    if _data_dir:
        return _data_dir / path
    return path.resolve()
