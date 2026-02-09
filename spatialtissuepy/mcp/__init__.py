"""
MCP (Model Context Protocol) server for spatialtissuepy.

This module provides an MCP server that exposes spatialtissuepy functionality
to LLMs and coding agents through a standardized protocol.

Installation
------------
pip install spatialtissuepy[mcp]

Usage
-----
Command line::

    spatialtissuepy-mcp [--data-dir PATH] [--session-dir PATH] [--debug]

Programmatic::

    from spatialtissuepy.mcp import create_server

    server = create_server(
        data_dir="/path/to/data",
        session_dir="~/.spatialtissuepy/mcp_sessions",
    )
    server.run()

Claude Desktop Configuration
----------------------------
Add to your Claude Desktop config (claude_desktop_config.json)::

    {
        "mcpServers": {
            "spatialtissuepy": {
                "command": "spatialtissuepy-mcp",
                "args": ["--data-dir", "/path/to/tissue/data"]
            }
        }
    }

Available Tool Categories
-------------------------
- data_* : Load, save, and inspect spatial tissue data
- spatial_* : Distance and neighborhood operations
- statistics_* : Spatial statistics (Ripley's K, colocalization, etc.)
- network_* : Graph-based analysis
- lda_* : Spatial topic modeling
- topology_* : Mapper/TDA analysis
- summary_* : Feature extraction for ML
- synthetic_* : PhysiCell ABM integration
- viz_* : Visualization (returns base64 PNG)
"""

from __future__ import annotations

__all__ = [
    "HAS_MCP",
    "create_server",
    "SessionManager",
    "MCPSerializer",
]

# Check for MCP dependencies
try:
    import fastmcp
    import pydantic
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _check_mcp_dependencies() -> None:
    """Raise ImportError if MCP dependencies are not installed."""
    if not HAS_MCP:
        raise ImportError(
            "MCP dependencies are not installed. "
            "Install with: pip install spatialtissuepy[mcp]"
        )


def create_server(
    name: str = "spatialtissuepy",
    session_dir: str | None = None,
    data_dir: str | None = None,
):
    """
    Create and configure the spatialtissuepy MCP server.

    Parameters
    ----------
    name : str, optional
        Server name for identification. Default: "spatialtissuepy"
    session_dir : str or Path, optional
        Directory for session persistence.
        Default: ~/.spatialtissuepy/mcp_sessions/
    data_dir : str or Path, optional
        Default directory for data file access.

    Returns
    -------
    FastMCP
        Configured server instance ready to run.

    Examples
    --------
    >>> from spatialtissuepy.mcp import create_server
    >>> server = create_server(data_dir="/path/to/data")
    >>> server.run()
    """
    _check_mcp_dependencies()
    from .server import create_server as _create_server
    return _create_server(name=name, session_dir=session_dir, data_dir=data_dir)


# Lazy imports for optional components
def __getattr__(name: str):
    if name == "SessionManager":
        _check_mcp_dependencies()
        from .session import SessionManager
        return SessionManager
    elif name == "MCPSerializer":
        _check_mcp_dependencies()
        from .serialization import MCPSerializer
        return MCPSerializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
