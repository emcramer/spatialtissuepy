"""
CLI entry point for spatialtissuepy MCP server.

Usage
-----
    spatialtissuepy-mcp [OPTIONS]

Options
-------
    --data-dir PATH     Default directory for data files
    --session-dir PATH  Directory for session persistence
    --transport TYPE    Transport type: stdio (default)
    --debug             Enable debug logging
    --version           Show version and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for MCP server."""
    parser = argparse.ArgumentParser(
        prog="spatialtissuepy-mcp",
        description="spatialtissuepy MCP server for spatial tissue analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spatialtissuepy-mcp
      Start server with default settings (stdio transport)

  spatialtissuepy-mcp --data-dir /path/to/data
      Start with default data directory

  spatialtissuepy-mcp --debug
      Start with debug logging enabled

Claude Desktop Configuration:
  Add to your claude_desktop_config.json:

  {
    "mcpServers": {
      "spatialtissuepy": {
        "command": "spatialtissuepy-mcp",
        "args": ["--data-dir", "/path/to/tissue/data"]
      }
    }
  }
""",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Default directory for data files",
    )

    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Session persistence directory (default: ~/.spatialtissuepy/mcp_sessions)",
    )

    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport type (default: stdio)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    # Handle version flag
    if args.version:
        try:
            from spatialtissuepy import __version__
            print(f"spatialtissuepy-mcp version {__version__}")
        except ImportError:
            print("spatialtissuepy-mcp (version unknown)")
        return 0

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr to avoid interfering with MCP protocol
    )

    logger = logging.getLogger(__name__)

    # Check for MCP dependencies
    try:
        from . import HAS_MCP
        if not HAS_MCP:
            print(
                "Error: MCP dependencies not installed.\n"
                "Install with: pip install spatialtissuepy[mcp]",
                file=sys.stderr,
            )
            return 1
    except ImportError as e:
        print(f"Error importing spatialtissuepy.mcp: {e}", file=sys.stderr)
        return 1

    # Create and run server
    try:
        from .server import create_server

        logger.info("Creating spatialtissuepy MCP server...")

        server = create_server(
            session_dir=args.session_dir,
            data_dir=args.data_dir,
        )

        logger.info(f"Starting server with {args.transport} transport...")

        # Run the server
        server.run(transport=args.transport)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=args.debug)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
