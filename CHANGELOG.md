# Changelog

All notable changes to spatialtissuepy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-26

First beta release. This release adds AI agent access via an MCP server, a
complete tutorial suite, continuous integration, and a number of bug fixes and
numerical-stability improvements across the analysis modules.

### Added

- **MCP server integration** (`spatialtissuepy.mcp`): a Model Context Protocol
  server exposing **97 tools** across 9 categories (data, spatial, statistics,
  network, lda, topology, summary, synthetic, viz), installable via
  `pip install spatialtissuepy[mcp]` and launchable through the
  `spatialtissuepy-mcp` CLI entry point. Includes disk-based session persistence
  at `~/.spatialtissuepy/mcp_sessions/`.
- **Documentation**: Sphinx API reference and user-guide chapters for the MCP
  server (`api/mcp.rst`, `user_guide/mcp_integration.rst`).
- **Tutorials**: complete suite of 13 Jupyter notebooks covering quickstart,
  data loading, spatial analysis, statistics, neighborhoods, networks, spatial
  LDA, topology, visualization, multi-sample analysis, PhysiCell integration,
  advanced workflows, and extended Mapper analysis.
- **Continuous integration**: GitHub Actions workflow running the test suite.
- **Custom metrics**: support for registering user-defined metrics in the
  summary/panel system.
- **`CITATION.cff`** for academic citation.

### Changed

- Bumped development status from Alpha to Beta.
- Synchronized documentation version with the package version (0.2.0).
- Replaced placeholder repository URLs in the documentation with the canonical
  `github.com/emcramer/spatialtissuepy`.
- Documented the `microenvironment` module as a planned feature for v0.3.0.

### Fixed

- SpatialLDA serialization now supports full model reconstruction and
  transformation.
- Network and LDA API mismatches and related bugs.
- Spatial and statistics bug fixes and numerical-stability improvements.
- Runtime bugs in the MCP synthetic and viz tools.

[0.2.0]: https://github.com/emcramer/spatialtissuepy/releases/tag/v0.2.0
