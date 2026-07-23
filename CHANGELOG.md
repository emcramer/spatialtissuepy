# Changelog

All notable changes to spatialtissuepy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-23

PhysiCell reader correctness fixes, reported against v0.2.0 by the
llm-abm-consistency harness (OHSU ChangLab) and reproduced against both their
PhysiCell 1.14.2 output and this repository's bundled example simulation. This
release also repairs the CI lint gate, which had never passed, and raises the
Python floor to match what the dependencies actually require.

### Fixed

- **Cell matrix orientation is now resolved from the frame's declared variable
  count** rather than by comparing the matrix dimensions. The old heuristic
  (`if shape[0] > shape[1]: transpose`) assumed every frame holds more cells
  than the model has variables, and silently returned transposed garbage for
  any frame that does not — sparse initial conditions, early time steps, small
  explants, and simulations approaching extinction. A frame that matches
  neither orientation now raises `ValueError` instead of returning data. When
  no labels are available, the old heuristic still applies but emits a
  `UserWarning` naming the ambiguity.
- **`PhysiCellTimeStep.to_dataframe()` now honors `include_dead_cells`.** It
  previously ignored the flag while `n_cells`, `positions`, `to_spatial_data()`
  and `cell_counts_by_type()` all respected it, so the same object reported
  different populations depending on the accessor and row indices did not align
  between `to_dataframe()` and `positions`.
- **`parse_microenvironment_mat` validates matrix orientation** against the
  declared substrate count when `substrate_names` is supplied, converting a
  would-be silent misread into a clear error.
- **`__version__` is now read from the installed distribution metadata**, so it
  cannot drift from `pyproject.toml`. It reported `0.1.0` at the `v0.2.0` tag,
  which misattributed results in downstream provenance capture. A regression
  test asserts the two agree.

### Added

- **`parse_cells_mat(..., labels=...)`**: derive column positions from the
  MultiCellDS `<labels>` block that `parse_physicell_xml` already parses. The
  parsed labels were previously stored at `metadata.extra['custom_labels']` and
  never read anywhere in the package. Precedence is `index_mapping` >
  `labels` > the existing row-count autodetect, so behavior is unchanged when
  labels are not supplied.
- **`'columns'` key in the `parse_cells_mat` result**: every labelled column,
  including standard PhysiCell fields absent from the hard-coded index tables
  (`is_motile`, `migration_speed`) and model-specific custom variables. The
  return dict was previously a closed set, so anything else the model wrote was
  unreachable.
- **`'orientation'` key** reporting the on-disk layout of `raw_data`, either
  `'variables_x_cells'` or `'cells_x_variables'`.
- **`expand_cell_labels()` and `declared_variable_count()`** helpers for
  working with `<labels>` blocks. Vector labels expand as `{name}_x/_y/_z` for
  `size == 3` and `{name}_0 … {name}_{n-1}` otherwise.
- **`to_dataframe(extra_columns=True)`** appends every labelled column. Off by
  default, since a full PhysiCell frame carries 150+ variables.
- **`to_dataframe(include_dead_cells=...)`** to override the instance attribute
  per call.
- Regression tests (`tests/test_physicell_labels.py`, 25 tests) covering label
  expansion, all four orientation cases, cross-model label isolation, the
  microenvironment guard, dead-cell filtering, and version consistency.

### Changed

- **`raw_data` from `parse_cells_mat` is now returned exactly as loaded from
  disk.** It was previously the reoriented matrix, so callers could not use it
  to recover the truth when the orientation heuristic misfired. Callers relying
  on the transposed `raw_data` should consult the new `'orientation'` key.
- `PhysiCellTimeStep._load_cell_data()` and `to_trajectory_dataframe()` now
  resolve columns from each frame's own XML, so models with differing variable
  counts parse correctly in the same session.
- **Raised the minimum Python version to 3.10** (from a nominal 3.8).
  `requires-python` claimed `>=3.8`, but the `mcp` extra depends on `fastmcp`,
  which requires 3.10+, so `pip install spatialtissuepy[all]` could never
  succeed on 3.8 or 3.9. The two had drifted since the MCP server was added and
  nothing surfaced it, because the CI lint gate failed before any install ran.
  Python 3.8 reached end of life in October 2024 and 3.9 in October 2025, and
  current releases of the core scientific stack already require 3.11+. The CI
  matrix now runs 3.10-3.13.

### Build

- **The `ruff` lint job passes for the first time.** It reported ~6,100 errors
  on every branch, and because the test job declares `needs: lint`, no pull
  request had been able to run the test suite. The linter was also flagging
  real defects behind the noise: `Union` used in annotations but never imported
  across seven modules (latent under `from __future__ import annotations`, but
  raised `NameError` under `typing.get_type_hints`), a matplotlib symbol
  referenced in module-level annotations while imported only lazily, and three
  broken forward references. The bulk of the remaining diff was trailing
  whitespace, import ordering, and unused locals.
- Moved the deprecated top-level `ruff` config under `[tool.ruff.lint]` and
  aligned the `black`, `ruff`, and `mypy` target versions with the new floor.
- **Decoupled the `lint` and `test` CI jobs.** The test job no longer declares
  `needs: lint`, so the two run as independent siblings. Previously a lint
  failure skipped the entire test matrix, meaning a style regression could
  silently suppress all test signal — which is how the `fastmcp` install
  incompatibility went undetected.

### Known gaps

- `parse_microenvironment_mat` is implemented and exported but still not wired
  into `PhysiCellTimeStep` / `PhysiCellSimulation`; substrate fields remain
  reachable only through the parser. Planned for a future release.

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

[0.3.0]: https://github.com/emcramer/spatialtissuepy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emcramer/spatialtissuepy/releases/tag/v0.2.0
