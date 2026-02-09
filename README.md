# spatialtissuepy

**Spatial analysis tools for tissue biology**

A Python package for analyzing spatial organization of cells in tissue samples, with support for spatial statistics, network analysis, neighborhood analysis, Spatial LDA/TDA, agent-based modeling, and comprehensive visualization tools.

## Features

- **Core Data Structures**: Efficient storage and manipulation of spatial cell data
- **Spatial Statistics**: Ripley's K, G-function, co-localization analysis, hotspot detection
- **Spatial LDA**: Latent Dirichlet Allocation for discovering recurrent cellular neighborhoods
  - Multi-sample mode: treats each sample as a document
  - Poisson disk ROI sampling: maximally samples ROIs from a single FOV
- **Topological Data Analysis**: Mapper algorithm for building filtered spatial graphs and neighborhood detection
- **Network Analysis**: Graph-based spatial analysis with centrality, clustering, and assortativity metrics
- **Agent-based Modeling (ABM) Integration**: I/O for analysing and summarizing spatial output from ABM frameworks
  - Currently supporting: PhysiCell
- **Visualization Suite**: Comprehensive plotting tools for all spatial analyses
- **MCP Server Integration**: Model Context Protocol server for LLM/AI agent access (97 tools)
- **Multi-format I/O**: CSV, JSON, HDF5, AnnData support

## Installation

```bash
pip install spatialtissuepy
```

For visualization support:
```bash
pip install spatialtissuepy[viz]
```

For all optional dependencies:
```bash
pip install spatialtissuepy[all]
```

For MCP server support (LLM/AI agent integration):
```bash
pip install spatialtissuepy[mcp]
```

## Quick Start

```python
from spatialtissuepy import SpatialTissueData
from spatialtissuepy.io import read_csv

# Load data
data = SpatialTissueData.from_csv(
    'cells.csv',
    x_col='X_centroid',
    y_col='Y_centroid',
    celltype_col='phenotype'
)

# Basic info
print(data)
# SpatialTissueData
#   Cells: 10000
#   Dimensions: 2D
#   Cell types: 8
#   Bounds: x=[0.0, 1000.0], y=[0.0, 1000.0]

# Access data
print(data.cell_type_counts)
print(data.bounds)

# Spatial queries
neighbors = data.query_radius(point=[500, 500], radius=50)
distances, indices = data.query_knn(point=[500, 500], k=10)

# Subset data
t_cells = data.subset(cell_types=['T_cell', 'CD8_T_cell'])

# Export
data.to_csv('output.csv')
```

## Multi-sample Support

```python
# Load multi-sample data
data = SpatialTissueData.from_csv(
    'all_samples.csv',
    x_col='x',
    y_col='y',
    celltype_col='cell_type',
    sample_col='patient_id'
)

# Iterate over samples
for sample_id, sample_data in data.iter_samples():
    print(f"{sample_id}: {sample_data.n_cells} cells")
```

## MCP Server (LLM/AI Integration)

spatialtissuepy includes a Model Context Protocol (MCP) server that enables LLMs and AI agents to perform spatial tissue analysis through 97 tools across 9 categories.

**Start the server:**
```bash
spatialtissuepy-mcp --data-dir /path/to/data
```

**Claude Desktop configuration** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "spatialtissuepy": {
      "command": "spatialtissuepy-mcp",
      "args": ["--data-dir", "/path/to/tissue/data"]
    }
  }
}
```

**Tool categories:**
- `data_*` (14 tools): Load, save, inspect, subset data
- `spatial_*` (7 tools): Distances, neighbors, density
- `statistics_*` (10 tools): Ripley's K/L/H, colocalization, hotspots
- `network_*` (14 tools): Graph construction, centrality, clustering
- `lda_*` (8 tools): Spatial topic modeling
- `topology_*` (10 tools): Mapper/TDA analysis
- `summary_*` (8 tools): Feature extraction for ML
- `synthetic_*` (9 tools): PhysiCell ABM integration
- `viz_*` (17 tools): Visualization (returns base64 PNG)

## Requirements

- Python ≥ 3.8
- numpy ≥ 1.20
- scipy ≥ 1.7
- pandas ≥ 1.3
- scikit-learn ≥ 1.0

## Development Status

This package is under active development. 

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spatialtissuepy,
  title = {spatialtissuepy: Spatial analysis tools for tissue biology},
  author = {Eric Cramer},
  year = {2025},
  url = {https://github.com/spatialtissuepy/spatialtissuepy}
}
```
