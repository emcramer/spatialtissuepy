# Testing Infrastructure for spatialtissuepy

This directory contains the test suite for spatialtissuepy.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── fixtures/                # Static test data files
├── test_core.py            # ✅ Core data structures
├── test_io.py              # ✅ I/O operations (NEW)
├── test_spatial.py         # 🔴 Next: Spatial operations
├── test_statistics.py      # Spatial statistics
├── test_summary.py         # Summary module
├── test_network.py         # Network analysis
├── test_lda.py             # Spatial LDA
├── test_topology.py        # Mapper/TDA
├── test_synthetic.py       # ABM/PhysiCell
├── test_viz.py             # Visualization
├── test_integration.py     # End-to-end workflows
└── test_reproducibility.py # Random seed tests
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_io.py -v
```

### Run specific test class
```bash
pytest tests/test_io.py::TestReadCSV -v
```

### Run specific test
```bash
pytest tests/test_io.py::TestReadCSV::test_read_csv_basic -v
```

### Run with coverage
```bash
pytest tests/ --cov=spatialtissuepy --cov-report=html
```

### Run fast tests only (skip slow ones)
```bash
pytest tests/ -m "not slow"
```

## Test Categories

Tests are organized by priority:

### 🔴 Critical (Must Pass)
- **Core**: Data structures, validators
- **I/O**: Loading/saving data
- **Spatial**: Distance, neighborhoods

### 🟡 Important (Should Pass)
- **Statistics**: Ripley's K, co-localization
- **Summary**: Feature extraction
- **Network**: Graph operations

### 🟢 Nice to Have
- **Visualization**: Plotting
- **LDA**: Topic modeling
- **Topology**: Mapper/TDA
- **Synthetic**: ABM integration

## Fixtures

Shared test fixtures are defined in `conftest.py`:

### Size Variants
- `tiny_tissue`: 10 cells
- `small_tissue`: 100 cells  
- `medium_tissue`: 1,000 cells
- `large_tissue`: 10,000 cells

### Data Types
- `simple_tissue_2d`: Basic 2D tissue
- `simple_tissue_3d`: Basic 3D tissue
- `tissue_with_markers`: With expression data
- `multisample_tissue`: Multiple samples

### Spatial Patterns
- `clustered_pattern`: Clustered cells
- `random_pattern`: CSR pattern
- `regular_grid`: Grid pattern

### File Fixtures
- `temp_csv_file`: Temporary CSV
- `temp_json_file`: Temporary JSON
- `sample_data_dir`: Real sample data

## Writing New Tests

### Test Structure Template

```python
class TestMyFeature:
    """Tests for my_feature function."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic use case."""
        result = my_feature(fixture_name)
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case or error handling."""
        with pytest.raises(ValueError):
            my_feature(bad_input)
    
    def test_reproducibility(self):
        """Test deterministic behavior."""
        result1 = my_feature(data, random_state=42)
        result2 = my_feature(data, random_state=42)
        np.testing.assert_array_equal(result1, result2)
```

### Best Practices

1. **Use descriptive names**: `test_ripleys_k_uniform_random` not `test1`
2. **One assertion per test**: Tests should be focused
3. **Use fixtures**: Don't recreate data in every test
4. **Test edge cases**: Empty input, NaN, negative values
5. **Test errors**: Use `pytest.raises(ExceptionType)`
6. **Use numpy testing**: `np.testing.assert_array_almost_equal`

## Current Test Status

| Module | Tests | Status |
|--------|-------|--------|
| core | ✅ Complete | Validators, Cell, SpatialTissueData |
| io | ✅ Complete | CSV, JSON read/write |
| spatial | ⏳ Next | Distances, neighborhoods |
| statistics | ⏳ Pending | Ripley's K, co-loc |
| summary | ⏳ Pending | Panels, metrics |
| network | ⏳ Pending | Graphs, centrality |
| viz | ⏳ Pending | Plotting functions |
| lda | ⏳ Pending | Topic modeling |
| topology | ⏳ Pending | Mapper |
| synthetic | ⏳ Pending | PhysiCell I/O |

## Test Coverage Goals

- Overall: >80%
- Critical modules (core, io, spatial): >90%
- Advanced modules (lda, topology): >70%

## Performance Benchmarks

Some tests include performance benchmarks (marked with `@pytest.mark.benchmark`):

```bash
pytest tests/ -m benchmark
```

Expected performance targets:
- Neighborhoods (10k cells): <5 seconds
- Ripley's K (1k cells): <2 seconds
- Network construction (10k cells): <10 seconds

## Continuous Integration

Tests run automatically on:
- Every push to main
- Every pull request
- Nightly (with larger datasets)

## Troubleshooting

### Tests fail with import errors
```bash
# Reinstall package in editable mode
pip install -e .
```

### Tests fail with missing dependencies
```bash
# Install test dependencies
pip install -e ".[dev]"
```

### Tests are slow
```bash
# Run only fast tests
pytest tests/ -m "not slow" -x
```

---

*Last updated: December 2024*
