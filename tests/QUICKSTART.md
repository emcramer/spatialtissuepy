# Testing Quick Reference

## 🚀 Quick Start

### Install test dependencies (if needed)
```bash
pip install pytest pytest-cov
```

### Run the new tests
```bash
cd /Users/cramere/spatialtissuepy

# Run I/O tests
pytest tests/test_io.py -v

# Run spatial tests  
pytest tests/test_spatial.py -v

# Run both with coverage
pytest tests/test_io.py tests/test_spatial.py --cov=spatialtissuepy.io --cov=spatialtissuepy.spatial -v

# See detailed coverage
pytest tests/test_io.py tests/test_spatial.py --cov=spatialtissuepy.io --cov=spatialtissuepy.spatial --cov-report=term-missing
```

## 📊 Expected Results

### Test Counts
- **test_io.py**: 38 tests
  - TestReadCSV: 9 tests
  - TestWriteCSV: 5 tests
  - TestReadJSON: 9 tests
  - TestWriteJSON: 4 tests
  - TestRealDataLoading: 1 test
  - TestCrossFormat: 1 test

- **test_spatial.py**: 53 tests
  - TestPairwiseDistances: 5 tests
  - TestPairwiseDistancesBetween: 2 tests
  - TestCondensedDistances: 2 tests
  - TestKDTree: 2 tests
  - TestNearestNeighbors: 6 tests
  - TestRadiusNeighbors: 5 tests
  - TestNearestNeighborDistances: 3 tests
  - TestDistanceToType: 3 tests
  - TestDistanceToNearestDifferentType: 2 tests
  - TestDistanceMatrixByType: 3 tests
  - TestCentroid: 2 tests
  - TestCentroidByType: 1 test
  - TestBoundingBox: 2 tests
  - TestConvexHullArea: 4 tests
  - TestPointDensity: 3 tests
  - TestSpatialIntegration: 2 tests
  - TestSpatialPerformance: 2 tests (marked slow)

### Expected Output
```
tests/test_io.py::TestReadCSV::test_read_csv_basic PASSED           [ 2%]
tests/test_io.py::TestReadCSV::test_read_csv_custom_columns PASSED  [ 5%]
...
================================ 91 passed in 2.5s =================================
```

## 🐛 Common Issues & Fixes

### Issue: Import errors
```bash
# Solution: Install package in editable mode
pip install -e .
```

### Issue: Missing pytest
```bash
# Solution: Install pytest
pip install pytest pytest-cov
```

### Issue: Can't find fixtures/sample data
```bash
# Solution: Make sure you're in the package root
cd /Users/cramere/spatialtissuepy
pytest tests/test_io.py -v
```

### Issue: Tests fail with "No module named 'spatialtissuepy'"
```bash
# Solution: Install package
pip install -e .
# Or ensure PYTHONPATH is set
export PYTHONPATH=/Users/cramere/spatialtissuepy:$PYTHONPATH
```

## 📈 Coverage Report

### Generate HTML coverage report
```bash
pytest tests/test_io.py tests/test_spatial.py \
  --cov=spatialtissuepy.io \
  --cov=spatialtissuepy.spatial \
  --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Expected coverage
- **spatialtissuepy.io**: >90%
- **spatialtissuepy.spatial**: >85%

## 🎯 Next Commands

After confirming these tests pass:

### 1. Create test environment (recommended)
```bash
conda create -n spatialtissue-test python=3.10
conda activate spatialtissue-test
cd /Users/cramere/spatialtissuepy
pip install .
pip install pytest pytest-cov
pytest tests/test_io.py tests/test_spatial.py -v
```

### 2. Run all existing tests
```bash
pytest tests/ -v
```

### 3. Check overall coverage
```bash
pytest tests/ --cov=spatialtissuepy --cov-report=term-missing
```

## 📝 What to Report Back

Please run this and send me the output:

```bash
cd /Users/cramere/spatialtissuepy
pytest tests/test_io.py tests/test_spatial.py -v --tb=short 2>&1 | head -100
```

This will show:
- Which tests pass/fail
- Any import errors
- First 100 lines of output

## ✅ Success Criteria

Tests are working correctly if you see:
- ✅ All tests PASSED (green)
- ✅ No import errors
- ✅ Coverage >80% for tested modules
- ✅ Tests run in <5 seconds

## 🔄 Iteration Plan

Once we confirm these work:
1. ✅ Fix any failing tests
2. 📝 Implement test_statistics.py
3. 📝 Implement test_summary.py
4. 📝 Implement test_network.py
5. 📊 Achieve >80% overall coverage
6. 📚 Create tutorials (Phase 10b)
7. 📖 Complete documentation (Phase 10c)

---

**Current Status**: Awaiting test run results ⏳
