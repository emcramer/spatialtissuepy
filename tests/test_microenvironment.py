"""
Tests for the microenvironment analysis module (v0.4.0).

Covers niche identification, boundary detection, and gradient estimation.
Where possible, correctness is checked against a construction with a known
answer (two separated composition regions; a linear field with a known
gradient) rather than a golden value.
"""

from pathlib import Path

import numpy as np
import pytest

from spatialtissuepy.core import SpatialTissueData
from spatialtissuepy.microenvironment import (
    BoundaryResult,
    GradientField,
    NicheResult,
    density_gradient,
    detect_boundaries,
    identify_niches,
    spatial_gradient,
    substrate_gradient,
)


@pytest.fixture
def two_region_data():
    """Left half tumor-dominant, right half immune-dominant."""
    rng = np.random.default_rng(0)
    left = np.column_stack([rng.uniform(0, 50, 100), rng.uniform(0, 100, 100)])
    right = np.column_stack([rng.uniform(50, 100, 100), rng.uniform(0, 100, 100)])
    coords = np.vstack([left, right])
    types = np.array(
        ['Tumor'] * 85 + ['Immune'] * 15 + ['Immune'] * 85 + ['Tumor'] * 15
    )
    return SpatialTissueData(coordinates=coords, cell_types=types)


@pytest.fixture
def example_timestep():
    path = (
        Path(__file__).parent.parent
        / 'examples' / 'sample_data' / 'example_physicell_sim'
    )
    if not path.exists():
        pytest.skip("Example PhysiCell data not found")
    from spatialtissuepy.synthetic.physicell import read_physicell_timestep
    xmls = sorted(path.glob('output*.xml'))
    return read_physicell_timestep(xmls[len(xmls) // 2])


class TestNiches:
    def test_labels_shape_and_range(self, two_region_data):
        res = identify_niches(two_region_data, n_niches=3, radius=20.0,
                              random_state=0)
        assert isinstance(res, NicheResult)
        assert res.labels.shape == (two_region_data.n_cells,)
        assert set(np.unique(res.labels)).issubset(set(range(3)))

    def test_separates_two_regions(self, two_region_data):
        """With two composition regions, 2 niches should split them."""
        res = identify_niches(two_region_data, n_niches=2, radius=20.0,
                              random_state=0)
        # Each niche's dominant type should differ between the two niches.
        dom = res.dominant_types(top=1)
        assert dom[0][0][0] != dom[1][0][0]

    def test_profiles_are_distributions(self, two_region_data):
        res = identify_niches(two_region_data, n_niches=3, radius=20.0,
                              random_state=0)
        # Non-empty niche profiles sum to ~1 (composition proportions).
        for niche, size in res.niche_sizes().items():
            if size > 0:
                assert np.isclose(res.profiles[niche].sum(), 1.0, atol=1e-6)

    def test_profiles_dataframe(self, two_region_data):
        res = identify_niches(two_region_data, n_niches=2, radius=20.0,
                              random_state=0)
        df = res.profiles_dataframe()
        assert list(df.columns) == list(two_region_data.cell_types_unique)
        assert len(df) == 2

    def test_reproducible(self, two_region_data):
        a = identify_niches(two_region_data, n_niches=3, radius=20.0,
                            random_state=42)
        b = identify_niches(two_region_data, n_niches=3, radius=20.0,
                            random_state=42)
        np.testing.assert_array_equal(a.labels, b.labels)

    def test_requires_radius(self, two_region_data):
        with pytest.raises(ValueError, match="radius is required"):
            identify_niches(two_region_data, n_niches=2, method='radius')

    def test_rejects_bad_n_niches(self, two_region_data):
        with pytest.raises(ValueError, match="n_niches"):
            identify_niches(two_region_data, n_niches=0, radius=20.0)
        with pytest.raises(ValueError, match="n_niches"):
            identify_niches(two_region_data, n_niches=10 ** 6, radius=20.0)


class TestBoundaries:
    def test_interface_has_higher_foreignness(self, two_region_data):
        res = detect_boundaries(two_region_data, radius=15.0)
        assert isinstance(res, BoundaryResult)
        # Cells near the x=50 interface should be more "foreign" on average
        # than cells deep in a region.
        coords = two_region_data.coordinates
        near = np.abs(coords[:, 0] - 50) < 10
        deep = np.abs(coords[:, 0] - 50) > 35
        assert res.foreignness[near].mean() > res.foreignness[deep].mean()

    def test_foreignness_bounded(self, two_region_data):
        res = detect_boundaries(two_region_data, radius=15.0)
        assert np.all(res.foreignness >= 0) and np.all(res.foreignness <= 1)

    def test_boundary_indices_and_fraction(self, two_region_data):
        res = detect_boundaries(two_region_data, radius=15.0)
        assert res.boundary_indices.shape[0] == res.is_boundary.sum()
        assert 0.0 <= res.boundary_fraction() <= 1.0

    def test_accepts_custom_labels(self, two_region_data):
        niche = identify_niches(two_region_data, n_niches=2, radius=20.0,
                                random_state=0)
        res = detect_boundaries(two_region_data, radius=15.0,
                                labels=niche.labels)
        np.testing.assert_array_equal(res.labels, niche.labels)

    def test_rejects_mismatched_labels(self, two_region_data):
        with pytest.raises(ValueError, match="labels length"):
            detect_boundaries(two_region_data, radius=15.0,
                              labels=np.array([0, 1, 2]))

    def test_threshold_reduces_boundaries(self, two_region_data):
        low = detect_boundaries(two_region_data, radius=15.0, threshold=0.0)
        high = detect_boundaries(two_region_data, radius=15.0, threshold=0.5)
        assert high.is_boundary.sum() <= low.is_boundary.sum()

    def test_min_neighbors_excludes_isolated(self):
        # Two far-apart cells: neither has neighbors within radius.
        data = SpatialTissueData(
            coordinates=np.array([[0.0, 0.0], [1000.0, 1000.0]]),
            cell_types=np.array(['A', 'B']),
        )
        res = detect_boundaries(data, radius=5.0, min_neighbors=1)
        assert not res.is_boundary.any()
        assert np.all(res.foreignness == 0)


class TestSpatialGradient:
    def test_linear_field_recovers_exact_gradient(self):
        """A local linear fit is exact for a globally linear field."""
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 100, (300, 2))
        values = 3.0 * pos[:, 0] + 2.0 * pos[:, 1] + 5.0  # grad = (3, 2)
        gf = spatial_gradient(pos, values, k=8)
        # Interior points recover (3, 2) closely; use median for robustness.
        np.testing.assert_allclose(np.median(gf.gradients, axis=0), [3, 2],
                                   atol=1e-6)

    def test_magnitude_and_direction(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = pos[:, 0]  # gradient (1, 0)
        gf = spatial_gradient(pos, values, k=4)
        assert np.allclose(gf.magnitude, 1.0)
        # direction unit vectors
        norms = np.linalg.norm(gf.direction, axis=1)
        assert np.allclose(norms, 1.0)

    def test_zero_field_direction_is_zero(self):
        pos = np.random.default_rng(0).uniform(0, 10, (20, 2))
        gf = spatial_gradient(pos, np.ones(20), k=5)
        assert np.allclose(gf.gradients, 0.0)
        assert np.allclose(gf.direction, 0.0)  # no NaNs

    def test_query_points(self):
        pos = np.random.default_rng(1).uniform(0, 10, (50, 2))
        values = pos[:, 0] * 2
        q = np.array([[5.0, 5.0], [2.0, 8.0]])
        gf = spatial_gradient(pos, values, query_points=q, k=8)
        assert gf.points.shape == (2, 2)
        assert gf.gradients.shape == (2, 2)
        np.testing.assert_allclose(gf.gradients, [[2, 0], [2, 0]], atol=1e-6)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            spatial_gradient(np.zeros((0, 2)), np.zeros(0))
        with pytest.raises(ValueError):
            spatial_gradient(np.zeros((5, 2)), np.zeros(4))

    def test_too_few_points_is_nan_not_wrong(self):
        """2 points in 2-D can't determine a 2-D gradient -> NaN, not (g, 0)."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        values = np.array([0.0, 3.0])  # true gradient underdetermined in y
        gf = spatial_gradient(pos, values, k=2)
        assert np.isnan(gf.gradients).all()

    def test_collinear_neighbors_are_nan(self):
        """Collinear points give a rank-deficient fit -> NaN, not a silent
        wrong minimum-norm gradient."""
        pos = np.column_stack([np.arange(5.0), np.zeros(5)])  # all on y=0
        values = 3.0 * pos[:, 0]
        gf = spatial_gradient(pos, values, k=5)
        assert np.isnan(gf.gradients).all()


class TestSubstrateGradient:
    def test_planar_field_gives_2d_gradient(self, example_timestep):
        gf = substrate_gradient(example_timestep, 'oxygen')
        assert gf.gradients.shape[1] == 2  # z is constant, dropped
        assert gf.points.shape[0] == example_timestep.voxel_positions.shape[0]

    def test_agrees_with_finite_difference(self, example_timestep):
        """Local-fit gradient correlates strongly with a grid finite diff."""
        vox = example_timestep.voxel_positions[:, :2]
        ox = example_timestep.substrates['oxygen']
        xs, ys = np.unique(vox[:, 0]), np.unique(vox[:, 1])
        if len(xs) * len(ys) != len(vox):
            pytest.skip("Non-grid microenvironment")
        ix = np.searchsorted(xs, vox[:, 0])
        iy = np.searchsorted(ys, vox[:, 1])
        grid = np.full((len(xs), len(ys)), np.nan)
        grid[ix, iy] = ox
        gx_true, gy_true = np.gradient(grid, xs, ys, edge_order=2)

        gf = substrate_gradient(example_timestep, 'oxygen')
        gxm = np.full_like(grid, np.nan)
        gxm[ix, iy] = gf.gradients[:, 0]
        interior = np.zeros_like(grid, dtype=bool)
        interior[2:-2, 2:-2] = True
        corr = np.corrcoef(gxm[interior], gx_true[interior])[0, 1]
        assert corr > 0.9

    def test_unknown_substrate_raises(self, example_timestep):
        with pytest.raises(ValueError, match="Unknown substrate"):
            substrate_gradient(example_timestep, 'nope')


class TestDensityGradient:
    def test_shape(self, two_region_data):
        gf = density_gradient(two_region_data, radius=20.0)
        assert isinstance(gf, GradientField)
        assert gf.gradients.shape == (two_region_data.n_cells, 2)

    def test_points_toward_denser_region(self):
        """A cluster next to sparse space: density gradient points inward."""
        rng = np.random.default_rng(0)
        dense = rng.uniform(0, 20, (200, 2))
        sparse = rng.uniform(60, 100, (10, 2))
        coords = np.vstack([dense, sparse])
        data = SpatialTissueData(
            coordinates=coords,
            cell_types=np.array(['A'] * len(coords)),
        )
        gf = density_gradient(data, radius=15.0)
        # At a dense-cluster edge cell (near x=20), gradient x-component should
        # be negative (pointing back into the dense cluster at lower x).
        edge = np.argmin(np.abs(coords[:200, 0] - 20))
        assert gf.gradients[edge, 0] <= 0
