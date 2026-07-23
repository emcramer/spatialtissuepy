"""
Tests for PhysiCell microenvironment / substrate access (v0.4.0).

Covers two distinct notions of "how much substrate":

- environmental concentration at a location, sampled from the voxel field via
  ``PhysiCellTimeStep.substrate_at`` (nearest-voxel, KD-tree);
- the amount a cell has actually internalized, read per-cell from PhysiCell's
  ``internalized_total_substrates`` field via
  ``PhysiCellTimeStep.internalized_substrates``.

The bundled example simulation is used for the equivalence checks; small
synthetic inputs cover ties, missing files, and error paths.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from spatialtissuepy.synthetic.physicell import (
    PhysiCellTimeStep,
    discover_physicell_timesteps,
    read_physicell_simulation,
    read_physicell_timestep,
)
from spatialtissuepy.synthetic.physicell.reader import _find_microenvironment_mat


@pytest.fixture
def example_physicell_dir():
    path = (
        Path(__file__).parent.parent
        / 'examples' / 'sample_data' / 'example_physicell_sim'
    )
    if not path.exists():
        pytest.skip(f"Example PhysiCell data not found at {path}")
    return path


@pytest.fixture
def frame_xml(example_physicell_dir):
    xmls = sorted(example_physicell_dir.glob('output*.xml'))
    if not xmls:
        pytest.skip("No frames in example simulation")
    return xmls[len(xmls) // 2]


def _injected_timestep(voxels, concentrations):
    """A PhysiCellTimeStep with a microenvironment injected, no files read."""
    ts = PhysiCellTimeStep(time=0.0, time_index=0, source_path=Path('none.xml'))
    ts._microenvironment = {
        'voxel_positions': np.asarray(voxels, dtype=float),
        'concentrations': {k: np.asarray(v, dtype=float)
                           for k, v in concentrations.items()},
        'raw_data': None,
    }
    ts._me_loaded = True
    return ts


def _build_substrate_frame(dir_path, n_subs, start=6, n_cells=8):
    """
    Write a minimal PhysiCell-like frame with `n_subs` substrates and an
    internalized_total_substrates field of size n_subs. Value in substrate i,
    cell j is (i+1)*100 + j, so the mapping can be checked exactly.

    Returns the substrate names.
    """
    subs = [f'sub{i}' for i in range(n_subs)]
    n_rows = start + n_subs  # declared variable count == matrix rows
    matrix = np.zeros((n_rows, n_cells))
    matrix[0] = np.arange(n_cells)
    matrix[1] = np.linspace(0, 70, n_cells)
    matrix[2] = np.linspace(0, 70, n_cells)
    for i in range(n_subs):
        matrix[start + i] = (i + 1) * 100 + np.arange(n_cells)
    savemat(str(dir_path / 'output00000000_cells.mat'), {'cells': matrix})

    var_xml = ''.join(
        f'<variable name="{s}" ID="{i}"/>' for i, s in enumerate(subs)
    )
    labels = [
        ('ID', 0, 1), ('position', 1, 3), ('cell_type', 5, 1),
        ('internalized_total_substrates', start, n_subs),
    ]
    label_xml = ''.join(
        f'<label index="{idx}" size="{sz}">{nm}</label>' for nm, idx, sz in labels
    )
    (dir_path / 'output00000000.xml').write_text(
        '<MultiCellDS><microenvironment><domain><variables>'
        f'{var_xml}</variables></domain></microenvironment>'
        '<cellular_information><simplified_data><labels>'
        f'{label_xml}</labels></simplified_data></cellular_information>'
        '</MultiCellDS>'
    )
    return subs


class TestSubstrateFields:
    def test_substrate_names(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        assert 'oxygen' in ts.substrate_names

    def test_substrates_shape(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        oxygen = ts.substrates['oxygen']
        assert oxygen.ndim == 1
        assert oxygen.shape[0] == ts.voxel_positions.shape[0]

    def test_voxel_positions_are_3d(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        assert ts.voxel_positions.shape[1] == 3


class TestSubstrateAt:
    def test_matches_bruteforce_argmin(self, frame_xml):
        """The report's acceptance criterion: exact match to nearest-voxel."""
        ts = read_physicell_timestep(frame_xml)
        coords = ts.to_spatial_data().coordinates
        xs, ys = coords[:, 0], coords[:, 1]

        got = ts.substrate_at('oxygen', xs, ys)

        vox = ts.voxel_positions[:, :2]
        field = ts.substrates['oxygen']
        brute = np.array([
            field[np.argmin((vox[:, 0] - x) ** 2 + (vox[:, 1] - y) ** 2)]
            for x, y in zip(xs, ys)
        ])
        np.testing.assert_array_equal(got, brute)

    def test_scalar_query_returns_scalar(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        val = ts.substrate_at('oxygen', 0.0, 0.0)
        assert np.isscalar(val) or val.ndim == 0

    def test_array_query_preserves_length(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        out = ts.substrate_at('oxygen', np.array([0.0, 10.0, 20.0]),
                              np.array([0.0, 10.0, 20.0]))
        assert out.shape == (3,)

    def test_unknown_substrate_raises(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        with pytest.raises(ValueError, match="Unknown substrate"):
            ts.substrate_at('unobtanium', 0.0, 0.0)

    def test_tie_is_deterministic_and_valid(self):
        """A point equidistant between two voxels returns one of their values.

        KD-tree tie-breaking need not match np.argmin, so this asserts the
        result is one of the tied voxels and is stable, rather than pinning a
        specific choice.
        """
        voxels = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ts = _injected_timestep(voxels, {'oxygen': [1.0, 2.0]})

        first = ts.substrate_at('oxygen', 5.0, 0.0)
        second = ts.substrate_at('oxygen', 5.0, 0.0)

        assert first in (1.0, 2.0)
        assert first == second  # deterministic

    def test_3d_query(self):
        voxels = [[0, 0, 0], [0, 0, 10]]
        ts = _injected_timestep(voxels, {'o2': [5.0, 9.0]})
        assert ts.substrate_at('o2', 0.0, 0.0, 1.0) == 5.0
        assert ts.substrate_at('o2', 0.0, 0.0, 9.0) == 9.0

    def test_output_shape_mirrors_input(self):
        # 4 voxels on a line; query with a 2-D grid of points.
        voxels = [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]]
        ts = _injected_timestep(voxels, {'o2': [1.0, 2.0, 3.0, 4.0]})

        grid = ts.substrate_at('o2', np.array([[0.0, 10.0], [20.0, 30.0]]),
                               np.array([[0.0, 0.0], [0.0, 0.0]]))
        assert grid.shape == (2, 2)
        np.testing.assert_array_equal(grid, [[1.0, 2.0], [3.0, 4.0]])

    def test_broadcasts_scalar_and_array(self):
        voxels = [[0, 0, 0], [0, 10, 0]]
        ts = _injected_timestep(voxels, {'o2': [1.0, 2.0]})
        # scalar x, array y -> broadcast to array
        out = ts.substrate_at('o2', 0.0, np.array([0.0, 10.0]))
        assert out.shape == (2,)
        np.testing.assert_array_equal(out, [1.0, 2.0])


class TestInternalizedSubstrates:
    def test_columns_match_raw_matrix(self, frame_xml, example_physicell_dir):
        """internalized_total_substrates maps to rows 92+ in matrix order."""
        ts = read_physicell_timestep(frame_xml, include_dead_cells=True)
        matrix = loadmat(str(ts.cells_mat_path))['cells']
        if matrix.shape[0] < 97:
            pytest.skip("Frame lacks internalized_total_substrates rows")

        inte = ts.internalized_substrates(include_dead_cells=True)
        for i, name in enumerate(ts.substrate_names):
            np.testing.assert_array_equal(inte[name].to_numpy(), matrix[92 + i, :])

    def test_columns_named_by_substrate(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        inte = ts.internalized_substrates()
        assert list(inte.columns) == ts.substrate_names

    def test_row_count_follows_dead_filter(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        assert len(ts.internalized_substrates()) == ts.n_cells
        assert len(ts.internalized_substrates(include_dead_cells=True)) == \
            ts.n_cells_total

    def test_row_order_matches_positions(self, frame_xml):
        ts = read_physicell_timestep(frame_xml)
        inte = ts.internalized_substrates()
        assert len(inte) == ts.positions.shape[0]

    def test_raises_without_field(self, tmp_path):
        """A frame whose cell data lacks the field raises informatively."""
        # Wider than tall so orientation is unambiguous; no <labels>, so the
        # parser exposes no 'columns' and no substrate names are declared.
        savemat(str(tmp_path / 'output00000000_cells.mat'),
                {'cells': np.zeros((5, 20))})
        (tmp_path / 'output00000000.xml').write_text(
            "<MultiCellDS><cellular_information/></MultiCellDS>"
        )
        ts = read_physicell_timestep(tmp_path / 'output00000000.xml')
        with pytest.raises(ValueError, match="internalized"):
            ts.internalized_substrates()

    @pytest.mark.parametrize('n_subs', [1, 3, 5])
    def test_maps_field_for_any_substrate_count(self, tmp_path, n_subs):
        """The field's columns are named by the label's own size convention:
        bare name for size 1, _x/_y/_z for size 3, _0.._n otherwise. Hardcoding
        one convention breaks the single- and triple-substrate cases (the two
        most common PhysiCell setups)."""
        subs = _build_substrate_frame(tmp_path, n_subs)
        ts = read_physicell_timestep(
            tmp_path / 'output00000000.xml', include_dead_cells=True
        )
        inte = ts.internalized_substrates(include_dead_cells=True)

        assert list(inte.columns) == subs
        for i, name in enumerate(subs):
            expected = (i + 1) * 100 + np.arange(len(inte))
            np.testing.assert_array_equal(inte[name].to_numpy(), expected)


class TestFileResolution:
    def test_resolves_from_xml_filename(self, frame_xml, example_physicell_dir):
        expected = example_physicell_dir / f'{frame_xml.stem}_microenvironment0.mat'
        got = _find_microenvironment_mat(
            frame_xml, f'{frame_xml.stem}_microenvironment0.mat'
        )
        assert got == expected

    def test_falls_back_to_naming_convention(self, frame_xml):
        # No filename hint -> should still find it by convention.
        got = _find_microenvironment_mat(frame_xml, None)
        assert got is not None and got.exists()

    def test_returns_none_when_absent(self, tmp_path):
        (tmp_path / 'output00000000.xml').write_text("<x/>")
        assert _find_microenvironment_mat(
            tmp_path / 'output00000000.xml', None
        ) is None


class TestEmptyMicroenvironment:
    def test_missing_file_yields_empty(self, tmp_path):
        """No microenvironment file -> empty substrates, no exception."""
        savemat(str(tmp_path / 'output00000000_cells.mat'),
                {'cells': np.zeros((30, 4))})
        (tmp_path / 'output00000000.xml').write_text(
            "<MultiCellDS><cellular_information/></MultiCellDS>"
        )
        ts = read_physicell_timestep(tmp_path / 'output00000000.xml')

        assert ts.substrates == {}
        assert ts.voxel_positions.shape == (0, 3)

    def test_substrate_at_on_empty_raises(self, tmp_path):
        savemat(str(tmp_path / 'output00000000_cells.mat'),
                {'cells': np.zeros((30, 4))})
        (tmp_path / 'output00000000.xml').write_text("<MultiCellDS/>")
        ts = read_physicell_timestep(tmp_path / 'output00000000.xml')
        with pytest.raises(ValueError):
            ts.substrate_at('oxygen', 0.0, 0.0)


class TestSimulationIntegration:
    def test_substrate_names(self, example_physicell_dir):
        sim = read_physicell_simulation(example_physicell_dir)
        assert 'oxygen' in sim.substrate_names

    def test_timesteps_have_substrate_access(self, example_physicell_dir):
        sim = read_physicell_simulation(example_physicell_dir)
        ts = sim.get_timestep(0)
        assert ts.voxel_positions.shape[0] > 0


class TestBackwardCompatibility:
    def test_discover_still_returns_3_tuples(self, example_physicell_dir):
        """Adding microenvironment support must not change discover's shape."""
        steps = discover_physicell_timesteps(example_physicell_dir)
        assert steps and all(len(s) == 3 for s in steps)
        index, xml_path, mat_path = steps[0]  # unpacks as before
        assert xml_path.suffix == '.xml'
        assert mat_path.suffix == '.mat'
