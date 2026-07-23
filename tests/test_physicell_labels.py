"""
Regression tests for PhysiCell reader defects fixed in v0.3.0.

These cover the label-driven column extraction and orientation handling
reported by the llm-abm-consistency harness (OHSU ChangLab, 2026-07-21).

The bundled example simulation cannot exercise the orientation bug on its own:
it writes 154 variables per frame and its smallest frame holds 906 cells, so
``n_cells > n_variables`` holds everywhere. These tests build small matrices
where that assumption fails.
"""

import importlib.metadata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat, savemat

import spatialtissuepy
from spatialtissuepy.synthetic.physicell import (
    discover_physicell_timesteps,
    read_physicell_timestep,
)
from spatialtissuepy.synthetic.physicell.parser import (
    declared_variable_count,
    expand_cell_labels,
    parse_cells_mat,
    parse_microenvironment_mat,
    parse_physicell_xml,
)

N_VARS = 87


@pytest.fixture
def example_physicell_dir():
    """Path to example PhysiCell simulation output folder."""
    path = (
        Path(__file__).parent.parent
        / 'examples' / 'sample_data' / 'example_physicell_sim'
    )
    if not path.exists():
        pytest.skip(f"Example PhysiCell data not found at {path}")
    return path


@pytest.fixture
def labels():
    """A minimal <labels> block in the shape parse_physicell_xml produces."""
    return {
        0: ('ID', 1),
        1: ('position', 3),
        4: ('total_volume', 1),
        5: ('cell_type', 1),
        7: ('current_phase', 1),
        26: ('dead', 1),
        37: ('radius', 1),
        50: ('is_motile', 1),
        52: ('migration_speed', 1),
        86: ('phenotype_state', 1),
    }


@pytest.fixture
def cell_matrix():
    """A (87, 40) matrix with each cell's values keyed off its column index."""
    matrix = np.zeros((N_VARS, 40), dtype=float)
    for cell in range(40):
        matrix[:, cell] = np.arange(N_VARS) * 1000.0 + cell
    return matrix


def _write(tmp_path, name, matrix):
    path = tmp_path / name
    savemat(str(path), {'cells': matrix})
    return path


class TestLabelExpansion:
    """Expansion of the MultiCellDS <labels> block into flat columns."""

    def test_scalar_labels_keep_their_name(self):
        assert expand_cell_labels({0: ('ID', 1)}) == {0: 'ID'}

    def test_size_three_expands_to_xyz(self):
        assert expand_cell_labels({1: ('position', 3)}) == {
            1: 'position_x', 2: 'position_y', 3: 'position_z'
        }

    def test_other_vector_sizes_expand_positionally(self):
        assert expand_cell_labels({5: ('state', 2)}) == {
            5: 'state_0', 6: 'state_1'
        }

    def test_variable_count_accounts_for_trailing_vector_width(self):
        # max(labels) + 1 would undercount this by two.
        assert declared_variable_count({0: ('ID', 1), 1: ('position', 3)}) == 4


class TestOrientation:
    """Issue #2: orientation must come from the label count, not magnitude."""

    def test_fewer_cells_than_variables(self, tmp_path, labels, cell_matrix):
        """An 87x40 matrix with 87 declared labels parses as 40 cells."""
        path = _write(tmp_path, 'small.mat', cell_matrix)
        result = parse_cells_mat(path, labels=labels)

        assert result['positions'].shape[0] == 40
        np.testing.assert_array_equal(
            result['positions'][:, 0], cell_matrix[1, :]
        )

    def test_genuine_on_disk_transpose(self, tmp_path, labels, cell_matrix):
        """A 40x87 matrix also parses as 40 cells."""
        path = _write(tmp_path, 'transposed.mat', cell_matrix.T)
        result = parse_cells_mat(path, labels=labels)

        assert result['positions'].shape[0] == 40
        np.testing.assert_array_equal(
            result['positions'][:, 0], cell_matrix[1, :]
        )
        assert result['orientation'] == 'cells_x_variables'

    def test_square_matrix_prefers_variables_on_axis_zero(self, tmp_path, labels):
        square = np.zeros((N_VARS, N_VARS), dtype=float)
        for cell in range(N_VARS):
            square[:, cell] = np.arange(N_VARS) * 1000.0 + cell

        result = parse_cells_mat(_write(tmp_path, 'sq.mat', square), labels=labels)

        assert result['positions'].shape[0] == N_VARS
        np.testing.assert_array_equal(result['positions'][:, 0], square[1, :])
        assert result['orientation'] == 'variables_x_cells'

    def test_neither_orientation_raises(self, tmp_path, labels):
        path = _write(tmp_path, 'bad.mat', np.zeros((99, 33)))

        with pytest.raises(ValueError, match='matches neither orientation'):
            parse_cells_mat(path, labels=labels)

    def test_raw_data_round_trips(self, tmp_path, labels, cell_matrix):
        """raw_data is what loadmat returned, not the reoriented view."""
        path = _write(tmp_path, 'raw.mat', cell_matrix)
        result = parse_cells_mat(path, labels=labels)

        np.testing.assert_array_equal(result['raw_data'], loadmat(str(path))['cells'])
        assert result['orientation'] == 'variables_x_cells'

    def test_warns_when_orientation_must_be_guessed(self, tmp_path, cell_matrix):
        path = _write(tmp_path, 'noguess.mat', cell_matrix)

        with pytest.warns(UserWarning, match='Guessing cell matrix orientation'):
            parse_cells_mat(path)


class TestLabelledColumns:
    """Issue #1: every labelled column must be reachable."""

    def test_non_indexed_standard_fields(self, tmp_path, labels, cell_matrix):
        result = parse_cells_mat(
            _write(tmp_path, 'cols.mat', cell_matrix), labels=labels
        )

        np.testing.assert_array_equal(
            result['columns']['is_motile'], cell_matrix[50, :]
        )
        np.testing.assert_array_equal(
            result['columns']['migration_speed'], cell_matrix[52, :]
        )

    def test_model_specific_custom_column(self, tmp_path, labels, cell_matrix):
        result = parse_cells_mat(
            _write(tmp_path, 'custom.mat', cell_matrix), labels=labels
        )

        np.testing.assert_array_equal(
            result['columns']['phenotype_state'], cell_matrix[86, :]
        )

    def test_differing_label_counts_do_not_leak(self, tmp_path, labels, cell_matrix):
        """Two models parsed in one session each resolve from their own XML."""
        model_b_labels = dict(labels)
        model_b_labels[87] = ('hif', 1)
        model_b_labels[88] = ('hypoxia_timer', 1)
        model_b_matrix = np.vstack([
            cell_matrix,
            np.full((2, cell_matrix.shape[1]), 7.0),
        ])

        result_a = parse_cells_mat(
            _write(tmp_path, 'a.mat', cell_matrix), labels=labels
        )
        result_b = parse_cells_mat(
            _write(tmp_path, 'b.mat', model_b_matrix), labels=model_b_labels
        )

        assert 'hif' not in result_a['columns']
        assert 'hif' in result_b['columns']
        # Index 86 carries a different meaning in each model.
        assert result_a['positions'].shape[0] == result_b['positions'].shape[0]

    def test_no_labels_preserves_legacy_output(self, tmp_path):
        """With labels=None, output matches v0.2.0 for a default-layout frame."""
        wide = np.zeros((N_VARS, 200), dtype=float)
        for cell in range(200):
            wide[:, cell] = np.arange(N_VARS) * 1000.0 + cell
        path = _write(tmp_path, 'wide.mat', wide)

        result = parse_cells_mat(path)

        assert result['columns'] == {}
        np.testing.assert_array_equal(result['positions'][:, 0], wide[1, :])
        np.testing.assert_array_equal(result['dead_flags'], wide[26, :].astype(int))


class TestMicroenvironmentGuard:
    """Issue #6: validate the microenvironment matrix orientation."""

    def test_correct_orientation_parses(self, tmp_path):
        me = np.arange(6 * 50, dtype=float).reshape(6, 50)
        path = tmp_path / 'me.mat'
        savemat(str(path), {'multiscale_microenvironment': me})

        result = parse_microenvironment_mat(path, ['oxygen', 'glucose'])

        assert result['voxel_positions'].shape == (50, 3)
        np.testing.assert_array_equal(result['concentrations']['oxygen'], me[4, :])

    def test_transposed_input_is_corrected(self, tmp_path):
        me = np.arange(6 * 50, dtype=float).reshape(6, 50)
        path = tmp_path / 'me_t.mat'
        savemat(str(path), {'multiscale_microenvironment': me.T})

        result = parse_microenvironment_mat(path, ['oxygen', 'glucose'])

        np.testing.assert_array_equal(result['concentrations']['oxygen'], me[4, :])

    def test_mismatched_substrate_count_raises(self, tmp_path):
        path = tmp_path / 'me_bad.mat'
        savemat(str(path), {'multiscale_microenvironment': np.zeros((9, 11))})

        with pytest.raises(ValueError, match='matches neither orientation'):
            parse_microenvironment_mat(path, ['oxygen', 'glucose'])


class TestToDataFrameDeadCells:
    """Issue #4: to_dataframe must honour include_dead_cells."""

    @pytest.fixture
    def timestep_with_dead(self, example_physicell_dir):
        for _, xml_path, _ in discover_physicell_timesteps(example_physicell_dir):
            timestep = read_physicell_timestep(xml_path)
            if timestep.n_dead_cells > 0:
                return timestep
        pytest.skip("No frame with dead cells in the example simulation")

    def test_excludes_dead_by_default(self, timestep_with_dead):
        df = timestep_with_dead.to_dataframe()

        assert len(df) == timestep_with_dead.n_cells
        assert not df['is_dead'].any()

    def test_includes_dead_when_requested(self, timestep_with_dead):
        df = timestep_with_dead.to_dataframe(include_dead_cells=True)

        assert len(df) == timestep_with_dead.n_cells_total

    def test_row_order_matches_positions(self, timestep_with_dead):
        df = timestep_with_dead.to_dataframe()

        np.testing.assert_allclose(
            df[['x', 'y', 'z']].to_numpy(), timestep_with_dead.positions
        )

    def test_extra_columns_are_opt_in(self, timestep_with_dead):
        base = timestep_with_dead.to_dataframe()
        extended = timestep_with_dead.to_dataframe(extra_columns=True)

        assert extended.shape[1] > base.shape[1]
        assert 'is_motile' in extended.columns
        assert len(extended) == len(base)

    def test_no_fragmentation_warning(self, timestep_with_dead):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            timestep_with_dead.to_dataframe(extra_columns=True)

        assert not [
            w for w in record
            if issubclass(w.category, pd.errors.PerformanceWarning)
        ]


class TestReaderUsesFrameLabels:
    """The reader resolves columns from each frame's own XML."""

    def test_custom_column_reachable_through_reader(self, example_physicell_dir):
        _, xml_path, _ = discover_physicell_timesteps(example_physicell_dir)[0]
        timestep = read_physicell_timestep(xml_path)

        labels = parse_physicell_xml(xml_path).extra['custom_labels']
        expected = expand_cell_labels(labels)

        data = timestep._load_cell_data()
        assert data['columns'], "reader should pass labels through to the parser"
        assert set(data['columns']) == set(expected.values())

    def test_reader_emits_no_orientation_warning(self, example_physicell_dir):
        _, xml_path, _ = discover_physicell_timesteps(example_physicell_dir)[0]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            read_physicell_timestep(xml_path)._load_cell_data()

        assert not [
            w for w in record
            if 'Guessing cell matrix orientation' in str(w.message)
        ]


class TestVersionSingleSource:
    """Issue #5: __version__ must not drift from distribution metadata."""

    def test_version_matches_package_metadata(self):
        assert spatialtissuepy.__version__ == importlib.metadata.version(
            "spatialtissuepy"
        )
