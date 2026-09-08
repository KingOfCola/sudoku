"""Tests for :mod:`checker.checker`, focused on 4x4 grids (order ``k = 2``)."""

import numpy as np
import pytest

from checker.checker import check_blocks, check_latin


# A valid 4x4 sudoku: rows, columns and 2x2 blocks each hold {0, 1, 2, 3}.
VALID_4x4 = np.array(
    [
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [3, 2, 1, 0],
    ]
)

# A 4x4 cyclic Latin square: valid rows/columns, but the 2x2 blocks are broken
# (e.g. the top-left block is [0, 1, 1, 2]).
LATIN_BUT_BAD_BLOCKS_4x4 = np.array(
    [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ]
)


# --- check_latin (4x4) --------------------------------------------------------


def test_check_latin_accepts_valid_4x4():
    assert check_latin(VALID_4x4) is True


def test_check_latin_accepts_cyclic_4x4():
    assert check_latin(LATIN_BUT_BAD_BLOCKS_4x4) is True


def test_check_latin_rejects_repeated_value_in_row():
    grid = VALID_4x4.copy()
    grid[0] = [0, 0, 2, 3]

    assert check_latin(grid) is False


def test_check_latin_rejects_repeated_value_in_column():
    grid = VALID_4x4.copy()
    grid[:, 1] = [1, 1, 0, 2]

    assert check_latin(grid) is False


def test_check_latin_rejects_out_of_range_value():
    grid = VALID_4x4.copy()
    grid[2, 2] = 4

    assert check_latin(grid) is False


def test_check_latin_rejects_non_square_grid():
    grid = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    assert check_latin(grid) is False


# --- check_blocks (k = 2) ---------------------------------------------------


def test_check_blocks_accepts_valid_4x4():
    assert check_blocks(VALID_4x4, k=2) is True


def test_check_blocks_rejects_wrong_shape():
    assert check_blocks(np.zeros((9, 9), dtype=int), k=2) is False


def test_check_blocks_rejects_repeated_value_in_a_block():
    grid = VALID_4x4.copy()
    # Break only the top-left block: [0, 1, 2, 3] -> [0, 1, 0, 3].
    grid[1, 0] = 0

    assert check_blocks(grid, k=2) is False


def test_check_blocks_rejects_out_of_range_value():
    grid = VALID_4x4.copy()
    grid[0, 0] = 4

    assert check_blocks(grid, k=2) is False


def test_check_blocks_is_independent_of_rows_and_columns():
    # Rows and columns are fine, but the 2x2 blocks are not.
    assert check_latin(LATIN_BUT_BAD_BLOCKS_4x4) is True
    assert check_blocks(LATIN_BUT_BAD_BLOCKS_4x4, k=2) is False


@pytest.mark.parametrize(
    "grid",
    [
        np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]]),
        np.array([[0, 1, 2, 3], [2, 3, 1, 0], [1, 2, 3, 0], [3, 0, 0, 1]]),
    ],
)
def test_full_4x4_grid_is_sudoku_iff_latin_and_blocks(grid):
    is_sudoku = check_latin(grid) and check_blocks(grid, k=2)
    # Reference check: sudoku == every row, column and block is a
    # permutation of range(4).
    def is_perm(vals):
        return sorted(int(v) for v in vals) == [0, 1, 2, 3]

    rows_ok = all(is_perm(row) for row in grid)
    cols_ok = all(is_perm(grid[:, c]) for c in range(4))
    blocks_ok = all(
        is_perm(grid[r:r + 2, c:c + 2].flatten())
        for r in (0, 2)
        for c in (0, 2)
    )
    assert is_sudoku == (rows_ok and cols_ok and blocks_ok)
