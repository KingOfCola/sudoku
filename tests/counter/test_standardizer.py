"""Tests for :mod:`counter.standardizer`.

Grids, bands and blocks are 2D ndarrays with matrix indexing (``x[i, j]`` is
row ``i``, column ``j``). Cell values are 0-based (digits 0 to k**2 - 1). Every
function under test mutates its array argument in place and returns ``None``.
"""

import numpy as np

from counter.standardizer import (
    lexicograph_block,
    relabel_band,
    standardize_band,
)

IDENTITY_BLOCK = np.arange(9).reshape(3, 3)


def band(*rows):
    return np.array(rows)


# --- lexicograph_block -----------------------------------------------------


def test_sorts_columns_by_first_row():
    # Block 0 laid out as:
    #   3 7 1
    #   4 5 6
    #   2 0 8
    # Sorting its columns so the first row ascends (1, 3, 7) reorders every row
    # with the same column permutation.
    grid = band([3, 7, 1, 0, 0, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 0, 0, 0],
                [2, 0, 8, 0, 0, 0, 0, 0, 0])

    result = lexicograph_block(grid, 0, 0)

    assert result is None
    np.testing.assert_array_equal(grid[:, :3], [[1, 3, 7], [6, 4, 5], [8, 2, 0]])


def test_first_row_is_ascending_after_call():
    grid = band([8, 0, 4, 1, 7, 3, 6, 2, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0])

    for block_col in range(3):
        lexicograph_block(grid, 0, block_col)
        first_row = grid[0, block_col * 3:block_col * 3 + 3]
        assert list(first_row) == sorted(first_row)


def test_already_sorted_block_is_unchanged():
    grid = band([0, 1, 2, 0, 0, 0, 0, 0, 0],
                [3, 4, 5, 0, 0, 0, 0, 0, 0],
                [6, 7, 8, 0, 0, 0, 0, 0, 0])
    original = grid.copy()

    lexicograph_block(grid, 0, 0)

    np.testing.assert_array_equal(grid, original)


def test_is_idempotent():
    grid = band([3, 7, 1, 0, 0, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 0, 0, 0],
                [2, 0, 8, 0, 0, 0, 0, 0, 0])

    lexicograph_block(grid, 0, 0)
    once = grid.copy()
    lexicograph_block(grid, 0, 0)

    np.testing.assert_array_equal(grid, once)


def test_column_contents_are_preserved():
    grid = band([3, 7, 1, 0, 0, 0, 0, 0, 0],
                [4, 5, 6, 0, 0, 0, 0, 0, 0],
                [2, 0, 8, 0, 0, 0, 0, 0, 0])
    before_columns = {frozenset(grid[:, c].tolist()) for c in range(3)}

    lexicograph_block(grid, 0, 0)

    after_columns = {frozenset(grid[:, c].tolist()) for c in range(3)}
    assert before_columns == after_columns


def test_only_the_selected_blocks_columns_move():
    grid = band([0, 1, 2, 5, 3, 4, 6, 7, 8],
                [3, 4, 5, 8, 6, 7, 0, 1, 2],
                [6, 7, 8, 2, 0, 1, 3, 4, 5])
    before = grid.copy()

    lexicograph_block(grid, 0, 1)  # block 1 spans columns 3, 4, 5

    np.testing.assert_array_equal(grid[:, 0:3], before[:, 0:3])
    np.testing.assert_array_equal(grid[:, 6:9], before[:, 6:9])
    np.testing.assert_array_equal(grid[0, 3:6], [3, 4, 5])


def test_reorders_the_whole_grid_not_just_the_first_band():
    grid = np.array([[3, 7, 1, 0, 0, 0, 0, 0, 0],
                     [4, 5, 6, 0, 0, 0, 0, 0, 0],
                     [2, 0, 8, 0, 0, 0, 0, 0, 0],
                     [10, 11, 12, 0, 0, 0, 0, 0, 0],
                     [13, 14, 15, 0, 0, 0, 0, 0, 0],
                     [16, 17, 18, 0, 0, 0, 0, 0, 0]])

    lexicograph_block(grid, 0, 0)  # order argsort([3, 7, 1]) == [2, 0, 1]

    # The column permutation derived from band 0's first row is applied to the
    # second band's rows too.
    np.testing.assert_array_equal(
        grid[3:6, 0:3], [[12, 10, 11], [15, 13, 14], [18, 16, 17]]
    )


def test_block_row_selects_which_band_defines_the_order():
    grid = np.array([[0, 1, 2, 0, 0, 0, 0, 0, 0],
                     [0, 1, 2, 0, 0, 0, 0, 0, 0],
                     [0, 1, 2, 0, 0, 0, 0, 0, 0],
                     [7, 3, 5, 0, 0, 0, 0, 0, 0],
                     [1, 2, 0, 0, 0, 0, 0, 0, 0],
                     [8, 4, 6, 0, 0, 0, 0, 0, 0]])

    lexicograph_block(grid, 1, 0)  # order from row 3: argsort([7, 3, 5]) == [1, 2, 0]

    np.testing.assert_array_equal(grid[3, 0:3], [3, 5, 7])
    np.testing.assert_array_equal(grid[0, 0:3], [1, 2, 0])  # same permutation, band 0


# --- relabel_band ---------------------------------------------------------


def test_relabel_makes_first_block_identity():
    b = band([3, 7, 1, 0, 5, 8, 2, 4, 6],
             [4, 5, 6, 1, 3, 7, 0, 5, 8],
             [2, 0, 8, 2, 4, 6, 1, 3, 7])

    result = relabel_band(b)

    assert result is None
    np.testing.assert_array_equal(b[:, :3], IDENTITY_BLOCK)


def test_relabel_applies_the_same_token_map_to_the_whole_band():
    b = band([3, 7, 1, 0, 5, 8, 2, 4, 6],
             [4, 5, 6, 1, 3, 7, 0, 5, 8],
             [2, 0, 8, 2, 4, 6, 1, 3, 7])
    original = b.copy()

    relabel_band(b)

    # Token -> value map read straight from the original first block.
    mapping = {tok: i for i, tok in enumerate(original[:, :3].ravel().tolist())}
    expected = np.array([[mapping[v] for v in row] for row in original.tolist()])
    np.testing.assert_array_equal(b, expected)


def test_relabel_is_identity_when_first_block_already_sorted():
    b = band([0, 1, 2, 4, 8, 2, 5, 6, 7],
             [3, 4, 5, 0, 6, 1, 3, 1, 0],
             [6, 7, 8, 5, 6, 7, 3, 1, 0])
    original = b.copy()

    relabel_band(b)

    np.testing.assert_array_equal(b, original)


def test_relabel_preserves_equal_tokens():
    b = band([5, 2, 8, 8, 8, 2, 2, 5, 5],
             [0, 3, 7, 0, 0, 3, 3, 7, 7],
             [1, 6, 4, 1, 1, 6, 6, 4, 4])
    original = b.copy()

    relabel_band(b)

    for value in np.unique(original):
        positions = original == value
        assert len(np.unique(b[positions])) == 1


def test_relabel_works_for_k_equals_2():
    b = np.array([[3, 1, 2, 0],
                  [0, 2, 1, 3]])

    relabel_band(b)

    np.testing.assert_array_equal(b[:, :2], [[0, 1], [2, 3]])


# --- standardize_band ---------------------------------------------------------


def test_standardize_first_block_becomes_identity():
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])

    standardize_band(b)

    np.testing.assert_array_equal(b[:, :3], IDENTITY_BLOCK)


def test_standardize_other_blocks_have_ascending_first_row():
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])

    standardize_band(b)

    for block_col in (1, 2):
        first_row = b[0, block_col * 3:block_col * 3 + 3]
        assert list(first_row) == sorted(first_row)


def test_standardize_matches_relabel_then_lexicograph():
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])
    manual = b.copy()

    relabel_band(manual)
    lexicograph_block(manual, 0, 1)
    lexicograph_block(manual, 0, 2)
    standardize_band(b)

    np.testing.assert_array_equal(b, manual)


def test_standardize_known_result():
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])

    standardize_band(b)

    np.testing.assert_array_equal(
        b,
        [[0, 1, 2, 3, 7, 8, 4, 5, 6],
         [3, 4, 5, 0, 2, 1, 8, 6, 7],
         [6, 7, 8, 5, 4, 1, 2, 3, 0]],
    )


def test_standardize_is_idempotent():
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])

    standardize_band(b)
    once = b.copy()
    standardize_band(b)

    np.testing.assert_array_equal(b, once)


def test_standardize_preserves_row_membership():
    # Relabelling + column reordering never moves a value across rows.
    b = band([3, 7, 1, 0, 4, 8, 2, 5, 6],
             [4, 5, 6, 1, 3, 7, 0, 8, 2],
             [2, 0, 8, 5, 6, 7, 3, 1, 4])
    manual = b.copy()
    relabel_band(manual)  # relabel keeps every value on its own row

    standardize_band(b)

    for row in range(3):
        assert set(b[row].tolist()) == set(manual[row].tolist())


def test_standardize_already_canonical_is_unchanged():
    b = band([0, 1, 2, 3, 4, 5, 6, 7, 8],
             [3, 4, 5, 6, 7, 8, 0, 1, 2],
             [6, 7, 8, 0, 1, 2, 3, 4, 5])
    original = b.copy()

    standardize_band(b)

    np.testing.assert_array_equal(b, original)
