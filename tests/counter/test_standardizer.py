"""Tests for :mod:`counter.standardizer`.

Cell values are 0-based (digits 0 to 8).
"""

import numpy as np
import pytest

from counter.standardizer import lexicograph_block


def test_sorts_columns_by_first_row():
    # Block laid out as:
    #   3 7 1
    #   4 5 6
    #   2 0 8
    # Sorting the columns so the first row ascends (1, 3, 7) reorders every row
    # with the same column permutation.
    block = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    expected = np.array([1, 3, 7, 6, 4, 5, 8, 2, 0])

    np.testing.assert_array_equal(lexicograph_block(block), expected)


def test_first_row_is_ascending_after_call():
    block = np.array([8, 0, 4, 1, 7, 3, 6, 2, 5])
    result = lexicograph_block(block)

    assert list(result[:3]) == sorted(result[:3])


def test_already_sorted_block_is_unchanged():
    block = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    np.testing.assert_array_equal(lexicograph_block(block), block)


def test_is_idempotent():
    block = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])

    once = lexicograph_block(block)
    twice = lexicograph_block(once)

    np.testing.assert_array_equal(once, twice)


def test_does_not_mutate_input():
    block = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    original = block.copy()

    lexicograph_block(block)

    np.testing.assert_array_equal(block, original)


def test_permutation_is_preserved_per_row():
    block = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    result = lexicograph_block(block)

    for row in range(3):
        src = set(block[row * 3:(row + 1) * 3].tolist())
        dst = set(result[row * 3:(row + 1) * 3].tolist())
        assert src == dst


@pytest.mark.parametrize(
    "block, expected_first_row",
    [
        (np.array([2, 0, 1, 0, 0, 0, 0, 0, 0]), [0, 1, 2]),
        (np.array([6, 8, 7, 0, 0, 0, 0, 0, 0]), [6, 7, 8]),
    ],
)
def test_first_row_reordered_ascending(block, expected_first_row):
    result = lexicograph_block(block)

    assert list(result[:3]) == expected_first_row
