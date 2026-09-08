"""Tests for :mod:`counter.standardizer`."""

import numpy as np
import pytest

from counter.standardizer import lexicograph_block


def test_sorts_columns_by_first_row():
    # Block laid out as:
    #   4 8 2
    #   5 6 7
    #   3 1 9
    # Sorting the columns so the first row ascends (2, 4, 8) reorders every row
    # with the same column permutation.
    block = np.array([4, 8, 2, 5, 6, 7, 3, 1, 9])
    expected = np.array([2, 4, 8, 7, 5, 6, 9, 3, 1])

    np.testing.assert_array_equal(lexicograph_block(block), expected)


def test_first_row_is_ascending_after_call():
    block = np.array([9, 1, 5, 2, 8, 4, 7, 3, 6])
    result = lexicograph_block(block)

    assert list(result[:3]) == sorted(result[:3])


def test_already_sorted_block_is_unchanged():
    block = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    np.testing.assert_array_equal(lexicograph_block(block), block)


def test_is_idempotent():
    block = np.array([4, 8, 2, 5, 6, 7, 3, 1, 9])

    once = lexicograph_block(block)
    twice = lexicograph_block(once)

    np.testing.assert_array_equal(once, twice)


def test_does_not_mutate_input():
    block = np.array([4, 8, 2, 5, 6, 7, 3, 1, 9])
    original = block.copy()

    lexicograph_block(block)

    np.testing.assert_array_equal(block, original)


def test_permutation_is_preserved_per_row():
    block = np.array([4, 8, 2, 5, 6, 7, 3, 1, 9])
    result = lexicograph_block(block)

    for row in range(3):
        src = set(block[row * 3:(row + 1) * 3].tolist())
        dst = set(result[row * 3:(row + 1) * 3].tolist())
        assert src == dst


@pytest.mark.parametrize(
    "block, expected_first_row",
    [
        (np.array([3, 1, 2, 0, 0, 0, 0, 0, 0]), [1, 2, 3]),
        (np.array([7, 9, 8, 0, 0, 0, 0, 0, 0]), [7, 8, 9]),
    ],
)
def test_first_row_reordered_ascending(block, expected_first_row):
    result = lexicograph_block(block)

    assert list(result[:3]) == expected_first_row
