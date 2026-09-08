"""Tests for :mod:`counter.standardizer`.

Cell values are 0-based (digits 0 to 8).
"""

import numpy as np
import pytest

from counter.standardizer import (
    lexicograph_block,
    relabel_blocks,
    standardize_blocks,
)


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


# --- relabel_blocks ---------------------------------------------------------


def test_relabel_makes_first_block_identity():
    b1 = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    b2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    r1, r2 = relabel_blocks(b1, b2)

    np.testing.assert_array_equal(r1, np.arange(9))


def test_relabel_applies_same_token_map_to_other_blocks():
    b1 = np.array([2, 0, 1])
    b2 = np.array([1, 2, 0])

    # token->digit map from b1: 2->0, 0->1, 1->2
    r1, r2 = relabel_blocks(b1, b2)

    np.testing.assert_array_equal(r1, [0, 1, 2])
    np.testing.assert_array_equal(r2, [2, 0, 1])


def test_relabel_preserves_equal_tokens_across_blocks():
    b1 = np.array([5, 2, 8, 0, 3, 7, 1, 6, 4])
    b2 = np.array([8, 8, 2, 2, 5, 5, 0, 0, 3])  # not a permutation, but tokens still map

    r1, r2 = relabel_blocks(b1, b2)

    # Build the expected token->digit map straight from b1.
    mapping = {tok: i for i, tok in enumerate(b1.tolist())}
    np.testing.assert_array_equal(r2, [mapping[t] for t in b2.tolist()])


def test_relabel_identity_when_first_block_already_sorted():
    b1 = np.arange(9)
    b2 = np.array([4, 8, 2, 5, 6, 7, 3, 1, 0])

    r1, r2 = relabel_blocks(b1, b2)

    np.testing.assert_array_equal(r1, b1)
    np.testing.assert_array_equal(r2, b2)


def test_relabel_single_block():
    (r1,) = relabel_blocks(np.array([3, 1, 2, 0]))

    np.testing.assert_array_equal(r1, [0, 1, 2, 3])


def test_relabel_returns_one_array_per_input():
    blocks = [np.random.permutation(9) for _ in range(4)]

    result = relabel_blocks(*blocks)

    assert len(result) == 4
    for original, relabelled in zip(blocks, result):
        assert relabelled.shape == original.shape


def test_relabel_does_not_mutate_inputs():
    b1 = np.array([2, 0, 1, 3])
    b2 = np.array([1, 3, 0, 2])
    b1_copy, b2_copy = b1.copy(), b2.copy()

    relabel_blocks(b1, b2)

    np.testing.assert_array_equal(b1, b1_copy)
    np.testing.assert_array_equal(b2, b2_copy)


# --- standardize_blocks ---------------------------------------------------------


def test_standardize_first_block_becomes_identity():
    b1 = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    b2 = np.array([0, 4, 8, 1, 5, 6, 2, 3, 7])

    result = standardize_blocks(b1, b2)

    np.testing.assert_array_equal(result[0], np.arange(9))


def test_standardize_other_blocks_have_ascending_first_row():
    b1 = np.array([5, 2, 8, 0, 3, 7, 1, 6, 4])
    b2 = np.array([8, 1, 4, 0, 6, 2, 7, 3, 5])
    b3 = np.array([2, 7, 0, 5, 8, 3, 6, 1, 4])

    result = standardize_blocks(b1, b2, b3)

    for block in result[1:]:
        assert list(block[:3]) == sorted(block[:3])


def test_standardize_matches_relabel_then_lexicograph():
    b1 = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    b2 = np.array([0, 4, 8, 1, 5, 6, 2, 3, 7])

    r1, r2 = relabel_blocks(b1, b2)
    expected = [r1, lexicograph_block(r2)]

    result = standardize_blocks(b1, b2)

    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_standardize_known_result():
    b1 = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
    b2 = np.array([0, 4, 8, 1, 5, 6, 2, 3, 7])

    s1, s2 = standardize_blocks(b1, b2)

    np.testing.assert_array_equal(s1, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(s2, [3, 7, 8, 4, 2, 5, 0, 6, 1])


def test_standardize_preserves_row_membership_of_other_blocks():
    # Relabelling + column reordering never moves a value across rows.
    b1 = np.array([5, 2, 8, 0, 3, 7, 1, 6, 4])
    b2 = np.array([8, 1, 4, 0, 6, 2, 7, 3, 5])

    r1, r2 = relabel_blocks(b1, b2)
    result = standardize_blocks(b1, b2)

    for row in range(3):
        before = set(r2[row * 3:(row + 1) * 3].tolist())
        after = set(result[1][row * 3:(row + 1) * 3].tolist())
        assert before == after


def test_standardize_is_idempotent():
    b1 = np.array([5, 2, 8, 0, 3, 7, 1, 6, 4])
    b2 = np.array([8, 1, 4, 0, 6, 2, 7, 3, 5])

    once = standardize_blocks(b1, b2)
    twice = standardize_blocks(*once)

    for a, b in zip(once, twice):
        np.testing.assert_array_equal(a, b)


def test_standardize_returns_one_block_per_input():
    blocks = [np.random.permutation(9) for _ in range(4)]

    result = standardize_blocks(*blocks)

    assert len(result) == len(blocks)


def test_standardize_already_canonical_is_unchanged():
    b1 = np.arange(9)
    b2 = np.array([2, 5, 7, 0, 3, 8, 1, 4, 6])  # first row already ascending

    s1, s2 = standardize_blocks(b1, b2)

    np.testing.assert_array_equal(s1, b1)
    np.testing.assert_array_equal(s2, b2)
