"""Tests for :mod:`counter.collapser`."""

import itertools

import numpy as np

from counter.collapser import collapser_by_column, encode_columns
from counter.first_block_row import enumerate_canonical_first_band


def make_band(row0, row1, row2):
    return np.concatenate([np.array(row0), np.array(row1), np.array(row2)])


# --- encode ----------------------------------------------------------------


def test_encode_known_small_case():
    # n = 3, k = 2. Columns are {0,2}, {1,0}, {2,1}, concatenated MSB-first
    # with 3 bits per column.
    band = [0, 1, 2,
            2, 0, 1]
    expected = (0b101 << 6) | (0b011 << 3) | 0b110

    assert encode_columns(band, 3) == expected


def test_encode_returns_int_within_n_squared_bits():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])

    result = encode_columns(band, 9)

    assert isinstance(result, int)
    assert 0 <= result < (1 << (9 * 9))
    # 9 columns each need at least one bit set, so the top column's mask is
    # non-zero -> the value uses more than 8 * 9 bits when digit 8 is present.
    assert result.bit_length() > 8 * 9


def test_encode_ignores_digit_order_within_a_column():
    a = encode_columns([0, 1, 2, 2, 0, 1], 3)
    b = encode_columns([2, 0, 1, 0, 1, 2], 3)  # each column holds the same pair, swapped

    assert a == b


def test_encode_ignores_row_arrangement():
    rows = [[0, 3, 6, 1, 4, 7, 2, 5, 8],
            [1, 4, 7, 2, 5, 8, 0, 3, 6],
            [2, 5, 8, 0, 3, 6, 1, 4, 7]]
    base = make_band(*rows)
    shuffled = make_band(rows[2], rows[0], rows[1])

    assert encode_columns(base, 9) == encode_columns(shuffled, 9)


def test_encode_distinguishes_different_column_sets():
    a = encode_columns([0, 1, 2, 0, 1, 2], 3)   # columns {0}, {1}, {2}
    b = encode_columns([0, 1, 2, 1, 2, 0], 3)   # columns {0,1}, {1,2}, {2,0}

    assert a != b


def test_encode_infers_row_count_from_length():
    two_rows = encode_columns([0, 1, 2, 3, 4, 5], 3)          # k = 2
    three_rows = encode_columns([0, 1, 2, 3, 4, 5, 6, 7, 8], 3)  # k = 3

    assert isinstance(two_rows, int) and isinstance(three_rows, int)
    # The third row adds digits 6, 7, 8 to the column masks, so it must differ.
    assert two_rows != three_rows


# --- collapser -----------------------------------------------------------------


def test_collapser_empty_input():
    assert collapser_by_column([]) == {}


def test_collapser_merges_bands_with_the_same_signature():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])
    row_swapped = make_band([3, 4, 5, 6, 7, 8, 0, 1, 2],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            [6, 7, 8, 0, 1, 2, 3, 4, 5])

    result = collapser_by_column([band, row_swapped])

    assert len(result) == 1


def test_collapser_keeps_the_first_representative():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])
    row_swapped = make_band([3, 4, 5, 6, 7, 8, 0, 1, 2],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8],
                            [6, 7, 8, 0, 1, 2, 3, 4, 5])

    (kept,) = collapser_by_column([band, row_swapped]).values()

    assert kept is band


def test_collapser_separates_different_signatures():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])
    other = (band + 1) % 9

    result = collapser_by_column([band, other])

    assert len(result) == 2


def test_collapser_keys_are_the_encoding_of_their_value():
    bands = list(itertools.islice(enumerate_canonical_first_band(), 500))

    result = collapser_by_column(bands)

    for key, band in result.items():
        assert key == encode_columns(band, 9)


def test_collapser_deduplicates_repeated_band():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])

    result = collapser_by_column([band, band.copy(), band.copy()])

    assert len(result) == 1
    assert next(iter(result.values())) is band


def test_collapser_on_canonical_first_bands_actually_collapses():
    bands = list(enumerate_canonical_first_band())

    result = collapser_by_column(bands)

    assert len(result) == len({encode_columns(b, 9) for b in bands})
    assert len(result) < len(bands)


def test_collapser_is_idempotent():
    bands = list(itertools.islice(enumerate_canonical_first_band(), 2000))

    once = collapser_by_column(bands)
    twice = collapser_by_column(list(once.values()))

    assert set(once.keys()) == set(twice.keys())
