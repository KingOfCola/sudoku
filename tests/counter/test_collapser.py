"""Tests for :mod:`counter.collapser`.

Bands are 2D ndarrays of shape ``(3, 9)`` with matrix indexing.
"""

import itertools

import numpy as np

from counter.collapser import (
    collapse_by_permutation,
    collapser_by_column,
    encode_by_band,
    encode_columns,
    generate_permutations,
    permute_columns,
    permute_rows,
)
from counter.first_block_row import enumerate_canonical_first_band
from counter.standardizer import standardize_band


def make_band(row0, row1, row2):
    return np.array([row0, row1, row2])


def first_canonical_bands(count):
    return list(itertools.islice(enumerate_canonical_first_band(), count))


def standardized(band):
    """Return a standardized copy of ``band`` (``standardize_band`` mutates)."""
    result = np.array(band)
    standardize_band(result)
    return result


def is_valid_band(band):
    rows_ok = all(
        sorted(band[r].tolist()) == list(range(9)) for r in range(band.shape[0])
    )
    blocks_ok = all(
        sorted(band[:, c:c + 3].ravel().tolist()) == list(range(9))
        for c in range(0, band.shape[1], 3)
    )
    return rows_ok and blocks_ok


# --- encode_columns ------------------------------------------------------------


def test_encode_columns_known_small_case():
    # A (2, 3) band. Columns are {0,2}, {1,0}, {2,1}, concatenated MSB-first
    # with 3 bits per column.
    band = np.array([[0, 1, 2],
                     [2, 0, 1]])
    expected = (0b101 << 6) | (0b011 << 3) | 0b110

    assert encode_columns(band) == expected


def test_encode_columns_returns_int_within_n_squared_bits():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])

    result = encode_columns(band)

    assert isinstance(result, int)
    assert 0 <= result < (1 << (9 * 9))
    # The most significant column's mask is non-zero, so the value uses more
    # than 8 * 9 bits.
    assert result.bit_length() > 8 * 9


def test_encode_columns_ignores_digit_order_within_a_column():
    a = encode_columns(np.array([[0, 1, 2], [2, 0, 1]]))
    b = encode_columns(np.array([[2, 0, 1], [0, 1, 2]]))  # same pairs, swapped

    assert a == b


def test_encode_columns_ignores_row_arrangement():
    rows = [[0, 3, 6, 1, 4, 7, 2, 5, 8],
            [1, 4, 7, 2, 5, 8, 0, 3, 6],
            [2, 5, 8, 0, 3, 6, 1, 4, 7]]
    base = np.array(rows)
    shuffled = np.array([rows[2], rows[0], rows[1]])

    assert encode_columns(base) == encode_columns(shuffled)


def test_encode_columns_distinguishes_different_column_sets():
    a = encode_columns(np.array([[0, 1, 2], [0, 1, 2]]))   # columns {0}, {1}, {2}
    b = encode_columns(np.array([[0, 1, 2], [1, 2, 0]]))   # columns {0,1}, {1,2}, {2,0}

    assert a != b


def test_encode_columns_uses_every_row():
    two_rows = encode_columns(np.array([[0, 1, 2], [3, 4, 5]]))
    three_rows = encode_columns(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

    assert isinstance(two_rows, int) and isinstance(three_rows, int)
    # The third row adds digits 6, 7, 8 to the column masks, so it must differ.
    assert two_rows != three_rows


# --- collapser_by_column -----------------------------------------------------


def test_collapser_empty_input():
    assert collapser_by_column([]) == {}


def test_collapser_merges_bands_with_the_same_signature():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])
    row_swapped = band[[1, 0, 2], :]

    result = collapser_by_column([band, row_swapped])

    assert len(result) == 1


def test_collapser_keeps_the_first_representative():
    band = make_band([0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [3, 4, 5, 6, 7, 8, 0, 1, 2],
                     [6, 7, 8, 0, 1, 2, 3, 4, 5])
    row_swapped = band[[1, 0, 2], :]

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
    bands = first_canonical_bands(500)

    result = collapser_by_column(bands)

    for key, band in result.items():
        assert key == encode_columns(band)


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

    assert len(result) == len({encode_columns(b) for b in bands})
    assert len(result) < len(bands)


def test_collapser_is_idempotent():
    bands = first_canonical_bands(2000)

    once = collapser_by_column(bands)
    twice = collapser_by_column(list(once.values()))

    assert set(once.keys()) == set(twice.keys())


# --- encode_by_band ----------------------------------------------------------


def test_encode_by_band_known_small_case():
    band = np.array([[0, 1, 2],
                     [3, 4, 5]])

    assert encode_by_band(band) == 12345


def test_encode_by_band_is_injective_on_canonical_bands():
    bands = first_canonical_bands(2000)

    encodings = [encode_by_band(b) for b in bands]

    assert len(set(encodings)) == len(bands)


# --- permute_columns / permute_rows ----------------------------------------


def test_permute_columns_only_touches_its_own_block():
    band = first_canonical_bands(1)[0]
    original = band.copy()

    permute_columns(band, 1, [2, 0, 1])  # block 1 spans columns 3, 4, 5

    assert np.array_equal(band[:, 0:3], original[:, 0:3])   # block 0 untouched
    assert np.array_equal(band[:, 6:9], original[:, 6:9])   # block 2 untouched
    assert sorted(band.ravel().tolist()) == sorted(original.ravel().tolist())
    assert is_valid_band(band)


def test_permute_columns_identity_order_is_a_no_op():
    band = first_canonical_bands(1)[0]
    original = band.copy()

    permute_columns(band, 1, [0, 1, 2])

    assert np.array_equal(band, original)


def test_permute_columns_moves_the_expected_column():
    band = first_canonical_bands(5)[4]
    before = band.copy()

    # New block-0 column i <- old block-0 column [2, 0, 1][i].
    permute_columns(band, 0, [2, 0, 1])

    assert np.array_equal(band[:, 0], before[:, 2])
    assert np.array_equal(band[:, 1], before[:, 0])
    assert np.array_equal(band[:, 2], before[:, 1])


def test_permute_rows_reorders_whole_rows():
    band = first_canonical_bands(3)[2]
    before = band.copy()

    permute_rows(band, [1, 2, 0])

    after_rows = {tuple(row) for row in band}
    before_rows = {tuple(row) for row in before}
    assert after_rows == before_rows
    assert np.array_equal(band[0], before[1])
    assert is_valid_band(band)


# --- standardize_band (from counter.standardizer, exercised via collapser) ---


def test_standardize_band_output_is_a_valid_band():
    for band in first_canonical_bands(200):
        assert is_valid_band(standardized(band))


def test_standardize_band_makes_block_one_the_identity():
    for band in first_canonical_bands(200):
        result = standardized(band)
        assert result[:, 0:3].tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_standardize_band_makes_every_first_block_row_ascending():
    for band in first_canonical_bands(200):
        result = standardized(band)
        for c in (0, 3, 6):
            assert list(result[0, c:c + 3]) == sorted(result[0, c:c + 3])


def test_standardize_band_is_idempotent():
    for band in first_canonical_bands(200):
        once = standardized(band)
        twice = standardized(once)
        assert np.array_equal(once, twice)


def test_standardize_band_fixes_canonical_first_bands():
    # Bands from enumerate_canonical_first_band are already block-standard.
    for band in first_canonical_bands(200):
        assert np.array_equal(standardized(band), band)


def test_standardize_band_undoes_a_relabelling():
    canonical = first_canonical_bands(30)[29]
    relabel = np.array([4, 0, 7, 2, 8, 1, 5, 3, 6])  # a permutation of 0..8
    relabelled = relabel[canonical]

    assert not np.array_equal(relabelled, canonical)
    assert np.array_equal(standardized(relabelled), canonical)


def test_standardize_band_invariant_under_block1_and_block2_column_permutations():
    band = first_canonical_bands(50)[49]
    reference = standardized(band)

    for p1 in itertools.permutations(range(3)):
        for p2 in itertools.permutations(range(3)):
            permuted = band.copy()
            permute_columns(permuted, 1, p1)
            permute_columns(permuted, 2, p2)
            assert np.array_equal(standardized(permuted), reference)


# --- generate_permutations ------------------------------------------------------


def test_generate_permutations_yields_1296_standardized_valid_bands():
    band = first_canonical_bands(10)[9]

    produced = list(generate_permutations(band))

    assert len(produced) == 6 ** 4  # one column perm per block + one row perm
    for candidate in produced:
        assert is_valid_band(candidate)
        assert np.array_equal(standardized(candidate), candidate)


def test_generate_permutations_contains_the_bands_own_standard_form():
    band = first_canonical_bands(10)[9]

    encodings = {encode_by_band(p) for p in generate_permutations(band)}

    assert encode_by_band(standardized(band)) in encodings


# --- collapse_by_permutation --------------------------------------------------


def test_collapse_by_permutation_empty_input():
    assert collapse_by_permutation([]) == {}


def test_collapse_by_permutation_key_is_the_orbit_minimum():
    band = first_canonical_bands(10)[9]

    result = collapse_by_permutation([band])

    expected_key = min(encode_by_band(p) for p in generate_permutations(band))
    assert list(result) == [expected_key]


def test_collapse_by_permutation_representatives_are_standardized():
    result = collapse_by_permutation(first_canonical_bands(24))

    for key, band in result.items():
        assert key == encode_by_band(band)
        assert np.array_equal(standardized(band), band)


def test_collapse_by_permutation_groups_a_band_with_its_permutations():
    band = first_canonical_bands(10)[9]
    variants = list(itertools.islice(generate_permutations(band), 0, 1296, 111))

    result = collapse_by_permutation([band, *variants])

    assert len(result) == 1


def test_collapse_by_permutation_key_is_stable_across_permutations():
    band = first_canonical_bands(20)[19]
    (base_key,) = collapse_by_permutation([band])

    for variant in itertools.islice(generate_permutations(band), 0, 1296, 129):
        (variant_key,) = collapse_by_permutation([variant])
        assert variant_key == base_key


def test_collapse_by_permutation_reduces_and_is_idempotent():
    bands = first_canonical_bands(24)

    once = collapse_by_permutation(bands)
    twice = collapse_by_permutation(list(once.values()))

    assert len(once) < len(bands)
    assert set(once) == set(twice)
