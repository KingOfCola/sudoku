"""Tests for :mod:`counter.collapser`."""

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
    standardize_band,
)
from counter.first_block_row import enumerate_canonical_first_band


def make_band(row0, row1, row2):
    return np.concatenate([np.array(row0), np.array(row1), np.array(row2)])


def first_canonical_bands(count):
    return list(itertools.islice(enumerate_canonical_first_band(), count))


def as_grid(band):
    return np.asarray(band).reshape(3, 9)


def is_valid_band(band):
    grid = as_grid(band)
    rows_ok = all(sorted(grid[r].tolist()) == list(range(9)) for r in range(3))
    blocks_ok = all(
        sorted(grid[:, c:c + 3].flatten().tolist()) == list(range(9))
        for c in (0, 3, 6)
    )
    return rows_ok and blocks_ok


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


# --- encode_by_band ----------------------------------------------------------


def test_encode_by_band_known_small_case():
    band = [0, 1, 2,
            3, 4, 5]

    assert encode_by_band(band, 3) == 12345


def test_encode_by_band_is_injective_on_canonical_bands():
    bands = first_canonical_bands(2000)

    encodings = [encode_by_band(b, 9) for b in bands]

    assert len(set(encodings)) == len(bands)


# --- permute_columns / permute_rows ----------------------------------------


def test_permute_columns_only_touches_its_own_block():
    band = first_canonical_bands(1)[0]
    original = band.copy()

    permute_columns(band, np.array([5, 3, 4]), 3)  # block 2 (columns 3, 4, 5)

    changed = as_grid(band)
    before = as_grid(original)
    assert np.array_equal(changed[:, 0:3], before[:, 0:3])   # block 1 untouched
    assert np.array_equal(changed[:, 6:9], before[:, 6:9])   # block 3 untouched
    assert sorted(band.tolist()) == sorted(original.tolist())
    assert is_valid_band(band)


def test_permute_columns_identity_order_is_a_no_op():
    band = first_canonical_bands(1)[0]
    original = band.copy()

    permute_columns(band, np.array([3, 4, 5]), 3)

    assert np.array_equal(band, original)


def test_permute_columns_moves_the_expected_column():
    band = first_canonical_bands(5)[4]
    before = as_grid(band).copy()

    # New block-1 column order (0, 1, 2) <- old (2, 0, 1).
    permute_columns(band, np.array([2, 0, 1]), 3)

    after = as_grid(band)
    assert np.array_equal(after[:, 0], before[:, 2])
    assert np.array_equal(after[:, 1], before[:, 0])
    assert np.array_equal(after[:, 2], before[:, 1])


def test_permute_rows_reorders_whole_rows():
    band = first_canonical_bands(3)[2]
    before = as_grid(band).copy()

    permute_rows(band, np.array([1, 2, 0]), 9)

    after_rows = {tuple(row) for row in as_grid(band)}
    before_rows = {tuple(row) for row in before}
    assert after_rows == before_rows
    assert is_valid_band(band)


# --- standardize_band -------------------------------------------------------


def test_standardize_band_output_is_a_valid_band():
    for band in first_canonical_bands(200):
        assert is_valid_band(standardize_band(band, 3))


def test_standardize_band_makes_block_one_the_identity():
    for band in first_canonical_bands(200):
        grid = as_grid(standardize_band(band, 3))
        assert grid[:, 0:3].flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_standardize_band_makes_every_first_block_row_ascending():
    for band in first_canonical_bands(200):
        grid = as_grid(standardize_band(band, 3))
        for c in (0, 3, 6):
            assert list(grid[0, c:c + 3]) == sorted(grid[0, c:c + 3])


def test_standardize_band_is_idempotent():
    for band in first_canonical_bands(200):
        once = standardize_band(band, 3)
        twice = standardize_band(np.asarray(once), 3)
        assert np.array_equal(once, twice)


def test_standardize_band_fixes_canonical_first_bands():
    # Bands from enumerate_canonical_first_band are already block-standard.
    for band in first_canonical_bands(200):
        assert np.array_equal(standardize_band(band, 3), band)


def test_standardize_band_undoes_a_relabelling():
    canonical = first_canonical_bands(30)[29]
    relabel = np.array([4, 0, 7, 2, 8, 1, 5, 3, 6])  # a permutation of 0..8
    relabelled = relabel[canonical]

    assert not np.array_equal(relabelled, canonical)
    assert np.array_equal(standardize_band(relabelled, 3), canonical)


def test_standardize_band_invariant_under_block2_and_block3_column_permutations():
    band = first_canonical_bands(50)[49]
    reference = standardize_band(band, 3)

    for p2 in itertools.permutations(range(3)):
        for p3 in itertools.permutations(range(3)):
            permuted = band.copy()
            permute_columns(permuted, np.array(p2) + 3, 3)
            permute_columns(permuted, np.array(p3) + 6, 3)
            assert np.array_equal(standardize_band(permuted, 3), reference)


# --- generate_permutations ------------------------------------------------------


def test_generate_permutations_yields_1296_standardized_valid_bands():
    band = first_canonical_bands(10)[9]

    produced = list(generate_permutations(band))

    assert len(produced) == 6 ** 4  # 3 in-block column perms + 1 row perm
    for candidate in produced:
        assert is_valid_band(candidate)
        assert np.array_equal(standardize_band(np.asarray(candidate), 3), candidate)


def test_generate_permutations_contains_the_bands_own_standard_form():
    band = first_canonical_bands(10)[9]

    encodings = {encode_by_band(p, 9) for p in generate_permutations(band)}

    assert encode_by_band(standardize_band(band, 3), 9) in encodings


# --- collapse_by_permutation --------------------------------------------------


def test_collapse_by_permutation_empty_input():
    assert collapse_by_permutation([]) == {}


def test_collapse_by_permutation_key_is_the_orbit_minimum():
    band = first_canonical_bands(10)[9]

    result = collapse_by_permutation([band])

    expected_key = min(encode_by_band(p, 9) for p in generate_permutations(band))
    assert list(result) == [expected_key]


def test_collapse_by_permutation_representatives_are_standardized():
    result = collapse_by_permutation(first_canonical_bands(24))

    for key, band in result.items():
        assert key == encode_by_band(band, 9)
        assert np.array_equal(standardize_band(np.asarray(band), 3), band)


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
