"""Collapse first bands that are interchangeable for the last two bands.

Once the first band of a 9x9 grid is fixed, the only thing the remaining two
bands "see" of it is, column by column, *which* three digits it already uses
(and therefore which six are still available). Neither the order of those
three digits within the column nor the way the first band arranges its rows
changes the universe of possibilities for bands 2 and 3.

So two first bands are equivalent for grid counting when, for every column,
they place the same set of three digits. :func:`encode_columns` maps a band to
that per-column signature and :func:`collapser_by_column` keeps one
representative per distinct signature.

:func:`collapse_by_permutation` collapses a coarser equivalence instead: bands
that match after permuting the three rows and the columns within each block.
"""

import itertools

import numpy as np

from counter.standardizer import standardize_blocks
from tqdm import tqdm

def collapser_by_column(bands):
    """Group first bands by their column signature, keeping one of each.

    Args:
        bands: Iterable of first bands. Each band is a length ``k * n``
            sequence (27 for the 9x9 case) of digits, flattened row-major
            with ``n = 9`` columns.

    Returns:
        dict: maps each signature integer (see :func:`encode`, called with
        ``n = 9``) to the first band seen carrying that signature. Its length
        is the number of distinct signatures; the multiplicity of each class
        is not retained.
    """
    uncollapsed = {}

    for band in bands:
        encoding = encode_columns(band, 9)
        if encoding not in uncollapsed.keys():
            uncollapsed[encoding] = band

    return uncollapsed

def encode_columns(band, n):
    """Return the per-column digit signature of a band as a single integer.

    Each column is reduced to an ``n``-bit mask (bit ``d`` set iff digit ``d``
    appears somewhere in that column). The ``n`` masks are then concatenated,
    most significant column first, into one ``n * n``-bit integer -- Python
    ints are arbitrary precision, so this is exact for any ``n``. The number
    of rows ``k`` is inferred as ``len(band) // n``.

    Args:
        band: Length ``k * n`` sequence of digits, flattened row-major.
        n: Number of columns.

    Returns:
        int: a value in ``[0, 2 ** (n * n))``. Two bands compare equal here
        exactly when they hold the same set of digits in every column --
        regardless of digit order within a column or of how the rows are
        arranged.
    """
    k = len(band) // n
    encoding = 0
    for col in range(n):
        col_encoding = 0
        for row in range(k):
            # int() keeps this a Python (arbitrary precision) int even when
            # band is a numpy array -- an n*n-bit value overflows numpy ints.
            col_encoding |= 1 << int(band[row * n + col])
        encoding = (encoding << n) | col_encoding
    return encoding

def collapse_by_permutation(bands):
    """Collapse first bands equivalent under row and in-block column permutations.

    Two first bands are treated as equivalent when one can be turned into the
    other by permuting the three band rows and/or permuting the three columns
    within any block, after both are put in block-standard form by
    :func:`standardize_band` (identity block 1, ascending first row per block).

    Args:
        bands: Iterable of first bands. Each band is a length ``k * n``
            sequence (27 for the 9x9 case) of digits, flattened row-major
            with ``n = 9`` columns.

    Returns:
        dict: maps the orbit key -- ``min`` over :func:`encode_by_band` of every
        standardized permutation in the orbit -- to the standardized band
        achieving that minimum. One entry per equivalence class; class sizes
        are not retained.
    """
    collapsed = {}
    seen = set()

    for band in tqdm(bands, total=len(bands), desc="Collapsing by permutation"):
        key = encode_by_band(standardize_band(band, 3), 9)
        if key in seen:
            continue

        orbit = {}
        for permuted_band in generate_permutations(band):
            orbit[encode_by_band(permuted_band, 9)] = permuted_band

        seen.update(orbit)

        base_encoding = min(orbit)
        if base_encoding not in collapsed:
            collapsed[base_encoding] = orbit[base_encoding]

    return collapsed

def encode_by_band(band, n):
    """Encode a band as a base-10 integer, one digit per cell, row-major.

    Injective for digits ``0 .. 9``; used only to give each band a comparable
    key so an orbit can be summarised by its smallest member.
    """
    k = len(band) // n
    encoding = 0
    for row in range(k):
        for col in range(n):
            encoding = encoding * 10 + int(band[row * n + col])
    return encoding

def generate_permutations(band):
    for perm_block_1 in itertools.permutations(range(3)):
        for perm_block_2 in itertools.permutations(range(3)):
            for perm_block_3 in itertools.permutations(range(3)):
                for perm_rows in itertools.permutations(range(3)):
                    permuted_band = np.copy(band)
                    permute_columns(permuted_band, np.array(perm_block_1), 3)
                    permute_columns(permuted_band, np.array(perm_block_2) + 3, 3)
                    permute_columns(permuted_band, np.array(perm_block_3) + 6, 3)
                    permute_rows(permuted_band, np.array(perm_rows), 9)

                    st_band = standardize_band(permuted_band, 3)
                    yield st_band

def permute_columns(band, order, k):
    """In place: rearrange one block's columns; ``order`` holds that block's
    absolute column indices in their new order (so block column ``sorted(order)[i]``
    takes the old contents of column ``order[i]``)."""
    n = k ** 2
    order = np.asarray(order)
    block_cols = np.sort(order)
    for row in range(k):
        base = row * n
        band[base + block_cols] = band[base + order]

def permute_rows(band, order, n):
    k = len(band) // n
    for col in range(n):
        idx1 = np.arange(k) * n + col
        idx2 = np.array(order) * n + col
        band[idx1], band[idx2] = band[idx2], band[idx1]

def standardize_band(band, k):
    n = k ** 2
    # Each block is flattened row-major (top-left to bottom-right), which is the
    # layout counter.standardizer expects and the same layout used to write the
    # standardized blocks back below.
    blocks = [
        [band[row * n + block * k + col] for row in range(k) for col in range(k)]
        for block in range(k)
    ]

    standardized_blocks = standardize_blocks(*blocks)
    standardized_band = np.zeros_like(band)
    for block_idx, block in enumerate(standardized_blocks):
        for row in range(k):
            for col in range(k):
                standardized_band[row * n + block_idx * k + col] = block[row * k + col]

    return standardized_band