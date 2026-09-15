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

Bands here are 2D ``ndarray`` of shape ``(k, k**2)`` (3x9) with ordinary matrix
indexing.
"""

import itertools
from math import isqrt

from tqdm import tqdm

from counter.standardizer import standardize_band

def collapser_by_column(bands):
    """Group first bands by their column signature, keeping one of each.

    Args:
        bands: Iterable of first bands, each a ``(k, k**2)`` int array.

    Returns:
        dict: maps each signature integer (see :func:`encode_columns`) to the
            first band seen carrying that signature. Its length is the number of
            distinct signatures; the multiplicity of each class is not retained.
    """
    uncollapsed = {}

    for band in bands:
        encoding = encode_columns(band)
        if encoding not in uncollapsed.keys():
            uncollapsed[encoding] = band

    return uncollapsed

def encode_columns(band):
    """Return the per-column digit signature of a band as a single integer.

    Each column is reduced to an ``n``-bit mask (bit ``d`` set iff digit ``d``
    appears somewhere in that column), where ``n = band.shape[1]``. The ``n``
    masks are then concatenated, most significant column first, into one
    ``n * n``-bit integer -- Python ints are arbitrary precision, so this is
    exact for any ``n``.

    Returns:
        int: a value in ``[0, 2 ** (n * n))``. Two bands compare equal here
            exactly when they hold the same set of digits in every column --
            regardless of digit order within a column or of how the rows are
            arranged.
    """
    n = band.shape[1]
    encoding = 0
    for col in range(n):
        col_encoding = 0
        for value in band[:, col]:
            # int() keeps this a Python (arbitrary precision) int even though
            # band holds numpy ints -- an n*n-bit value overflows numpy ints.
            col_encoding |= 1 << int(value)
        encoding = (encoding << n) | col_encoding
    return encoding

def collapse_by_permutation(bands):
    """Collapse first bands equivalent under row and in-block column permutations.

    Two first bands are treated as equivalent when one can be turned into the
    other by permuting the three band rows and/or permuting the three columns
    within any block, after both are put in block-standard form by
    :func:`counter.standardizer.standardize_band` (identity block 1, ascending
    first row per block).

    Args:
        bands: Iterable of first bands, each a ``(k, k**2)`` int array.

    Returns:
        dict: maps the orbit key -- ``min`` over :func:`encode_by_band` of
            every standardized permutation in the orbit -- to a dict with
            ``"representative"`` (the standardized band achieving that
            minimum) and ``"orbit_size"`` (the number of distinct
            standardized permutations in the orbit). One entry per
            equivalence class.
    """
    collapsed = {}
    seen = set()

    for band in tqdm(bands, total=len(bands), desc="Collapsing by permutation"):
        standardized = band.copy()
        standardize_band(standardized)
        key = encode_by_band(standardized)
        if key in seen:
            continue

        orbit = {}
        for permuted_band in generate_permutations(band):
            orbit[encode_by_band(permuted_band)] = permuted_band

        seen.update(orbit)

        base_encoding = min(orbit)
        if base_encoding not in collapsed:
            collapsed[base_encoding] = {
                "representative": orbit[base_encoding],
                "orbit_size": len(orbit),
            }

    return collapsed

def encode_by_band(band):
    """Encode a band as a base-10 integer, one digit per cell, row-major.

    Injective for digits ``0 .. 9``; used only to give each band a comparable
    key so an orbit can be summarised by its smallest member.
    """
    encoding = 0
    for value in band.ravel():
        encoding = encoding * 10 + int(value)
    return encoding

def generate_permutations(band):
    """Yield every standardized band reachable from ``band`` by permuting its
    rows and the columns within each block.

    ``6 ** 4`` bands are produced (one row permutation and one column
    permutation per block), with duplicates once :func:`standardize_band`
    folds equivalent arrangements together.
    """
    k = isqrt(band.shape[1])
    for block_perms in itertools.product(itertools.permutations(range(k)), repeat=k):
        for row_perm in itertools.permutations(range(k)):
            permuted_band = band.copy()
            for block_index, order in enumerate(block_perms):
                permute_columns(permuted_band, block_index, order)
            permute_rows(permuted_band, row_perm)

            standardize_band(permuted_band)
            yield permuted_band

def permute_columns(band, block_index, order):
    """In place: rearrange the columns of one block.

    ``order`` is a within-block permutation of ``0 .. k-1``: the block's
    column ``i`` receives the old contents of the block's column ``order[i]``.
    """
    k = isqrt(band.shape[1])
    cols = slice(block_index * k, (block_index + 1) * k)
    band[:, cols] = band[:, cols][:, list(order)]

def permute_rows(band, order):
    """In place: reorder the rows of ``band`` -- row ``i`` receives the old
    row ``order[i]``."""
    band[:] = band[list(order), :]
