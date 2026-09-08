"""Collapse first bands that are interchangeable for the last two bands.

Once the first band of a 9x9 grid is fixed, the only thing the remaining two
bands "see" of it is, column by column, *which* three digits it already uses
(and therefore which six are still available). Neither the order of those
three digits within the column nor the way the first band arranges its rows
changes the universe of possibilities for bands 2 and 3.

So two first bands are equivalent for grid counting when, for every column,
they place the same set of three digits. :func:`encode` maps a band to that
per-column signature and :func:`collapser` keeps one representative per
distinct signature.
"""

import numpy as np

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

