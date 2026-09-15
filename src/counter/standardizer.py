"""Standardization helpers for sudoku grids and bands.

Every structure is a 2D ``ndarray`` with ordinary matrix indexing
(``x[i, j]`` is row ``i``, column ``j``):

* a **grid** has shape ``(k**2, k**2)`` (9x9 for ``k = 3``);
* a **band** is one block-row, shape ``(k, k**2)`` (3x9 for ``k = 3``);
* a **block** is a ``(k, k)`` slice ``x[r:r+k, c:c+k]``.

Cell values are 0-based, i.e. the digits ``0 .. k**2 - 1``. ``k`` is never
passed explicitly -- it is recovered as ``math.isqrt(x.shape[1])``, which is
correct for both a band and a full grid.

The functions here mutate their array argument in place and return ``None``.
"""

from math import isqrt

import numpy as np


def relabel_band(band):
    """Relabel ``band`` so its first block becomes the identity block.

    The first block ``band[:k, :k]`` is read as a sequence of opaque tokens;
    the token at position ``i`` (row-major) is renamed to ``i``. The same
    token->value map is then applied to every cell of ``band``, so equal
    tokens stay equal. The first block is assumed to contain every value
    ``0 .. k**2 - 1`` exactly once.

    Mutates ``band`` in place.

    Example:
        >>> band = np.array([[3, 7, 1, 0, 5, 8, 2, 4, 6],
        ...                  [4, 5, 6, 1, 3, 7, 0, 5, 8],
        ...                  [2, 0, 8, 2, 4, 6, 1, 3, 7]])
        >>> relabel_band(band)
        >>> band[:, :3]
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
    """
    k = isqrt(band.shape[1])
    label = np.empty(k * k, dtype=band.dtype)
    label[band[:k, :k].ravel()] = np.arange(k * k)
    band[:] = label[band]


def lexicograph_block(grid, block_row, block_col):
    """Reorder the columns of one block so that block's first row ascends.

    The block is the one spanning rows ``block_row*k : (block_row+1)*k`` and
    columns ``block_col*k : (block_col+1)*k``. The column permutation that
    sorts that block's top row into ascending order is applied to the
    **whole grid** (all rows), so columns keep their contents and only change
    places.

    Mutates ``grid`` in place.

    Example:
        Block   3 7 1        columns sorted by       1 3 7
                4 5 6   -->  their top value    -->  6 4 5
                2 0 8        (1 < 3 < 7)             8 2 0
        >>> grid = np.array([[3, 7, 1, 0, 0, 0, 0, 0, 0],
        ...                  [4, 5, 6, 0, 0, 0, 0, 0, 0],
        ...                  [2, 0, 8, 0, 0, 0, 0, 0, 0]])
        >>> lexicograph_block(grid, 0, 0)
        >>> grid[:, :3]
        array([[1, 3, 7],
               [6, 4, 5],
               [8, 2, 0]])
    """
    k = isqrt(grid.shape[1])
    cols = slice(block_col * k, (block_col + 1) * k)
    order = np.argsort(grid[block_row * k, cols])
    grid[:, cols] = grid[:, cols][:, order]


def standardize_band(band):
    """Put ``band`` into canonical form, in place.

    Two steps:

    1. :func:`relabel_band` -- the first block becomes ``0, 1, ..., k**2-1``
       and equal tokens stay equal across blocks.
    2. :func:`lexicograph_block` on every block after the first -- each gets an
       ascending first row.

    The first block is left as the plain identity block (its first row is
    already sorted).

    Example:
        >>> band = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
        ...                  [3, 4, 5, 6, 7, 8, 0, 1, 2],
        ...                  [6, 7, 8, 0, 1, 2, 3, 4, 5]])
        >>> standardize_band(band)
        >>> band
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
               [3, 4, 5, 6, 7, 8, 0, 1, 2],
               [6, 7, 8, 0, 1, 2, 3, 4, 5]])
    """
    k = isqrt(band.shape[1])
    relabel_band(band)
    for block_col in range(1, k):
        lexicograph_block(band, 0, block_col)
