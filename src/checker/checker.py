"""Validity checks for (partially) filled sudoku-like grids.

A grid of order ``k`` is a 2D ``ndarray`` of shape ``(k**2, k**2)`` whose cells
hold values in ``0 .. k**2 - 1`` and use ordinary matrix indexing
(``grid[i, j]`` is row ``i``, column ``j``). The classic 9x9 sudoku is
``k = 3``; a 4x4 "shidoku" is ``k = 2``. ``k`` is recovered from the grid shape
as ``math.isqrt(grid.shape[0])``.
"""

from math import isqrt

import numpy as np


def check_latin(grid):
    """Check that ``grid`` is a valid ``n x n`` Latin square.

    A Latin square is an ``n x n`` grid filled with ``n`` different symbols
    (here the integers ``0 .. n-1``), each occurring exactly once in every row
    and exactly once in every column. ``n`` is taken from ``grid.shape[0]``.

    Args:
        grid: A 2D numpy array. It must be square and every value must lie in
            ``0 .. n-1`` for the grid to be considered valid.

    Returns:
        ``True`` if the grid is a valid Latin square, ``False`` otherwise
            (wrong shape, a repeated value in a row/column, or an
            out-of-range value).

    Example:
        >>> check_latin(np.array([[0, 1], [1, 0]]))
        True
        >>> check_latin(np.array([[0, 1], [0, 1]]))
        False
    """
    n = grid.shape[0]

    # Check that the grid is square.
    if grid.shape != (n, n):
        return False

    # Check that each row contains every value 0 .. n-1 exactly once.
    for row in grid:
        if len(set(row)) != n or not all(0 <= val < n for val in row):
            return False

    # Check that each column contains every value 0 .. n-1 exactly once.
    for col in range(n):
        column_values = grid[:, col]
        if len(set(column_values)) != n or not all(0 <= val < n for val in column_values):
            return False

    return True


def check_blocks(grid):
    """Check that every ``k x k`` block of ``grid`` holds all values ``0 .. k**2-1``.

    ``grid`` is split into a ``k`` by ``k`` arrangement of non-overlapping
    ``k x k`` blocks (``k = isqrt(grid.shape[0])``). Each block must contain
    every value in ``0 .. k**2 - 1`` exactly once -- the "box" constraint of
    sudoku. Rows and columns are not checked here; use :func:`check_latin` for
    those.

    Args:
        grid: A 2D numpy array. It must be square with a side length that is a
            perfect square (``(k**2, k**2)``).

    Returns:
        ``True`` if every block is a permutation of ``0 .. k**2 - 1``,
            ``False`` otherwise (wrong shape, a repeated value in a block,
            or an out-of-range value).

    Example:
        >>> grid = np.array([[0, 1, 2, 3],
        ...                  [2, 3, 0, 1],
        ...                  [1, 0, 3, 2],
        ...                  [3, 2, 1, 0]])
        >>> check_blocks(grid)
        True
    """
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        return False

    m = grid.shape[0]
    k = isqrt(m)
    if k * k != m:
        return False

    for row in range(k):
        for col in range(k):
            block_values = grid[row * k:(row + 1) * k, col * k:(col + 1) * k].flatten()

            if len(set(block_values)) != m or not all(0 <= val < m for val in block_values):
                return False

    return True
