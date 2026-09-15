"""Matplotlib rendering of a sudoku(-like) grid.

:func:`show_grid` draws one grid (2D array, ordinary matrix indexing --
``grid[i, j]`` is row ``i``, column ``j`` -- the same convention as
:mod:`checker.checker` and :mod:`counter.standardizer`) onto a matplotlib
``Axes``: a heavy solid line around the grid and around every ``k x k``
block, a light dashed line between every other pair of adjacent cells, and
each cell's value drawn as centered text.

Matplotlib's y-axis increases upward while grid row ``0`` is drawn at the
top, so row indices are flipped (``n - row``) when converted to plot
coordinates; see the inline comments in :func:`show_grid`.
"""

from math import isqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

HEAVY_LINE_SETTINGS = {"color": "black", "linewidth": 2}
LIGHT_LINE_SETTINGS = {"color": "black", "linewidth": 1, "linestyle": "--"}
CELL_FONT_SIZE = 12


def show_grid(grid: np.ndarray, ax: Axes | None = None) -> Axes:
    """Draw ``grid`` as a sudoku board on ``ax``.

    Args:
        grid: A 2D array, normally shape ``(k**2, k**2)``, holding the
            values to display. Values are drawn as-is (0-based digits
            ``0 .. k**2 - 1``, the project's usual convention -- not
            shifted to 1-based).
        ax: The ``Axes`` to draw on. If ``None`` (the default), a new
            figure and axes are created; retrieve the figure afterwards via
            the returned axes' ``.figure`` attribute.

    Returns:
        matplotlib.axes.Axes: ``ax``, for chaining (e.g.
            ``show_grid(grid).figure.savefig(...)``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n = grid.shape[1]
    k = isqrt(n)

    # One horizontal + one vertical line per grid coordinate 0 .. n: heavy
    # at block boundaries (multiples of k, which includes the outer border
    # at 0 and n), light dashed everywhere else.
    for idx in range(n + 1):
        style = HEAVY_LINE_SETTINGS if idx % k == 0 else LIGHT_LINE_SETTINGS
        ax.plot([0, n], [n - idx, n - idx], **style)
        ax.plot([idx, idx], [n, 0], **style)

    # Cell values. Row i is drawn at y = n - i - 0.5 (flipped, see module
    # docstring) so row 0 ends up at the top, matching how a grid is read.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j + 0.5, n - i - 0.5, str(grid[i, j]), ha="center", va="center", fontsize=CELL_FONT_SIZE)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
