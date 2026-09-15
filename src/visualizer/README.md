# src/visualizer

Matplotlib rendering of grids (same convention as
[../checker/](../checker/README.md), [../counter/](../counter/README.md) and
[../storage/](../storage/README.md): 2D arrays, ordinary matrix indexing,
0-based values).

## Files

- [display.py](display.py)
  - `show_grid(grid, ax=None)` — draws a grid on a matplotlib `Axes` (heavy
    lines around the grid and every `k x k` block, light dashed lines
    between other cells, each cell's value as centered text). Creates a new
    figure/axes when `ax` is omitted; returns the `Axes` either way.

## Related

- [../README.md](../README.md) — sibling packages.
- [../../scripts/README.md](../../scripts/README.md) — `recounter.py` uses `show_grid` to plot bands/grids interactively.
- [../../CLAUDE.md](../../CLAUDE.md) — agent rules, including keeping this file in sync with `display.py`.
