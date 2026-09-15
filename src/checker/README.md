# src/checker

Validity checks for (partially) filled sudoku-like grids of order `k`
(`k=3` → 9x9, `k=2` → 4x4 "shidoku"). Grids are 2D numpy arrays, shape
`(k**2, k**2)`, 0-based values.

## Files

- [checker.py](checker.py)
  - `check_latin(grid)` — row/column constraint (Latin square).
  - `check_blocks(grid)` — box/block constraint.

## Related

- [../../tests/checker/README.md](../../tests/checker/README.md) — tests for this module.
- [../README.md](../README.md) — sibling packages.
- [../../CLAUDE.md](../../CLAUDE.md) — agent rules, including keeping this file in sync with `checker.py`.
