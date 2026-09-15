# src/counter

Symmetry-reduced enumeration of 9x9 sudoku "first bands" (the top block-row,
shape `(k, k**2)`), implementing the method from
[the Felgenhauer & Jarvis paper](../../docs/README.md).

## Files

- [first_block_row.py](first_block_row.py) — `enumerate_canonical_first_band()`
  yields every canonical first band (block 1 = identity block, ascending
  first row per block).
- [standardizer.py](standardizer.py) — puts a band into canonical form
  in place: `relabel_band`, `lexicograph_block`, `standardize_band`.
- [collapser.py](collapser.py) — reduces enumerated bands to inequivalent
  representatives: `collapser_by_column` (per-column digit signature) and
  `collapse_by_permutation` (full row/column permutation orbit); uses
  `standardize_band` from `standardizer.py`.

## Related

- [../../scripts/README.md](../../scripts/README.md) — driver script (`first_block_row_enumerator.py`) that ties these modules together.
- [../../tests/counter/README.md](../../tests/counter/README.md) — tests for this package.
- [../checker/README.md](../checker/README.md) — sibling package (grid validity checks).
- [../../docs/README.md](../../docs/README.md) — reference paper for the method.
- [../../CLAUDE.md](../../CLAUDE.md) — agent rules, including keeping this file in sync with this package's contents.
