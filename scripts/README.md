# scripts

Runnable drivers that exercise the library code in [../src/](../src/README.md)
end to end (as opposed to `src/`, which holds importable modules only).

## Files

- [first_block_row_enumerator.py](first_block_row_enumerator.py) — enumerates
  every canonical first band, collapses it by permutation then by column
  signature, and prints the resulting counts plus a sample of bands. Uses:
  - [`counter.first_block_row.enumerate_canonical_first_band`](../src/counter/first_block_row.py)
  - [`counter.collapser.collapse_by_permutation`](../src/counter/collapser.py)
  - [`counter.collapser.collapser_by_column`](../src/counter/collapser.py)
- [recounter.py](recounter.py) — exploratory/WIP script for visualizing
  grids via [`visualizer.display.show_grid`](../src/visualizer/display.py).

Run it via the project's venv (see [../CLAUDE.md](../CLAUDE.md)):

```
./.venv/Scripts/python.exe scripts/first_block_row_enumerator.py
```

## Related

- [../src/counter/README.md](../src/counter/README.md) — the modules this script drives.
- [../README.md](../README.md) — project overview.
