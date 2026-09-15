# src/storage

On-disk storage for lists of grids (same convention as
[../checker/](../checker/README.md) and [../counter/](../counter/README.md):
2D arrays, shape `(k**2, k**2)`, 0-based values).

## Files

- [grid_store.py](grid_store.py)
  - `save_grids_text` / `load_grids_text` — human-readable text format: one
    base-36 character per cell, one row per line, small header, blank line
    between grids. Limited to `n = k**2 <= 36`.
  - `save_grids_binary` / `load_grids_binary` — compact binary format: fixed
    header + cells bit-packed to `ceil(log2(n))` bits each (MSB first,
    row-major, grids concatenated). No alphabet-size limit.
  - Both pairs accept a path or an already-open file object (text mode for
    the text format, binary mode for the binary format), so callers can pass
    real paths or in-memory buffers (`io.StringIO`/`io.BytesIO`).

## Related

- [../../tests/storage/README.md](../../tests/storage/README.md) — tests for this module (round trips via `tmp_path` and in-memory buffers).
- [../README.md](../README.md) — sibling packages.
- [../../CLAUDE.md](../../CLAUDE.md) — agent rules, including keeping this file in sync with `grid_store.py`.
