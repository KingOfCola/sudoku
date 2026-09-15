# tests/storage

Tests for [../../src/storage/grid_store.py](../../src/storage/grid_store.py).

## Files

- [test_grid_store.py](test_grid_store.py) — round-trips (save then load) for
  both the text and binary formats, plus error handling (empty input,
  mismatched shapes, out-of-range values, alphabet overflow, corrupted
  headers/magic, truncated files). Uses only pytest's `tmp_path` (temporary
  directory) or in-memory `io.StringIO`/`io.BytesIO` buffers — never writes
  into the repository.

## Related

- [../../src/storage/README.md](../../src/storage/README.md) — module under test.
- [../README.md](../README.md) — sibling test packages.
