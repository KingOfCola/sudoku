# tests/counter

Tests for [../../src/counter/](../../src/counter/README.md).

## Files

- [test_collapser.py](test_collapser.py) — `counter.collapser` (both collapse
  strategies, `encode_columns`, `encode_by_band`, `permute_columns`,
  `permute_rows`, `generate_permutations`), plus `standardize_band`
  properties exercised through it.
- [test_standardizer.py](test_standardizer.py) — `counter.standardizer`
  (`lexicograph_block`, `relabel_band`, `standardize_band`) directly.

Note: `counter.first_block_row` has no dedicated test file yet; it's
exercised indirectly as a band source in the tests above. If you add one,
list it here.

## Related

- [../../src/counter/README.md](../../src/counter/README.md) — package under test.
- [../README.md](../README.md) — sibling test packages.
