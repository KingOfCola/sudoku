# src

Importable library code (on `PYTHONPATH` via root [`.env`](../.env) and
[`pyproject.toml`](../pyproject.toml)'s `pythonpath = ["src"]`).

## Packages

- [checker/README.md](checker/README.md) — validity checks for (partial) sudoku grids.
- [counter/README.md](counter/README.md) — band enumeration, standardization and symmetry-collapsing for grid counting.
- [storage/README.md](storage/README.md) — text and binary on-disk storage for lists of grids.

## Related

- [../scripts/README.md](../scripts/README.md) — runnable scripts built on these packages.
- [../tests/README.md](../tests/README.md) — tests, mirroring this tree.
- [../conftest.py](../conftest.py) — makes `.env`'s `PYTHONPATH` importable for pytest.
- [../CLAUDE.md](../CLAUDE.md) — agent rules, including keeping this file in sync with `src/` contents.
