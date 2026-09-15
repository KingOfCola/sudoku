# tests

Test suite, mirroring [../src/](../src/README.md) package by package. Run via
the project's venv (mandatory rule, see [../CLAUDE.md](../CLAUDE.md)):

```
./.venv/Scripts/python.exe -m pytest
```

Config lives in [../pyproject.toml](../pyproject.toml)
(`--import-mode=importlib --doctest-modules`, so `src/` docstring examples are
also run as tests).

## Packages

- [checker/README.md](checker/README.md) — tests for `src/checker`.
- [counter/README.md](counter/README.md) — tests for `src/counter`.

## Related

- [../src/README.md](../src/README.md) — code under test.
- [../CLAUDE.md](../CLAUDE.md) — agent rules, including keeping this file in sync with `tests/` contents.
