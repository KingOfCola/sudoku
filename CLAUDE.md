# SUDOKU — Project Guide

Read this file first in every session on this repo. It gives the minimum context needed to work here correctly.

## What this project is

A research/counting project computing the number of distinct 9x9 sudoku grids by symmetry-reduced enumeration, following the approach in Felgenhauer & Jarvis, *Enumerating possible Sudoku grids* (`docs/2006_felgenhauer_SudokuNbGrilles.pdf` — the source reference for the method).

The core idea: enumerate canonical "first bands" (the top block-row of 3x9 cells), collapse them into equivalence classes under symmetries that don't affect the final count (relabeling, row permutations within a band, column permutations within a block), then use the reduced set to derive the total grid count without brute-forcing all possibilities.

Grids of general order `k` are supported (`k=3` is the classic 9x9 sudoku; `k=2` is 4x4 "shidoku"), with `k` typically recovered from array shape via `math.isqrt`.

## Layout

- `src/checker/checker.py` — validity checks for filled/partial grids: `check_latin` (row/column constraint) and `check_blocks` (box constraint). Grids are 2D numpy arrays, shape `(k**2, k**2)`, 0-based values, standard `grid[i, j]` indexing.
- `src/counter/first_block_row.py` — enumerates every canonical first band (shape `(k, k**2)`) for `k=3`.
- `src/counter/standardizer.py` — puts a band into canonical form in place: `relabel_band` (first block becomes the identity block), `lexicograph_block` (ascending first row per block), `standardize_band` (both steps).
- `src/counter/collapser.py` — reduces the enumerated bands to inequivalent representatives: `collapser_by_column` (group by per-column digit signature) and `collapse_by_permutation` (group by full row/column permutation orbit).
- `src/storage/grid_store.py` — saves/loads lists of grids to/from a human-readable text format and a compact bit-packed binary format (`save_grids_text`/`load_grids_text`, `save_grids_binary`/`load_grids_binary`); accepts paths or already-open file objects.
- `src/visualizer/display.py` — `show_grid` draws a grid on a matplotlib `Axes` (block/cell grid lines + cell values).
- `scripts/first_block_row_enumerator.py` — runnable driver that ties enumeration + collapsing together and prints counts.
- `scripts/recounter.py` — exploratory/WIP script using `show_grid` to visualize a grid.
- `tests/` mirrors `src/` (`tests/checker/`, `tests/counter/`, `tests/storage/`).
- `conftest.py` — parses `.env`'s `PYTHONPATH` so the IDE and pytest agree on import paths.

## Per-directory documentation

Every real directory (root, `docs/`, `scripts/`, `src/`, `src/checker/`,
`src/counter/`, `src/storage/`, `tests/`, `tests/checker/`, `tests/counter/`,
`tests/storage/`) has its own
small `README.md` acting as a local index: what the directory holds, links to
its important files, and "Related" links to sibling/parent/child docs and the
modules they cover. Start from the root [README.md](README.md) and follow the
links rather than re-deriving the tree from scratch. Generated directories
(`.venv/`, `__pycache__/`, `.pytest_cache/`) are excluded and get no README.

## Docstring style

Docstrings use a Google-style `Args:` / `Returns:` / `Yields:` / `Raises:`
layout with a strict three-level indent hierarchy, relative to the
docstring's own base indent (4 spaces under a `def`):

1. **Section keyword** (`Args:`, `Returns:`, `Yields:`, `Raises:`) sits at
   the base indent.
2. **Each entry** in that section is indented one level deeper than the
   section keyword — one entry per parameter for `Args:`; for
   `Returns:`/`Yields:`/`Raises:`, the `name:`/`type:` label, or (if there's
   no label) the single sentence itself, treated as that section's one
   implicit entry.
3. **Wrapped/continuation lines** of an entry's description are indented one
   level deeper still — two levels past the section keyword. This is what
   visually distinguishes "still describing this entry" from "a new entry
   starts here", and it applies even to an unlabeled single-sentence
   `Returns:`/`Raises:` (nest its continuation lines too, per point 2).

Example (`src/counter/collapser.py`):

```
    Args:
        bands: Iterable of first bands, each a ``(k, k**2)`` int array.

    Returns:
        dict: maps each signature integer (see :func:`encode_columns`) to
            the first band seen carrying that signature. Its length is the
            number of distinct signatures; the multiplicity of each class
            is not retained.
```

Example of the unlabeled case (`src/checker/checker.py`):

```
    Returns:
        ``True`` if the grid is a valid Latin square, ``False`` otherwise
            (wrong shape, a repeated value in a row/column, or an
            out-of-range value).
```

## Mandatory rules for all agents working in this repo

1. **Always use the project's `.venv/` for any Python execution or pytest run.** Never use a system/global Python interpreter for this project.
   - Run scripts: `./.venv/Scripts/python.exe <script>` (Git Bash) or `.venv\Scripts\python.exe <script>` (PowerShell).
   - Run tests: `./.venv/Scripts/python.exe -m pytest` (no extra args needed — config already targets `src` + `tests`).
   - New dependencies go in `requirements.txt`, then reinstall with `./.venv/Scripts/python.exe -m pip install -r requirements.txt`.
2. **Do not change the pytest import mode.** `pyproject.toml` sets `--import-mode=importlib` deliberately — the default prepend mode breaks because `src/checker/checker.py` clashes with the `checker` namespace package. Keep `addopts` as-is unless the user explicitly asks to change it.
3. **Keep doctests passing.** `addopts` includes `--doctest-modules`, so docstring examples in `src/` are executed as tests — any docstring `>>>` example must stay correct.
4. **Don't hand-edit `.pytest_cache/` or `__pycache__/` artifacts** — they're generated and git-ignored.
5. **Every code change updates its directory's `README.md` in the same commit/turn.** If you add, remove, rename, or change the behavior/signature of a file, function, or link target in a directory, update that directory's `README.md` (and any other README whose "Related"/links section points at what changed) so the docs never drift from the code. If you create a new directory with real content, give it a `README.md` following the existing pattern (contents list + links to important files + "Related" section) and link it from its parent's `README.md`.
6. **Follow the [Docstring style](#docstring-style) indentation hierarchy** for every `Args:`/`Returns:`/`Yields:`/`Raises:` section you write or edit — including nesting continuation lines under an unlabeled single-sentence entry.
7. **Do not add tests for code under `scripts/`.** `tests/` only mirrors `src/`; `scripts/` holds runnable drivers/exploration tools without a stable contract, and the user has explicitly asked for them to stay untested.

If the user gives you additional standing rules for this repo, add them to this file (or to persistent memory, per the memory system) rather than only applying them ad hoc.
