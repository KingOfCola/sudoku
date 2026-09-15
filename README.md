# sudoku

A small project for counting the number of different sudoku grids, following
the symmetry-reduction approach in
[Felgenhauer & Jarvis](docs/2006_felgenhauer_SudokuNbGrilles.pdf).

See [CLAUDE.md](CLAUDE.md) for the full project guide and mandatory rules for
agents working in this repo.

## Project structure

Every directory has its own `README.md` documenting its contents and linking
to related modules/docs. Start here and follow the links:

- [docs/README.md](docs/README.md) — reference paper the method is based on.
- [src/README.md](src/README.md) — library code.
  - [src/checker/README.md](src/checker/README.md) — grid validity checks.
  - [src/counter/README.md](src/counter/README.md) — band enumeration, standardization, collapsing.
- [scripts/README.md](scripts/README.md) — runnable driver script(s).
- [tests/README.md](tests/README.md) — test suite.
  - [tests/checker/README.md](tests/checker/README.md)
  - [tests/counter/README.md](tests/counter/README.md)

## Running

See [CLAUDE.md](CLAUDE.md) for the mandatory `.venv/` usage and test-running
instructions.
