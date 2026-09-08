"""Enumerate the canonical first bands of a 9x9 sudoku.

A *band* is a block-row: the top three rows of the grid, made of three
horizontally adjacent K x K blocks. Here it is stored flattened row-major in a
length ``K ** 3`` (27) integer array with values ``0 .. N-1``:

    indices  0..8  -> row 0 (first row of blocks 1, 2, 3)
    indices  9..17 -> row 1
    indices 18..26 -> row 2

A band is *canonical* (the standard form produced by
:mod:`counter.standardizer`) when:

* block 1 (columns 0..K-1) is the identity block::

      0 1 2
      3 4 5
      6 7 8

* every block's first row is in ascending order.

These canonical bands are the representatives used for symmetry-reduced grid
counting; the full band total is recovered afterwards by multiplying by the
size of each representative's orbit under relabelling, in-block column
permutations and block permutations.

Every band produced satisfies the two sudoku constraints that apply within a
band: each row is a permutation of ``0 .. N-1`` and each K x K block contains
every value ``0 .. N-1`` exactly once. (Column distinctness within the band
then follows from the block constraint.)
"""

import numpy as np
import itertools

K = 3
N = K ** 2

def enumerate_first_row():
    """Yield every canonical first row of a band.

    Block 1's first row is fixed to ``(0, 1, ..., K-1)``. The remaining digits
    ``K .. N-1`` are split between the first rows of blocks 2 and 3, and each
    block's ``K`` digits are emitted in ascending order. This gives
    ``C(N-K, K)`` rows (20 for ``K = 3``).

    Yields:
        tuple[int]: a length-``N`` tuple, the concatenated first rows of
        blocks 1, 2 and 3.
    """
    remaining_digits = set(range(K, K ** 2))
    first_row_block_1 = tuple(range(K))

    for digits_block_2 in itertools.combinations(remaining_digits, K):
        first_row_block_2 = tuple(sorted(digits_block_2))
        first_row_block_3 = tuple(sorted(remaining_digits - set(first_row_block_2)))

        yield tuple(first_row_block_1 + first_row_block_2 + first_row_block_3)

def enumerate_blocks_first_br():
    """Yield every canonical first band of a 9x9 sudoku.

    This is the module's main entry point. ``base`` is seeded with the fixed
    identity block 1 (its lower rows never change), then for each canonical
    first row from :func:`enumerate_first_row` the remaining cells -- rows 1
    and 2 of blocks 2 and 3 -- are filled in every way allowed by the row and
    block constraints.

    Yields:
        numpy.ndarray: a fresh length-``K ** 3`` (27) int array, the band
        flattened row-major. Each array is a copy and may be retained.
    """
    base = np.zeros(K ** 3, dtype=int)
    for row in range(K):
        base[row * N: row * N + K] = range(K * row, K * (row + 1))


    for first_row in enumerate_first_row():
        base[:N] = first_row
        for block in iterate_blocks_first_rb(base, 1, 1):
            yield block

def iterate_blocks_first_rb(base, row, block):
    """Recursively fill cell-group ``(row, block)`` and everything after it.

    Tries every ordering of every valid ``K``-subset for the group, then
    advances to the next block in the row, or to block 1 of the next row,
    yielding ``base.copy()`` once the last group ``(K-1, K-1)`` is placed.
    Mutates ``base`` in place while recursing. ``block`` starts at 1 because
    block 0 stays fixed as the identity block.
    """
    possible_values = get_possible_values_row_3(base, row, block)

    for values in itertools.combinations(possible_values, K):
        for permutation in itertools.permutations(values):
            base[row * N + block * K: row * N + block * K + K] = permutation

            # If we're at the last column of the last row, yield the block. Otherwise, move to the next cell.
            if block == K - 1:
                if row == K - 1:
                    yield base.copy()
                else:
                    # Move to the next row and reset column to 0
                    yield from iterate_blocks_first_rb(base, row + 1, 1)
            else:
                # Move to the next column in the same row
                yield from iterate_blocks_first_rb(base, row, block + 1)


def get_possible_values_row_3(base, row, block):
    """Return the digits still allowed in cell-group ``(row, block)``.

    That is ``0 .. N-1`` minus the digits already placed in earlier rows of
    the same block (block constraint) and in earlier blocks of the same row
    (row constraint).
    """
    forbidden_values = set()
    for r in range(row):
        for col in range(K):
            forbidden_values.add(base[r * N + block * K + col])
    for b in range(block):
        for col in range(K):
            forbidden_values.add(base[row * N + b * K + col])

    return set(range(N)) - forbidden_values