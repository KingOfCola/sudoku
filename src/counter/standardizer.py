"""Standardization functions for handling blocks of a sudoku.

Blocks correspond to the 3x3 subgrids of a sudoku puzzle. The functions in this module are used to standardize and relabel these blocks for further processing.
They are stored as a 1D array of length 9, where each element corresponds to a cell in the block, ordered from top-left to bottom-right.
Cell values are 0-based, i.e. the digits 0 to 8 (not 1 to 9).
For example, a block represented as:
    [0, 1, 2,
     3, 4, 5,
     6, 7, 8]

"""

import numpy as np

N = 9
K = 3

def standardize_blocks(*blocks):
    """Put a group of blocks into canonical form.

    Standardization has two steps:

    1. Relabel all blocks with :func:`relabel_blocks` so the first block
       becomes the identity sequence ``0, 1, ..., n-1`` while equal tokens
       stay equal across blocks.
    2. Reorder the columns of every block *after the first* with
       :func:`lexicograph_block` so that its first row is ascending.

    The first block is left as the plain identity sequence (its first row is
    already sorted, so step 2 would be a no-op).

    All blocks are assumed to be permutations of ``0 .. n-1``.

    Args:
        *blocks: One or more length-n int arrays, each a permutation of 0..n-1.

    Returns:
        List of standardized blocks, in the same order. The first entry is
        ``array([0, 1, ..., n-1])``; every other entry has an ascending first
        row.

    Example:
        >>> b1 = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
        >>> b2 = np.array([0, 4, 8, 1, 5, 6, 2, 3, 7])
        >>> s1, s2 = standardize_blocks(b1, b2)
        >>> s1
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        >>> s2
        array([3, 7, 8, 4, 2, 5, 0, 6, 1])
    """
    relabeled_blocks = relabel_blocks(*blocks)
    return [relabeled_blocks[0]] + [lexicograph_block(block) for block in relabeled_blocks[1:]]

def relabel_blocks(*blocks):
    """Relabel every block so that the first one becomes 0, 1, ..., n-1.

    The original values are treated as opaque tokens. The first block defines
    the relabelling: the token sitting at position ``i`` of ``blocks[0]`` is
    renamed to ``i``. The same token-to-digit map is then applied to every
    block, so equal tokens across blocks stay equal after relabelling.

    All blocks are assumed to be permutations of ``0 .. n-1``.

    Args:
        *blocks: One or more length-n int arrays, each a permutation of 0..n-1.

    Returns:
        Tuple of relabelled blocks, in the same order. The first is always
        ``array([0, 1, ..., n-1])``.

    Example:
        >>> a, b = relabel_blocks(np.array([2, 0, 1]), np.array([1, 2, 0]))
        >>> a
        array([0, 1, 2])
        >>> b            # token 1 -> 2, token 2 -> 0, token 0 -> 1
        array([2, 0, 1])
    """
    n = len(blocks[0])

    label = np.arange(n)
    label[blocks[0]] = np.arange(n)

    return tuple(label[block] for block in blocks)


def lexicograph_block(block):
    """Reorder a block's 3 columns so its first row reads in ascending order.

    The block is a length-9 array viewed as a 3x3 grid (row-major). The three
    columns are permuted as a whole -- whatever permutation sorts the first
    row is applied identically to the second and third rows -- so the columns
    keep their contents and only change places.

    Args:
        block: Length-9 int array, the 3x3 grid flattened row by row.

    Returns:
        A new length-9 int array with the columns reordered.

    Example:
        Grid    3 7 1        columns sorted by       1 3 7
                4 5 6   -->  their top value    -->  6 4 5
                2 0 8        (1 < 3 < 7)             8 2 0
        >>> block = np.array([3, 7, 1, 4, 5, 6, 2, 0, 8])
        >>> lexicograph_block(block)
        array([1, 3, 7, 6, 4, 5, 8, 2, 0])
    """
    order = np.argsort(block[:K])
    new_block = np.zeros_like(block)

    for i in range(K):
        new_block[i*K:(i+1)*K] = block[i*K:(i+1)*K][order]
    
    return new_block