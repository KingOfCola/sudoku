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

def standardize_blocks(b1, b2, b3):
    pass

def relabel_blocks(b1, b2, b3):
    label = np.arange(N)
    label[b1] = np.arange(N)

    return label[b2], label[b3]

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