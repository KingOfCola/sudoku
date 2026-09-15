"""Efficient on-disk storage for lists of sudoku(-like) grids.

Two file formats are supported for a non-empty list of grids that all share
the same order ``k`` (each grid a 2D ``ndarray`` of shape ``(k**2, k**2)``,
0-based values ``0 .. k**2 - 1``, ordinary matrix indexing -- the same
convention as :mod:`checker.checker` and :mod:`counter.standardizer`):

* **Text** (:func:`save_grids_text` / :func:`load_grids_text`) -- one
  character per cell (base-36: ``0-9`` then ``A-Z``), one line per row, a
  small header line, and a blank line between grids. Human readable (each
  grid prints out looking like an actual sudoku) and trivially parsable:
  fixed line length, one row per line. Limited to ``n = k**2 <= 36`` (the
  alphabet size); classic 9x9 sudoku (``n = 9``) is well within that.

* **Binary** (:func:`save_grids_binary` / :func:`load_grids_binary`) -- a
  short fixed-size header followed by every cell value bit-packed to
  ``ceil(log2(n))`` bits (MSB first, row-major within a grid, grids
  concatenated in order). E.g. 4 bits/cell for classic 9x9 grids instead of
  one character plus a newline per cell in the text form -- several times
  more compact, and not limited to an alphabet size.

Both ``save_*`` functions accept either a path (``str``/``os.PathLike``) or
an already-open file object (text mode for the text format, binary mode for
the binary format); likewise for the ``load_*`` functions. Passing a file
object (e.g. ``io.StringIO``/``io.BytesIO``) is convenient for testing
without touching the filesystem.
"""

import struct
from math import isqrt

import numpy as np

_TEXT_MAGIC = "SUDOKU-GRIDS-TEXT"
_TEXT_VERSION = 1
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ALPHABET_INDEX = {ch: i for i, ch in enumerate(_ALPHABET)}

_BINARY_MAGIC = b"SGRB"
_BINARY_VERSION = 1
_BINARY_HEADER = struct.Struct("<BBQ")  # version, k, count


def save_grids_text(file, grids):
    """Write ``grids`` to ``file`` in the human-readable text format.

    Args:
        file: A path (``str``/``os.PathLike``) or an already-open text-mode
            file object to write to.
        grids: A non-empty iterable of square 2D int arrays, all the same
            shape ``(n, n)`` with ``n = k**2`` a perfect square, holding
            values ``0 .. n-1``.

    Raises:
        ValueError: if ``grids`` is empty, the grids have mismatched or
            non-square shapes, a grid holds an out-of-range value, ``n`` is
            not a perfect square, or ``n`` exceeds the alphabet size (36).

    Example:
        >>> import io
        >>> import numpy as np
        >>> grid = np.array([[0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]])
        >>> buf = io.StringIO()
        >>> save_grids_text(buf, [grid])
        >>> print(buf.getvalue(), end="")
        SUDOKU-GRIDS-TEXT v1 k=2 count=1
        0123
        2301
        1032
        3210
        <BLANKLINE>
    """
    grids, n, k = _check_grids(grids)
    if n > len(_ALPHABET):
        raise ValueError(f"n={n} exceeds the text format's alphabet size ({len(_ALPHABET)})")

    handle, should_close = _open_for_write(file, "w")
    try:
        handle.write(f"{_TEXT_MAGIC} v{_TEXT_VERSION} k={k} count={len(grids)}\n")
        for grid in grids:
            for row in grid:
                handle.write("".join(_ALPHABET[int(value)] for value in row))
                handle.write("\n")
            handle.write("\n")
    finally:
        if should_close:
            handle.close()


def load_grids_text(file):
    """Read back a list of grids written by :func:`save_grids_text`.

    Args:
        file: A path or an already-open text-mode file object to read from.

    Returns:
        list[numpy.ndarray]: the grids, each shape ``(k**2, k**2)`` dtype
        ``int``, in the order they were written.

    Raises:
        ValueError: if the header is missing/malformed, a row has the wrong
            length or an invalid character, or the file ends before
            ``count`` grids have been read.
    """
    handle, should_close = _open_for_read(file, "r")
    try:
        k, count = _parse_text_header(handle.readline())
        n = k * k

        grids = []
        for _ in range(count):
            grids.append(_read_one_text_grid(handle, n))
        return grids
    finally:
        if should_close:
            handle.close()


def save_grids_binary(file, grids):
    """Write ``grids`` to ``file`` in the compact bit-packed binary format.

    Each cell is packed into ``ceil(log2(n))`` bits (``n = k**2``), MSB
    first, cells in row-major order, grids concatenated in the given order.

    Args:
        file: A path or an already-open binary-mode file object to write to.
        grids: Same requirements as :func:`save_grids_text` (the alphabet
            size limit does not apply here).

    Raises:
        ValueError: same cases as :func:`save_grids_text`, minus the
            alphabet size limit.
    """
    grids, n, k = _check_grids(grids)
    bits_per_cell = _bits_per_cell(n)

    values = np.stack(grids).astype(np.uint32).reshape(-1)
    packed = _pack_bits(values, bits_per_cell)

    handle, should_close = _open_for_write(file, "wb")
    try:
        handle.write(_BINARY_MAGIC)
        handle.write(_BINARY_HEADER.pack(_BINARY_VERSION, k, len(grids)))
        handle.write(packed.tobytes())
    finally:
        if should_close:
            handle.close()


def load_grids_binary(file):
    """Read back a list of grids written by :func:`save_grids_binary`.

    Args:
        file: A path or an already-open binary-mode file object to read from.

    Returns:
        list[numpy.ndarray]: the grids, each shape ``(k**2, k**2)`` dtype
        ``int``, in the order they were written.

    Raises:
        ValueError: if the magic bytes or format version are not recognised.
    """
    handle, should_close = _open_for_read(file, "rb")
    try:
        magic = handle.read(len(_BINARY_MAGIC))
        if magic != _BINARY_MAGIC:
            raise ValueError(f"not a recognised sudoku grid binary file (bad magic {magic!r})")

        version, k, count = _BINARY_HEADER.unpack(handle.read(_BINARY_HEADER.size))
        if version != _BINARY_VERSION:
            raise ValueError(f"unsupported binary format version: {version}")

        n = k * k
        bits_per_cell = _bits_per_cell(n)
        total_cells = count * n * n
        values = _unpack_bits(handle.read(), bits_per_cell, total_cells)

        return [
            values[i * n * n:(i + 1) * n * n].reshape(n, n).astype(int)
            for i in range(count)
        ]
    finally:
        if should_close:
            handle.close()


# --- shared helpers ----------------------------------------------------------


def _check_grids(grids):
    """Validate ``grids`` as a non-empty list of equal-shape square grids.

    Returns:
        tuple[list, int, int]: the grids (as a list), their common side
        length ``n``, and the block size ``k = isqrt(n)``.

    Raises:
        ValueError: if ``grids`` is empty, a grid is not square, grids have
            mismatched shapes, a value is out of range, or ``n`` is not a
            perfect square.
    """
    grids = list(grids)
    if not grids:
        raise ValueError("grids must not be empty")

    n = grids[0].shape[0]
    k = isqrt(n)
    if k * k != n:
        raise ValueError(f"grid side length {n} is not a perfect square")

    for grid in grids:
        if grid.ndim != 2 or grid.shape != (n, n):
            raise ValueError(f"all grids must be square and the same shape, got {grid.shape}")
        if grid.min() < 0 or grid.max() >= n:
            raise ValueError(f"grid values must be in 0 .. {n - 1}")

    return grids, n, k


def _bits_per_cell(n):
    return max(1, (n - 1).bit_length())


def _open_for_write(file, mode):
    if hasattr(file, "write"):
        return file, False
    return open(file, mode), True


def _open_for_read(file, mode):
    if hasattr(file, "read"):
        return file, False
    return open(file, mode), True


def _parse_text_header(header):
    parts = header.strip().split()
    if len(parts) != 4 or parts[0] != _TEXT_MAGIC:
        raise ValueError(f"not a recognised sudoku grid text file header: {header!r}")

    _magic, version_token, k_token, count_token = parts
    if version_token != f"v{_TEXT_VERSION}":
        raise ValueError(f"unsupported text format version: {version_token!r}")
    if not k_token.startswith("k=") or not count_token.startswith("count="):
        raise ValueError(f"malformed header: {header!r}")

    return int(k_token[len("k="):]), int(count_token[len("count="):])


def _read_one_text_grid(handle, n):
    rows = []
    while len(rows) < n:
        line = handle.readline()
        if line == "":
            raise ValueError("unexpected end of file while reading a grid")
        line = line.rstrip("\n")
        if line == "":
            continue  # blank separator line between grids
        if len(line) != n:
            raise ValueError(f"expected a row of length {n}, got {line!r}")
        try:
            rows.append([_ALPHABET_INDEX[ch] for ch in line])
        except KeyError as exc:
            raise ValueError(f"invalid character {exc} in grid row {line!r}") from exc
    return np.array(rows, dtype=int)


def _pack_bits(values, bits_per_cell):
    """Bit-pack the 1D uint array ``values`` to ``bits_per_cell`` bits each, MSB first."""
    shifts = np.arange(bits_per_cell - 1, -1, -1, dtype=np.uint32)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits)


def _unpack_bits(payload, bits_per_cell, total_cells):
    """Inverse of :func:`_pack_bits`: recover ``total_cells`` values from packed bytes."""
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), count=total_cells * bits_per_cell)
    weights = (1 << np.arange(bits_per_cell - 1, -1, -1)).astype(np.uint32)
    return bits.reshape(total_cells, bits_per_cell).astype(np.uint32) @ weights
