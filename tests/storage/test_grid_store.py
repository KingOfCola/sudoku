"""Tests for :mod:`storage.grid_store`.

All tests write only to pytest's ``tmp_path`` (a temporary directory removed
after the test run) or to in-memory file objects (``io.StringIO`` /
``io.BytesIO``) -- never to files inside the repository.
"""

import io

import numpy as np
import pytest

from storage.grid_store import (
    load_grids_binary,
    load_grids_text,
    save_grids_binary,
    save_grids_text,
)

VALID_4x4 = np.array([
    [0, 1, 2, 3],
    [2, 3, 0, 1],
    [1, 0, 3, 2],
    [3, 2, 1, 0],
])

VALID_4x4_OTHER = np.array([
    [1, 0, 3, 2],
    [3, 2, 1, 0],
    [0, 1, 2, 3],
    [2, 3, 0, 1],
])

VALID_9x9 = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [3, 4, 5, 6, 7, 8, 0, 1, 2],
    [6, 7, 8, 0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6, 7, 8, 0],
    [4, 5, 6, 7, 8, 0, 1, 2, 3],
    [7, 8, 0, 1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7, 8, 0, 1],
    [5, 6, 7, 8, 0, 1, 2, 3, 4],
    [8, 0, 1, 2, 3, 4, 5, 6, 7],
])


# --- text format: round trip via tmp_path -------------------------------------


def test_text_round_trip_single_grid_tmp_path(tmp_path):
    path = tmp_path / "grids.txt"
    save_grids_text(path, [VALID_4x4])

    result = load_grids_text(path)

    assert len(result) == 1
    assert np.array_equal(result[0], VALID_4x4)


def test_text_round_trip_multiple_grids_tmp_path(tmp_path):
    path = tmp_path / "grids.txt"
    save_grids_text(path, [VALID_4x4, VALID_4x4_OTHER])

    result = load_grids_text(path)

    assert len(result) == 2
    assert np.array_equal(result[0], VALID_4x4)
    assert np.array_equal(result[1], VALID_4x4_OTHER)


def test_text_round_trip_9x9_tmp_path(tmp_path):
    path = tmp_path / "grids.txt"
    save_grids_text(path, [VALID_9x9])

    (result,) = load_grids_text(path)

    assert np.array_equal(result, VALID_9x9)


# --- text format: round trip via in-memory (abstract) file objects -------------


def test_text_round_trip_stringio():
    buf = io.StringIO()
    save_grids_text(buf, [VALID_4x4, VALID_4x4_OTHER])

    buf.seek(0)
    result = load_grids_text(buf)

    assert len(result) == 2
    assert np.array_equal(result[0], VALID_4x4)
    assert np.array_equal(result[1], VALID_4x4_OTHER)


def test_text_format_is_one_char_per_cell_and_human_readable():
    buf = io.StringIO()
    save_grids_text(buf, [VALID_4x4])

    lines = buf.getvalue().splitlines()

    assert lines[0] == "SUDOKU-GRIDS-TEXT v1 k=2 count=1"
    assert lines[1:5] == ["0123", "2301", "1032", "3210"]


# --- binary format: round trip via tmp_path -------------------------------------


def test_binary_round_trip_single_grid_tmp_path(tmp_path):
    path = tmp_path / "grids.bin"
    save_grids_binary(path, [VALID_4x4])

    (result,) = load_grids_binary(path)

    assert np.array_equal(result, VALID_4x4)


def test_binary_round_trip_multiple_grids_tmp_path(tmp_path):
    path = tmp_path / "grids.bin"
    save_grids_binary(path, [VALID_4x4, VALID_4x4_OTHER])

    result = load_grids_binary(path)

    assert len(result) == 2
    assert np.array_equal(result[0], VALID_4x4)
    assert np.array_equal(result[1], VALID_4x4_OTHER)


def test_binary_round_trip_9x9_tmp_path(tmp_path):
    path = tmp_path / "grids.bin"
    save_grids_binary(path, [VALID_9x9])

    (result,) = load_grids_binary(path)

    assert np.array_equal(result, VALID_9x9)


# --- binary format: round trip via in-memory (abstract) file objects -----------


def test_binary_round_trip_bytesio():
    buf = io.BytesIO()
    save_grids_binary(buf, [VALID_4x4, VALID_4x4_OTHER])

    buf.seek(0)
    result = load_grids_binary(buf)

    assert len(result) == 2
    assert np.array_equal(result[0], VALID_4x4)
    assert np.array_equal(result[1], VALID_4x4_OTHER)


def test_binary_format_uses_ceil_log2_n_bits_per_cell(tmp_path):
    # k=3, n=9 -> bits_per_cell=4 -> payload is ceil(81*4/8) = 41 bytes.
    path = tmp_path / "grids.bin"
    save_grids_binary(path, [VALID_9x9])

    header_size = len(b"SGRB") + 10  # magic + struct("<BBQ") = 4 + (1+1+8)
    payload_size = path.stat().st_size - header_size
    assert payload_size == (9 * 9 * 4 + 7) // 8


def test_binary_is_smaller_than_text_for_9x9(tmp_path):
    text_path = tmp_path / "grids.txt"
    binary_path = tmp_path / "grids.bin"
    grids = [VALID_9x9] * 5

    save_grids_text(text_path, grids)
    save_grids_binary(binary_path, grids)

    assert binary_path.stat().st_size < text_path.stat().st_size


# --- error handling --------------------------------------------------------------


def test_save_text_rejects_empty_list():
    with pytest.raises(ValueError):
        save_grids_text(io.StringIO(), [])


def test_save_binary_rejects_empty_list():
    with pytest.raises(ValueError):
        save_grids_binary(io.BytesIO(), [])


def test_save_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        save_grids_text(io.StringIO(), [VALID_4x4, VALID_9x9])


def test_save_rejects_out_of_range_values():
    bad = VALID_4x4.copy()
    bad[0, 0] = 4  # only 0..3 valid for a 4x4 grid

    with pytest.raises(ValueError):
        save_grids_text(io.StringIO(), [bad])


def test_save_text_rejects_alphabet_overflow():
    # A 49x49 grid (n=49) exceeds the 36-symbol text alphabet.
    grid = np.array([np.roll(np.arange(49), i) for i in range(49)])

    with pytest.raises(ValueError):
        save_grids_text(io.StringIO(), [grid])


def test_load_text_rejects_bad_header():
    buf = io.StringIO("NOT-A-SUDOKU-FILE v1 k=2 count=1\n0123\n2301\n1032\n3210\n")

    with pytest.raises(ValueError):
        load_grids_text(buf)


def test_load_binary_rejects_bad_magic():
    buf = io.BytesIO(b"NOPE" + b"\x00" * 10)

    with pytest.raises(ValueError):
        load_grids_binary(buf)


def test_load_text_rejects_truncated_file():
    buf = io.StringIO("SUDOKU-GRIDS-TEXT v1 k=2 count=2\n0123\n2301\n1032\n3210\n")

    with pytest.raises(ValueError):
        load_grids_text(buf)
