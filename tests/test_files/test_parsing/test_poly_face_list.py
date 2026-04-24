from pathlib import Path

import numpy as np
import pytest
from foamlib import FoamFile
from foamlib._files._parsing._parser import (
    ParseError,
    _parse_ascii_float_list_list,
    _parse_ascii_integer_list_list,
)

faces_contents = r"""
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2206                                  |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       faceList;
    location    "constant/polyMesh";
    object      faces;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

3
(
3(0 1 2)
4(3 4 5 6)
5(7 8 9 10 11)
)

// ************************************************************************* //
"""


def test_parse_poly_faces(tmp_path: Path) -> None:
    """Test that ascii faceList with triangles, quads, and pentagons is parsed correctly."""
    path = tmp_path / "faces"
    path.write_text(faces_contents)

    file = FoamFile(path)
    faces = file[None]

    assert len(faces) == 3
    assert np.array_equal(faces[0], [0, 1, 2])
    assert np.array_equal(faces[1], [3, 4, 5, 6])
    assert np.array_equal(faces[2], [7, 8, 9, 10, 11])


float_list_list_contents = r"""
3
(
2(0.1 0.2)
3(0.3 0.4 0.5)
1(0.6)
)
"""


def test_parse_float_list_list(tmp_path: Path) -> None:
    """Test that a standalone ascii numeric list-of-lists with float values is parsed correctly."""
    path = tmp_path / "floats"
    path.write_text(float_list_list_contents)

    file = FoamFile(path)
    data = file[None]

    assert len(data) == 3
    assert np.allclose(data[0], [0.1, 0.2])
    assert np.allclose(data[1], [0.3, 0.4, 0.5])
    assert np.allclose(data[2], [0.6])


commented_faces_contents = r"""
3
(
3(0 1 2) // triangle
4 /* quad */ (3 4 5 6)
5(
  7 // comment inside
  8
  9
  10
  11
)
)
"""


def test_parse_commented_faces(tmp_path: Path) -> None:
    """Test that ascii faceList with inline comments is parsed correctly."""
    path = tmp_path / "faces_commented"
    path.write_text(commented_faces_contents)

    file = FoamFile(path)
    faces = file[None]

    assert len(faces) == 3
    assert np.array_equal(faces[0], [0, 1, 2])
    assert np.array_equal(faces[1], [3, 4, 5, 6])
    assert np.array_equal(faces[2], [7, 8, 9, 10, 11])


# --- Sub-list count validation: no outer count (per-sublist path) ---
# When no outer count is present OpenFOAM omits it for short lists.
# The parser validates each n(...) individually in this path.


def test_no_outer_count_correct() -> None:
    ret, _ = _parse_ascii_integer_list_list(b"(\n3(0 1 2)\n4(3 4 5 6)\n5(7 8 9 10 11)\n)", 0)
    assert [list(s) for s in ret] == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11]]


def test_no_outer_count_zero_length_sublist() -> None:
    ret, _ = _parse_ascii_integer_list_list(b"(0())", 0, empty_ok=True)
    assert len(ret) == 1
    assert len(ret[0]) == 0


def test_no_outer_count_overfull_sublist() -> None:
    """2(1 2 3): count says 2 but 3 values are present."""
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"(2(1 2 3))", 0)


def test_no_outer_count_underfull_sublist() -> None:
    """4(1 2 3): count says 4 but only 3 values are present."""
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"(4(1 2 3))", 0)


def test_no_outer_count_overfull_first_correct_second() -> None:
    """2(1 2 3) 4(10 20 30 40): first sub-list has a wrong count."""
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"(2(1 2 3) 4(10 20 30 40))", 0)


def test_no_outer_count_negative_sublist_count() -> None:
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"(-1(0 1 2))", 0)


def test_no_outer_count_float_list_overfull_sublist() -> None:
    """Same validation applies to float list-of-lists."""
    with pytest.raises(ParseError):
        _parse_ascii_float_list_list(b"(2(0.1 0.2 0.3))", 0)


# --- Sub-list count validation: outer count present (fast-path + guards) ---
# When an outer count is present the parser uses the fast flat-array loop.
# The outer count acts as the primary net check; two in-loop guards cover
# crash scenarios (negative count, data shorter than declared length).


def test_outer_count_correct() -> None:
    ret, _ = _parse_ascii_integer_list_list(b"3\n(\n3(0 1 2)\n4(3 4 5 6)\n5(7 8 9 10 11)\n)", 0)
    assert [list(s) for s in ret] == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11]]


def test_outer_count_mismatch() -> None:
    """Outer count says 3 but only 2 sub-lists are present."""
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"3\n(\n4(0 1 2 3)\n4(4 5 6 7)\n)", 0)


def test_outer_count_underfull_sublist_crash_guard() -> None:
    """4(1 2 3): declared length exceeds available data — crash guard triggers."""
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"1\n(\n4(1 2 3)\n)", 0)


def test_outer_count_negative_sublist_count_crash_guard() -> None:
    with pytest.raises(ParseError):
        _parse_ascii_integer_list_list(b"1\n(\n-1(0 1 2)\n)", 0)
