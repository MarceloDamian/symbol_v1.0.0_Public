"""
Render interpreter results (APLArray) back into canonical APL source.

The output is re-evaluable: running `to_apl(run(p))` yields an array identical
to `run(p)`. That gives the model exactly one target string per value. Works
on APLArray rather than plain lists because lists cannot distinguish a matrix
from a vector of vectors.
"""

from __future__ import annotations

import math
from typing import Any

from symbol.sandbox.apl_interp import APLArray

FLOAT_DIGITS = 6


class FormatError(ValueError):
    """Raised for values that have no canonical APL rendering (inf, nan)."""


def _number(value: Any) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int) and abs(value) > 2 ** 53:
        value = float(value)  # past exact-int range APL shows E notation: 2.65253E32
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormatError("non-finite number %r" % value)
        if value == int(value) and abs(value) < 2 ** 53:
            value = int(value)
        else:
            text = "%.*g" % (FLOAT_DIGITS, value)
            if "e" in text:
                mantissa, exp = text.split("e")
                text = mantissa + "E" + str(int(exp))
            return text.replace("-", "¯")
    if not isinstance(value, int):
        raise FormatError("cannot render cell %r" % (value,))
    return str(value).replace("-", "¯")


def _is_char(cell: Any) -> bool:
    return isinstance(cell, str) and len(cell) == 1


def _quote(chars) -> str:
    return "'" + "".join("''" if c == "'" else c for c in chars) + "'"


def _scalar_cell(cell: Any) -> str:
    """A simple (non-array) cell on its own."""
    return _quote([cell]) if _is_char(cell) else _number(cell)


def _item(cell: Any) -> str:
    """One item of a mixed/nested vector, safe to juxtapose with its neighbours."""
    if isinstance(cell, APLArray):
        return "(" + to_apl(cell) + ")"
    if _is_char(cell):
        return _quote([cell])
    return "(" + _number(cell) + ")"


def _vector(cells) -> str:
    n = len(cells)
    if n == 0:
        return "⍬"
    if all(_is_char(c) for c in cells):
        return ("," if n == 1 else "") + _quote(cells)
    if all(not isinstance(c, APLArray) and not _is_char(c) for c in cells):
        return ("," if n == 1 else "") + " ".join(_number(c) for c in cells)
    if n == 1:
        cell = cells[0]
        inner = to_apl(cell) if isinstance(cell, APLArray) else _scalar_cell(cell)
        return ",⊂" + inner
    return "".join(_item(c) for c in cells)


def to_apl(array: APLArray) -> str:
    """Render an APLArray of any rank as APL source."""
    if array.rank == 0:
        cell = array.data[0]
        if isinstance(cell, APLArray):
            return "⊂" + to_apl(cell)
        return _scalar_cell(cell)
    if array.rank == 1:
        return _vector(array.data)
    shape = " ".join(str(d) for d in array.shape)
    return shape + "⍴" + _vector(array.data)


def same_array(a: Any, b: Any, tol: float = 1e-5) -> bool:
    """Structural equality: same shape, same cells, floats within `tol`."""
    if isinstance(a, APLArray) or isinstance(b, APLArray):
        if not (isinstance(a, APLArray) and isinstance(b, APLArray)):
            return False
        if a.shape != b.shape:
            return False
        return all(same_array(x, y, tol) for x, y in zip(a.data, b.data))
    if _is_char(a) or _is_char(b):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b  # exact, and safe for ints too big for float (a model can predict 10**400)
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
    except (TypeError, ValueError, OverflowError):
        return False
