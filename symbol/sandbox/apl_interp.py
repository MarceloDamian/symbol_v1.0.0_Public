"""
A pure-Python interpreter for a practical subset of APL.

This exists so the compression pipeline can verify semantic equivalence of two
APL expressions without requiring Dyalog/GNU APL to be installed. It covers the
primitives that show up in the overwhelming majority of real compression pairs:
scalar arithmetic with pervasion, reduce/scan, each, outer product, commute,
compose/atop/over, selection, and 2- and 3-train tacit forms.

Evaluation follows APL's right-to-left rule, and functions bind long right /
short left. Index origin is 1 (Dyalog default).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, List, Optional, Sequence, Union

Number = Union[int, float]


class APLError(Exception):
    """Raised for any parse or evaluation failure in the APL subset."""


# Arrays are built eagerly, so the step budget alone cannot stop `1E6⍴0`.
MAX_CELLS = 1_000_000


def _check_size(total: int, glyph: str) -> None:
    if total > MAX_CELLS:
        raise APLError("%s: WS FULL, %d cells exceeds limit %d" % (glyph, total, MAX_CELLS))


# ---------------------------------------------------------------------------
# Array model
# ---------------------------------------------------------------------------


class APLArray:
    """A multidimensional array: a shape plus a flat, row-major cell list.

    Cells are numbers, single-character strings, or nested APLArrays. A scalar
    has shape `()`; a vector has shape `(n,)`.
    """

    __slots__ = ("shape", "data")

    def __init__(self, shape: Sequence[int], data: Sequence[Any]):
        self.shape = tuple(int(d) for d in shape)
        self.data = list(data)
        expected = 1
        for dim in self.shape:
            expected *= dim
        if len(self.data) != expected:
            raise APLError(
                "shape %s needs %d cells, got %d" % (self.shape, expected, len(self.data))
            )

    # -- constructors ------------------------------------------------------

    @staticmethod
    def scalar(value: Any) -> "APLArray":
        return APLArray((), [value])

    @staticmethod
    def vector(values: Sequence[Any]) -> "APLArray":
        return APLArray((len(values),), list(values))

    @staticmethod
    def from_nested(obj: Any) -> "APLArray":
        """Build an array from Python scalars / lists (lists become vectors)."""
        if isinstance(obj, APLArray):
            return obj
        if isinstance(obj, (list, tuple)):
            return APLArray.vector([APLArray.from_nested(x).demote() for x in obj])
        return APLArray.scalar(obj)

    # -- basic queries -----------------------------------------------------

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def count(self) -> int:
        return len(self.data)

    def is_scalar(self) -> bool:
        return self.rank == 0

    def demote(self) -> Any:
        """Return the bare cell if this is a simple scalar, else the array."""
        if self.rank == 0:
            return self.data[0]
        return self

    def scalar_value(self) -> Any:
        if self.count != 1:
            raise APLError("expected a single value, got shape %s" % (self.shape,))
        cell = self.data[0]
        return cell.scalar_value() if isinstance(cell, APLArray) else cell

    def to_nested(self) -> Any:
        """Convert to plain Python for comparison and display."""
        cells = [c.to_nested() if isinstance(c, APLArray) else c for c in self.data]
        if self.rank == 0:
            return cells[0]
        if self.rank == 1:
            return cells
        return _reshape_nested(cells, self.shape)

    # -- structural helpers ------------------------------------------------

    def major_cells(self) -> List["APLArray"]:
        """Split along the leading axis (rows of a matrix, items of a vector)."""
        if self.rank == 0:
            return [self]
        cell_shape = self.shape[1:]
        stride = 1
        for dim in cell_shape:
            stride *= dim
        out = []
        for i in range(self.shape[0]):
            chunk = self.data[i * stride : (i + 1) * stride]
            out.append(APLArray(cell_shape, chunk))
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "APLArray(shape=%s, data=%r)" % (self.shape, self.data)


def _reshape_nested(flat: List[Any], shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return list(flat)
    stride = 1
    for dim in shape[1:]:
        stride *= dim
    return [
        _reshape_nested(flat[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Function objects
# ---------------------------------------------------------------------------


class APLFunction:
    """A callable with optional monadic and dyadic cases."""

    def __init__(
        self,
        name: str,
        monad: Optional[Callable[[APLArray], APLArray]] = None,
        dyad: Optional[Callable[[APLArray, APLArray], APLArray]] = None,
    ):
        self.name = name
        self._monad = monad
        self._dyad = dyad

    def call_monad(self, omega: APLArray) -> APLArray:
        if self._monad is None:
            raise APLError("%s has no monadic form" % self.name)
        return self._monad(omega)

    def call_dyad(self, alpha: APLArray, omega: APLArray) -> APLArray:
        if self._dyad is None:
            raise APLError("%s has no dyadic form" % self.name)
        return self._dyad(alpha, omega)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "APLFunction(%s)" % self.name


class Dfn(APLFunction):
    """A brace-delimited function `{...}` using ⍺ and ⍵ (and ⍵⍵/⍺⍺ operands)."""

    def __init__(self, body: Any, interp: "Interpreter", left_op=None, right_op=None):
        super().__init__("dfn")
        self.body = body
        self.interp = interp
        self.left_op = left_op
        self.right_op = right_op

    def call_monad(self, omega: APLArray) -> APLArray:
        return self.interp.run_dfn(self, omega, None)

    def call_dyad(self, alpha: APLArray, omega: APLArray) -> APLArray:
        return self.interp.run_dfn(self, omega, alpha)


# ---------------------------------------------------------------------------
# Scalar pervasion
# ---------------------------------------------------------------------------


def pervade_monadic(fn: Callable[[Any], Any], name: str):
    def apply(omega: APLArray) -> APLArray:
        def walk(cell: Any) -> Any:
            if isinstance(cell, APLArray):
                return APLArray(cell.shape, [walk(c) for c in cell.data])
            try:
                return fn(cell)
            except (TypeError, ValueError) as exc:
                raise APLError("%s: bad argument %r (%s)" % (name, cell, exc))
            except ZeroDivisionError:
                raise APLError("%s: division by zero" % name)

        return APLArray(omega.shape, [walk(c) for c in omega.data])

    return apply


def pervade_dyadic(fn: Callable[[Any, Any], Any], name: str):
    """Apply `fn` cellwise, extending a single-cell argument over the other."""

    def apply(alpha: APLArray, omega: APLArray) -> APLArray:
        def combine(a: Any, b: Any) -> Any:
            a_arr = isinstance(a, APLArray)
            b_arr = isinstance(b, APLArray)
            if a_arr or b_arr:
                left = a if a_arr else APLArray.scalar(a)
                right = b if b_arr else APLArray.scalar(b)
                return apply(left, right).demote()
            try:
                return fn(a, b)
            except ZeroDivisionError:
                raise APLError("%s: division by zero" % name)
            except (TypeError, ValueError) as exc:
                raise APLError("%s: bad arguments %r %r (%s)" % (name, a, b, exc))

        # Scalar (or one-cell) extension in either direction.
        if alpha.count == 1 and omega.count != 1:
            a_cell = alpha.data[0]
            return APLArray(omega.shape, [combine(a_cell, c) for c in omega.data])
        if omega.count == 1 and alpha.count != 1:
            b_cell = omega.data[0]
            return APLArray(alpha.shape, [combine(c, b_cell) for c in alpha.data])

        if alpha.count == 1 and omega.count == 1:
            shape = omega.shape if alpha.rank == 0 else alpha.shape
            return APLArray(shape, [combine(alpha.data[0], omega.data[0])])

        if alpha.shape != omega.shape:
            raise APLError(
                "%s: length error, shapes %s and %s" % (name, alpha.shape, omega.shape)
            )
        return APLArray(
            alpha.shape, [combine(a, b) for a, b in zip(alpha.data, omega.data)]
        )

    return apply


MAX_EXACT_INT = 2 ** 53  # beyond this APL holds numbers as floats, like 2.65253E32


def _tidy(value: Number) -> Number:
    """Collapse float results that are exactly integral, as APL displays them."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return value
        if abs(value) < MAX_EXACT_INT and abs(value - round(value)) < 1e-12:
            return int(round(value))
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > MAX_EXACT_INT:
        try:
            return float(value)
        except OverflowError:
            raise APLError("domain error: number too large")
    return value


def _num(value: Any) -> Number:
    if isinstance(value, bool):
        return int(value)
    if _is_number(value):
        return value
    raise APLError("expected a number, got %r" % (value,))


def _add(a, b):
    return _tidy(_num(a) + _num(b))


def _sub(a, b):
    return _tidy(_num(a) - _num(b))


def _mul(a, b):
    return _tidy(_num(a) * _num(b))


def _div(a, b):
    a_n, b_n = _num(a), _num(b)
    if b_n == 0:
        if a_n == 0:
            return 1  # APL: 0÷0 is 1
        raise APLError("÷: division by zero")
    return _tidy(a_n / b_n)


def _residue(a, b):
    a_n, b_n = _num(a), _num(b)
    if a_n == 0:
        return _tidy(b_n)
    return _tidy(b_n - a_n * math.floor(b_n / a_n))


def _power(a, b):
    a_n, b_n = _num(a), _num(b)
    # Refuse before computing: 9*9*9 as an exact int has ~370 million digits.
    if abs(a_n) > 1 and b_n > 0 and b_n * math.log10(abs(a_n)) > 300:
        raise APLError("*: overflow")
    try:
        return _tidy(a_n ** b_n)
    except OverflowError:
        raise APLError("*: overflow")


def _log(a, b):
    a_n, b_n = _num(a), _num(b)
    if a_n <= 0 or b_n <= 0:
        raise APLError("⍟: domain error")
    return _tidy(math.log(b_n) / math.log(a_n))


def _bool(value: Any, glyph: str) -> int:
    n = _num(value)
    if n not in (0, 1):
        raise APLError("%s: domain error, expected boolean, got %r" % (glyph, n))
    return int(n)


def _int_arg(value: Any, glyph: str) -> int:
    n = _num(value)
    if n != int(n):
        raise APLError("%s: domain error, non-integer %r" % (glyph, n))
    return int(n)


def _and(a, b):
    """∧ is LCM; on booleans that coincides with logical AND."""
    x, y = _int_arg(a, "∧"), _int_arg(b, "∧")
    if x == 0 or y == 0:
        return 0
    lcm = abs(x * y) // math.gcd(x, y)
    return lcm if (x < 0) == (y < 0) else -lcm


def _or(a, b):
    """∨ is GCD; on booleans that coincides with logical OR."""
    return math.gcd(_int_arg(a, "∨"), _int_arg(b, "∨"))


def _compare(value: Any) -> Any:
    """Normalize a cell for comparison: numbers stay, chars stay."""
    return value


def _equal(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return int(a == b)
    return int(_num(a) == _num(b))


def _not_equal(a, b):
    return 1 - _equal(a, b)


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------


def _iota(omega: APLArray) -> APLArray:
    """⍳ monadic: index generator (origin 1)."""
    if omega.count == 1:
        n = _num(omega.scalar_value())
        if n != int(n) or n < 0:
            raise APLError("⍳: domain error")
        _check_size(int(n), "⍳")
        return APLArray.vector(list(range(1, int(n) + 1)))
    dims = [int(_num(c)) for c in omega.data]
    total = 1
    for d in dims:
        total *= d
    _check_size(total, "⍳")
    coords = []
    for flat in range(total):
        rest, coord = flat, []
        for dim in reversed(dims):
            coord.append(rest % dim + 1)
            rest //= dim
        coords.append(APLArray.vector(list(reversed(coord))))
    return APLArray(tuple(dims), coords)


def _index_of(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍳ dyadic: index of each item of ⍵ in ⍺ (not-found = ≢⍺ + 1)."""
    haystack = [c.to_nested() if isinstance(c, APLArray) else c for c in alpha.data]
    miss = len(haystack) + 1

    def lookup(cell: Any) -> int:
        key = cell.to_nested() if isinstance(cell, APLArray) else cell
        for i, item in enumerate(haystack):
            if item == key:
                return i + 1
        return miss

    return APLArray(omega.shape, [lookup(c) for c in omega.data])


def _shape(omega: APLArray) -> APLArray:
    return APLArray.vector(list(omega.shape))


def _reshape(alpha: APLArray, omega: APLArray) -> APLArray:
    dims = [int(_num(c)) for c in alpha.data]
    if any(d < 0 for d in dims):
        raise APLError("⍴: domain error")
    total = 1
    for d in dims:
        total *= d
    _check_size(total, "⍴")
    if total and not omega.data:
        raise APLError("⍴: cannot reshape an empty array to a non-empty shape")
    cells = [omega.data[i % omega.count] for i in range(total)] if total else []
    return APLArray(tuple(dims), cells)


def _tally(omega: APLArray) -> APLArray:
    return APLArray.scalar(1 if omega.rank == 0 else omega.shape[0])


def _depth(omega: APLArray) -> APLArray:
    def walk(cell: Any) -> int:
        if isinstance(cell, APLArray):
            inner = [walk(c) for c in cell.data]
            return 1 + (max(inner) if inner else 0)
        return 0

    if omega.rank == 0 and not isinstance(omega.data[0], APLArray):
        return APLArray.scalar(0)
    inner = [walk(c) for c in omega.data]
    return APLArray.scalar(1 + (max(inner) if inner else 0))


def _ravel(omega: APLArray) -> APLArray:
    return APLArray.vector(list(omega.data))


def _catenate(alpha: APLArray, omega: APLArray) -> APLArray:
    if alpha.rank <= 1 and omega.rank <= 1:
        return APLArray.vector(list(alpha.data) + list(omega.data))
    left = alpha.major_cells() if alpha.rank > 0 else [alpha]
    right = omega.major_cells() if omega.rank > 0 else [omega]
    cell_shape = left[0].shape if left else right[0].shape
    for cell in left + right:
        if cell.shape != cell_shape:
            raise APLError(", : mismatched cell shapes")
    flat: List[Any] = []
    for cell in left + right:
        flat.extend(cell.data)
    return APLArray((len(left) + len(right),) + cell_shape, flat)


def _reverse(omega: APLArray) -> APLArray:
    if omega.rank == 0:
        return omega
    cells = omega.major_cells()
    cells.reverse()
    flat: List[Any] = []
    for cell in cells:
        flat.extend(cell.data)
    return APLArray(omega.shape, flat)


def _rotate(alpha: APLArray, omega: APLArray) -> APLArray:
    n = int(_num(alpha.scalar_value()))
    if omega.rank == 0:
        return omega
    cells = omega.major_cells()
    if not cells:
        return omega
    k = n % len(cells)
    cells = cells[k:] + cells[:k]
    flat: List[Any] = []
    for cell in cells:
        flat.extend(cell.data)
    return APLArray(omega.shape, flat)


def _enclose(omega: APLArray) -> APLArray:
    if omega.rank == 0 and not isinstance(omega.data[0], APLArray):
        return omega  # enclosing a simple scalar is a no-op
    return APLArray.scalar(omega)


def _disclose(omega: APLArray) -> APLArray:
    if omega.count == 0:
        return APLArray.scalar(0)
    first = omega.data[0]
    return first if isinstance(first, APLArray) else APLArray.scalar(first)


def _take(alpha: APLArray, omega: APLArray) -> APLArray:
    n = int(_num(alpha.scalar_value()))
    cells = omega.major_cells() if omega.rank > 0 else [omega]
    _check_size(abs(n) * max(1, _prod(cells[0].shape) if cells else 1), "↑")
    fill: Any = 0
    if cells and cells[0].rank == 0 and isinstance(cells[0].data[0], str):
        fill = " "
    if n >= 0:
        picked = cells[:n]
        pad = n - len(picked)
        if pad > 0:
            blank = cells[0].shape if cells else ()
            picked = picked + [APLArray(blank, [fill] * max(1, _prod(blank)))] * pad
    else:
        picked = cells[n:] if -n <= len(cells) else cells
        pad = -n - len(picked)
        if pad > 0:
            blank = cells[0].shape if cells else ()
            picked = [APLArray(blank, [fill] * max(1, _prod(blank)))] * pad + picked
    flat: List[Any] = []
    for cell in picked:
        flat.extend(cell.data)
    cell_shape = cells[0].shape if cells else ()
    return APLArray((len(picked),) + cell_shape, flat)


def _drop(alpha: APLArray, omega: APLArray) -> APLArray:
    n = int(_num(alpha.scalar_value()))
    cells = omega.major_cells() if omega.rank > 0 else [omega]
    kept = cells[n:] if n >= 0 else cells[: len(cells) + n] if -n <= len(cells) else []
    flat: List[Any] = []
    for cell in kept:
        flat.extend(cell.data)
    cell_shape = cells[0].shape if cells else ()
    return APLArray((len(kept),) + cell_shape, flat)


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _first_axis_pairs(omega: APLArray) -> List[APLArray]:
    return omega.major_cells() if omega.rank > 0 else [omega]


def _compress(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺/⍵ — replicate/filter ⍵ by the counts in ⍺."""
    cells = _first_axis_pairs(omega)
    counts = [int(_num(c)) for c in alpha.data]
    if len(counts) == 1:
        counts = counts * len(cells)
    if len(counts) != len(cells):
        raise APLError("/: length error between %d and %d" % (len(counts), len(cells)))
    if any(count < 0 for count in counts):
        raise APLError("/: negative replicate count")
    cell_size = len(cells[0].data) if cells else 1
    _check_size(sum(counts) * max(1, cell_size), "/")  # before building: {⍵/⍵}5*15 is 3×10¹⁰ cells
    picked: List[APLArray] = []
    for count, cell in zip(counts, cells):
        picked.extend([cell] * count)
    flat: List[Any] = []
    for cell in picked:
        flat.extend(cell.data)
    cell_shape = cells[0].shape if cells else ()
    return APLArray((len(picked),) + cell_shape, flat)


def _grade_up(omega: APLArray) -> APLArray:
    if omega.rank == 0:
        raise APLError("grade: rank error, scalar argument")
    keys = [_sort_key(c) for c in _first_axis_pairs(omega)]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return APLArray.vector([i + 1 for i in order])


def _grade_down(omega: APLArray) -> APLArray:
    if omega.rank == 0:
        raise APLError("grade: rank error, scalar argument")
    keys = [_sort_key(c) for c in _first_axis_pairs(omega)]
    order = sorted(range(len(keys)), key=lambda i: keys[i], reverse=True)
    return APLArray.vector([i + 1 for i in order])


def _sort_key(cell: APLArray) -> Any:
    nested = cell.to_nested()
    if isinstance(nested, list):
        return [_flatten_key(x) for x in nested]
    return [_flatten_key(nested)]


def _flatten_key(value: Any) -> Any:
    if isinstance(value, str):
        return (1, value)
    if isinstance(value, list):
        return (2, [_flatten_key(v) for v in value])
    return (0, value)


def _membership(alpha: APLArray, omega: APLArray) -> APLArray:
    pool = [c.to_nested() if isinstance(c, APLArray) else c for c in omega.data]
    def has(cell: Any) -> int:
        key = cell.to_nested() if isinstance(cell, APLArray) else cell
        return int(key in pool)
    return APLArray(alpha.shape, [has(c) for c in alpha.data])


def _unique(omega: APLArray) -> APLArray:
    seen: List[Any] = []
    out: List[Any] = []
    for cell in omega.data:
        key = cell.to_nested() if isinstance(cell, APLArray) else cell
        if key not in seen:
            seen.append(key)
            out.append(cell)
    return APLArray.vector(out)


def _without(alpha: APLArray, omega: APLArray) -> APLArray:
    pool = [c.to_nested() if isinstance(c, APLArray) else c for c in omega.data]
    kept = [
        c for c in alpha.data
        if (c.to_nested() if isinstance(c, APLArray) else c) not in pool
    ]
    return APLArray.vector(kept)


def _flat_vector(array: APLArray, glyph: str) -> List[Any]:
    """Cells of a scalar or vector; higher ranks are not supported by `glyph`."""
    if array.rank > 1:
        raise APLError("%s: rank error, expected a vector" % glyph)
    return array.data


def _nub_sieve(omega: APLArray) -> APLArray:
    """≠ monadic: 1 where an item appears for the first time."""
    seen: List[Any] = []
    marks = []
    for cell in _flat_vector(omega, "≠"):
        key = cell.to_nested() if isinstance(cell, APLArray) else cell
        marks.append(int(key not in seen))
        seen.append(key)
    return APLArray(omega.shape, marks)


def _where(omega: APLArray) -> APLArray:
    """⍸ monadic: each index repeated by its count, so ⍸1 0 1 is 1 3."""
    counts = [_int_arg(c, "⍸") for c in _flat_vector(omega, "⍸")]
    if any(n < 0 for n in counts):
        raise APLError("⍸: domain error")
    _check_size(sum(counts), "⍸")  # before building: ⍸1E9 must not allocate a billion cells
    out: List[Any] = []
    for i, n in enumerate(counts, 1):
        out.extend([i] * n)
    return APLArray.vector(out)


def _intersection(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺∩⍵: items of ⍺ that are also in ⍵, in ⍺'s order."""
    pool = [c.to_nested() if isinstance(c, APLArray) else c for c in omega.data]
    kept = [
        c for c in _flat_vector(alpha, "∩")
        if (c.to_nested() if isinstance(c, APLArray) else c) in pool
    ]
    return APLArray.vector(kept)


def _decode(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺⊥⍵: digits ⍵ read in base ⍺ (a scalar base, or one radix per digit).
    A matrix ⍵ decodes each column, so 2⊥⍉C reads every row of a bit matrix C."""
    if omega.rank == 2:
        r, c = omega.shape
        columns = [APLArray((r,), omega.data[j::c]) for j in range(c)]
        return APLArray((c,), [_decode(alpha, col).data[0] for col in columns])
    digits = [_num(c) for c in _flat_vector(omega, "⊥")]
    radix = [_num(c) for c in _flat_vector(alpha, "⊥")]
    if len(radix) == 1:
        radix = radix * len(digits)
    elif len(digits) == 1:
        digits = digits * len(radix)
    if len(radix) != len(digits):
        raise APLError("⊥: length error")
    total = 0
    for base, digit in zip(radix, digits):
        total = total * base + digit
    return APLArray.scalar(_tidy(total))


def _encode(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺⊤⍵: ⍵ written in the mixed radix ⍺, so 2 2 2⊤5 is 1 0 1.
    A vector ⍵ gives one column per item: (4⍴2)⊤0 1 2 is a 4×3 bit matrix."""
    if omega.rank > 1:
        raise APLError("⊤: rank error, expected a scalar or vector right argument")
    if omega.rank == 1:
        rows = len(alpha.data)
        _check_size(rows * len(omega.data), "⊤")  # before building: (⍳1000)⊤⍳1000000 is 10⁹ cells
        columns = [_encode(alpha, APLArray.scalar(c)).data for c in omega.data]
        return APLArray((rows, len(columns)), [col[r] for r in range(rows) for col in columns])
    value = _num(omega.data[0])
    out = []
    for base in reversed([_num(c) for c in _flat_vector(alpha, "⊤")]):
        if base == 0:
            out.append(value)
            value = 0
        else:
            digit = _tidy(value % base)
            out.append(digit)
            value = _tidy((value - digit) / base)
    return APLArray.vector(list(reversed(out)))


def _find(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺⍷⍵: 1 where the vector ⍺ starts as a run inside the vector ⍵."""
    needle = [c.to_nested() if isinstance(c, APLArray) else c for c in _flat_vector(alpha, "⍷")]
    hay = [c.to_nested() if isinstance(c, APLArray) else c for c in _flat_vector(omega, "⍷")]
    n = len(needle)
    marks = [int(hay[i : i + n] == needle) for i in range(len(hay))]
    return APLArray(omega.shape, marks)


def _binomial(a, b):
    """⍺!⍵: ways to choose ⍺ items from ⍵."""
    a_n, b_n = _num(a), _num(b)
    if isinstance(a_n, int) and isinstance(b_n, int) and 0 <= a_n and 0 <= b_n <= 10_000:
        return _tidy(math.comb(b_n, a_n))  # bounded: math.comb on huge ⍵ cannot be interrupted
    try:
        return _tidy(math.gamma(b_n + 1) / (math.gamma(a_n + 1) * math.gamma(b_n - a_n + 1)))
    except (ValueError, ZeroDivisionError, OverflowError):
        raise APLError("!: domain error")


def _index(alpha: APLArray, omega: APLArray) -> APLArray:
    """⍺⌷⍵ — select major cells of ⍵ at the indices in ⍺ (origin 1)."""
    cells = _first_axis_pairs(omega)
    picked: List[APLArray] = []
    for cell in alpha.data:
        if isinstance(cell, APLArray):
            raise APLError("⌷: nested index not supported")
        i = int(_num(cell))
        if i < 1 or i > len(cells):
            raise APLError("⌷: index %d out of range" % i)
        picked.append(cells[i - 1])
    flat: List[Any] = []
    for cell in picked:
        flat.extend(cell.data)
    cell_shape = cells[0].shape if cells else ()
    if alpha.rank == 0:
        return picked[0] if picked else APLArray.scalar(0)
    return APLArray((len(picked),) + cell_shape, flat)


def _transpose(omega: APLArray) -> APLArray:
    if omega.rank < 2:
        return omega
    if omega.rank > 2:
        raise APLError("⍉: only rank <= 2 supported")
    rows, cols = omega.shape
    flat = [omega.data[r * cols + c] for c in range(cols) for r in range(rows)]
    return APLArray((cols, rows), flat)


def _match(alpha: APLArray, omega: APLArray) -> APLArray:
    return APLArray.scalar(int(alpha.to_nested() == omega.to_nested() and alpha.shape == omega.shape))


def _not_match(alpha: APLArray, omega: APLArray) -> APLArray:
    return APLArray.scalar(1 - _match(alpha, omega).scalar_value())


# ---------------------------------------------------------------------------
# Primitive tables
# ---------------------------------------------------------------------------

MONADIC_SCALARS = {
    "+": lambda x: _tidy(_num(x)),
    "-": lambda x: _tidy(-_num(x)),
    "×": lambda x: (0 if _num(x) == 0 else (1 if _num(x) > 0 else -1)),
    "÷": lambda x: _div(1, x),
    "|": lambda x: _tidy(abs(_num(x))),
    "⌈": lambda x: int(math.ceil(_num(x))),
    "⌊": lambda x: int(math.floor(_num(x))),
    "*": lambda x: _tidy(math.exp(_num(x))),
    "⍟": lambda x: _tidy(math.log(_num(x))),
    "~": lambda x: 1 - _bool(x, "~"),
    "!": lambda x: _tidy(math.gamma(_num(x) + 1)),
    "○": lambda x: _tidy(math.pi * _num(x)),
}

DYADIC_SCALARS = {
    "+": _add,
    "-": _sub,
    "×": _mul,
    "÷": _div,
    "|": _residue,
    "*": _power,
    "⍟": _log,
    "⌈": lambda a, b: _tidy(max(_num(a), _num(b))),
    "⌊": lambda a, b: _tidy(min(_num(a), _num(b))),
    "=": _equal,
    "≠": _not_equal,
    "<": lambda a, b: int(_num(a) < _num(b)),
    "≤": lambda a, b: int(_num(a) <= _num(b)),
    ">": lambda a, b: int(_num(a) > _num(b)),
    "≥": lambda a, b: int(_num(a) >= _num(b)),
    "∧": _and,
    "∨": _or,
    "!": _binomial,
    "⍲": lambda a, b: 1 - (_bool(a, "⍲") & _bool(b, "⍲")),
    "⍱": lambda a, b: 1 - (_bool(a, "⍱") | _bool(b, "⍱")),
}

# Primitives whose structural (non-pervasive) behaviour we define directly.
STRUCTURAL = {
    "⍳": (_iota, _index_of),
    "⍴": (_shape, _reshape),
    "≢": (_tally, _not_match),
    "≡": (_depth, _match),
    ",": (_ravel, _catenate),
    "⌽": (_reverse, _rotate),
    "⊂": (_enclose, None),
    "⊃": (_disclose, None),
    "↑": (_disclose, _take),
    "↓": (None, _drop),
    "⍋": (_grade_up, None),
    "⍒": (_grade_down, None),
    "∊": (_ravel, _membership),
    "∪": (_unique, None),
    "~": (None, _without),
    "⌷": (None, _index),
    "⍉": (_transpose, None),
    "≠": (_nub_sieve, None),
    "⍸": (_where, None),
    "∩": (None, _intersection),
    "⊥": (None, _decode),
    "⊤": (None, _encode),
    "⍷": (None, _find),
    "⊢": (lambda w: w, lambda a, w: w),
    "⊣": (lambda w: w, lambda a, w: a),
}

MONADIC_OPERATORS = {"/", "\\", "¨", "⌿", "⍀", "⍨"}
DYADIC_OPERATORS = {"∘", ".", "⍤", "⍥", "⍢", "⍣"}

IDENTITY_ELEMENTS = {
    "+": 0, "-": 0, "×": 1, "÷": 1, "⌈": float("-inf"), "⌊": float("inf"),
    "∨": 0, "∧": 1, "≠": 0, "=": 1,
}


def build_primitive(glyph: str) -> APLFunction:
    """Assemble a primitive function, combining scalar and structural cases."""
    monad = None
    dyad = None
    if glyph in STRUCTURAL:
        s_monad, s_dyad = STRUCTURAL[glyph]
        monad, dyad = s_monad, s_dyad
    if monad is None and glyph in MONADIC_SCALARS:
        monad = pervade_monadic(MONADIC_SCALARS[glyph], glyph)
    if dyad is None and glyph in DYADIC_SCALARS:
        dyad = pervade_dyadic(DYADIC_SCALARS[glyph], glyph)
    if monad is None and dyad is None:
        raise APLError("unsupported primitive: %s" % glyph)
    return APLFunction(glyph, monad, dyad)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

ALL_GLYPHS = (
    set(MONADIC_SCALARS) | set(DYADIC_SCALARS) | set(STRUCTURAL)
    | MONADIC_OPERATORS | DYADIC_OPERATORS
)

NAME_START = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
NAME_CHARS = NAME_START | set("0123456789")
DIGITS = set("0123456789")


class Token:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: Any):
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Token(%s, %r)" % (self.kind, self.value)


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]

        if ch in " \t\n":
            i += 1
            continue

        if ch == "⍝":  # comment to end of line
            while i < n and source[i] != "\n":
                i += 1
            continue

        if ch == "'":  # character literal / string
            i += 1
            chars: List[str] = []
            while i < n:
                if source[i] == "'":
                    if i + 1 < n and source[i + 1] == "'":
                        chars.append("'")
                        i += 2
                        continue
                    break
                chars.append(source[i])
                i += 1
            if i >= n:
                raise APLError("unterminated string literal")
            i += 1
            tokens.append(Token("string", chars))
            continue

        # Numeric literal, possibly a vector of them. ¯ is APL's negative sign.
        if ch in DIGITS or ch == "¯" or (
            ch == "." and i + 1 < n and source[i + 1] in DIGITS
        ):
            numbers: List[Number] = []
            while i < n:
                start = i
                if source[i] == "¯":
                    i += 1
                while i < n and source[i] in DIGITS:
                    i += 1
                if i < n and source[i] == ".":
                    i += 1
                    while i < n and source[i] in DIGITS:
                        i += 1
                if i < n and source[i] in "eE":
                    probe = i + 1
                    if probe < n and source[probe] in "¯-":
                        probe += 1
                    if probe < n and source[probe] in DIGITS:
                        i = probe
                        while i < n and source[i] in DIGITS:
                            i += 1
                raw = source[start:i].replace("¯", "-")
                if raw in ("", "-"):
                    raise APLError("malformed numeric literal near %r" % source[start:start + 6])
                numbers.append(_tidy(float(raw)) if ("." in raw or "e" in raw or "E" in raw) else int(raw))

                # Keep consuming if a space separates another number (vector literal).
                probe = i
                while probe < n and source[probe] == " ":
                    probe += 1
                if probe < n and (
                    source[probe] in DIGITS or source[probe] == "¯"
                    or (source[probe] == "." and probe + 1 < n and source[probe + 1] in DIGITS)
                ):
                    i = probe
                    continue
                break
            tokens.append(Token("number", numbers))
            continue

        if ch in NAME_START:
            start = i
            while i < n and source[i] in NAME_CHARS:
                i += 1
            tokens.append(Token("name", source[start:i]))
            continue

        if ch in "⍺⍵":
            if source[i : i + 2] in ("⍺⍺", "⍵⍵"):
                tokens.append(Token("operand", source[i : i + 2]))
                i += 2
            else:
                tokens.append(Token("arg", ch))
                i += 1
            continue

        if ch in "(){}[]":
            tokens.append(Token(ch, ch))
            i += 1
            continue

        if ch == "←":
            tokens.append(Token("assign", ch))
            i += 1
            continue

        if ch == "⋄" or ch == ";":
            tokens.append(Token("sep", ch))
            i += 1
            continue

        if ch == ":":
            tokens.append(Token("guard", ch))
            i += 1
            continue

        if ch == "∇":
            tokens.append(Token("self", ch))
            i += 1
            continue

        if ch == "⍬":
            tokens.append(Token("number", []))
            i += 1
            continue

        if ch in ALL_GLYPHS:
            if ch in MONADIC_OPERATORS:
                tokens.append(Token("mop", ch))
            elif ch in DYADIC_OPERATORS:
                tokens.append(Token("dop", ch))
            else:
                tokens.append(Token("prim", ch))
            i += 1
            continue

        raise APLError("unexpected character %r" % ch)

    return tokens


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Train:
    """A tacit composition of 2 or 3 tines, e.g. `(+/÷≢)` or `(⌽∘⍳)`."""

    def __init__(self, tines: List[Any]):
        self.tines = tines


class Interpreter:
    """Evaluates APL source against an environment of named arrays."""

    def __init__(self, env: Optional[dict] = None, step_budget: int = 200_000):
        self.env = dict(env or {})
        self.step_budget = step_budget
        self._steps = 0
        self._primitives: dict = {}

    # -- public API --------------------------------------------------------

    def run(self, source: str) -> Optional[APLArray]:
        """Evaluate possibly multiple ⋄-separated statements; return the last value."""
        result: Optional[APLArray] = None
        for statement in _split_statements(tokenize(source)):
            if statement:
                result = self._eval_statement(statement)
        return result

    def primitive(self, glyph: str) -> APLFunction:
        if glyph not in self._primitives:
            self._primitives[glyph] = build_primitive(glyph)
        return self._primitives[glyph]

    # -- stepping / guards -------------------------------------------------

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > self.step_budget:
            raise APLError("evaluation exceeded step budget")

    # -- statements --------------------------------------------------------

    def _eval_statement(self, tokens: List[Token]) -> APLArray:
        # Name assignment: `name ← expr`
        if len(tokens) >= 2 and tokens[0].kind == "name" and tokens[1].kind == "assign":
            value = self._eval_tokens(tokens[2:])
            self.env[tokens[0].value] = value
            return value
        return self._eval_tokens(tokens)

    def _eval_tokens(self, tokens: List[Token]) -> APLArray:
        value = self._eval_expr(self._parse_items(tokens))
        if isinstance(value, (APLFunction, Train)):
            raise APLError("expression evaluated to a function, not an array")
        return value

    # -- parsing into items -----------------------------------------------

    def _parse_items(self, tokens: List[Token]) -> List[Any]:
        """Turn a flat token list into a list of arrays / functions / operators."""
        items: List[Any] = []
        i, n = 0, len(tokens)
        while i < n:
            tok = tokens[i]

            if tok.kind == "number":
                items.append(_number_literal(tok.value))
                i += 1
            elif tok.kind == "string":
                chars = tok.value
                items.append(
                    APLArray.scalar(chars[0]) if len(chars) == 1 else APLArray.vector(list(chars))
                )
                i += 1
            elif tok.kind == "name":
                items.append(("name", tok.value))
                i += 1
            elif tok.kind == "arg":
                items.append(("arg", tok.value))
                i += 1
            elif tok.kind == "operand":
                items.append(("operand", tok.value))
                i += 1
            elif tok.kind == "prim":
                items.append(self.primitive(tok.value))
                i += 1
            elif tok.kind in ("mop", "dop"):
                items.append(("op", tok.kind, tok.value))
                i += 1
            elif tok.kind == "self":
                if "∇" not in self.env:
                    raise APLError("∇ outside a dfn")
                items.append(self.env["∇"])
                i += 1
            elif tok.kind == "assign":
                items.append(("assign", None))
                i += 1
            elif tok.kind == "guard":
                items.append(("guard", None))
                i += 1
            elif tok.kind == "(":
                body, i = _match_group(tokens, i, "(", ")")
                items.append(self._parse_paren(body))
            elif tok.kind == "{":
                body, i = _match_group(tokens, i, "{", "}")
                items.append(Dfn(body, self))
            elif tok.kind == "[":
                # Bracket indexing binds to the array item just left: v[2], m[1;].
                body, i = _match_group(tokens, i, "[", "]")
                if not items or _is_function(items[-1]) or (
                    isinstance(items[-1], tuple) and items[-1][0] in ("op", "assign", "guard")
                ):
                    raise APLError("[ ] needs an array on its left")
                axes = [self._parse_items(part) if part else None for part in _split_statements(body)]
                items.append(("index", items.pop(), axes))
            else:
                raise APLError("unexpected token %r" % (tok,))

        return self._bind_operators(items)

    def _parse_paren(self, body: List[Token]) -> Any:
        """A parenthesized group is either an array expression or a tacit train."""
        inner = self._parse_items(body)
        if not inner:
            raise APLError("empty parentheses")
        if all(_is_function(x) for x in inner) and len(inner) > 1:
            return Train(list(inner))
        if len(inner) == 1:
            return _plain(inner[0])  # `(1 2)` is one strand item
        return ("group", inner)

    def _bind_operators(self, items: List[Any]) -> List[Any]:
        """Apply operators to their operands, left to right (APL binds ops leftward)."""
        out: List[Any] = []
        i, n = 0, len(items)
        while i < n:
            item = items[i]
            if isinstance(item, tuple) and item and item[0] == "op":
                _, kind, glyph = item
                # ∘.f is outer product: ∘ is a placeholder operand, not compose.
                if glyph == "∘" and i + 2 < n and items[i + 1] == ("op", "dop", "."):
                    out.append(self._make_inner_or_outer(_jot_marker(), items[i + 2]))
                    i += 3
                    continue
                # X/⍨Y and X⌿⍨Y: here / is the replicate function, so this is Y/X.
                if glyph in ("/", "⌿") and i + 1 < n and items[i + 1] == ("op", "mop", "⍨"):
                    out.append(self._make_commute(APLFunction("replicate", None, _compress)))
                    i += 2
                    continue
                if not out:
                    raise APLError("operator %s has no left operand" % glyph)
                left = out.pop()
                if kind == "mop":
                    out.append(self._apply_monadic_operator(glyph, left))
                    i += 1
                else:
                    if i + 1 >= n:
                        raise APLError("operator %s has no right operand" % glyph)
                    right = items[i + 1]
                    out.append(self._apply_dyadic_operator(glyph, left, right))
                    i += 2
                continue
            out.append(item)
            i += 1
        return out

    # -- operators ---------------------------------------------------------

    def _apply_monadic_operator(self, glyph: str, operand: Any) -> Any:
        if glyph in ("/", "⌿"):
            if _is_function(operand):
                return self._make_reduce(operand, first_axis=(glyph == "⌿"))
            return APLFunction(
                "compress",
                lambda w, o=operand: _compress(self._as_array(o), w),
                None,
            )
        if glyph in ("\\", "⍀"):
            if not _is_function(operand):
                raise APLError("\\ expects a function operand here")
            return self._make_scan(operand, first_axis=(glyph == "⍀"))
        if glyph == "¨":
            return self._make_each(operand)
        if glyph == "⍨":
            return self._make_commute(operand)
        raise APLError("unsupported monadic operator %s" % glyph)

    def _apply_dyadic_operator(self, glyph: str, left: Any, right: Any) -> Any:
        if glyph == "∘":
            return self._make_compose(left, right)
        if glyph == "⍤":
            return self._make_atop(left, right)
        if glyph == "⍥":
            return self._make_over(left, right)
        if glyph == ".":
            return self._make_inner_or_outer(left, right)
        if glyph == "⍣":
            return self._make_power(left, right)
        raise APLError("unsupported dyadic operator %s" % glyph)

    def _make_power(self, left: Any, right: Any) -> APLFunction:
        """`f⍣n` applies f n times; `f⍣g` repeats until `new g old` is true (e.g. ⍣=)."""
        fn = self._resolve(left)
        until = self._resolve(right)
        if not _is_function(fn):
            raise APLError("⍣ needs a function on its left")

        def run(omega: APLArray, step) -> APLArray:
            if _is_function(until):
                while True:
                    self._tick()
                    new = step(omega)
                    if _truthy(self._call_dyad(until, new, omega)):
                        return new
                    omega = new
            count = _int_arg(self._as_array(until).scalar_value(), "⍣")
            if count < 0:
                raise APLError("⍣: negative count needs an inverse")
            for _ in range(count):
                omega = step(omega)
            return omega

        return APLFunction(
            "power",
            lambda w: run(w, lambda x: self._call_monad(fn, x)),
            lambda a, w: run(w, lambda x: self._call_dyad(fn, a, x)),
        )

    def _fold_right(self, fn: Any, cells: List[APLArray]) -> APLArray:
        """a f (b f (c f d)) — APL reduction order."""
        if not cells:
            glyph = getattr(fn, "name", None)
            if glyph in IDENTITY_ELEMENTS:
                return APLArray.scalar(IDENTITY_ELEMENTS[glyph])
            raise APLError("reduce over empty array with no identity")
        acc = cells[-1]
        for cell in reversed(cells[:-1]):
            self._tick()
            acc = self._call_dyad(fn, cell, acc)
        return acc

    def _make_reduce(self, fn: Any, first_axis: bool = False) -> APLFunction:
        def monad(omega: APLArray) -> APLArray:
            self._tick()
            if omega.rank == 0:
                return omega
            if first_axis or omega.rank == 1:
                return self._fold_right(fn, omega.major_cells())
            # f/ on a matrix reduces each row (along the last axis).
            out = [self._fold_right(fn, row.major_cells()).demote() for row in _last_axis_rows(omega)]
            return APLArray(omega.shape[:-1], out)

        def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
            """n f/ ⍵ — windowed reduction."""
            self._tick()
            width = int(_num(alpha.scalar_value()))
            cells = omega.major_cells() if omega.rank > 0 else [omega]
            if width < 0:
                raise APLError("f/: negative window")
            results = []
            for start in range(0, len(cells) - width + 1):
                window = cells[start : start + width]
                if not window:
                    raise APLError("f/: empty window")
                acc = window[-1]
                for cell in reversed(window[:-1]):
                    self._tick()
                    acc = self._call_dyad(fn, cell, acc)
                results.append(acc.demote())
            return APLArray.vector(results)

        return APLFunction("reduce", monad, dyad)

    def _make_scan(self, fn: Any, first_axis: bool = False) -> APLFunction:
        def prefixes(cells: List[APLArray]) -> List[Any]:
            # Each item is the full right-to-left reduction of its prefix, so
            # -\1 2 3 is 1 (1-2) (1-2-3) = 1 ¯1 2, not a running left fold.
            return [self._fold_right(fn, cells[: i + 1]).demote() for i in range(len(cells))]

        def monad(omega: APLArray) -> APLArray:
            self._tick()
            if omega.rank == 0:
                return omega
            if first_axis or omega.rank == 1:
                cells = omega.major_cells()
                out = prefixes(cells)
                if omega.rank == 1:
                    return APLArray(omega.shape, out)
                return APLArray(omega.shape, [c for cell in out for c in cell.data])
            rows = [prefixes(row.major_cells()) for row in _last_axis_rows(omega)]
            return APLArray(omega.shape, [c for row in rows for c in row])

        return APLFunction("scan", monad, None)

    def _make_each(self, fn: Any) -> APLFunction:
        def monad(omega: APLArray) -> APLArray:
            self._tick()
            out = []
            for cell in omega.data:
                self._tick()
                arg = cell if isinstance(cell, APLArray) else APLArray.scalar(cell)
                out.append(self._call_monad(fn, arg).demote())
            return APLArray(omega.shape, out)

        def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
            self._tick()
            if alpha.count == 1:
                pairs = [(alpha.data[0], c) for c in omega.data]
                shape = omega.shape
            elif omega.count == 1:
                pairs = [(c, omega.data[0]) for c in alpha.data]
                shape = alpha.shape
            else:
                if alpha.shape != omega.shape:
                    raise APLError("¨: length error")
                pairs = list(zip(alpha.data, omega.data))
                shape = alpha.shape
            out = []
            for a, b in pairs:
                self._tick()
                left = a if isinstance(a, APLArray) else APLArray.scalar(a)
                right = b if isinstance(b, APLArray) else APLArray.scalar(b)
                out.append(self._call_dyad(fn, left, right).demote())
            return APLArray(shape, out)

        return APLFunction("each", monad, dyad)

    def _make_commute(self, fn: Any) -> APLFunction:
        def monad(omega: APLArray) -> APLArray:
            self._tick()
            return self._call_dyad(fn, omega, omega)

        def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
            self._tick()
            return self._call_dyad(fn, omega, alpha)

        return APLFunction("commute", monad, dyad)

    def _make_compose(self, left: Any, right: Any) -> APLFunction:
        left_fn, right_fn = _is_function(left), _is_function(right)

        if left_fn and right_fn:
            def monad(omega: APLArray) -> APLArray:
                self._tick()
                return self._call_monad(left, self._call_monad(right, omega))

            def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
                self._tick()
                return self._call_dyad(left, alpha, self._call_monad(right, omega))

            return APLFunction("compose", monad, dyad)

        if left_fn and not right_fn:
            bound = self._as_array(right)

            def monad_r(omega: APLArray) -> APLArray:
                self._tick()
                return self._call_dyad(left, omega, bound)

            return APLFunction("bind-right", monad_r, None)

        if right_fn and not left_fn:
            bound = self._as_array(left)

            def monad_l(omega: APLArray) -> APLArray:
                self._tick()
                return self._call_dyad(right, bound, omega)

            return APLFunction("bind-left", monad_l, None)

        raise APLError("∘: needs at least one function operand")

    def _make_atop(self, left: Any, right: Any) -> APLFunction:
        def monad(omega: APLArray) -> APLArray:
            self._tick()
            return self._call_monad(left, self._call_monad(right, omega))

        def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
            self._tick()
            return self._call_monad(left, self._call_dyad(right, alpha, omega))

        return APLFunction("atop", monad, dyad)

    def _make_over(self, left: Any, right: Any) -> APLFunction:
        def monad(omega: APLArray) -> APLArray:
            self._tick()
            return self._call_monad(left, self._call_monad(right, omega))

        def dyad(alpha: APLArray, omega: APLArray) -> APLArray:
            self._tick()
            return self._call_dyad(
                left, self._call_monad(right, alpha), self._call_monad(right, omega)
            )

        return APLFunction("over", monad, dyad)

    def _make_inner_or_outer(self, left: Any, right: Any) -> APLFunction:
        # Outer product: ∘.f  (left operand is the jot primitive-less marker)
        if isinstance(left, APLFunction) and left.name == "jot-marker":
            def outer(alpha: APLArray, omega: APLArray) -> APLArray:
                self._tick()
                out = []
                for a in alpha.data:
                    for b in omega.data:
                        self._tick()
                        la = a if isinstance(a, APLArray) else APLArray.scalar(a)
                        rb = b if isinstance(b, APLArray) else APLArray.scalar(b)
                        out.append(self._call_dyad(right, la, rb).demote())
                return APLArray(alpha.shape + omega.shape, out)

            return APLFunction("outer", None, outer)

        # Inner product: f.g
        def inner(alpha: APLArray, omega: APLArray) -> APLArray:
            self._tick()
            left_rows = alpha.major_cells() if alpha.rank > 1 else [alpha]
            right_cols = _columns(omega)
            out = []
            for row in left_rows:
                for col in right_cols:
                    self._tick()
                    products = self._call_dyad(right, row, col)
                    out.append(self._call_monad(self._make_reduce(left), products).demote())
            shape = (
                (len(left_rows), len(right_cols))
                if alpha.rank > 1 and omega.rank > 1
                else (len(out),) if len(out) != 1 else ()
            )
            return APLArray(shape, out)

        return APLFunction("inner", None, inner)

    # -- calling -----------------------------------------------------------

    def _call_monad(self, fn: Any, omega: APLArray) -> APLArray:
        self._tick()
        if isinstance(fn, Train):
            return self._call_train_monad(fn, omega)
        if isinstance(fn, APLFunction):
            return fn.call_monad(omega)
        raise APLError("not a function: %r" % (fn,))

    def _call_dyad(self, fn: Any, alpha: APLArray, omega: APLArray) -> APLArray:
        self._tick()
        if isinstance(fn, Train):
            return self._call_train_dyad(fn, alpha, omega)
        if isinstance(fn, APLFunction):
            return fn.call_dyad(alpha, omega)
        raise APLError("not a function: %r" % (fn,))

    def _call_train_monad(self, train: Train, omega: APLArray) -> APLArray:
        tines = train.tines
        if len(tines) == 2:
            f, g = tines
            if _is_function(f) and _is_function(g):
                return self._call_monad(f, self._call_monad(g, omega))
        if len(tines) == 3:
            f, g, h = tines
            if not _is_function(f):  # (A g h) — bound left argument
                return self._call_dyad(g, self._as_array(f), self._call_monad(h, omega))
            return self._call_dyad(
                g, self._call_monad(f, omega), self._call_monad(h, omega)
            )
        if len(tines) == 1:
            return self._call_monad(tines[0], omega)
        # Longer trains resolve right-to-left in 3-tine groups.
        return self._call_train_monad(Train(tines[:-2] + [Train(tines[-2:])]), omega)

    def _call_train_dyad(self, train: Train, alpha: APLArray, omega: APLArray) -> APLArray:
        tines = train.tines
        if len(tines) == 2:
            f, g = tines
            return self._call_monad(f, self._call_dyad(g, alpha, omega))
        if len(tines) == 3:
            f, g, h = tines
            if not _is_function(f):
                return self._call_dyad(g, self._as_array(f), self._call_dyad(h, alpha, omega))
            return self._call_dyad(
                g, self._call_dyad(f, alpha, omega), self._call_dyad(h, alpha, omega)
            )
        if len(tines) == 1:
            return self._call_dyad(tines[0], alpha, omega)
        return self._call_train_dyad(Train(tines[:-2] + [Train(tines[-2:])]), alpha, omega)

    # -- dfn execution -----------------------------------------------------

    def run_dfn(self, dfn: Dfn, omega: APLArray, alpha: Optional[APLArray]) -> APLArray:
        self._tick()
        saved = self.env
        self.env = dict(saved)
        self.env["⍵"] = omega
        self.env["∇"] = dfn
        if alpha is not None:
            self.env["⍺"] = alpha
        else:
            self.env.pop("⍺", None)
        try:
            result: Optional[APLArray] = None
            for statement in _split_statements(dfn.body):
                if not statement:
                    continue
                guarded = _split_guard(statement)
                if guarded is not None:
                    cond_tokens, body_tokens = guarded
                    cond = self._eval_tokens(cond_tokens)
                    if _truthy(cond):
                        return self._eval_statement(body_tokens)
                    continue
                result = self._eval_statement(statement)
                if statement[0].kind == "name" and len(statement) > 1 and statement[1].kind == "assign":
                    continue  # assignment is not a return value
                return result
            if result is None:
                raise APLError("dfn produced no result")
            return result
        finally:
            self.env = saved

    # -- expression evaluation --------------------------------------------

    def _resolve(self, item: Any) -> Any:
        """Resolve names/args/groups into arrays or functions."""
        if isinstance(item, tuple):
            tag = item[0]
            if tag == "name":
                name = item[1]
                if name in self.env:
                    return _plain(self.env[name])
                raise APLError("undefined name: %s" % name)
            if tag == "arg":
                sym = item[1]
                if sym in self.env:
                    return _plain(self.env[sym])
                raise APLError("%s has no value here" % sym)
            if tag == "operand":
                raise APLError("operand %s outside an operator" % item[1])
            if tag == "group":
                return self._eval_expr(item[1])
            if tag == "index":
                target = self._as_array(item[1])
                axes = [None if part is None else self._as_array(self._eval_expr(part)) for part in item[2]]
                return _bracket_index(target, axes)
        return item

    def _as_array(self, item: Any) -> APLArray:
        value = self._resolve(item)
        if isinstance(value, APLArray):
            return value
        raise APLError("expected an array, got %r" % (value,))

    def _eval_expr(self, items: List[Any]) -> Any:
        """Evaluate right-to-left, the core APL rule.

        Walks leftward keeping one accumulated value. A function found to the
        left of that value is dyadic when an array sits immediately to *its*
        left (left arguments bind short), and monadic otherwise. Adjacent
        arrays strand into a vector, as in `2 3⍴⍳6` or `(1 2)(3 4)`.
        """
        self._tick()
        if not items:
            raise APLError("empty expression")

        items = list(items)
        if len(items) == 1:
            return self._resolve(items[0])

        index = len(items) - 1
        acc = self._resolve(items[index])
        index -= 1

        while index >= 0:
            self._tick()
            item = items[index]
            resolved = item if _is_function(item) else self._resolve(item)

            if _is_function(resolved):
                if not isinstance(acc, APLArray):
                    # Function applied to a function: compose them as a 2-train.
                    acc = Train([resolved, acc])
                    index -= 1
                    continue

                # Dyadic iff an array (or a strand of arrays) is immediately left.
                left_end = index - 1
                if left_end >= 0:
                    probe = items[left_end]
                    probe_val = probe if _is_function(probe) else self._resolve(probe)
                    if isinstance(probe_val, APLArray):
                        strand, next_index = self._strand_left(items, left_end)
                        acc = self._call_dyad(resolved, strand, acc)
                        index = next_index
                        continue

                acc = self._call_monad(resolved, acc)
                index -= 1
                continue

            # An array to the left of an array: strand them into a vector.
            if isinstance(acc, APLArray) and isinstance(resolved, APLArray):
                values, next_index = self._arrays_left(items, index)
                acc = _strand(_splice(values + [acc]))
                index = next_index
                continue

            raise APLError("cannot combine %r with %r" % (resolved, acc))

        return acc

    def _strand_left(self, items: List[Any], end: int):
        """Collect the maximal run of adjacent arrays ending at `end`.

        Returns (stranded_array, index_before_the_run).
        """
        values, i = self._arrays_left(items, end)
        if len(values) == 1:
            return values[0], i
        return _strand(_splice(values)), i

    def _arrays_left(self, items: List[Any], end: int):
        """Return (arrays_in_source_order, index_before_the_run)."""
        values: List[APLArray] = []
        i = end
        while i >= 0:
            item = items[i]
            if _is_function(item):
                break
            value = self._resolve(item)
            if not isinstance(value, APLArray):
                break
            values.append(value)
            i -= 1
        if not values:
            raise APLError("expected an array operand")
        values.reverse()
        return values, i


def _splice(values: List[APLArray]) -> List[APLArray]:
    """Split source number vectors back into one strand item per number."""
    out: List[APLArray] = []
    for value in values:
        if isinstance(value, _NumberLiteral):
            out.extend(APLArray.scalar(x) for x in value.data)
        else:
            out.append(value)
    return out


def _strand(values: List[APLArray]) -> APLArray:
    """Juxtaposed arrays form a vector with one item per array: (1 2)(3 4 5)."""
    return APLArray.vector([v.demote() for v in values])


def _bracket_index(array: APLArray, axes: List[Optional[APLArray]]) -> APLArray:
    """`A[I;J]`: one index array (or None for "all") per axis, origin 1."""
    if array.rank == 0 or len(axes) != array.rank:
        raise APLError("[ ]: rank error (%d indices for rank %d)" % (len(axes), array.rank))
    shape: List[int] = []
    picks: List[List[int]] = []
    for dim, axis in zip(array.shape, axes):
        if axis is None:
            shape.append(dim)
            picks.append(list(range(dim)))
            continue
        shape.extend(axis.shape)
        chosen = []
        for cell in axis.data:
            k = _int_arg(cell, "[ ]")
            if not 1 <= k <= dim:
                raise APLError("[ ]: index error (%d not in 1..%d)" % (k, dim))
            chosen.append(k - 1)
        picks.append(chosen)
    strides = [_prod(array.shape[k + 1 :]) for k in range(array.rank)]
    _check_size(_prod(shape), "[ ]")
    cells = [
        array.data[sum(i * st for i, st in zip(combo, strides))]
        for combo in itertools.product(*picks)
    ]
    return APLArray(tuple(shape), cells)


def _last_axis_rows(array: APLArray) -> List[APLArray]:
    """Split into vectors along the last axis, in row-major order."""
    width = array.shape[-1]
    return [
        APLArray((width,), array.data[i : i + width])
        for i in range(0, len(array.data), width)
    ] if width else [APLArray((0,), []) for _ in range(_prod(array.shape[:-1]))]


def _columns(array: APLArray) -> List[APLArray]:
    if array.rank <= 1:
        return [array]
    if array.rank > 2:
        raise APLError("inner product: rank > 2 unsupported")
    rows, cols = array.shape
    out = []
    for c in range(cols):
        out.append(APLArray((rows,), [array.data[r * cols + c] for r in range(rows)]))
    return out


class _NumberLiteral(APLArray):
    """A numeric vector written in the source, like `1 2`.

    APL strands its numbers one by one, so `1 2(3 4)` has three items; the
    lexer reads `1 2` as one token, and this marker lets stranding split it
    again. Values that pass through a name, a paren or a function are plain.
    """


def _plain(value: Any) -> Any:
    if isinstance(value, _NumberLiteral):
        return APLArray(value.shape, value.data)
    return value


def _number_literal(values: List[Number]) -> APLArray:
    if len(values) == 1:
        return APLArray.scalar(values[0])
    return _NumberLiteral((len(values),), list(values))


def _is_function(item: Any) -> bool:
    return isinstance(item, (APLFunction, Train))


def _truthy(value: APLArray) -> bool:
    if value.count == 0:
        return False
    return bool(_num(value.scalar_value()))


def _match_group(tokens: List[Token], start: int, open_kind: str, close_kind: str):
    """Return (body_tokens, index_after_close) for a bracketed group."""
    depth = 0
    i = start
    while i < len(tokens):
        kind = tokens[i].kind
        if kind == open_kind:
            depth += 1
        elif kind == close_kind:
            depth -= 1
            if depth == 0:
                return tokens[start + 1 : i], i + 1
        i += 1
    raise APLError("unbalanced %s" % open_kind)


def _split_statements(tokens: List[Token]) -> List[List[Token]]:
    """Split on ⋄ / ; at the top nesting level."""
    out: List[List[Token]] = []
    current: List[Token] = []
    depth = 0
    for tok in tokens:
        if tok.kind in ("(", "{", "["):
            depth += 1
        elif tok.kind in (")", "}", "]"):
            depth -= 1
        if tok.kind == "sep" and depth == 0:
            out.append(current)
            current = []
            continue
        current.append(tok)
    out.append(current)
    return out


def _split_guard(tokens: List[Token]):
    """Split `cond : body` at the top level, or return None."""
    depth = 0
    for i, tok in enumerate(tokens):
        if tok.kind in ("(", "{", "["):
            depth += 1
        elif tok.kind in (")", "}", "]"):
            depth -= 1
        elif tok.kind == "guard" and depth == 0:
            return tokens[:i], tokens[i + 1 :]
    return None


# Jot used as the left operand of ∘. (outer product) parses as a bare glyph; we
# register it so `∘.×` resolves to the outer-product path.
def _jot_marker() -> APLFunction:
    return APLFunction("jot-marker", None, None)


def evaluate(source: str, env: Optional[dict] = None, step_budget: int = 200_000):
    """Evaluate `source` and return the result as plain Python data."""
    interp = Interpreter(env=env, step_budget=step_budget)
    result = interp.run(source)
    if result is None:
        return None
    return result.to_nested()
