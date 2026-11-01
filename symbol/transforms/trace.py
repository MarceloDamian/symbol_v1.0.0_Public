"""
Small-step APL traces: break one big evaluation into single applications.

A transformer cannot add ten numbers in one forward pass, but it can do one
`a+b` at a time if each intermediate state is written out. `trace` rewrites a
program right to left, one function application per step, until only the
canonical result remains:

    +/⍳4         ->  +/1 2 3 4 ⋄ 1+2+3+4 ⋄ 1+2+7 ⋄ 1+9 ⋄ 10
    (+/÷≢)2 4 6  ->  (+/2 4 6)÷≢2 4 6 ⋄ (+/2 4 6)÷3 ⋄ (2+4+6)÷3 ⋄ ...

Every step is valid APL with the same value as the original program; each
rewrite is checked with the interpreter, and a rewrite that fails the check
falls back to evaluating that one application directly. Joined with ⋄ the
whole trace is itself an APL program whose value is the result.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from symbol.sandbox.apl_interp import (
    DYADIC_OPERATORS,
    MONADIC_OPERATORS,
    APLArray,
    APLError,
    Interpreter,
)
from symbol.transforms.format import FormatError, same_array, to_apl

STEP_SEP = "⋄"
MAX_STEPS = 40
MAX_STEP_CHARS = 160  # dfn substitution can grow text exponentially; give up instead
STEP_BUDGET = 5_000

NUMBER_START = set("0123456789¯")
NAME_START = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")
NAME_CHARS = NAME_START | set("0123456789")

_SCALAR_TEXT = re.compile(r"^(¯?\d+(\.\d+)?(E¯?\d+)?|'(''|[^'])')$")
_NUM_VECTOR_TEXT = re.compile(r"^¯?\d+(\.\d+)?(E¯?\d+)?( ¯?\d+(\.\d+)?(E¯?\d+)?)+$")
_STRING_TEXT = re.compile(r"^'(''|[^'])*'$")
_SCALAR_DYADS = "+-×÷⌈⌊|=≠<>≤≥∧∨*"
_ELEMENTWISE = set("+-×÷⌈⌊|∧∨*")  # arithmetic worth one step per item; comparisons are easy
MAX_ELEMENTWISE = 4  # longer vectors would blow past the trace length cap


class TraceError(ValueError):
    """The program could not be traced (unsupported shape or a failed check)."""


def _run(src: str) -> APLArray:
    value = Interpreter(step_budget=STEP_BUDGET).run(src)
    if not isinstance(value, APLArray):
        raise TraceError("not an array expression: %r" % src)
    return value


# ---------------------------------------------------------------------------
# Lexing and segmentation into array / function units
# ---------------------------------------------------------------------------


def _match(src: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index just past the bracket matching src[i], skipping string literals."""
    depth = 0
    while i < len(src):
        ch = src[i]
        if ch == "'":
            i += 1
            while i < len(src) and not (src[i] == "'" and src[i + 1 : i + 2] != "'"):
                i += 2 if src[i] == "'" else 1
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise TraceError("unbalanced %s in %r" % (open_ch, src))


def _lex(src: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == " ":
            i += 1
        elif ch == "'":
            j = i + 1
            while j < n:
                if src[j] == "'":
                    if src[j + 1 : j + 2] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(("array", src[i : j + 1]))
            i = j + 1
        elif ch in NUMBER_START:
            j = i
            while True:
                while j < n and (src[j] in NUMBER_START or src[j] in ".E"):
                    j += 1
                k = j
                while k < n and src[k] == " ":
                    k += 1
                if k > j and k < n and src[k] in NUMBER_START:
                    j = k
                    continue
                break
            out.append(("array", src[i:j]))
            i = j
        elif ch == "⍬":
            out.append(("array", ch))
            i += 1
        elif ch in NAME_START:
            j = i
            while j < n and src[j] in NAME_CHARS:
                j += 1
            out.append(("array", src[i:j]))
            i = j
        elif ch == "(":
            j = _match(src, i, "(", ")")
            out.append(("paren", src[i + 1 : j - 1]))
            i = j
        elif ch == "{":
            j = _match(src, i, "{", "}")
            out.append(("fn", src[i:j]))
            i = j
        elif ch == "[":
            # Indexing binds to the array just left: `v[2]`, `(⍳5)[2 3]` are one unit.
            j = _match(src, i, "[", "]")
            if not out or out[-1][0] not in ("array", "paren"):
                raise TraceError("[ ] without an array in %r" % src)
            kind, text = out.pop()
            out.append(("array", ("(" + text + ")" if kind == "paren" else text) + src[i:j]))
            i = j
        elif ch in MONADIC_OPERATORS:
            out.append(("mop", ch))
            i += 1
        elif ch in DYADIC_OPERATORS:
            out.append(("dop", ch))
            i += 1
        else:
            out.append(("fn", ch))
            i += 1
    return out


def _is_function_text(text: str) -> bool:
    try:
        return not isinstance(Interpreter(step_budget=STEP_BUDGET).run(text), APLArray)
    except APLError:
        return True


def segment(src: str) -> List[Tuple[str, str]]:
    """Split an expression into ("array" | "fn", text) units, operators bound."""
    lex = _lex(src)
    units: List[Tuple[str, str]] = []
    i = 0
    while i < len(lex):
        kind, text = lex[i]
        if kind == "dop" and text == "∘" and i + 2 < len(lex) and lex[i + 1] == ("dop", "."):
            units.append(("fn", "∘." + _unit_text(lex[i + 2])))
            i += 3
        elif kind == "dop":
            if not units or i + 1 >= len(lex):
                raise TraceError("operator %s without operands" % text)
            prev = units.pop()[1]
            units.append(("fn", prev + text + _unit_text(lex[i + 1])))
            i += 2
        elif kind == "mop":
            if text in "/⌿\\⍀" and (not units or units[-1][0] == "array"):
                units.append(("fn", text))  # replicate / expand with an array left arg
            elif units and units[-1][0] == "fn":
                units[-1] = ("fn", units[-1][1] + text)
            else:
                raise TraceError("operator %s without a function operand" % text)
            i += 1
        elif kind == "paren":
            units.append(("fn" if _is_function_text(text) else "array", "(" + text + ")"))
            i += 1
        else:
            units.append((kind, text))
            i += 1
    return units


def _unit_text(lexeme: Tuple[str, str]) -> str:
    kind, text = lexeme
    return "(" + text + ")" if kind == "paren" else text


def join(texts: List[str]) -> str:
    """Concatenate unit texts, adding a space only where tokens would fuse."""
    out = ""
    for text in texts:
        if out and text and (out[-1] in NAME_CHARS or out[-1] == ".") and (
            text[0] in NAME_CHARS or text[0] in "¯."
        ):
            out += " "
        out += text
    return out


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------


def _is_simple_literal(text: str) -> bool:
    return bool(
        _SCALAR_TEXT.match(text) or _NUM_VECTOR_TEXT.match(text) or _STRING_TEXT.match(text)
    ) or text == "⍬"


def _q(text: str) -> str:
    """Make an argument safe to splice: bare if it is a plain literal."""
    return text if _is_simple_literal(text) else "(" + text + ")"


def _strip_outer(text: str) -> str:
    """`(x)` -> `x` when the parentheses wrap the whole text."""
    if text.startswith("(") and _match(text, 0, "(", ")") == len(text):
        return text[1:-1]
    return text


def _matrix_cells(text: str) -> Optional[Tuple[int, int, List[str]]]:
    """(rows, cols, cells) for a canonical numeric matrix literal `R C⍴v…`."""
    m = re.match(r"^(\d+) (\d+)⍴(.+)$", _strip_outer(text))
    if not m or not (_NUM_VECTOR_TEXT.match(m.group(3)) or _SCALAR_TEXT.match(m.group(3))):
        return None
    r, c, cells = int(m.group(1)), int(m.group(2)), m.group(3).split(" ")
    return (r, c, cells) if len(cells) == r * c and not cells[0].startswith("'") else None


def _rows(mat: Tuple[int, int, List[str]]) -> List[str]:
    r, c, cells = mat
    return [" ".join(cells[i * c : (i + 1) * c]) for i in range(r)]


def _items(text: str) -> Optional[List[str]]:
    """Items of a numeric vector or a strand of parenthesised items, else None."""
    if _NUM_VECTOR_TEXT.match(text):
        return text.split(" ")
    try:
        units = segment(text)
    except (TraceError, APLError):
        return None
    if len(units) < 2 or any(k != "array" for k, _ in units):
        return None
    out = []
    for _, t in units:
        inner = _strip_outer(t)
        out.append(inner if _is_simple_literal(inner) else t)
    return out


def _pairs(left: str, right: str) -> Optional[List[Tuple[str, str]]]:
    """Matched scalar pairs for `left fn right` on short numeric arguments (2-4 items)."""
    def items(text):
        if _NUM_VECTOR_TEXT.match(text):
            return text.split(" ")
        if _SCALAR_TEXT.match(text) and not text.startswith("'"):
            return [text]
        return None

    a, b = items(left), items(right)
    if a is None or b is None or max(len(a), len(b)) > MAX_ELEMENTWISE:
        return None
    if len(a) == 1 and len(b) > 1:
        a = a * len(b)
    elif len(b) == 1 and len(a) > 1:
        b = b * len(a)
    return list(zip(a, b)) if len(a) == len(b) and len(a) > 1 else None


def _pick_branch(body: str, L: Optional[str], R: str) -> Optional[str]:
    """The statement a guarded dfn body returns for these arguments, or None."""
    for statement in body.split("⋄"):
        if ":" not in statement:
            return statement
        cond, expr = statement.split(":", 1)
        cond = cond.replace("⍵", R)
        if L is not None:
            cond = cond.replace("⍺", L)
        if "⍺" in cond or "∇" in cond:
            return None
        try:
            value = _run(cond)
        except (APLError, TraceError):
            return None
        if value.count != 1 or value.data[0] not in (0, 1):
            return None
        if value.data[0] == 1:
            return expr
    return None


def _rewrites(left: Optional[str], fn: str, right: str) -> List[str]:
    """Candidate one-step rewrites of `left fn right`, most instructive first."""
    out: List[str] = []
    R = _q(right)
    L = _q(left) if left is not None else None

    # `(f)` around a single function: try the rules for f itself first.
    if fn.startswith("(") and _match(fn, 0, "(", ")") == len(fn):
        try:
            tines = segment(fn[1:-1])
        except (TraceError, APLError):
            tines = []
        if len(tines) == 1 and tines[0][0] == "fn":
            out.extend(_rewrites(left, tines[0][1], right))

    # Power: f⍣3 x -> f f f x (small counts only).
    m = re.match(r"^(.+)⍣(\d)$", fn)
    if m and int(m.group(2)) <= 4:
        f = m.group(1)
        if len(f) > 1 and not (f.startswith("{") or f.startswith("(")):
            f = "(" + f + ")"
        k = int(m.group(2))
        out.append(join(([L, f] if L is not None else [f]) * k + [R]) if k else right)

    # Guarded / recursive dfn: take the branch whose guard holds, ∇ is the dfn itself.
    m = re.match(r"^\{([^{}←]*)\}$", fn)
    if m and any(ch in m.group(1) for ch in "⋄:∇"):
        branch = _pick_branch(m.group(1), L, R)
        if branch is not None:
            body = branch.replace("⍵", R)
            if L is not None:
                body = body.replace("⍺", L)
            if "⍺" not in body:
                out.append(body.replace("∇", fn))

    m = re.match(r"^([%s])[/⌿]$" % re.escape(_SCALAR_DYADS), fn)
    if m and left is None and _NUM_VECTOR_TEXT.match(right):
        out.append(m.group(1).join(right.split(" ")))
    m = re.match(r"^(\{[^{}]*\})/$", fn)  # dfn reduce: 1{⍺×⍵}2{⍺×⍵}3
    if m and left is None and _NUM_VECTOR_TEXT.match(right):
        out.append(m.group(1).join(right.split(" ")))

    # Matrices: split into rows / columns so each step is a vector-sized job.
    mat = _matrix_cells(right)
    m = re.match(r"^([%s])/$" % re.escape(_SCALAR_DYADS), fn)
    if m and left is None and mat and mat[0] >= 2 and mat[1] >= 2:
        out.append(join(["(" + m.group(0) + row + ")" for row in _rows(mat)]))
    m = re.match(r"^([%s])⌿$" % re.escape(_SCALAR_DYADS), fn)
    if m and left is None and mat and mat[0] >= 2 and mat[1] >= 2:
        out.append(m.group(1).join(_rows(mat)))
    if fn == "⍉" and left is None and mat and mat[0] >= 2 and mat[1] >= 2:
        r, c, cells = mat
        cols = [" ".join(cells[j::c]) for j in range(c)]
        out.append("%d %d⍴" % (c, r) + ",".join(cols))

    # Outer product: one row per left item, `l g R`, then catenate and reshape.
    m = re.match(r"^∘\.([%s])$" % re.escape(_SCALAR_DYADS), fn)
    if m and left is not None and _NUM_VECTOR_TEXT.match(left) and _NUM_VECTOR_TEXT.match(right):
        items = left.split(" ")
        rows = [join([x, m.group(1), right]) for x in items]
        shape = "%d %d⍴" % (len(items), len(right.split(" ")))
        out.append(shape + ",".join("(" + r + ")" for r in rows[:-1]) + "," + rows[-1])

    # Short elementwise arithmetic: one scalar application per item.
    if fn in _ELEMENTWISE and left is not None:
        pairs = _pairs(left, right)
        if pairs:
            out.append("".join("(%s%s%s)" % (a, fn, b) for a, b in pairs))

    # Replicate: drop the zero counts, then spell out each run.
    if fn == "/" and left is not None and _NUM_VECTOR_TEXT.match(left) and _NUM_VECTOR_TEXT.match(right):
        counts, values = left.split(" "), right.split(" ")
        if len(counts) == len(values) and all(c.isdigit() for c in counts):
            kept = [(int(c), v) for c, v in zip(counts, values) if c != "0"]
            parts: List[str] = []  # neighbouring single copies stay one vector: 1 2,(3⍴4)
            for idx, (c, v) in enumerate(kept):
                if c == 1 and idx and kept[idx - 1][0] == 1:
                    parts[-1] += " " + v
                else:
                    parts.append(v if c == 1 else "(%d⍴%s)" % (c, v))
            if len(parts) >= 2:
                out.append(",".join(parts))

    # Decode: 10⊥1 2 3 -> +/1 2 3×100 10 1 (place values, then the usual sum).
    if fn == "⊥" and left is not None and _pairs(left, right):
        radix, digits = zip(*_pairs(left, right))
        if all(re.match(r"^\d+$", x) for x in radix + digits):
            weights, place = [], 1
            for base in reversed(radix):
                weights.append(str(place))
                place *= int(base)
            out.append("+/" + " ".join(digits) + "×" + " ".join(reversed(weights)))

    # Inner product of vectors: a+.×b -> +/a×b. (Row by row, a matrix overflows the trace cap.)
    if (fn == "+.×" and left is not None and _NUM_VECTOR_TEXT.match(left) and _NUM_VECTOR_TEXT.match(right)
            and len(left.split(" ")) == len(right.split(" "))):
        out.append("+/" + left + "×" + right)

    # Two-row inner product, one row at a time: (2 3⍴…)+.×v -> (r1+.×v)(r2+.×v). More rows overflow the trace.
    lmat = _matrix_cells(left) if left is not None else None
    if fn == "+.×" and lmat and lmat[0] == 2 and _NUM_VECTOR_TEXT.match(right) and lmat[1] == len(right.split(" ")):
        out.append(join(["(" + row + "+.×" + right + ")" for row in _rows(lmat)]))

    # Decode a matrix column by column: 2⊥2 2⍴1 0 1 1 -> (2⊥1 1)(2⊥0 1).
    mat = _matrix_cells(right)
    if fn == "⊥" and left is not None and _SCALAR_TEXT.match(left) and mat and mat[1] <= 3:
        r, c, cells = mat
        out.append(join(["(" + left + "⊥" + " ".join(cells[j::c]) + ")" for j in range(c)]))

    # Encode a number: 10 10⊤47 -> (10|⌊47÷10)(10|47). A 0 radix keeps the rest: 0 5⊤17 -> (⌊17÷5)(5|17).
    if fn == "⊤" and left is not None and _NUM_VECTOR_TEXT.match(left) and re.match(r"^\d+$", right):
        radix = left.split(" ")
        if len(radix) <= 4 and all(x.isdigit() for x in radix):
            digits, place = [], 1
            for base in reversed(radix):
                shifted = right if place == 1 else "⌊%s÷%d" % (right, place)
                digits.append(shifted if base == "0" else "%s|%s" % (base, shifted))
                place *= int(base) or 1
            out.append(join(["(" + d + ")" for d in reversed(digits)]))

    # Binomial: 3!6 -> (6×5×4)÷1×2×3, falling factorial over factorial.
    if fn == "!" and left is not None and re.match(r"^\d$", left) and re.match(r"^\d+$", right):
        k, n = int(left), int(right)
        if 2 <= k <= 4 and n >= k:
            out.append("(" + "×".join(str(n - i) for i in range(k)) + ")÷" + "×".join(str(i) for i in range(1, k + 1)))

    # Find: compare the needle with each window; windows past the end are 0. 1 2⍷3 1 2 -> (1 2≡3 1)(1 2≡1 2)0
    if fn == "⍷" and left is not None:
        for parse in (lambda t: t.split(" ") if _NUM_VECTOR_TEXT.match(t) else None,
                      lambda t: list(t[1:-1]) if _STRING_TEXT.match(t) and "'" not in t[1:-1] else None):
            needle, hay = parse(left), parse(right)
            if needle and hay and len(needle) >= 2 and len(hay) <= 6:
                quote = "'" if left.startswith("'") else ""
                spell = lambda xs: quote + ("" if quote else " ").join(xs) + quote
                n = len(needle)
                windows = ["(" + left + "≡" + spell(hay[i:i + n]) + ")" if i + n <= len(hay) else "0"
                           for i in range(len(hay))]
                out.append(join(windows))

    # Where: ⍸1 0 1 -> 1 0 1/⍳3. Intersection: a∩b -> (a∊b)/a.
    if fn == "⍸" and left is None and _NUM_VECTOR_TEXT.match(right) and "¯" not in right:
        out.append(right + "/⍳" + str(len(right.split(" "))))
    if fn == "∩" and left is not None and (_NUM_VECTOR_TEXT.match(left) or _STRING_TEXT.match(left)):
        out.append("(" + join([left, "∊", R]) + ")/" + left)

    # Each: apply the function to every item, written as a strand.
    m = re.match(r"^(.+)¨$", fn)
    if m and left is None:
        items = _items(right)
        if items:
            out.append(join(["(" + join([m.group(1), x]) + ")" for x in items]))

    m = re.match(r"^(.+)⍨$", fn)
    if m:
        g = m.group(1)
        out.append(join([R, g, L if L is not None else R]))

    m = re.match(r"^\{([^{}⋄:←]*)\}$", fn)
    if m:
        body = m.group(1).replace("⍵", R)
        if L is not None:
            body = body.replace("⍺", L)
        if "⍺" not in body:
            out.append(body)

    m = re.match(r"^\((¯?\d+)∘(.)\)$", fn)
    if m and left is None:
        out.append(join([m.group(1), m.group(2), R]))
    m = re.match(r"^\((.)∘(¯?\d+)\)$", fn)
    if m and left is None:
        out.append(join([R, m.group(1), m.group(2)]))
    m = re.match(r"^([^∘]+)∘([^∘.]+)$", fn)
    if m and left is None:
        out.append(join([m.group(1), m.group(2), R]))

    m = re.match(r"^(.)\.(.)$", fn)
    if m and L is not None and m.group(1) != "∘":
        out.append(join([m.group(1) + "/", L, m.group(2), R]))

    if fn.startswith("(") and fn.endswith(")"):
        try:
            tines = segment(fn[1:-1])
        except (TraceError, APLError):
            tines = []
        if tines and all(kind == "fn" for kind, _ in tines):
            t = [text for _, text in tines]
            if len(t) == 1:
                out.append(join(([L] if L else []) + [t[0], R]))
            elif len(t) == 2:
                out.append(join([t[0]] + ([L] if L else []) + [t[1], R]))
            elif len(t) == 3:
                lhs = join(([L] if L else []) + [t[0], R])
                rhs = join(([L] if L else []) + [t[2], R])
                out.append("(" + lhs + ")" + t[1] + rhs)
    return out


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def _unwrap(units: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drop parentheses around plain literals when no array neighbour could strand."""
    out = list(units)
    for idx, (kind, text) in enumerate(out):
        if kind != "array" or not text.startswith("("):
            continue
        inner = text[1:-1]
        if not _is_simple_literal(inner):
            continue
        left_ok = idx == 0 or out[idx - 1][0] == "fn"
        right_ok = idx == len(out) - 1 or out[idx + 1][0] == "fn"
        # A finished number inside a strand needs no parens: (1+4)(7)(9) -> (1+4)7 9.
        numeric_scalar = bool(_SCALAR_TEXT.match(inner)) and not inner.startswith("'")
        if (left_ok and right_ok) or numeric_scalar:
            out[idx] = ("array", inner)
    return out


def _canonical(src: str) -> str:
    return to_apl(_run(src))


def step(src: str, target: APLArray) -> Optional[str]:
    """One small step toward the value. None means `src` is already canonical."""
    if src == to_apl(target):
        return None

    m = re.match(r"^([A-Za-z_]\w*)←([^⋄]+)⋄(.+)$", src)
    if m:
        name, value, rest = m.groups()
        value = _canonical(value)
        return re.sub(r"(?<![\w])%s(?![\w])" % re.escape(name), _q(value).replace("\\", "\\\\"), rest)

    units = segment(src)
    fn_idx = max((i for i, (k, _) in enumerate(units) if k == "fn"), default=None)

    def non_canonical_paren(indices) -> Optional[int]:
        for i in reversed(list(indices)):
            kind, text = units[i]
            if kind != "array":
                continue
            if text.endswith("]") or (text.startswith("(") and _canonical(text[1:-1]) != text[1:-1]):
                return i
        return None

    if fn_idx is None:
        # Pure arrays: a strand of parenthesised expressions.
        idx = non_canonical_paren(range(len(units)))
        if idx is None:
            return _canonical(src)
        return _replace_paren(units, idx, target)

    if any(k != "array" for k, _ in units[fn_idx + 1 :]):
        raise TraceError("unexpected unit layout in %r" % src)
    if fn_idx == len(units) - 1:
        raise TraceError("function with no right argument in %r" % src)

    idx = non_canonical_paren(range(fn_idx + 1, len(units)))
    if idx is not None:
        return _replace_paren(units, idx, target)

    start = fn_idx
    while start > 0 and units[start - 1][0] == "array":
        start -= 1
    idx = non_canonical_paren(range(start, fn_idx))
    if idx is not None:
        return _replace_paren(units, idx, target)

    prefix = [t for _, t in units[:start]]
    left = join([t for _, t in units[start:fn_idx]]) if start < fn_idx else None
    fn = units[fn_idx][1]
    right = join([t for _, t in units[fn_idx + 1 :]])
    right = _canonical(right) if len(units) - fn_idx > 2 else right

    application = join(([left] if left is not None else []) + [fn, right])
    direct = _canonical(application)
    candidates = _rewrites(left, fn, right) + [direct if not prefix else _q(direct)]
    for cand in candidates:
        new = _finish(prefix + [cand])
        if new != src and _same_value(new, target):
            return new
    raise TraceError("no valid step from %r" % src)


def _split_index(text: str) -> Tuple[str, str]:
    """`base[axes]` -> (base, axes) for the outermost trailing bracket."""
    for k, ch in enumerate(text):
        if ch == "[" and _match(text, k, "[", "]") == len(text):
            return text[:k], text[k + 1 : -1]
    raise TraceError("not an indexed array: %r" % text)


def _step_index(units, idx, target) -> str:
    """One step on `base[axes]`: simplify the base, then the index, then pick."""
    text = units[idx][1]
    base, axes = _split_index(text)
    new_text = None
    inner = _strip_outer(base)
    if inner != base and _canonical(inner) != inner:
        nxt = step(inner, _run(inner))
        new_base = nxt if _is_simple_literal(nxt) else "(" + nxt + ")"
        new_text = new_base + "[" + axes + "]"
    elif ";" not in axes and _canonical(axes) != axes:
        new_text = base + "[" + step(axes, _run(axes)) + "]"
    candidates = ([new_text] if new_text else []) + [_q(_canonical(text))]
    for cand in candidates:
        new_units = list(units)
        new_units[idx] = ("array", cand)
        new = join([t for _, t in _unwrap(new_units)])
        if _same_value(new, target):
            return new
    raise TraceError("no valid index step from %r" % text)


def _replace_paren(units, idx, target) -> str:
    if units[idx][1].endswith("]"):
        return _step_index(units, idx, target)
    inner = units[idx][1][1:-1]
    inner_value = _run(inner)
    nxt = step(inner, inner_value)
    if nxt is None:
        raise TraceError("paren already canonical")
    new_units = list(units)
    new_units[idx] = ("array", "(" + nxt + ")")
    new = join([t for _, t in _unwrap(new_units)])
    if not _same_value(new, target):
        new = join([t for _, t in new_units])
        if not _same_value(new, target):
            raise TraceError("inner step changed the value: %r" % new)
    return new


def _finish(texts: List[str]) -> str:
    src = join(texts)
    try:
        return join([t for _, t in _unwrap(segment(src))])
    except TraceError:
        return src


def _same_value(src: str, target: APLArray) -> bool:
    try:
        return same_array(_run(src), target)
    except (APLError, TraceError, ArithmeticError, ValueError, TypeError, IndexError, RecursionError):
        return False


def trace(program: str) -> List[str]:
    """All intermediate states after `program`, ending with the canonical result."""
    target = _run(program)
    final = to_apl(target)
    steps: List[str] = []
    seen = {program}
    current = program
    for _ in range(MAX_STEPS):
        try:
            nxt = step(current, target)
        except (APLError, FormatError, ArithmeticError, IndexError, RecursionError) as exc:
            raise TraceError("cannot step %r: %s" % (current, exc))
        if nxt is None:
            break
        if len(nxt) > MAX_STEP_CHARS:
            raise TraceError("step longer than %d chars" % MAX_STEP_CHARS)
        if nxt in seen:
            raise TraceError("trace loops at %r" % nxt)
        seen.add(nxt)
        # A step that only adds parentheses carries no information; keep going from it silently.
        if _strip_parens(nxt) != _strip_parens(current):
            steps.append(nxt)
        current = nxt
    else:
        raise TraceError("trace exceeded %d steps" % MAX_STEPS)
    if not steps or steps[-1] != final:
        steps.append(final)
    return steps


def _strip_parens(src: str) -> str:
    return src.replace("(", "").replace(")", "")


def trace_text(program: str) -> str:
    return STEP_SEP.join(trace(program))


def final_answer(text: str) -> str:
    """The result part of a generated trace (text after the last ⋄)."""
    return text.rsplit(STEP_SEP, 1)[-1]
