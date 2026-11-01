"""Correctness tests for the APL subset interpreter.

Runs standalone (`python3 tests/test_apl_interp.py`) so no pytest dependency is
required, but the assert-style functions are also pytest-compatible.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbol.sandbox.apl_interp import APLError, evaluate  # noqa: E402

# (source, expected) pairs covering the primitives the compression rules rely on.
CASES = [
    # Scalar arithmetic and pervasion
    ("1+2", 3),
    # Control flow: ∇ recursion, guards, ⍣ power.
    ("{⍵≤1:1⋄⍵×∇⍵-1}5", 120),
    ("12{0=⍵:⍺⋄⍵∇⍵|⍺}8", 4),
    ("{⍵<5:⍵⋄∇⍵-5}17", 2),
    ("(×⍨⍣2)3", 81),
    ("({⍵+3}⍣4)0", 12),
    ("2(×⍣3)1", 8),
    # Round 10 primitives.
    ("⍸1 0 1 1", [1, 3, 4]),
    ("⍸2 0 1", [1, 1, 3]),
    ("1 2 3 4∩2 4 6", [2, 4]),
    ("2⊥1 0 1", 5),
    ("24 60⊥2 30", 150),
    ("2 2 2⊤5", [1, 0, 1]),
    ("0 10⊤123", [12, 3]),
    ("2 3⍷1 2 3 2 3", [0, 1, 0, 1, 0]),
    ("1 2 3!4", [4, 6, 4]),
    ("≠3 1 3 2 1", [1, 1, 0, 1, 0]),
    ("2 2⊤1 2", [[0, 1], [1, 0]]),
    ("1 2 3 4/⍨1 0 1 1", [1, 3, 4]),
    ("C←⍉(4⍴2)⊤¯1+⍳16⋄C⌿⍨15=C+.×3 5 7 11", [[1, 1, 1, 0]]),  # subset sum by bits
    ("2⊥2 2⍴1 0 1 1", [3, 1]),  # a matrix decodes column by column
    ("2⊥(3⍴2)⊤3 6", [3, 6]),  # decode undoes encode
    # Bracket indexing, origin 1; `;` separates axes, empty means all.
    ("10 20 30[2]", 20),
    ("10 20 30[3 1]", [30, 10]),
    ("(⍳5)[2 3]", [2, 3]),
    ("A←2 3⍴⍳6⋄A[2;]", [4, 5, 6]),
    ("A←2 3⍴⍳6⋄A[;1]", [1, 4]),
    ("A←2 3⍴⍳6⋄A[2;3]", 6),
    ("{⍵[⍋⍵]}3 1 2", [1, 2, 3]),
    # Stranding splits source number vectors: one item per number.
    ("≢1 2(3 4)", 3),
    ("≢(1 2)3 4", 3),
    ("≢(1 2)3", 2),
    ("A←1 2⋄≢A 3", 2),
    ("1 2(3 4)+1", [2, 3, [4, 5]]),
    ("1 2 3+4 5 6", [5, 7, 9]),
    ("10-1 2 3", [9, 8, 7]),
    ("2×3 4", [6, 8]),
    ("¯3", -3),
    ("|¯4", 4),
    ("÷4", 0.25),
    ("2*10", 1024),
    ("3|10", 1),
    ("⌈2.3", 3),
    ("⌊2.7", 2),
    ("2⌈5", 5),
    ("2⌊5", 2),

    # Comparison produces booleans
    ("1 2 3=2", [0, 1, 0]),
    ("1 2 3≠2", [1, 0, 1]),
    ("1 2 3<2", [1, 0, 0]),
    ("1 2 3≥2", [0, 1, 1]),

    # Index generator and shape
    ("⍳5", [1, 2, 3, 4, 5]),
    ("⍴1 2 3", [3]),
    ("≢1 2 3 4", 4),
    ("2 3⍴⍳6", [[1, 2, 3], [4, 5, 6]]),
    ("⍴2 3⍴⍳6", [2, 3]),

    # Reduce
    ("+/⍳10", 55),
    ("×/1 2 3 4", 24),
    ("-/1 2 3", 2),          # 1-(2-3)
    ("⌈/3 1 4 1 5", 5),
    ("⌊/3 1 4 1 5", 1),
    ("+/⍬", 0),              # identity element for empty reduce

    # Scan
    ("+\\1 2 3 4", [1, 3, 6, 10]),
    ("×\\1 2 3 4", [1, 2, 6, 24]),
    ("-\\1 2 3", [1, -1, 2]),        # prefix reductions, right to left
    ("+\\2 3⍴⍳6", [[1, 3, 6], [4, 9, 15]]),
    ("+⍀2 3⍴⍳6", [[1, 2, 3], [5, 7, 9]]),
    ("×\\0", 0),

    # Reduce along axes
    ("+/2 3⍴⍳6", [6, 15]),           # last axis: row sums
    ("+⌿2 3⍴⍳6", [5, 7, 9]),         # first axis: column sums
    ("+/5", 5),

    # Boolean / GCD-LCM
    ("1 0 1∧1 1 0", [1, 0, 0]),
    ("1 0 0∨0 0 1", [1, 0, 1]),
    ("4∧6", 12),
    ("4∨6", 2),
    ("~1 0", [0, 1]),

    # Each
    ("-¨1 2 3", [-1, -2, -3]),
    ("≢¨(1 2)(3 4 5)", [2, 3]),
    ("≢(1 2)(3 4)(5 6)", 3),   # three-way strand stays flat, not nested pairs

    # Commute
    ("-⍨3", 0),
    ("2-⍨10", 8),            # reversed: 10-2

    # Compose / bind
    ("+/∘⍳ 5", 15),
    ("(2∘×)3", 6),
    ("(×∘2)3", 6),

    # Outer and inner product
    ("1 2∘.×1 2 3", [[1, 2, 3], [2, 4, 6]]),
    ("⍴1 2 3∘.+1 2", [3, 2]),
    ("1 2 3+.×4 5 6", 32),

    # Structural
    (",1 2 3", [1, 2, 3]),
    ("1 2,3 4", [1, 2, 3, 4]),
    ("⌽1 2 3", [3, 2, 1]),
    ("2⌽1 2 3 4 5", [3, 4, 5, 1, 2]),
    ("3↑1 2 3 4 5", [1, 2, 3]),
    ("2↓1 2 3 4 5", [3, 4, 5]),
    ("¯2↑1 2 3 4 5", [4, 5]),
    ("⊃1 2 3", 1),
    ("≡5", 0),
    ("≡1 2", 1),
    ("≡⊂1 2", 2),
    ("≡⊂⊂1 2", 3),            # enclosing a non-simple scalar adds depth
    ("⍋3 1 2", [2, 3, 1]),
    ("⍒3 1 2", [1, 3, 2]),
    ("∪1 1 2 3 2", [1, 2, 3]),
    ("1 2 3∊2 3 4", [0, 1, 1]),
    ("1 2 3 4~2 4", [1, 3]),
    ("1 0 1/1 2 3", [1, 3]),
    ("3 1 4⍳4", 3),
    ("⍉2 3⍴⍳6", [[1, 4], [2, 5], [3, 6]]),
    ("1 2 3≡1 2 3", 1),
    ("1 2 3≡1 2 4", 0),
    ("2 3⌷1 2 3 4 5", [2, 3]),

    # Dfns
    ("{⍵+1}5", 6),
    ("3{⍺+⍵}2 3", [5, 6]),
    ("2{⍺+⍵}3", 5),
    ("{+/⍵}1 2 3", 6),
    ("{⍵×⍵}4", 16),

    # Dfn with guard
    ("{0=⍵:99 ⋄ ⍵}0", 99),
    ("{0=⍵:99 ⋄ ⍵}7", 7),

    # Tacit trains
    ("(+/÷≢)1 2 3 4", 2.5),   # mean, the canonical 3-train
    ("(⌽∘⍳)3", [3, 2, 1]),

    # Character data
    ("'abc'", ["a", "b", "c"]),
    ("≢'hello'", 5),
    ("⌽'abc'", ["c", "b", "a"]),

    # Right-to-left evaluation order
    ("2×3+4", 14),            # not 10
    ("1+2×3", 7),

    # Parens
    ("(2×3)+4", 10),

    # Assignment
    ("A←1 2 3 ⋄ +/A", 6),

    # Nested
    ("⊃⌽1 2 3", 3),
    ("+/1 0 1 0 1", 3),
]

# Expressions that must raise, not silently produce a wrong answer.
ERROR_CASES = [
    "⍸¯1 2",
    "2 3⊥1 2 3",
    "2 2⊤2 2⍴1",
    "(⍳1000)⊤⍳10000",  # 10⁷ cells: WS FULL before anything is built
    "{⍵/⍵}5*15",  # 3×10¹⁰ copies: WS FULL before anything is built
    "∇3",
    "(+⍣¯1)3",
    "1 2 3[4]",
    "1 2 3[1;1]",
    "5[1]",
    "1 2 3+4 5",       # length error
    "⍳¯1",             # domain error
    "undefinedName",   # unbound name
    "{⍺+⍵}5",          # ⍺ missing in monadic call
    "~3",              # ~ is only defined on booleans
    "2⍲3",
    "11 20 200 992⍴⍳4",  # WS FULL instead of allocating 44M cells
    "⍳1E9",
    "1E9↑1",
]


def run() -> int:
    passed = failed = 0
    failures = []

    for source, expected in CASES:
        try:
            actual = evaluate(source)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            failed += 1
            failures.append("%-24s raised %s: %s" % (source, type(exc).__name__, exc))
            continue
        if actual == expected:
            passed += 1
        else:
            failed += 1
            failures.append("%-24s got %r, want %r" % (source, actual, expected))

    for source in ERROR_CASES:
        try:
            actual = evaluate(source)
        except APLError:
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append("%-24s raised non-APLError %s: %s" % (source, type(exc).__name__, exc))
        else:
            failed += 1
            failures.append("%-24s should have errored, got %r" % (source, actual))

    print("passed: %d   failed: %d" % (passed, failed))
    if failures:
        print("\nfailures:")
        for line in failures:
            print("  " + line)
    return 1 if failed else 0


# pytest entry points -------------------------------------------------------

def test_cases():
    for source, expected in CASES:
        assert evaluate(source) == expected, source


def test_error_cases():
    for source in ERROR_CASES:
        try:
            evaluate(source)
        except APLError:
            continue
        raise AssertionError("%s should have raised APLError" % source)


if __name__ == "__main__":
    sys.exit(run())
