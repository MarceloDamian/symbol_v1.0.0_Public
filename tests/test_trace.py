"""Tests for small-step traces. Pure Python:  python3 tests/test_trace.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbol.dataset.canonical import BITS, CANONICAL, CONTROL, HARD, PRIMS  # noqa: E402
from symbol.dataset.generator import generate  # noqa: E402
from symbol.sandbox.apl_interp import Interpreter  # noqa: E402
from symbol.transforms.format import same_array, to_apl  # noqa: E402
from symbol.transforms.trace import STEP_SEP, final_answer, trace, trace_text  # noqa: E402

EXPECTED = {
    "2×3+4": ["2×7", "14"],
    "(2×3)+4": ["6+4", "10"],
    "+/⍳4": ["+/1 2 3 4", "1+2+3+4", "1+2+7", "1+9", "10"],
    "2-⍨10": ["10-2", "8"],
    "{⍵×⍵}4": ["4×4", "16"],
    "A←1 2 3⋄+/A": ["+/1 2 3", "1+2+3", "1+5", "6"],
    "1 2 3+.×4 5 6": ["+/1 2 3×4 5 6", "+/(1×4)(2×5)(3×6)", "+/(1×4)(2×5)18", "+/(1×4)10 18", "+/4 10 18", "4+10+18", "4+28", "32"],
    "8 2 3∧3 1 3": ["(8∧3)(2∧1)(3∧3)", "(8∧3)(2∧1)3", "(8∧3)2 3", "24 2 3"],
    "1 1 0 3/1 2 3 4": ["1 2,(3⍴4)", "1 2,4 4 4", "1 2 4 4 4"],
    "3 1/4 5": ["(3⍴4),5", "4 4 4,5", "4 4 4 5"],
    "⍴2 3⍴⍳6": ["⍴2 3⍴1 2 3 4 5 6", "2 3"],  # no parens-only step in between
    # Round 4: matrix and nested ops step through rows / columns / items.
    "+⌿2 3⍴⍳6": ["+⌿2 3⍴1 2 3 4 5 6", "1 2 3+4 5 6", "(1+4)(2+5)(3+6)", "(1+4)(2+5)9", "(1+4)7 9", "5 7 9"],
    "⍉2 3⍴⍳6": ["⍉2 3⍴1 2 3 4 5 6", "3 2⍴1 4,2 5,3 6", "3 2⍴1 4,2 5 3 6", "3 2⍴1 4 2 5 3 6"],
    "1 2∘.×1 2 3": ["2 3⍴(1×1 2 3),2×1 2 3", "2 3⍴(1×1 2 3),(2×1)(2×2)(2×3)", "2 3⍴(1×1 2 3),(2×1)(2×2)6", "2 3⍴(1×1 2 3),(2×1)4 6", "2 3⍴(1×1 2 3),2 4 6", "2 3⍴((1×1)(1×2)(1×3)),2 4 6", "2 3⍴((1×1)(1×2)3),2 4 6", "2 3⍴((1×1)2 3),2 4 6", "2 3⍴1 2 3,2 4 6", "2 3⍴1 2 3 2 4 6"],
    "≢¨(1 2)(3 4 5)": ["(≢1 2)(≢3 4 5)", "(≢1 2)3", "2 3"],
}


def test_known_traces():
    for program, steps in EXPECTED.items():
        assert trace(program) == steps, (program, trace(program))


def test_every_step_has_the_programs_value():
    for program in CANONICAL + HARD + CONTROL + PRIMS + BITS:
        want = Interpreter().run(program)
        for state in trace(program):
            assert same_array(Interpreter().run(state), want), (program, state)


def test_trace_ends_with_canonical_result():
    for program in CANONICAL + HARD + CONTROL + PRIMS + BITS:
        text = trace_text(program)
        assert final_answer(text) == to_apl(Interpreter().run(program)), program
        # Joined with ⋄ the whole trace is itself an APL program with the same value.
        assert same_array(Interpreter().run(text), Interpreter().run(program)), program


def test_generated_rows_carry_valid_traces():
    rows = list(generate(300, seed=11))
    traced = [r for r in rows if r["trace"]]
    assert len(traced) >= 0.95 * len(rows)
    for row in traced:
        assert final_answer(row["trace"]) == row["result"], row
        assert STEP_SEP not in row["result"]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   %s" % name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL %s: %r" % (name, exc))
    sys.exit(1 if failed else 0)
