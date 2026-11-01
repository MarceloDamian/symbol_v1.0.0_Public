"""Tests for tokenizer, result formatter, and dataset generator.

Pure Python (no torch). Runs standalone or under pytest:
    python3 tests/test_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbol.dataset.canonical import BITS, CANONICAL, CONTROL, HARD, PRIMS  # noqa: E402
from symbol.dataset.generator import ProgramGenerator, generate, split_of, verify  # noqa: E402
from symbol.sandbox.apl_interp import Interpreter  # noqa: E402
from symbol.transforms.format import same_array, to_apl  # noqa: E402
from symbol.transforms.tokenizer import APLTokenizer  # noqa: E402

ROUNDTRIP_PROGRAMS = [
    "1+2", "⍳5", "2 3⍴⍳6", "(1 2)(3 4)", "2 2⍴⍳4", "'abc'", "≢¨(1 2)(3 4 5)",
    "⊂1 2", "⊂⊂1 2", "÷3", "¯2.5×2", "2 2⍴'abcd'", "⍬", ",5", "(+/÷≢)1 2 3 4",
    "'it''s'", "(1 2)(3)", "1 2∘.×1 2 3", "-\\1 2 3",
]


def test_tokenizer_roundtrip():
    tok = APLTokenizer()
    for src in ROUNDTRIP_PROGRAMS + ["A←1 2 3 ⋄ +/A", "{0=⍵:99 ⋄ ⍵}0"]:
        assert tok.decode(tok.encode(src)) == src, src


def test_one_token_per_glyph():
    tok = APLTokenizer()
    assert len(tok.encode("+/÷≢")) == 4
    assert len(tok.encode("⍳10")) == 3


def test_pair_layout():
    tok = APLTokenizer()
    ids = tok.encode_pair("+/⍳3", "6")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.split_pair(ids) == ("+/⍳3", "6")
    assert tok.encode_prompt("+/⍳3") == ids[: ids.index(tok.sep_id) + 1]


def test_tokenizer_strict_and_save(tmp_path=None):
    tok = APLTokenizer()
    try:
        tok.encode("é")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-vocab char")
    assert tok.encode("é", strict=False) == [tok.unk_id]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "vocab.json")
        tok.save(path)
        assert APLTokenizer.load(path).itos == tok.itos


def test_format_reevaluates_to_same_array():
    for program in ROUNDTRIP_PROGRAMS:
        value = Interpreter().run(program)
        back = Interpreter().run(to_apl(value))
        assert same_array(value, back), (program, to_apl(value))


def test_format_distinguishes_matrix_from_nested():
    assert to_apl(Interpreter().run("2 2⍴⍳4")) == "2 2⍴1 2 3 4"
    assert to_apl(Interpreter().run("(1 2)(3 4)")) == "(1 2)(3 4)"


def test_verify_rejects_errors():
    assert verify("1 2 3+4 5") == (None, "apl_error")
    assert verify("+/⍳10") == ("55", "ok")


def test_generator_is_deterministic_and_valid():
    a = [ProgramGenerator(seed=7).program() for _ in range(1)]
    b = [ProgramGenerator(seed=7).program() for _ in range(1)]
    assert a == b
    rows = list(generate(200, seed=3))
    assert len(rows) == 200
    assert len({r["program"] for r in rows}) == 200
    for row in rows:
        value = Interpreter().run(row["program"])
        assert to_apl(value) == row["result"], row
        assert row["split"] == split_of(row["program"])


def test_canonical_programs_evaluate_and_stay_held_out():
    for program in CANONICAL + HARD + CONTROL + PRIMS + BITS:
        assert verify(program)[1] == "ok", program
    generated = {r["program"] for r in generate(2000, seed=5)}
    assert not generated & (set(CANONICAL) | set(HARD) | set(CONTROL) | set(PRIMS) | set(BITS))


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
