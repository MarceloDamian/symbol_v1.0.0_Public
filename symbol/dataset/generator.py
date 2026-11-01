"""
Random APL program generator with interpreter-verified results.

Produces (program, result) pairs where `result` is canonical APL source for
the program's value (see symbol.transforms.format). Every pair is checked:
the program must evaluate, and the rendered result must evaluate back to the
same array. Anything that errors, explodes in size, or fails the roundtrip is
rejected and counted in `stats`.

CLI:
    python -m symbol.dataset.generator --n 60000 --out data/apl_eval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
from collections import Counter
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

from symbol.dataset.canonical import BITS, CANONICAL, CONTROL, HARD, PRIMS
from symbol.sandbox.apl_interp import APLArray, APLError, Interpreter
from symbol.transforms.format import FormatError, same_array, to_apl
from symbol.transforms.trace import STEP_SEP, TraceError, trace

# Primitive pools. Kept to what the sandbox interpreter implements.
MONADIC_FNS = ["-", "|", "⌽", "≢", ",", "⊃", "⍋", "⍒", "∪", "×", "⌈", "⌊", "~", "≡", "⍴", "⊂", "⍉", "⍸", "≠"]
SCALAR_DYADS = ["+", "-", "×", "⌈", "⌊", "|", "=", "≠", "<", ">", "≤", "≥", "∧", "∨", "÷", "*"]
STRUCT_DYADS = [",", "⍴", "↑", "↓", "⌽", "∊", "~", "⍳", "/", "≡", "≢", "∩", "⍷", "⊥"]
REDUCIBLE = ["+", "×", "⌈", "⌊", "-", "∧", "∨"]
WORDS = [
    "apl", "abc", "hello", "data", "array", "rank", "glyph", "x", "ab", "world",
    "iverson", "tensor", "token", "model", "vector", "matrix", "a b c", "lambda",
]

# Fixed shapes of everyday APL with random fillers. Random grammar sampling
# alone almost never produces `+/⍳10` or `2 3⍴⍳6`, so these guarantee coverage.
#   {n} 1..12   {k} 0..9   {v} int vector   {w} word   {f} reducible fn
#   {g} scalar dyad   {m} monadic fn   {r} {c} small dims, {rc} = r×c
TEMPLATES = [
    "{f}/⍳{n}", "{f}/{v}", "{f}\\{v}", "{f}\\⍳{n}", "{f}⌿{r} {c}⍴⍳{rc}", "{f}/{r} {c}⍴⍳{rc}",
    "{r} {c}⍴⍳{rc}", "{r} {c}⍴{v}", "⍉{r} {c}⍴⍳{rc}", "⍴{r} {c}⍴{v}", "{r} {c}⍴'{w}'",
    "⌽'{w}'", "≢'{w}'", "∪'{w}'", "⌽{v}", "{k}⌽{v}", "{k}↑{v}", "{k}↓{v}", "¯{k}↑{v}",
    "(+/÷≢){v}", "(⌈/-⌊/){v}", "{v}∘.{g}{v}", "{v}+.×{v}", "{v}{g}{v}", "{k}{g}{v}",
    "{v}∊{v}", "{v}~{v}", "{v}⍳{k}", "⍋{v}", "⍒{v}", "{m}{v}", "{m}⍳{n}",
    "{{⍵{g}{k}}}{v}", "{k}{{⍺{g}⍵}}{v}", "{{{f}/⍵}}{v}", "({k}∘{g}){v}", "{k}{g}⍨{v}",
    "{v}/⍳{n}", "{k}{g}{k}{g}{v}", "({k}{g}{k}){g}{v}", "⊃{m}{v}",
    # Matrices filled from random vectors: `⍳{rc}` alone gives only 16 distinct shapes.
    "⍉{r} {c}⍴{v}", "{f}⌿{r} {c}⍴{v}", "{f}/{r} {c}⍴{v}", "{m}¨{v}", "{v}/{v}",
    # Round 5: boolean masks, nested inputs, more transposes.
    "{mask}", "+/{mask}", "{m}¨{nest}", "≢¨{nest}", "+/¨{nest}", "⌽¨{nest}", "≢{nest}",
    "⊃{nest}", "⍉{r} {c}⍴⍳{n}", "⍴⍉{r} {c}⍴{v}",
    # Round 6: bracket indexing ({i} is a small index, 1-4).
    "{v}[{i}]", "{v}[{i} {i}]", "(⍳{n})[{i} {i}]", "'{w}'[{i} {i}]", "{f}/{v}[{i} {i}]",
    "({r} {c}⍴{v})[{i};{i}]", "({r} {c}⍴{v})[{i};]", "({r} {c}⍴{v})[;{i}]",
    "{{⍵[⍋⍵]}}{v}", "{{⍵[⍒⍵]}}{v}", "A←{v}⋄A[{i}]{g}{k}", "{v}[⍋{v}]", "{{⍵[{i}]}}{v}",
    # Round 6: strings, dfn operands, power, precedence, ravel.
    "+/'{w}'='{ch}'", "'{w}'='{ch}'", "'{w}'∊'{w}'", "'{w}','{w}'", "⌽'{w}','{w}'",
    "{{⍺{g}⍵}}/{v}", "{{⍵{g}{k}}}¨{v}", "{{⍵{g}⍵}}¨{v}", "{f}/(⍳{n})*{i}", "(⍳{n})*{i}",
    "+/⍳{i}{g}{i}", "⍳{i}{g}{i}", ",⍉{r} {c}⍴{v}", ",{r} {c}⍴{v}", "{k}↓⌽{v}",
    "{{(⍵≠{k})/⍵}}{v}", "{{⍵/⍳≢⍵}}{bits}", "+/0={i}|⍳{n}",
    # Round 8: control flow. Guards pick a branch, ∇ recurses, ⍣ repeats.
    "{{⍵{cmp}{k}:{k}⋄⍵{g}{k}}}{k}", "{{⍵{cmp}{k}:⍵⋄{m}⍵}}{k}", "{{⍵{cmp}{k}:'{w}'⋄'{w}'}}{k}",
    "{{⍵{cmp}{k}:0⋄⍵}}¨{v}", "{{⍺{cmp}⍵:⍺⋄⍵}}/{v}", "{{0=2|⍵:⍵÷2⋄⍵{g}{k}}}{n}",
    "{{⍵≤1:{k}⋄⍵{g}∇⍵-1}}{i}", "{{⍵=0:{k}⋄⍵{g}∇⍵-1}}{i}", "{{⍵<{i}:⍵⋄∇⍵-{i}}}{n}",
    "{k}{{0=⍵:⍺⋄⍵∇⍵|⍺}}{k}", "{{⍵≤0:⍬⋄⍵,∇⍵-1}}{i}", "{{⍵≤1:⍵⋄∇⍵÷2}}{n}",
    "{{⍵≤{i}:{k}⋄⍵{g}∇⍵-{i}}}{n}", "{k}{{⍵≤0:⍺⋄⍺{g}⍺∇⍵-1}}{i}",  # more varied recursion
    "({m}⍣{i}){v}", "({{⍵{g}{k}}}⍣{i}){k}", "{k}({g}⍣{i}){k}", "({g}⍨⍣{i}){i}",
    # Round 10: newer primitives. ⍸ where, ∩ intersection, ⊥ decode, ⊤ encode, ⍷ find, ≠ nub sieve, ! binomial.
    "⍸{bits}", "+/⍸{bits}", "≢⍸{bits}", "⍸0=2|{v}", "{v}∩{v}", "'{w}'∩'{w}'", "(⍳{n})∩{v}",
    "2⊥{bits}", "10⊥{v}", "{n} {n}⊥{k} {k}", "2⊥⌽{bits}", "2 2 2⊤{k}", "2 2 2 2⊤{n}",
    "10 10⊤{n}", "0 {n}⊤{n}", "{i}!{n}", "{k}!{k}", "{i}!⍳{i}", "≠{v}", "≠'{w}'",
    "{{(≠⍵)/⍵}}{v}", "{v}⍷{v}", "'{ch}{ch}'⍷'{w}'", "+/'{ch}'⍷'{w}'",
    # Round 12: two-digit arithmetic ({nn} 10-99), indexes past 4 ({j}), two-digit vectors ({vv}),
    # mixed strands 1 2(3 4), and ⍳-filled matrices indexed and ravelled.
    "{nn}-{k}", "{nn}{g}{k}", "{nn}{g}{nn}", "{nn}-{nn}", "{k}|{nn}", "{n}|{nn}", "{nn}|{nn}",
    "{{⍵<{n}:⍵⋄∇⍵-{n}}}{nn}", "{{⍵{cmp}{nn}:'{w}'⋄'{w}'}}{nn}", "{{0=2|⍵:'{w}'⋄'{w}'}}{nn}",
    "'{w}'[{j}]", "'{w}'[{j} {j}]", "{v}[{j} {j}]", "{vv}[{i}]", "{vv}[{i} {i}]",
    "A←{vv}⋄A[{i} {i}]", "A←'{w}'⋄A[{j}]", "+/{vv}", "{vv}{g}{k}", "⌽{vv}",
    "≢{v}({v})", "≢{k}({v}){k}", "≢({v}){v}", "{v}({v})", "⍴{k} {k}({v})",
    "({r} {c}⍴⍳{rc})[;{i}]", "({r} {c}⍴⍳{rc})[{i};]", "+/,{r} {c}⍴⍳{rc}", ",{r} {c}⍴⍳{rc}",
    # Round 13: the binary pieces the bridge's programs are built from. Encode to bits, count and
    # flip them, combine bit vectors, select with /⍨ ⌿⍨, weigh rows with +.×, and small bit searches.
    "(3⍴2)⊤{k}", "(4⍴2)⊤{n}", "2 2 2⊤{k} {k}", "⍉2 2 2⊤{k} {k}", "+/(4⍴2)⊤{n}", "+⌿2 2 2⊤{k} {k}",
    "2⊥⌽(4⍴2)⊤{n}", "2⊥~(4⍴2)⊤{n}", "~(3⍴2)⊤{k}", "¯1+⍳{n}", "2⊥⍉2 2⍴{b} {b} {b} {b}",
    "{b} {b} {b}∧{b} {b} {b}", "{b} {b} {b} {b}∨{b} {b} {b} {b}", "{b} {b} {b} {b}≠{b} {b} {b} {b}",
    "2⊥{b} {b} {b}∨{b} {b} {b}", "2⊥{b} {b} {b}≠{b} {b} {b}",
    "{k} {k} {k}/⍨{b} {b} {b}", "'{ch}{ch}{ch}'/⍨{b} {b} {b}", "{{⍵/⍨0=2|⍵}}⍳{n}", "{nn} {nn} {nn}/⍨{b} {b} {b}",
    "(3 2⍴{v})⌿⍨{b} {b} {b}", "(2 {c}⍴{v})⌿⍨{b} {b}", "M←3 2⍴{v}⋄M⌿⍨{b} {b} {b}",
    "(2 3⍴{b} {b} {b} {b} {b} {b})+.×{k} {k} {k}", "(3 2⍴{v})+.×{k} {k}", "{b} {b} {b}+.×{k} {k} {k}",
    "C←⍉(2⍴2)⊤¯1+⍳4⋄C+.×{k} {k}", "C←⍉(2⍴2)⊤¯1+⍳4⋄C⌿⍨{k}=C+.×{k} {k}",
    "C←⍉(3⍴2)⊤¯1+⍳8⋄C+.×{k} {k} {k}", "C←⍉(3⍴2)⊤¯1+⍳8⋄+/{k}=C+.×{k} {k} {k}",
    "C←⍉(3⍴2)⊤¯1+⍳8⋄⍸{nn}=C+.×{k} {k} {k}", "C←⍉(2⍴2)⊤¯1+⍳4⋄⍸{k}=C+.×{k} {k}", "⊃⍒{v}",
    # Round 14: round 13's misses. Guards on negative numbers, gcd past 9, counting and comparing
    # longer vectors, two-row +.× (now traced row by row), and more ⊤ ⍷ ! (now spelled out).
    "{{⍵{cmp}0:⍵⋄-⍵}}¯{k}", "{{⍵{cmp}0:⍵⋄-⍵}}{k}", "{{⍵{cmp}0:'{w}'⋄'{w}'}}¯{k}", "{{⍵<0:-⍵⋄⍵}}¨¯{k} {k} ¯{k}",
    "-{v}", "|¯{k} {k} ¯{k}", "{{⍵{cmp}{k}:0⋄⍵}}¨{k} {k} {k}", "{{⍵{cmp}{k}:⍵⋄0}}¨{v}",
    "{nn}{{0=⍵:⍺⋄⍵∇⍵|⍺}}{k}", "{nn}{{0=⍵:⍺⋄⍵∇⍵|⍺}}{nn}", "{k}{{0=⍵:⍺⋄⍵∇⍵|⍺}}{nn}",
    "≢{v} {v}", "≢⍸{bits} {bits}", "+/{bits} {bits}", "{k}={v} {v}", "{k}<{v} {v}", "⍸{k}={v} {v}",
    "(2 2⍴{v})+.×{k} {k}", "(2 2⍴{v})+.×{b} {b}", "(2 2⍴{b} {b} {b} {b})+.×{nn} {nn}",
    "{n} {n}⊤{nn}", "10 10⊤{nn}", "0 {n}⊤{nn}", "2 2 2⊤{k}", "{i}!{nn}", "{v}⍷{v} {v}", "'{ch}{ch}'⍷'{w}'",
    "2⊥2 {c}⍴{bits} {bits} {bits}",
    # Round 15: basics round 14 lost (~ without, ∪ unique, row / column indexing, ∊ counts) and
    # (n⍴2)⊤ written with the reshape, as the bridge writes it.
    "{v}~{k}", "{v}~{k} {k}", "(⍳{n})~{v}", "{v} {v}~{v}", "'{w}'~'{ch}'", "≢{v}~{k}",
    "∪{v} {v}", "∪{k} {k} {k} {k}", "≢∪{v}", "∪⌽{v}", "+/{v}∊{v}", "+/{v}∊{k}", "{v}∊{k}",
    "+/({r} {c}⍴⍳{rc})[{i};]", "+/({r} {c}⍴⍳{rc})[;{i}]", "({r} {c}⍴⌽⍳{rc})[{i};]", "{r} {c}⍴⌽⍳{rc}",
    "({r} {c}⍴{vv})[{i};]", "M←{r} {c}⍴⍳{rc}⋄M[;{i}]", "M←{r} {c}⍴⍳{rc}⋄M[{i};]",
    "({i}⍴2)⊤{k}", "({i}⍴2)⊤{n}", "2⊥⌽({i}⍴2)⊤{k}", "2⊥~({i}⍴2)⊤{k}", "+/({i}⍴2)⊤{n}", "~({i}⍴2)⊤{k}",
    "(2⍴10)⊤{nn}", "({i}⍴2)⊤{k} {k}", "2⊥({i}⍴2)⊤{n}",
    # Round 16: arithmetic round 15 slipped inside otherwise correct traces (2|even, ÷ by 4 and 8,
    # 3-digit ÷ in spelled-out !), ≡ windows that do match, and 2×3 +.× (fits the 384 cap).
    "2|{nn}", "2|{n}", "2|⌊{nn}÷{i}", "⌊{nn}÷{i}", "{nn}÷2", "{nn}÷4", "{nn}÷8", "⌊{nn}÷8",
    "{nn}×{k}", "{nn}×{n}", "{n}×{n}×{n}", "{nn}+{nn}+{nn}", "+/{k} {k} {k} {k}",
    "3!{n}", "4!{n}", "2!{nn}", "3!{nn}", "(4⍴2)⊤{nn}", "+/(4⍴2)⊤{nn}",
    "{b} {b}≡{b} {b}", "{b} {b} {b}≡{b} {b} {b}", "{b} {b}⍷{b} {b} {b} {b} {b}", "1 {b}⍷{b} 1 {b} 1 {b}",
    "{i} {i}⍷{i} {i} {i} {i} {i}", "'{ch}{ch}'≡'{ch}{ch}'", "{nn}{{0=⍵:⍺⋄⍵∇⍵|⍺}}{n}",
    "(2 3⍴{b} {b} {b} {b} {b} {b})+.×{n} {n} {n}", "(2 3⍴{k} {k} {k} {k} {k} {k})+.×{b} {b} {b}",
    # Round 17: string windows that match ({ab} is a, b or n, so 'ab' turns up inside the word),
    # residues by two-digit numbers, 2⊥ of ⌽/~ of 4 bits (fits the 416 cap), and ⍸ of a count.
    "'{ab}{ab}'⍷'{ab}{ab}{ab}{ab}{ab}{ab}'", "'{ab}{ab}'⍷'{ab}{ab}{ab}{ab}{ab}'", "'{ab}{ab}{ab}'⍷'{ab}{ab}{ab}{ab}{ab}'",
    "+/'{ab}{ab}'⍷'{ab}{ab}{ab}{ab}{ab}{ab}'", "'{ab}{ab}'≡'{ab}{ab}'", "'{ab}'='{ab}{ab}{ab}{ab}{ab}'",
    "{nn}|{vv}", "{n}|{vv}", "({nn}|{nn})+{k}", "{nn}|{nn}×{k}",
    "2⊥⌽(4⍴2)⊤{nn}", "2⊥~(4⍴2)⊤{nn}", "2⊥~(4⍴2)⊤{n}", "⍸{n}", "≢⍸{n}", "⍸⌈{n}÷2",
]

# Hand-written idioms that random sampling rarely hits but real APL uses a lot.
IDIOM_FNS = [
    "(+/÷≢)",      # mean
    "(⌈/-⌊/)",     # range
    "(+/×⍨)",      # sum of squares: +/ (⍵×⍵)
    "(⊢/)",        # last
    "(⊣/)",        # first
    "{(⍋⍵)⌷⍵}",    # sort ascending
    "{(⍵≠' ')/⍵}",  # remove blanks
    "{+/⍵=⌈/⍵}",   # count of maxima
    "(≢∪)",        # number of distinct
    "{⌽⍵}",
]

MAX_PROGRAM_LEN = 48
MAX_RESULT_LEN = 48
MAX_TRACE_LEN = 416  # round 14: spelled-out ⊤ ⍷ ! (320); 16: 2×3 +.× (384); 17: 2⊥⌽(4⍴2)⊤nn (416)
MAX_RESULT_CELLS = 24
STEP_BUDGET = 5_000
TIME_LIMIT_S = 1.0


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float):
    """Wall-clock backstop for slow paths the step budget does not cover."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def on_alarm(signum, frame):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ProgramGenerator:
    """Grammar-driven sampler. Depth bounds the nesting of expressions."""

    def __init__(self, seed: int = 0, max_depth: int = 3):
        self.rng = random.Random(seed)
        self.max_depth = max_depth

    # -- leaves -----------------------------------------------------------

    def _int(self) -> str:
        r = self.rng.random()
        if r < 0.75:
            return str(self.rng.randint(0, 9))
        if r < 0.9:
            return "¯" + str(self.rng.randint(1, 5))
        return str(self.rng.randint(10, 20))

    def atom(self, leaves: Tuple[str, ...] = ()) -> str:
        if leaves and self.rng.random() < 0.6:
            return self.rng.choice(leaves)
        r = self.rng.random()
        if r < 0.3:
            return self._int()
        if r < 0.65:
            return " ".join(self._int() for _ in range(self.rng.randint(2, 5)))
        if r < 0.88:
            return "⍳" + str(self.rng.randint(1, 12))
        return "'" + self._word() + "'"

    def left_arg(self, depth: int, leaves: Tuple[str, ...]) -> str:
        """Left arguments bind short, so anything compound needs parentheses."""
        if depth > 0 and self.rng.random() < 0.2:
            return "(" + self.expr(depth - 1, leaves) + ")"
        a = self.atom(leaves)
        return "(" + a + ")" if a.startswith("⍳") else a

    # -- functions --------------------------------------------------------

    def monadic_fn(self, depth: int) -> str:
        r = self.rng.random()
        if r < 0.40:
            return self.rng.choice(MONADIC_FNS)
        if r < 0.58:
            return self.rng.choice(REDUCIBLE) + self.rng.choice(["/", "/", "\\"])
        if r < 0.66:
            return self.rng.choice(MONADIC_FNS + SCALAR_DYADS[:3]) + "¨"
        if r < 0.72:
            return self.rng.choice(SCALAR_DYADS[:5] + [","]) + "⍨"
        if r < 0.78:
            return self.rng.choice(MONADIC_FNS) + "∘" + self.rng.choice(MONADIC_FNS)
        if r < 0.83:
            return "(" + self._int() + "∘" + self.rng.choice(SCALAR_DYADS[:5]) + ")"
        if r < 0.92 and depth > 0:
            return "{" + self.expr(depth - 1, ("⍵",)) + "}"
        if r < 0.96:
            return self.rng.choice(IDIOM_FNS)
        f, h = self.rng.choice(MONADIC_FNS), self.rng.choice(MONADIC_FNS)
        return "(" + f + self.rng.choice(SCALAR_DYADS + [","]) + h + ")"

    def dyadic_fn(self, depth: int) -> str:
        r = self.rng.random()
        if r < 0.55:
            return self.rng.choice(SCALAR_DYADS)
        if r < 0.80:
            return self.rng.choice(STRUCT_DYADS)
        if r < 0.86:
            return self.rng.choice(SCALAR_DYADS[:5] + STRUCT_DYADS[:3]) + "⍨"
        if r < 0.91:
            return "∘." + self.rng.choice(SCALAR_DYADS[:5] + ["="])
        if r < 0.94:
            return "+.×"
        if depth > 0:
            return "{" + self.expr(depth - 1, ("⍺", "⍵")) + "}"
        return self.rng.choice(SCALAR_DYADS)

    # -- expressions ------------------------------------------------------

    def expr(self, depth: int, leaves: Tuple[str, ...] = ()) -> str:
        if depth <= 0:
            return self.atom(leaves)
        r = self.rng.random()
        if r < 0.15:
            return self.atom(leaves)
        if r < 0.55:
            return self.monadic_fn(depth) + self.expr(depth - 1, leaves)
        return self.left_arg(depth, leaves) + self.dyadic_fn(depth) + self.expr(depth - 1, leaves)

    def template(self) -> str:
        r, c = self.rng.randint(1, 4), self.rng.randint(1, 4)
        fill = {
            "n": str(self.rng.randint(1, 12)),
            "k": str(self.rng.randint(0, 9)),
            "v": " ".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(2, 6))),
            "w": self.rng.choice(WORDS),
            "f": self.rng.choice(REDUCIBLE),
            "g": self.rng.choice(SCALAR_DYADS),
            "m": self.rng.choice(MONADIC_FNS),
            "r": str(r), "c": str(c), "rc": str(r * c),
        }
        # Each placeholder gets a fresh value per occurrence, e.g. two different {v}.
        out = self.rng.choice(TEMPLATES)
        for key in ("rc", "mask", "nest", "bits", "b", "nn", "n", "k", "i", "j", "vv", "v", "w", "ch", "ab", "cmp", "f", "g", "m",
                    "r", "c"):
            while "{" + key + "}" in out:
                value = fill[key] if key in ("r", "c", "rc") else self._fresh(key)
                out = out.replace("{" + key + "}", value, 1)
        return out.replace("{{", "{").replace("}}", "}")

    def _word(self) -> str:
        """Half real words, half random letters: a fixed list gets memorised instead of copied."""
        if self.rng.random() < 0.5:
            return self.rng.choice(WORDS)
        return "".join(self.rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(self.rng.randint(2, 6)))

    def _fresh(self, key: str) -> str:
        if key == "n":
            return str(self.rng.randint(1, 12))
        if key == "k":
            return str(self.rng.randint(0, 9))
        if key == "i":
            return str(self.rng.randint(1, 4))
        if key == "ch":
            return self.rng.choice("aeiloprstx")
        if key == "ab":  # three letters only, so short strings repeat and windows match
            return self.rng.choice("abn")
        if key == "cmp":
            return self.rng.choice("<>≤≥=≠")
        if key == "b":  # one bit, for bit vectors of a set length: {b} {b} {b}
            return self.rng.choice("01")
        if key == "bits":
            return " ".join(self.rng.choice("01") for _ in range(self.rng.randint(2, 6)))
        if key == "v":
            return " ".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(2, 6)))
        if key == "w":
            return self._word()
        if key == "nn":
            return str(self.rng.randint(10, 99))
        if key == "j":  # indexes past 4, for longer strings and vectors
            return str(self.rng.randint(1, 6))
        if key == "vv":  # two-digit vector: 10 20 30
            return " ".join(str(self.rng.randint(10, 99)) for _ in range(self.rng.randint(2, 4)))
        if key == "f":
            return self.rng.choice(REDUCIBLE)
        if key == "mask":  # `1 0 1/4 5 6`: a 0/1 mask and values of the same length
            size = self.rng.randint(2, 6)
            mask = " ".join(self.rng.choice("01") for _ in range(size))
            return mask + "/" + " ".join(str(self.rng.randint(0, 9)) for _ in range(size))
        if key == "nest":  # `(1 2)(3 4 5)`: a strand of 2-3 short vectors
            return "".join("(" + self._fresh("v") + ")" for _ in range(self.rng.randint(2, 3)))
        if key == "g":
            return self.rng.choice(SCALAR_DYADS)
        return self.rng.choice(MONADIC_FNS)

    def program(self) -> str:
        """A top-level program always applies at least one function."""
        if self.rng.random() < 0.3:
            return self.template()
        depth = self.rng.randint(1, self.max_depth)
        if self.rng.random() < 0.05:
            name = self.rng.choice("ABXY")
            return name + "←" + self.atom() + "⋄" + self._applied(depth, (name,))
        return self._applied(depth)

    def _applied(self, depth: int, leaves: Tuple[str, ...] = ()) -> str:
        if self.rng.random() < 0.45:
            return self.monadic_fn(depth) + self.expr(depth - 1, leaves)
        return self.left_arg(depth, leaves) + self.dyadic_fn(depth) + self.expr(depth - 1, leaves)


def _cell_count(array: APLArray) -> int:
    total = 0
    for cell in array.data:
        total += _cell_count(cell) if isinstance(cell, APLArray) else 1
    return total


def trace_of(program: str) -> Tuple[Optional[str], str]:
    """Small-step trace joined with ⋄, or (None, reason) when it cannot be built."""
    try:
        with _time_limit(TIME_LIMIT_S):
            text = STEP_SEP.join(trace(program))
    except (_Timeout, TraceError, APLError, FormatError):
        return None, "failed"
    if len(text) > MAX_TRACE_LEN:
        return None, "too_long"
    return text, "ok"


def verify(program: str) -> Tuple[Optional[str], str]:
    """Evaluate and render. Returns (result_source, status)."""
    try:
        with _time_limit(TIME_LIMIT_S):
            return _verify(program)
    except _Timeout:
        return None, "timeout"


def _verify(program: str) -> Tuple[Optional[str], str]:
    if len(program) > MAX_PROGRAM_LEN:
        return None, "program_too_long"
    try:
        value = Interpreter(step_budget=STEP_BUDGET).run(program)
    except APLError:
        return None, "apl_error"
    except (ArithmeticError, ValueError, TypeError, IndexError, RecursionError):
        return None, "python_error"
    if value is None:
        return None, "no_value"
    if _cell_count(value) > MAX_RESULT_CELLS:
        return None, "result_too_big"
    try:
        result = to_apl(value)
    except FormatError:
        return None, "unformattable"
    if len(result) > MAX_RESULT_LEN:
        return None, "result_too_long"
    try:
        back = Interpreter(step_budget=STEP_BUDGET).run(result)
    except Exception:  # noqa: BLE001 - any failure here means the render is bad
        return None, "roundtrip_error"
    if not same_array(value, back):
        return None, "roundtrip_mismatch"
    return result, "ok"


def split_of(program: str, val_fraction: float = 0.05) -> str:
    """Deterministic train/val split by program hash, stable across runs."""
    digest = hashlib.sha1(program.encode("utf-8")).digest()
    return "val" if digest[0] < 256 * val_fraction else "train"


def generate(n: int, seed: int = 0, max_attempts_factor: int = 50) -> Iterator[Dict[str, str]]:
    """Yield up to `n` unique verified examples. `stats` is attached to the generator."""
    gen = ProgramGenerator(seed=seed)
    seen = set(CANONICAL) | set(HARD) | set(CONTROL) | set(PRIMS) | set(BITS)  # held out: the benchmarks must stay unseen
    stats: Counter = Counter()
    generate.stats = stats  # type: ignore[attr-defined]
    produced = 0
    for _ in range(n * max_attempts_factor):
        if produced >= n:
            break
        program = gen.program()
        if program in seen:
            stats["duplicate"] += 1
            continue
        seen.add(program)
        result, status = verify(program)
        stats[status] += 1
        if result is None:
            continue
        produced += 1
        steps, trace_status = trace_of(program)
        stats["trace_" + trace_status] += 1
        yield {"program": program, "result": result, "trace": steps, "split": split_of(program)}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/apl_eval.jsonl")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = Counter()
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in generate(args.n, seed=args.seed):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count[row["split"]] += 1

    stats = generate.stats  # type: ignore[attr-defined]
    attempts = sum(v for k, v in stats.items() if not k.startswith("trace_"))
    print("wrote %d examples to %s (train %d, val %d)" % (
        sum(count.values()), args.out, count["train"], count["val"]))
    print("acceptance: %.1f%% of %d attempts" % (100.0 * stats["ok"] / max(attempts, 1), attempts))
    for status, num in stats.most_common():
        print("  %-20s %d" % (status, num))


if __name__ == "__main__":
    main()
