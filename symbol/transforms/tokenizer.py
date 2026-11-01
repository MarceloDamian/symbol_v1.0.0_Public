"""
Glyph-level tokenizer: APL source <-> integer ids for a neural network.

APL is already maximally dense: one glyph is one concept (`⍳` is "iota", `/`
is "reduce"). So unlike BPE over English, the natural unit here is the single
character. Every APL glyph, digit and printable ASCII character gets exactly
one id, which makes encoding lossless and the vocabulary small (~200).

Sequence layout used for training (see `encode_pair`):

    <bos> program <sep> result <eos>

where `result` is itself APL source that evaluates to the program's value.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

PAD, BOS, EOS, SEP, UNK = "<pad>", "<bos>", "<eos>", "<sep>", "<unk>"
SPECIALS = [PAD, BOS, EOS, SEP, UNK]

# Order is part of the model contract: changing it invalidates checkpoints.
# Append new glyphs at the end only.
APL_GLYPHS = (
    "←→⋄⍝⍺⍵⍬¯×÷⌈⌊∣⍟○⍳⍴≢≡⍷⍸⌽⊖⍉↑↓⊂⊃⊆⊇∊∪∩~⍋⍒⌷⍕⍎⊥⊤⌹⍪"
    "¨⍨⍣⍤⍥⍢∘⌿⍀⌸⌺⊢⊣≠≤≥∧∨⍲⍱⎕⍞∆⍙"
)
ASCII = "".join(chr(c) for c in range(32, 127))
NEWLINE = "\n"
# Glyphs added after the first checkpoints; always append here so old ids keep their meaning.
LATER_GLYPHS = "∇"


def _build_alphabet() -> List[str]:
    seen = set()
    out = []
    for ch in APL_GLYPHS + ASCII + NEWLINE + LATER_GLYPHS:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


class APLTokenizer:
    """Maps APL text to ids and back. One character = one token."""

    def __init__(self, alphabet: Optional[List[str]] = None):
        self.itos: List[str] = SPECIALS + (alphabet if alphabet is not None else _build_alphabet())
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.sep_id = self.stoi[SEP]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # -- core -------------------------------------------------------------

    def encode(self, text: str, strict: bool = True) -> List[int]:
        """Text to ids. `strict` raises on characters outside the vocabulary."""
        ids = []
        for ch in text:
            idx = self.stoi.get(ch)
            if idx is None:
                if strict:
                    raise ValueError("character %r is not in the APL vocabulary" % ch)
                idx = self.unk_id
            ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> str:
        out = []
        for idx in ids:
            tok = self.itos[idx]
            if tok in SPECIALS:
                if skip_specials:
                    continue
            out.append(tok)
        return "".join(out)

    # -- training layout --------------------------------------------------

    def encode_pair(self, program: str, result: str) -> List[int]:
        """<bos> program <sep> result <eos>"""
        return (
            [self.bos_id] + self.encode(program) + [self.sep_id]
            + self.encode(result) + [self.eos_id]
        )

    def encode_prompt(self, program: str, strict: bool = True) -> List[int]:
        """<bos> program <sep>, ready for the model to continue with the result."""
        return [self.bos_id] + self.encode(program, strict) + [self.sep_id]

    def split_pair(self, ids: List[int]) -> Tuple[str, str]:
        """Inverse of encode_pair; tolerates a missing <eos> (truncated output)."""
        if self.sep_id not in ids:
            return self.decode(ids), ""
        cut = ids.index(self.sep_id)
        rest = ids[cut + 1 :]
        if self.eos_id in rest:
            rest = rest[: rest.index(self.eos_id)]
        return self.decode(ids[:cut]), self.decode(rest)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"itos": self.itos}, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "APLTokenizer":
        with open(path, encoding="utf-8") as fh:
            itos = json.load(fh)["itos"]
        if itos[: len(SPECIALS)] != SPECIALS:
            raise ValueError("vocab file does not start with the expected special tokens")
        return cls(alphabet=itos[len(SPECIALS) :])
