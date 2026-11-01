"""Model sanity tests. Skipped when torch is not installed (system Python).

    .venv/bin/python -m pytest tests/test_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from symbol.training.model import APLTransformer, ModelConfig  # noqa: E402
from symbol.transforms.tokenizer import APLTokenizer  # noqa: E402


def _tiny():
    tok = APLTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_len=32, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    torch.manual_seed(0)
    return tok, APLTransformer(cfg).eval()


def test_forward_shape():
    tok, model = _tiny()
    ids = torch.tensor([tok.encode_pair("+/⍳3", "6")])
    assert model(ids).shape == (1, ids.size(1), tok.vocab_size)


def test_causal_mask():
    """Changing a later token must not change logits at earlier positions."""
    tok, model = _tiny()
    a = torch.tensor([tok.encode_pair("+/⍳3", "6")])
    b = a.clone()
    b[0, -2] = tok.stoi["9"]
    la, lb = model(a), model(b)
    assert torch.allclose(la[0, :-2], lb[0, :-2], atol=1e-5)


def test_generate_respects_max_len():
    tok, model = _tiny()
    prompt = torch.tensor([tok.encode_prompt("⍳3")])
    out = model.generate(prompt, stop_id=tok.eos_id, max_new=100)
    assert prompt.size(1) + out.size(1) <= model.cfg.max_len


def test_overfits_one_example():
    """A few hundred steps on one pair should reproduce it exactly."""
    tok, model = _tiny()
    model.train()
    seq = torch.tensor([tok.encode_pair("+/⍳4", "10")])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        logits = model(seq[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits[0], seq[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = model.generate(torch.tensor([tok.encode_prompt("+/⍳4")]), stop_id=tok.eos_id, max_new=8)[0].tolist()
    assert tok.decode(out[: out.index(tok.eos_id)]) == "10"


def test_cached_generation_matches_full_recompute():
    """KV-cached decoding must pick the same tokens as recomputing every step."""
    tok, model = _tiny()
    prompt = torch.tensor([tok.encode_prompt("+/⍳4")])
    cached = model.generate(prompt, stop_id=-1, max_new=12)[0].tolist()
    out = prompt
    for _ in range(12):
        nxt = model(out)[:, -1].argmax(-1)
        out = torch.cat([out, nxt[:, None]], dim=1)
    assert cached == out[0, prompt.size(1) :].tolist()
