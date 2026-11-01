# Symbol: Semantic Density & Token Optimization Protocol

**Symbol** is an experimental framework designed to maximize semantic density and minimize token consumption for Large Language Models (LLMs). By replacing verbose natural language and boilerplate syntax with a high-density symbolic Intermediate Representation (IR), Symbol aims to:
1. **Drastically reduce token overhead** (targeting 40-70% reduction for repetitive context, code structures, and instructions).
2. **Improve precision and reasoning** by enforcing strict, unambiguous structural hierarchies and dense semantic mappings.
3. **Eliminate syntactic noise** (boilerplate, redundant keywords, verbose framing).

---

## Core Architecture

```
[ Verbose Input / Code ] ──> [ Encoder / Compressor ] ──> [ Dense Symbol Stream ] 
                                                                    │
[ Decoded Output ] <── [ Decoder / Expander ] <─────────────────────┘ (Sent to LLM)
```

1. **Symbol Dictionary (`sym_dict.json`)**: A mapping of frequent idioms, data structures, control flow patterns, and prompt directives to compact symbolic representations.
2. **Structural IR Parser**: Converts code/text into a hierarchical tree of symbols.
3. **LLM Context Adapter**: Prepares prompt blocks using the Symbol grammar so the model processes high-density tokens with lower latency and higher context retention.
4. **Roundtrip Transcoder**: Expands model outputs back into fully readable natural language or code.

---

## APL language model (working pipeline)

A small transformer that reads APL and writes its result. APL is the IR: one
glyph = one concept = one token, so the network sees dense, unambiguous input.

```
APL source ──tokenizer──> glyph ids ──transformer──> result ids ──> APL source
   "+/⍳10"     transforms/   [1,77,…]   training/      […,2]          "55"
```

| Part | File | What it does |
|---|---|---|
| Interpreter | `symbol/sandbox/apl_interp.py` | Pure-Python APL subset. Ground truth for every label. |
| Tokenizer | `symbol/transforms/tokenizer.py` | Lossless glyph-level ids (~167 vocab). Layout `<bos> program <sep> result <eos>`. |
| Formatter | `symbol/transforms/format.py` | Renders any result array back to canonical, re-evaluable APL. |
| Dataset | `symbol/dataset/generator.py` | Grammar sampler; keeps only programs the interpreter verifies. |
| Model | `symbol/training/model.py` | Decoder-only transformer (pre-norm GPT, tied embeddings). |
| Train / eval | `symbol/training/train.py` | Next-token training; scores exact / semantic / valid-APL on held-out programs. |
| Predict | `symbol/training/predict.py` | Ask the model, compare with the interpreter. |

### Small-step traces (best model)

`symbol/transforms/trace.py` rewrites a program one function application at a
time, right to left, and every step is valid APL checked by the interpreter:

```
2×3+4         ⇒  2×7 ⋄ 14
(+/÷≢)2 4 6   ⇒  (+/2 4 6)÷≢2 4 6 ⋄ (+/2 4 6)÷3 ⋄ (2+4+6)÷3 ⋄ (2+10)÷3 ⋄ 12÷3 ⋄ 4
```

Training on traces (`--target trace`) instead of bare answers:

| Model | Held-out exact | Canonical (46 classic programs, never trained on) |
|---|---|---|
| Round 2, answer only | 65.9% | 21/46 |
| Round 3, traces | 84.0% | 36/46 |
| Round 4, matrix / outer / each traces | 90.5% | 43/46 |

`HARD` (44 multi-step programs: indexing, idioms, nested data, also held
out) is the harder benchmark from round 6 on; round 4 scores 18/44 there.
`CONTROL` (15 programs: guards, `∇` recursion, `⍣`) is added in round 8; round 4 scores 1/15.
Score any checkpoint with `.venv/bin/python -m symbol.training.bench <ckpt>…`.

**Best checkpoint: round 16** (`bash scripts/restore.sh 16`, release `round-16`), 133/137 across all five
held-out suites. Round 17 scored 132/137; see [`runs/v17/NOTE.md`](runs/v17/NOTE.md).

| Round | Canonical (46) | Hard (44) | Control (15) | Prims (16) | Bits (16) | Total |
|---|---|---|---|---|---|---|
| 15 | 100% | 100% | 93.3% | 81.2% | 81.2% | 128/137 |
| **16** | **100%** | **95.5%** | **100%** | **93.8%** | **93.8%** | **133/137** |
| 17 | 100% | 95.5% | 100% | 87.5% | 93.8% | 132/137 |

### Run it

```bash
# one-time: deps for training (generator + tests need only Python 3.8+)
python3.12 -m venv .venv && .venv/bin/pip install torch numpy pytest

python3 -m symbol.dataset.generator --n 100000 --out data/apl_eval.jsonl   # ~70 s
.venv/bin/python -m symbol.training.train --data data/apl_eval.jsonl         # Apple GPU / CUDA / CPU
python3 -m symbol.dataset.generator --n 150000 --out data/apl_trace.jsonl  # adds a "trace" field
.venv/bin/python -m symbol.training.train --data data/apl_trace.jsonl --target trace --steps 15000 --out checkpoints/apl_trace.pt
.venv/bin/python -m symbol.training.predict --ckpt checkpoints/apl_trace.pt "2×3+4" "(+/÷≢)2 4 6"

python3 tests/test_apl_interp.py && python3 tests/test_pipeline.py          # or: .venv/bin/python -m pytest tests
```

### Interpreter limits worth knowing
- Bracket indexing `v[i]`, `m[i;j]`, `m[i;]`, dfn guards, `∇` recursion and `⍣` work; no axis brackets `f[k]`. Arrays over 1M cells raise WS FULL.
