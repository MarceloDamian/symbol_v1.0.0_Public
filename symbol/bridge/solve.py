"""
English problem -> APL (bits and arrays) -> exact answer.

Words only at the edges: Claude turns the problem into an APL program, our
interpreter runs it exactly (Claude may test drafts with `run_apl`), and the
verified program is logged to data/bridge_solved.jsonl, which later becomes
training data so our own glyph model can learn to write these programs.

    .venv/bin/python -m symbol.bridge.solve "Which of the coins 3 5 7 11 add up to exactly 15?"
    .venv/bin/python -m symbol.bridge.solve --glyph-model checkpoints/apl_trace_v8.pt "..."

Needs ANTHROPIC_API_KEY: in the environment, a line in symbol/.env, or ~/Keys/api-keys/anthropic.key.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from typing import Any, Dict, Optional, Tuple

import anthropic

from symbol.sandbox.apl_interp import APLError, Interpreter
from symbol.transforms.format import FormatError, to_apl

MODEL = "claude-opus-5-5"
MAX_TURNS = 12
LOG_PATH = "data/bridge_solved.jsonl"

SYSTEM = """You solve problems by writing APL, which a strict interpreter runs exactly.

Dialect (a Dyalog-style subset): index origin 1, right-to-left evaluation, statements joined
with ⋄, assignment A←..., dfns {⍺ ⍵} with guards (cond:result⋄...) and ∇ for recursion.
Functions: + - × ÷ | ⌈ ⌊ * ⍟ ! ○ = ≠ < > ≤ ≥ ∧ ∨ ⍲ ⍱ ~ ⍳ ⍴ ≢ ≡ , ⌽ ⍉ ↑ ↓ ⊂ ⊃ ∊ ∪ ∩ ⍋ ⍒ ⍸ ⊥ ⊤ ⍷ ⌷ ⊢ ⊣,
bracket indexing X[i] and M[i;j].
Operators: / ⌿ (reduce, or replicate with an array left), \\ ⍀ (scan), ¨ ⍨ ∘ ∘.f f.g ⍣.
X/⍨mask and M⌿⍨mask keep the items or rows a mask selects. Arrays are capped at 1,000,000 cells.
Not available: ⎕ system names, ⌸, ⍤ with non-scalar ranks, :If/:For control structures.

Think in bits and whole arrays, not loops. For a choice among n yes/no options (n ≤ 20):
  C←⍉(n⍴2)⊤¯1+⍳2*n        every combination as a row of n bits
  C⌿⍨T=C+.×W               keep the rows whose weighted sum is T
  ⊃⍒V or (V⍳⌈/V)           position of the best score
Encode, filter with boolean masks, and decode results with ⊥/⊤ when numbers are bit patterns.

Use run_apl to test drafts as often as you need: it returns the exact result or the error.
When a program gives the right answer, call submit with the final program and a one-sentence
plain-English answer that reads the result back. The program must compute the answer itself;
never hard-code the answer."""

TOOLS = [
    {
        "name": "run_apl",
        "description": "Run an APL program on the interpreter. Returns the result as canonical "
                       "APL text, or the error message.",
        "input_schema": {
            "type": "object",
            "properties": {"program": {"type": "string", "description": "APL source"}},
            "required": ["program"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit",
        "description": "Submit the final APL program and a plain-English answer. The program is "
                       "re-run to verify it before the answer is shown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "the final APL program"},
                "answer": {"type": "string", "description": "one sentence reading the result back"},
            },
            "required": ["program", "answer"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def run_apl(program: str) -> Tuple[bool, str]:
    """(ok, text): canonical APL result, or the error."""
    try:
        return True, to_apl(Interpreter(step_budget=200_000).run(program))
    except (APLError, FormatError, RecursionError, ZeroDivisionError, ValueError) as exc:
        return False, "error: %s" % exc


def solve(problem: str, client: anthropic.Anthropic, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Let Claude write and test APL until it submits a program that runs."""
    messages: list = [{"role": "user", "content": problem}]
    for _ in range(MAX_TURNS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            print("declined: %s" % (response.stop_details.category if response.stop_details else "no reason"))
            return None
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:  # auto tool choice may answer in text; ask for the tool explicitly
            messages.append({"role": "user", "content": "Please call submit with the final program."})
            continue

        results, submitted = [], None
        for call in calls:
            ok, text = run_apl(call.input["program"])
            if verbose:
                print("  %s %s\n      -> %s" % ("run" if call.name == "run_apl" else "submit",
                                              call.input["program"], text))
            if call.name == "submit" and ok:
                submitted = {"program": call.input["program"], "result": text, "answer": call.input["answer"]}
                text = "verified: " + text
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": not ok})
        if submitted:
            return submitted
        messages.append({"role": "user", "content": results})
    print("no verified program after %d turns" % MAX_TURNS)
    return None


def glyph_model_check(program: str, ckpt: str) -> str:
    """Ask our own glyph model to evaluate the program (CPU, so a training GPU is left alone)."""
    import torch

    from symbol.training.predict import load
    from symbol.transforms.trace import final_answer

    model, tok, _ = load(ckpt, torch.device("cpu"))
    ids = [i if i < model.cfg.vocab_size else tok.unk_id for i in tok.encode_prompt(program, strict=False)]
    if len(ids) >= model.cfg.max_len - 8:
        return "program too long for the glyph model (%d tokens)" % len(ids)
    gen = model.generate(torch.tensor([ids]), stop_id=tok.eos_id, max_new=model.cfg.max_len - len(ids))[0].tolist()
    return final_answer(tok.decode(gen[: gen.index(tok.eos_id)] if tok.eos_id in gen else gen))


ENV_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".env")  # symbol/.env, git-ignored


def load_env_file(path: str = ENV_FILE) -> None:
    """Read KEY=VALUE lines from symbol/.env into the environment, without overriding it."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


KEY_FILE = os.path.expanduser("~/Keys/api-keys/anthropic.key")  # one line: the key, next to the user's other keys


def load_key_file(path: str = KEY_FILE) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            os.environ["ANTHROPIC_API_KEY"] = fh.read().strip()


def main(argv=None) -> None:
    load_env_file()
    load_key_file()
    parser = argparse.ArgumentParser(description="Solve an English problem with APL")
    parser.add_argument("problem")
    parser.add_argument("--glyph-model", help="also ask this checkpoint to evaluate the program")
    parser.add_argument("--quiet", action="store_true", help="hide the drafts Claude tests")
    args = parser.parse_args(argv)

    found = solve(args.problem, anthropic.Anthropic(), verbose=not args.quiet)
    if not found:
        raise SystemExit(1)
    print("\nanswer:  %s\nprogram: %s\nresult:  %s   (computed by the interpreter)" % (
        found["answer"], found["program"], found["result"]))
    if args.glyph_model:
        guess = glyph_model_check(found["program"], args.glyph_model)
        print("glyph model: %s  %s" % (guess, "✓ agrees" if guess == found["result"] else "✗ differs"))

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"problem": args.problem, **found, "model": MODEL,
                             "time": datetime.datetime.now().isoformat(timespec="seconds")},
                            ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
