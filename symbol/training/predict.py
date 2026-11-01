"""
Ask a trained checkpoint to evaluate APL, and compare with the interpreter.

    .venv/bin/python -m symbol.training.predict "+/⍳10" "(+/÷≢)2 4 6"
"""

from __future__ import annotations

import argparse

import torch

from symbol.sandbox.apl_interp import APLError, Interpreter
from symbol.training.model import APLTransformer, ModelConfig
from symbol.training.train import pick_device
from symbol.transforms.format import to_apl
from symbol.transforms.tokenizer import SPECIALS, APLTokenizer
from symbol.transforms.trace import final_answer


def load(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    tok = APLTokenizer(alphabet=ckpt["vocab"][len(SPECIALS) :])
    model = APLTransformer(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    trace_mode = ckpt.get("args", {}).get("target") == "trace"
    return model, tok, trace_mode


def predict(model: APLTransformer, tok: APLTokenizer, program: str, device: torch.device) -> str:
    prompt = torch.tensor([tok.encode_prompt(program)], device=device)
    ids = model.generate(prompt, stop_id=tok.eos_id, max_new=model.cfg.max_len - prompt.size(1))[0].tolist()
    if tok.eos_id in ids:
        ids = ids[: ids.index(tok.eos_id)]
    return tok.decode(ids)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("programs", nargs="+")
    parser.add_argument("--ckpt", default="checkpoints/apl_eval.pt")
    args = parser.parse_args(argv)

    device = pick_device()
    model, tok, trace_mode = load(args.ckpt, device)
    for program in args.programs:
        generated = predict(model, tok, program, device)
        guess = final_answer(generated) if trace_mode else generated
        try:
            truth = to_apl(Interpreter().run(program))
        except APLError as exc:
            truth = "APL error: %s" % exc
        mark = "✓" if guess == truth else "✗"
        print("%s %-24s model: %-20s interpreter: %s" % (mark, program, guess, truth))
        if trace_mode:
            print("    steps: " + generated.replace("⋄", " ⋄ "))


if __name__ == "__main__":
    main()
