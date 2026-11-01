"""
Score saved checkpoints on the held-out benchmarks.

    .venv/bin/python -m symbol.training.bench checkpoints/apl_trace_v4.pt checkpoints/apl_trace_v5.pt
"""

from __future__ import annotations

import argparse

import torch

from symbol.dataset.canonical import BITS, CANONICAL, CONTROL, HARD, PRIMS
from symbol.training.predict import load
from symbol.transforms.trace import final_answer
from symbol.training.train import benchmark_rows, evaluate_model, pick_device, report_benchmark


def pass_at(model, tok, rows, device, k: int, temperature: float) -> float:
    """Share of programs where at least one of k sampled traces ends in the right answer.
    The ceiling for any answer-picker such as voting: errors it misses are systematic."""
    hits = 0
    for row in rows:
        ids = [i if i < model.cfg.vocab_size else tok.unk_id for i in tok.encode_prompt(row["program"], strict=False)]
        prompts = torch.tensor([ids] * k, device=device)
        gen = model.generate(prompts, stop_id=tok.eos_id, max_new=model.cfg.max_len - len(ids),
                             temperature=temperature).tolist()
        answers = {final_answer(tok.decode(g[: g.index(tok.eos_id)] if tok.eos_id in g else g)) for g in gen}
        hits += row["result"] in answers
    return hits / max(len(rows), 1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score checkpoints on CANONICAL, HARD and CONTROL")
    parser.add_argument("ckpts", nargs="+")
    parser.add_argument("--quiet", action="store_true", help="only print the totals")
    parser.add_argument("--votes", type=int, default=1, help="majority vote over this many answers")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", help="cpu keeps the GPU free for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pass-at", type=int, default=0,
                        help="instead: count programs where any of K sampled answers is right")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    vote = {"votes": args.votes, "temperature": args.temperature}
    suites = (("canonical", benchmark_rows(CANONICAL)), ("hard", benchmark_rows(HARD)),
              ("control", benchmark_rows(CONTROL)), ("prims", benchmark_rows(PRIMS)),
              ("bits", benchmark_rows(BITS)))
    totals = []
    for path in args.ckpts:
        model, tok, trace_mode = load(path, device)
        print("\n=== %s" % path)
        scores = {}
        for name, rows in suites:
            if args.pass_at:
                scores[name] = pass_at(model, tok, rows, device, args.pass_at, args.temperature)
            elif args.quiet:
                scores[name] = evaluate_model(model, tok, rows, device, trace_mode=trace_mode, **vote)["exact"]
            else:
                scores[name] = report_benchmark(model, tok, name, rows, device, trace_mode, **vote)
        totals.append((path, scores))

    print("\n%-40s" % "checkpoint" + "".join("%11s" % name for name, _ in suites))
    for path, scores in totals:
        print("%-40s" % path + "".join("%10.1f%%" % (100 * scores[name]) for name, _ in suites))


if __name__ == "__main__":
    main()
