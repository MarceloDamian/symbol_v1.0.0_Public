"""
Train the APL transformer to evaluate APL: read `program`, write `result`.

    .venv/bin/python -m symbol.training.train --data data/apl_eval.jsonl

Each example is laid out as `<bos> program <sep> result <eos>` (glyph tokens)
and trained with next-token cross-entropy. Validation decodes the result
greedily and scores it against the interpreter three ways:

  exact     predicted text == canonical result text
  semantic  predicted text evaluates to the same array as the program
  valid     predicted text is runnable APL at all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from symbol.dataset.canonical import BITS, CANONICAL, CONTROL, HARD, PRIMS
from symbol.sandbox.apl_interp import Interpreter
from symbol.training.model import APLTransformer, ModelConfig
from symbol.transforms.format import same_array, to_apl
from symbol.transforms.tokenizer import APLTokenizer
from symbol.transforms.trace import final_answer


def pick_device() -> torch.device:
    if os.environ.get("SYMBOL_DEVICE"):  # e.g. cpu, to test without touching a GPU that is training
        return torch.device(os.environ["SYMBOL_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rows(path: str) -> Tuple[List[dict], List[dict]]:
    train, val = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            (val if row["split"] == "val" else train).append(row)
    return train, val


def batches(
    seqs: List[List[int]], batch_size: int, pad_id: int, rng: random.Random,
    sep_id: int = -1, result_only: bool = False, max_tokens: int = 12_000,
):
    """Endless shuffled batches of right-padded (inputs, targets).

    With `result_only`, targets up to and including <sep> are set to pad (and
    so ignored by the loss): random programs are noise to predict, the result
    is the signal.

    Batches hold at most `batch_size` rows and `max_tokens` padded tokens, so
    long traces come in smaller batches and memory stays bounded. Widths are
    rounded up to a multiple of 16 to limit how many shapes the GPU compiles.
    """
    order = list(range(len(seqs)))
    pool = batch_size * 50
    while True:
        rng.shuffle(order)
        # Bucket by length inside large shuffled pools so batches pad little;
        # traces vary from ~10 to ~250 tokens.
        groups = []
        for p in range(0, len(order), pool):
            block = sorted(order[p : p + pool], key=lambda i: len(seqs[i]))
            current: List[int] = []
            for i in block:
                width = _round16(len(seqs[i]))
                if current and (len(current) + 1) * width > max_tokens or len(current) == batch_size:
                    groups.append(current)
                    current = []
                current.append(i)
            if current:
                groups.append(current)
        rng.shuffle(groups)
        for group in groups:
            chunk = [seqs[i] for i in group]
            width = _round16(max(len(s) for s in chunk) - 1) + 1
            padded = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            for row, seq in enumerate(chunk):
                padded[row, : len(seq)] = torch.tensor(seq)
            targets = padded[:, 1:].clone()
            if result_only:
                for row, seq in enumerate(chunk):
                    targets[row, : seq.index(sep_id)] = pad_id
            yield padded[:, :-1], targets


def free_memory_pct() -> int:
    """macOS's own free-memory level (0-100, as `memory_pressure` reports it); 100 elsewhere."""
    if sys.platform != "darwin":
        return 100
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                                  capture_output=True, text=True, timeout=5).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 100


LOW_MEMORY_EXIT = 3  # the queue sees this, waits for memory, then reruns with --resume


def _round16(n: int) -> int:
    return -(-n // 16) * 16


def score_prediction(program: str, expected: str, predicted: str) -> Dict[str, bool]:
    exact = predicted == expected
    if exact:
        return {"exact": True, "semantic": True, "valid": True}
    try:
        pred_val = Interpreter(step_budget=5000).run(predicted)
    except Exception:  # noqa: BLE001 - any failure means the output is not valid APL
        return {"exact": False, "semantic": False, "valid": False}
    want_val = Interpreter(step_budget=5000).run(program)
    return {"exact": False, "semantic": pred_val is not None and same_array(pred_val, want_val), "valid": True}


@torch.no_grad()
def evaluate_model(
    model: APLTransformer, tok: APLTokenizer, rows: List[dict], device: torch.device,
    batch_size: int = 128, show: int = 0, trace_mode: bool = False,
    votes: int = 1, temperature: float = 0.7,
) -> Dict[str, float]:
    """Decode results for `rows`. Prompts are bucketed by length so each batch is
    rectangular without padding (positions stay aligned).

    votes > 1: besides the greedy answer, sample votes-1 more at `temperature`
    and keep the most common final answer (ties go to the greedy one)."""
    # Not strict: an older checkpoint's vocabulary may lack a newer glyph (∇); it reads <unk>.
    by_len: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        by_len[len(tok.encode_prompt(row["program"], strict=False))].append(row)

    totals = {"exact": 0, "semantic": 0, "valid": 0}
    examples = []
    for length, group in by_len.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            # A model resumed from before a glyph was added has fewer embeddings: read it as <unk>.
            ids = [[i if i < model.cfg.vocab_size else tok.unk_id for i in tok.encode_prompt(r["program"], strict=False)]
                   for r in chunk]
            prompts = torch.tensor(ids, device=device)
            max_new = model.cfg.max_len - length
            greedy = model.generate(prompts, stop_id=tok.eos_id, max_new=max_new).tolist()
            sampled = [[] for _ in chunk]
            if votes > 1:
                many = model.generate(prompts.repeat_interleave(votes - 1, dim=0), stop_id=tok.eos_id,
                                      max_new=max_new, temperature=temperature).tolist()
                sampled = [many[i * (votes - 1) : (i + 1) * (votes - 1)] for i in range(len(chunk))]
            for row, first, rest in zip(chunk, greedy, sampled):
                answers = []
                for ids in [first] + rest:
                    if tok.eos_id in ids:
                        ids = ids[: ids.index(tok.eos_id)]
                    text = tok.decode(ids)
                    answers.append((final_answer(text) if trace_mode else text, text))
                # most_common keeps first-seen order on ties, so the greedy answer wins them.
                predicted = Counter(a for a, _ in answers).most_common(1)[0][0]
                generated = answers[0][1]
                if predicted != answers[0][0]:
                    generated += "  [vote -> %s]" % predicted
                scores = score_prediction(row["program"], row["result"], predicted)
                for key, ok in scores.items():
                    totals[key] += int(ok)
                if len(examples) < show:
                    examples.append((row["program"], row["result"], generated, scores["exact"]))
    n = max(len(rows), 1)
    metrics = {k: v / n for k, v in totals.items()}
    metrics["examples"] = examples  # type: ignore[assignment]
    return metrics


def benchmark_rows(programs: List[str]) -> List[dict]:
    return [{"program": p, "result": to_apl(Interpreter().run(p))} for p in programs]


def report_benchmark(model, tok, name: str, rows: List[dict], device, trace_mode: bool, **vote) -> float:
    """Score a held-out benchmark, print every program, return the exact rate."""
    res = evaluate_model(model, tok, rows, device, show=len(rows), trace_mode=trace_mode, **vote)
    print("\n%s benchmark (never trained on): exact %d/%d (%.1f%%)" % (
        name, round(res["exact"] * len(rows)), len(rows), 100 * res["exact"]))
    for program, want, got, ok in res["examples"]:
        print("  %s %-28s want %-22s got %s" % ("✓" if ok else "✗", program, want, got))
    return res["exact"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the APL evaluator transformer")
    parser.add_argument("--data", default="data/apl_eval.jsonl")
    parser.add_argument("--out", default="checkpoints/apl_eval.pt")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-on", choices=["result", "all"], default="result")
    parser.add_argument("--target", choices=["result", "trace"], default="result",
                        help="trace: write every small step (a⋄b⋄result) before the answer")
    parser.add_argument("--final-eval-size", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=12_000, help="padded tokens per batch")
    parser.add_argument("--accum", type=int, default=1,
                        help="split each batch into this many micro-batches (same update, less memory)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out>.partial, saved at every eval")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="also save <out>.partial this often, so a crash loses little")
    parser.add_argument("--min-free", type=int, default=10,
                        help="save and exit (code 3) when macOS free memory drops below this %%")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device()
    tok = APLTokenizer()

    train_rows, val_rows = load_rows(args.data)
    trace_mode = args.target == "trace"
    if trace_mode:
        train_rows = [r for r in train_rows if r.get("trace")]
        val_rows = [r for r in val_rows if r.get("trace")]
    train_seqs = [
        tok.encode_pair(r["program"], r["trace"] if trace_mode else r["result"]) for r in train_rows
    ]
    max_len = max(len(s) for s in train_seqs) + 8
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, max_len=max_len, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers, dropout=args.dropout,
    )
    partial = args.out + ".partial"
    state = torch.load(partial, map_location="cpu") if args.resume and os.path.exists(partial) else None
    if state is not None:
        cfg = ModelConfig(**state["config"])  # max_len must match the saved weights
    model = APLTransformer(cfg).to(device)
    print("device %s | params %.2fM | train %d | val %d | max_len %d" % (
        device, model.num_params() / 1e6, len(train_rows), len(val_rows), max_len))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    first_step, history = 1, []
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        first_step, history = state["step"] + 1, state["history"]
        print("resumed from %s at step %d" % (partial, state["step"]), flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    eval_rows = val_rows[: args.eval_size]
    stream = batches(
        train_seqs, max(1, args.batch_size // args.accum), tok.pad_id, rng,
        sep_id=tok.sep_id, result_only=(args.loss_on == "result"),
        max_tokens=args.max_tokens // args.accum,
    )
    canonical_rows = benchmark_rows(CANONICAL)
    hard_rows = benchmark_rows(HARD)
    control_rows = benchmark_rows(CONTROL)
    prims_rows = benchmark_rows(PRIMS)
    bits_rows = benchmark_rows(BITS)
    start = time.time()
    running = None

    def save_partial(step: int) -> None:
        # Write then rename: a crash mid-save must not destroy the last good partial.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "config": cfg.to_dict(),
                    "step": step, "history": history}, partial + ".tmp")
        os.replace(partial + ".tmp", partial)

    for step in range(first_step, args.steps + 1):
        model.train()
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            x, y = next(stream)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=tok.pad_id)
            (loss / args.accum).backward()
            step_loss += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running = step_loss if running is None else 0.98 * running + 0.02 * step_loss

        if step % 100 == 0:
            print("step %5d | loss %.4f | lr %.2e | %.0fs" % (step, running, lr_at(step), time.time() - start), flush=True)
            free = free_memory_pct()
            if free < args.min_free:
                save_partial(step)
                print("low memory: %d%% free < %d%%; saved %s at step %d, exiting" % (
                    free, args.min_free, partial, step), flush=True)
                sys.exit(LOW_MEMORY_EXIT)
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate_model(model, tok, eval_rows, device, trace_mode=trace_mode)
            m.pop("examples")
            c = evaluate_model(model, tok, canonical_rows, device, trace_mode=trace_mode)
            h = evaluate_model(model, tok, hard_rows, device, trace_mode=trace_mode)
            k = evaluate_model(model, tok, control_rows, device, trace_mode=trace_mode)
            p = evaluate_model(model, tok, prims_rows, device, trace_mode=trace_mode)
            b = evaluate_model(model, tok, bits_rows, device, trace_mode=trace_mode)
            m.update(step=step, loss=running, canonical_exact=c["exact"], hard_exact=h["exact"],
                     control_exact=k["exact"], prims_exact=p["exact"], bits_exact=b["exact"])
            history.append(m)
            print("  eval @%d: exact %.1f%% | semantic %.1f%% | valid APL %.1f%% | canonical %.1f%% | hard %.1f%%"
                  " | control %.1f%% | prims %.1f%% | bits %.1f%%" % (
                      step, 100 * m["exact"], 100 * m["semantic"], 100 * m["valid"],
                      100 * c["exact"], 100 * h["exact"], 100 * k["exact"], 100 * p["exact"], 100 * b["exact"]), flush=True)
            save_partial(step)
            if device.type == "mps":
                torch.mps.empty_cache()  # eval shapes differ from training; give the memory back
        elif step % args.save_every == 0:
            save_partial(step)

    val_rows = val_rows[: args.final_eval_size]
    final = evaluate_model(model, tok, val_rows, device, show=15, trace_mode=trace_mode)
    examples = final.pop("examples")
    print("\nfinal on all %d val examples: exact %.1f%% | semantic %.1f%% | valid APL %.1f%%" % (
        len(val_rows), 100 * final["exact"], 100 * final["semantic"], 100 * final["valid"]))
    for program, want, got, ok in examples:
        print("  %s %-28s want %-22s got %s" % ("✓" if ok else "✗", program, want, got))

    for name, rows in (("canonical", canonical_rows), ("hard", hard_rows), ("control", control_rows),
                       ("prims", prims_rows), ("bits", bits_rows)):
        final[name + "_exact"] = report_benchmark(model, tok, name, rows, device, trace_mode)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "vocab": tok.itos[: cfg.vocab_size],
        "final_metrics": final,
        "history": history,
        "args": vars(args),
    }, args.out)
    print("saved %s" % args.out)


if __name__ == "__main__":
    main()
