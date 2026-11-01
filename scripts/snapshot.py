"""
Write runs/v<round>/manifest.json: everything needed to re-run or check a round.

    python3 scripts/snapshot.py 15

Reads what gen.sh and queue.sh recorded (runs/v<round>/gen.json, train.json), hashes the
data and checkpoint, and copies the round's logs next to the manifest so git keeps them.
Safe to run again: it only rewrites the manifest and the log copies.
"""

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def bench_scores(path, ckpt):
    """{suite: percent} for this round's checkpoint, from the table bench.py prints last."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = [i for i, line in enumerate(lines) if line.startswith("checkpoint ")]
    if not header:
        return None
    names = lines[header[-1]].split()[1:]
    for line in lines[header[-1] + 1:]:
        if line.startswith(ckpt):
            return dict(zip(names, [float(x.rstrip("%")) for x in line.split()[1:]]))
    return None


def final_val(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        m = re.findall(r"final on all \d+ val examples: exact ([\d.]+)%", fh.read())
    return float(m[-1]) if m else None


def main(round_):
    os.chdir(ROOT)
    run_dir = "runs/v%s" % round_
    os.makedirs(run_dir, exist_ok=True)
    data, ckpt = "data/apl_trace_v%s.jsonl" % round_, "checkpoints/apl_trace_v%s.pt" % round_
    rows = sum(1 for _ in open(data, encoding="utf-8")) if os.path.exists(data) else None
    for log in ("gen_v%s.log", "train_v%s.log", "bench_v%s.log"):
        src = "checkpoints/" + log % round_
        if os.path.exists(src):
            shutil.copy(src, os.path.join(run_dir, log % round_))
    manifest = {
        "round": round_,
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "gen": read_json(os.path.join(run_dir, "gen.json")),
        "train": read_json(os.path.join(run_dir, "train.json")),
        "data": {"path": data, "rows": rows, "sha256": sha256(data)},
        "checkpoint": {"path": ckpt, "sha256": sha256(ckpt), "release": "round-%s" % round_},
        "val_exact": final_val("checkpoints/train_v%s.log" % round_),
        "bench": bench_scores("checkpoints/bench_v%s.log" % round_, ckpt),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("wrote %s/manifest.json" % run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
