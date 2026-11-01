# Runbook

How to pick this project up cold: a new session, a new machine, or after a crash.

## Where everything lives

| What | Where | Saved off this machine by |
|---|---|---|
| Code, scripts, tests | this repo, branch `main` | `git push` (private GitHub repo `MarceloDamian/symbol`) |
| Per-round record | `runs/v<N>/manifest.json` (+ gen/train/bench logs) | committed by `scripts/publish.sh` |
| Trained models | `checkpoints/apl_trace_v<N>.pt` (git-ignored) | GitHub release `round-<N>` |
| Training data | `data/apl_trace_v<N>.jsonl` (git-ignored) | GitHub release `round-<N>` (gzipped) |
| Resume points | `checkpoints/apl_trace_v<N>.pt.partial` (weights + optimizer, every 1000 steps) | local only (also iCloud, since Desktop syncs) |
| API key for the bridge | `~/Keys/api-keys/anthropic.key` | never committed |

A manifest records the data's rows, seed and SHA-256, the code commit that generated it
(round 15 on), the exact training command and steps, the checkpoint's SHA-256, and the scores.

## New machine

```bash
gh repo clone MarceloDamian/symbol && cd symbol
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/restore.sh 16          # best so far (runs/v17/NOTE.md); checkpoint + data, hashes checked
.venv/bin/python -m pytest -q tests && .venv/bin/python tests/test_apl_interp.py
.venv/bin/python -m symbol.training.bench --quiet checkpoints/apl_trace_v16.pt
```

The data generator runs on system `python3` (3.8 was used); training and benches use `.venv` (3.12, torch).

## Run a new round

1. Change the generator / trace / interpreter, then `pytest` until green.
2. Commit, so the round's code is recorded: `git commit -am "Round N: ..."`.
3. Launch detached, strictly one job at a time (several at once restarted the Mac):
   ```bash
   nohup perl -MPOSIX -e 'setsid(); exec @ARGV' bash -c \
     'bash scripts/gen.sh N 210000 N && STEPS=30000 BASELINE=checkpoints/apl_trace_v<N-1>.pt bash scripts/queue.sh N' \
     >> checkpoints/queue.log 2>&1 < /dev/null &
   ```
4. Watch `checkpoints/queue.log` and `checkpoints/train_vN.log`. When scored, queue.sh runs
   `scripts/publish.sh N` (manifest commit + push, checkpoint and data to release `round-N`).

`gen.sh` refuses to overwrite finished data (`FORCE=1` to redo) and records the commit in
`runs/vN/gen.json`; uncommitted changes are saved as `runs/vN/uncommitted.patch`.

## After a crash or reboot

- Training resumes from its `.partial`: rerun the same `queue.sh N` command; it waits for 30% free memory.
- A killed generator leaves only `data/apl_trace_vN.jsonl.tmp`; rerun `gen.sh` (the watchdog log says why it died).
- Publishing failed (offline)? `bash scripts/publish.sh N` any time; it is safe to repeat.

## Re-run a past round exactly

```bash
DATA_ONLY=1 bash scripts/rerun.sh 15   # regenerate with round 15's code, compare SHA-256
bash scripts/rerun.sh 15               # ... then train checkpoints/rerun_v15.pt
```

Rounds 4-14 predate the repo: their generator code is gone, so `rerun.sh` retrains them from the
saved data (`restore.sh N` first if it is missing).

## Rules learned the hard way

- One heavy job at a time. Training, data generation and benches together restarted the Mac.
- Every allocation in the interpreter checks `_check_size` *before* building (`⍸`, `⊤`, `/` all bit us).
- `pgrep -f pattern` inside `bash -c '...pattern...'` matches itself; wait on a PID with `kill -0`.
- Never edit a running bash script in place; write a new file and `mv` it over.
