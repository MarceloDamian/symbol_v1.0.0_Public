#!/bin/bash
# Rounds 5 -> 6 -> 8, strictly one job at a time (running several at once restarted the Mac).
# Each step runs only if the one before it saved its checkpoint. Log: checkpoints/queue.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
TRAIN="$PY -u -m symbol.training.train --target trace --steps 30000 --eval-every 5000 --eval-size 500
       --d-model 384 --layers 8 --heads 8 --accum 2"

need() { [ -f "$1" ] || { echo "$(date '+%H:%M') missing $1; stopping"; exit 1; }; }

echo "$(date '+%H:%M') round 5: resume from step 15000"
echo "--- resumed $(date) ---" >> checkpoints/train_v5.log
$TRAIN --data data/apl_trace_v5.jsonl --out checkpoints/apl_trace_v5.pt --resume >> checkpoints/train_v5.log 2>&1
need checkpoints/apl_trace_v5.pt

echo "$(date '+%H:%M') round 5 saved; scoring"
$PY -m symbol.training.bench --quiet checkpoints/apl_trace_v4.pt checkpoints/apl_trace_v5.pt \
    > checkpoints/bench_v5.log 2>&1
for v in 1 9; do
    echo "votes $v: $($PY -m symbol.training.bench --quiet --votes $v --temperature 1.0 \
        checkpoints/apl_trace_v5.pt 2>&1 | tail -1)"
done > checkpoints/bench_v9_votes.log
echo "pass@16: $($PY -m symbol.training.bench --quiet --pass-at 16 --temperature 1.0 \
    checkpoints/apl_trace_v5.pt 2>&1 | tail -1)" >> checkpoints/bench_v9_votes.log

echo "$(date '+%H:%M') round 6: training"
$TRAIN --data data/apl_trace_v6.jsonl --out checkpoints/apl_trace_v6.pt > checkpoints/train_v6.log 2>&1
need checkpoints/apl_trace_v6.pt
$PY -m symbol.training.bench --quiet checkpoints/apl_trace_v4.pt checkpoints/apl_trace_v5.pt \
    checkpoints/apl_trace_v6.pt > checkpoints/bench_v6.log 2>&1

echo "$(date '+%H:%M') round 8: training"
$TRAIN --data data/apl_trace_v8.jsonl --out checkpoints/apl_trace_v8.pt > checkpoints/train_v8.log 2>&1
need checkpoints/apl_trace_v8.pt
$PY -m symbol.training.bench --quiet checkpoints/apl_trace_v6.pt checkpoints/apl_trace_v8.pt \
    > checkpoints/bench_v8.log 2>&1
echo "$(date '+%H:%M') done: rounds 5, 6, 8 trained and scored"
