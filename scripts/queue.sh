#!/bin/bash
# Train and score rounds one at a time (several jobs at once restarted the Mac).
#   bash scripts/queue.sh 6 8      -> round 6, score, round 8, score
#   STEPS=45000 bash scripts/queue.sh 15   -> a longer run (default 30000 steps)
# A round that stops early (low-memory exit 3, crash, reboot) resumes from its
# .partial after free memory is back above 30%, up to 5 tries. Each scored round is
# published: manifest committed and pushed, checkpoint + data on GitHub release round-<r>.
# Run detached:  nohup perl -MPOSIX -e 'setsid(); exec @ARGV' bash scripts/queue.sh 6 8 \
#                    > checkpoints/queue.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
STEPS=${STEPS:-30000}
TRAIN="$PY -u -m symbol.training.train --target trace --steps $STEPS --eval-every 5000 --eval-size 500
       --d-model 384 --layers 8 --heads 8 --accum 2"

say() { echo "$(date '+%m-%d %H:%M') $*"; }

wait_for_memory() {
    until [ "$(sysctl -n kern.memorystatus_level)" -ge 30 ]; do sleep 60; done
}

train_round() {
    local out=checkpoints/apl_trace_v$1.pt
    while pgrep -f "out $out" >/dev/null; do sleep 60; done  # a run from an earlier queue finishes first
    mkdir -p "runs/v$1"
    [ -f "runs/v$1/train.json" ] || echo "{\"steps\": $STEPS, \"command\": \"$(echo $TRAIN) --data data/apl_trace_v$1.jsonl --out $out --resume\"}" \
        > "runs/v$1/train.json"
    for try in 1 2 3 4 5; do
        [ -f "$out" ] && return 0
        wait_for_memory
        say "round $1: training (try $try)"
        echo "--- queue try $try $(date) ---" >> "checkpoints/train_v$1.log"
        $TRAIN --data "data/apl_trace_v$1.jsonl" --out "$out" --resume >> "checkpoints/train_v$1.log" 2>&1
        say "round $1: exit $?"
    done
    [ -f "$out" ]
}

done_rounds=(${BASELINE:-checkpoints/apl_trace_v5.pt})  # scored alongside, for comparison
for r in "$@"; do
    train_round "$r" || { say "round $r failed 5 times; stopping"; exit 1; }
    done_rounds+=("checkpoints/apl_trace_v$r.pt")
    say "round $r saved; scoring"
    $PY -m symbol.training.bench --quiet "${done_rounds[@]}" > "checkpoints/bench_v$r.log" 2>&1
    tail -$((${#done_rounds[@]} + 1)) "checkpoints/bench_v$r.log"
    bash scripts/publish.sh "$r"
done
say "done: rounds $* trained and scored"
