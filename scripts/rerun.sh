#!/bin/bash
# Rebuild a round from scratch with the exact code that made it: check out the commit recorded
# in runs/v<round>/gen.json (plus any uncommitted patch), regenerate the data, compare its hash
# with the manifest, then train a fresh checkpoint. Nothing existing is overwritten.
#   bash scripts/rerun.sh 15            -> data/rerun_v15.jsonl, checkpoints/rerun_v15.pt
#   DATA_ONLY=1 bash scripts/rerun.sh 15  (just regenerate and compare)
# Rounds before 15 have no recorded code; they retrain from their saved data instead.
cd "$(dirname "$0")/.." || exit 1
r=$1
root=$PWD
gen=runs/v$r/gen.json
out_data=data/rerun_v$r.jsonl
field() { python3 -c "import json,sys; print(json.load(open('$1')).get('$2') or '')"; }

if [ -f "$gen" ] && [ -n "$(field "$gen" commit)" ]; then
    work=$(mktemp -d)/symbol
    git worktree add -q --detach "$work" "$(field "$gen" commit)" || exit 1
    [ -s "runs/v$r/uncommitted.patch" ] && git -C "$work" apply "$root/runs/v$r/uncommitted.patch"
    (cd "$work" && python3 -m symbol.dataset.generator --n "$(field "$root/$gen" rows)" \
        --seed "$(field "$root/$gen" seed)" --out "$root/$out_data") || exit 1
    git worktree remove --force "$work"
    want=$(python3 -c "import json; print(json.load(open('runs/v$r/manifest.json'))['data']['sha256'])")
    got=$(shasum -a 256 "$out_data" | cut -d' ' -f1)
    [ "$got" = "$want" ] && echo "data matches round $r exactly" ||
        echo "data differs from round $r (a 1-second generator timeout can land differently under load)"
else
    echo "round $r has no recorded code; training from its saved data/apl_trace_v$r.jsonl"
    out_data=data/apl_trace_v$r.jsonl
fi
[ -n "$DATA_ONLY" ] && exit 0

steps=30000
[ -f "runs/v$r/train.json" ] && steps=$(field "runs/v$r/train.json" steps)
.venv/bin/python -u -m symbol.training.train --target trace --steps "$steps" --eval-every 5000 \
    --eval-size 500 --d-model 384 --layers 8 --heads 8 --accum 2 \
    --data "$out_data" --out "checkpoints/rerun_v$r.pt" --resume
