#!/bin/bash
# Generate a dataset with a memory watchdog: kill the generator past MAX_MB of RSS
# instead of letting macOS kill it (or the whole machine) first.
#   bash scripts/gen.sh <round> <rows> <seed>   -> data/apl_trace_v<round>.jsonl, checkpoints/gen_v<round>.log
# Never overwrites a finished dataset (FORCE=1 to redo one). Writes to a .tmp file and renames it
# only on success, and records the code commit, seed and rows in runs/v<round>/gen.json.
cd "$(dirname "$0")/.." || exit 1
MAX_MB=${MAX_MB:-3000}
out="data/apl_trace_v$1.jsonl"
if [ -f "$out" ] && [ -z "$FORCE" ]; then
    echo "$out already exists; FORCE=1 to regenerate" >&2
    exit 1
fi
mkdir -p "runs/v$1"
git diff HEAD > "runs/v$1/uncommitted.patch"   # empty when the code is committed
[ -s "runs/v$1/uncommitted.patch" ] || rm -f "runs/v$1/uncommitted.patch"
cat > "runs/v$1/gen.json" <<JSON
{"rows": $2, "seed": $3, "commit": "$(git rev-parse HEAD)", "uncommitted_patch": $([ -f "runs/v$1/uncommitted.patch" ] && echo true || echo false),
 "python": "$(python3 --version 2>&1)", "started": "$(date '+%Y-%m-%dT%H:%M:%S')",
 "command": "python3 -m symbol.dataset.generator --n $2 --seed $3 --out $out"}
JSON
python3 -m symbol.dataset.generator --n "$2" --seed "$3" --out "$out.tmp" \
    > "checkpoints/gen_v$1.log" 2>&1 &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
    mb=$(( $(ps -o rss= -p $pid 2>/dev/null || echo 0) / 1024 ))
    [ "$mb" -gt "$peak" ] && peak=$mb
    if [ "$mb" -gt "$MAX_MB" ]; then
        kill -9 $pid
        echo "watchdog: generator used ${mb} MB > ${MAX_MB} MB; killed" >> "checkpoints/gen_v$1.log"
        exit 1
    fi
    sleep 2
done
wait $pid; code=$?
echo "peak RSS ${peak} MB" >> "checkpoints/gen_v$1.log"
[ $code -eq 0 ] && mv "$out.tmp" "$out"
exit $code
