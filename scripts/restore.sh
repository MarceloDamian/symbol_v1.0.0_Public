#!/bin/bash
# Get a round's checkpoint and data back from its GitHub release (new machine, deleted files),
# then check both against the hashes in runs/v<round>/manifest.json.
#   bash scripts/restore.sh 14
cd "$(dirname "$0")/.." || exit 1
r=$1
mkdir -p checkpoints data
gh release download "round-$r" -p "apl_trace_v$r.pt" -D checkpoints --skip-existing || exit 1
gh release download "round-$r" -p "apl_trace_v$r.jsonl.gz" -D data --skip-existing &&
    gunzip -kf "data/apl_trace_v$r.jsonl.gz" && rm -f "data/apl_trace_v$r.jsonl.gz"
python3 - "$r" <<'EOF'
import hashlib, json, sys
r = sys.argv[1]
m = json.load(open("runs/v%s/manifest.json" % r))
for part in ("checkpoint", "data"):
    want, path = m[part]["sha256"], m[part]["path"]
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print("%-36s %s" % (path, "ok" if got == want else "HASH MISMATCH"))
EOF
