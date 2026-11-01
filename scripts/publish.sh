#!/bin/bash
# Save a finished round off this machine: commit runs/v<round> to main and push, then put the
# checkpoint and the gzipped data on the GitHub release round-<round>.
#   bash scripts/publish.sh 15
# Failures are reported, never fatal: the files stay on disk and the next publish retries.
cd "$(dirname "$0")/.." || exit 1
r=$1
say() { echo "$(date '+%m-%d %H:%M') publish $r: $*"; }

python3 scripts/snapshot.py "$r" || say "snapshot failed"
git add "runs/v$r" && git commit -q -m "Round $r: manifest, logs and scores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016YQUCqC8JZHsUeUS1sEtUV" && say "committed"
git push -q origin main && say "pushed" || say "push failed"

ckpt=checkpoints/apl_trace_v$r.pt
data=data/apl_trace_v$r.jsonl
[ -f "$ckpt" ] || { say "no $ckpt; skipping release"; exit 0; }
gzip -kf "$data" 2>/dev/null
gh release view "round-$r" >/dev/null 2>&1 ||
    gh release create "round-$r" --title "Round $r" --notes "Checkpoint and data for round $r. See runs/v$r/manifest.json." \
        >/dev/null && say "release round-$r ready"
gh release upload "round-$r" "$ckpt" "$data.gz" runs/v$r/manifest.json --clobber && say "uploaded" || say "upload failed"
rm -f "$data.gz"
