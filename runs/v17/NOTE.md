# Round 17: not the best round

**Round 16 appears to be the best checkpoint** (`checkpoints/apl_trace_v16.pt`, release `round-16`). Use it, not this one.

| | canonical | hard | control | prims | bits | total |
|---|---|---|---|---|---|---|
| v16 | 100.0 | 95.5 | 100.0 | 93.8 | 93.8 | 133/137 |
| v17 | 100.0 | 95.5 | 100.0 | 87.5 | 93.8 | 132/137 |

Round 17 fixed what it targeted ('an'⍷'banana', 11|91, 2⊥⌽(4⍴2)⊤12, ⍸∘⌈15), but new slips appeared
(3!6 -> 130, ∩ duplicates, V[3 1], 3|9⍳⍳10). More templates seem to have hit a plateau at 132-133.
Kept as a record so that approach is not repeated blindly.
