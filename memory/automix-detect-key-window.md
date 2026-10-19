---
name: automix-detect-key-window
description: automix.py detect_key analysis window is kept conservative on purpose — do not widen it
metadata: 
  node_type: memory
  type: project
  originSessionId: 77ebd06e-5cdd-44ae-84a8-2dd0757864fb
---

In `automix.py`, `detect_key()` intentionally keeps its short fallback threshold
(`len(y) < sr`, ~1 s) and its `offset=30 / duration=60` analysis window. Do NOT align it
with `detect_bpm()`'s wider `sr*30` fallback for "consistency."

**Why:** A past experiment where a track's *detected key* changed produced very bad mixes.
Key feeds the harmonic-sort tie-break (`score − 50` when Camelot-compatible), so shifting
detected keys reshuffles track order. Roman prefers the occasional short-window key read
over the risk of keys changing.

**How to apply:** When touching BPM/key analysis windows in automix, `detect_bpm` may be
changed freely (uncached BPM drift is an accepted P1.4 exception), but leave `detect_key`'s
window and threshold alone unless Roman explicitly asks. See [[automix-bpm-analysis-window]].
