## v0.2.1 — Meaning-Preservation Fixes

### Bug Fixes

**FIX — `normalizer.py`: hedge phrases stripped mid-sentence** (Critical)

In v0.2.0, patterns like `is non-negotiable`, `under any circumstances`,
`at all times`, `without exception` were stripped *anywhere* they appeared —
including inside meaningful instruction sentences:

```
# Before (broken):
"Risk management is non-negotiable — protecting capital..."
→ "Risk management  — protecting capital..."   ← corrupted

"Stop losses are mandatory without exception whatsoever"
→ "Stop losses are mandatory  whatsoever"       ← corrupted
```

Fix: all hedge phrase patterns are removed entirely from the rule list.
Sentence-opener patterns (`Furthermore,`, `Additionally,`, meta-commentary)
are now anchored with `(?:^|\n\s*)` so they only fire at sentence boundaries.

**FIX — `deduplicator.py`: numeric-protected chunks could be dropped** (High)

Chunks containing specific numeric values — risk-reward ratios (`1:2`),
volume multipliers (`1.5x`), named moving averages (`10-EMA`, `20-EMA`),
risk multiples (`1R`, `2R`), and the keyword `breakeven` — could previously
be chosen as the victim to drop when paired with a similar chunk.

These sentences carry precise, unique rules with no equivalent elsewhere.
Dropping them causes silent loss of critical instructions.

Fix: a new `_NUMERIC_GUARD` regex identifies protected chunks. In `_greedy_dedup`:
- Both chunks protected → keep both (skip this pair)
- One chunk protected → drop the OTHER one, always
- Neither protected → drop shorter (existing behaviour, unchanged)

```python
# New victim selection logic in _greedy_dedup:
if i_protected and j_protected:
    continue             # keep both
elif i_protected:
    victim = j           # always drop j
elif j_protected:
    victim = i           # always drop i
else:
    victim = shorter_of(i, j)   # existing behaviour
```

### Migration from v0.2.0

No API changes. Drop-in replacement. If you were relying on hedge phrase
removal for compression, note those phrases are no longer stripped —
they were causing meaning corruption on instruction-heavy prompts.

---

# Changelog

## v0.2.0 — Bug Fixes & Token Savings Improvements

### Bug Fixes

**BUG #1 — `deduplicator.py:108` wrong victim selection** (Critical)
- Old: `len(embeddings[i])` always returned `384` (the vector dimension) for every
  chunk equally — so the comparison was always equal and always dropped chunk `j`.
- Fix: `len(chunks[i].text)` — correctly keeps the longer, more informative chunk.
- Impact: every dedup removal now picks the right victim.

**BUG #2 — Overlap inflated output token count** (Critical)
- Old default `overlap=20` caused adjacent chunks to share 20 words. When joined,
  those words appeared twice in the output, adding ~80 tokens back to a 684-token prompt.
- Fix: default `overlap=0`. Overlap parameter still available for retrieval use cases.
- Impact: eliminates token inflation; output is now strictly ≤ input.

### New Features

**GAP #3 — Stage 0: `TextNormalizer`** (new `tokenmesh/core/normalizer.py`)
- Strips filler phrases (`"It is important to note"`, `"Furthermore,"`, `"under any
  circumstances"`, etc.), markdown artifacts, and redundant whitespace before chunking.
- 6–12% free token reduction with zero semantic loss on typical LLM system prompts.
- Toggle with `TokenMesh(normalize=False)` to disable.

**GAP #4 — Embeddings shared between deduplicator and scorer**
- Old: chunks embedded twice — once in `SemanticDeduplicator`, once in `ImportanceScorer`.
- Fix: `deduplicate(return_embeddings=True)` returns L2-normalized vectors; scorer
  accepts `precomputed_embeddings=` and only re-encodes the query.
- Impact: ~40–50% faster pipeline latency (one fewer `model.encode()` call).

**GAP #5 — Default `chunk_size` lowered from 200 → 40**
- `chunk_size=200` produced only 3–4 chunks from a 684-token prompt — giving
  the deduplicator almost nothing to compare.
- `chunk_size=40` (~1–2 sentences) gives 20–40 chunks and unlocks real dedup gains.

**GAP #6 — `token_budget` parameter**
- `TokenMesh(token_budget=2000)` enforces a hard output token cap.
- Pipeline progressively drops lowest-scoring chunks until budget is met.
- Safety floor: never drops below 20% of original token count.

**GAP #7 — Smart separator**
- Old: `separator="\n\n"` always — 2 tokens × N chunks wasted.
- Fix: auto-detects plain text (uses `" "`, 0 tokens) vs markdown (uses `"\n"`, 1 token).

### Migration from v0.1

```python
# v0.1
tm = TokenMesh(chunk_size=200, overlap=20)

# v0.2 (equivalent behaviour, better defaults)
tm = TokenMesh()  # chunk_size=40, overlap=0, normalize=True
```
