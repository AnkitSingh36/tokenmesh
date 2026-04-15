# 🧵 TokenMesh

**why send 8000 tokens when 3000 do trick**

<p align="center">
  ⚡ ~6ms / 1K tokens &nbsp; • &nbsp; 💸 Up to 75% cost reduction &nbsp; • &nbsp; 🧠 Minimal semantic loss
</p>

[Install](#install) · [Live Demo](#-try-it-live--no-install-needed) · [Quick Start](#quick-start) · [Two Modes](#two-modes-lite--aggressive) · [Claude Integration](#claude-integration) · [Benchmarks](#benchmarks) · [How It Works](#how-it-works) · [Contributing](#contributing)

---

A Python library that removes duplicate, irrelevant, and filler content from LLM prompts **before the API call is made** — cutting 40–75% of tokens while keeping full semantic accuracy.

## Before / After

| | |
|---|---|
| **Original system prompt** (684 tokens) · *"You are an expert swing trading assistant specializing in NSE and BSE equity markets. You are a knowledgeable financial co-pilot helping retail Indian investors trade profitably. Your role is to assist with trade analysis... Always apply strict risk management: every trade must have a defined stop loss. Never let a losing trade run beyond the defined stop loss level under any circumstances. Risk management is non-negotiable..."* | **TokenMesh output** (~290 tokens) · *"Expert swing trading assistant for NSE/BSE. Assist with trade analysis, scanner interpretation, risk management. Every trade needs a stop loss. Never enter without 1:2 R:R minimum. EMA 20/50 + RSI confirmation for high-probability signals..."* |

**Same instructions. 57% fewer tokens. Claude understands it fine.**

---

## 🚀 Try It Live — No Install Needed

**[→ Open Live Demo](https://AnkitSingh36.github.io/tokenmesh/)** — runs entirely in your browser.

Or download and open locally (zero setup):

```bash
# After cloning the repo:
open demo.html          # Mac
start demo.html         # Windows
xdg-open demo.html      # Linux
```

Paste any prompt → click Optimize → see exact token savings instantly.  
Supports all three modes: Default · Lite · Aggressive.

---

## Install

```bash
pip install tokenmesh
```

Works on Python 3.9+, CPU-only, Windows/Mac/Linux.

> **First run** downloads `all-MiniLM-L6-v2` (22 MB) once. Cached forever after.

---

## Quick Start

### Compress any text

```python
from tokenmesh import TokenMesh

tm = TokenMesh()
result = tm.optimize(your_text, query="what is the refund policy?")

print(result.optimized_text)       # send this to Claude
print(result.reduction_percent)    # e.g. 54.2
print(result.summary())
# TokenMesh | 4,200 → 1,930 tokens (54.0% reduction) | $0.0068 saved | 38ms
```

### With Claude API

```python
from tokenmesh.integrations.claude import TokenMeshClaude

client = TokenMeshClaude()   # reads ANTHROPIC_API_KEY from env

response = client.chat(
    system=your_long_system_prompt,
    user="SHRIRAMFIN broke above 20-EMA with volume. Entry?",
    model="claude-sonnet-4-20250514",
)

print(response.content)
print(f"Saved {response.token_savings} tokens (${response.savings_usd:.4f})")
```

---

## Two Modes: Lite & Aggressive

### 🟢 Lite — safe, conservative

```python
from tokenmesh import TokenMeshLite

result = TokenMeshLite().optimize(your_system_prompt)
```

| | |
|---|---|
| Dedup threshold | 0.92 — only near-identical duplicates |
| Filler removal | Yes |
| Typical reduction | **20–40%** |
| Best for | System prompts, legal text, financial rules |

### 🔴 Aggressive — maximum compression

```python
from tokenmesh import TokenMeshAggressive

result = TokenMeshAggressive().optimize(long_document, query="key risks")
```

| | |
|---|---|
| Dedup threshold | 0.72 — catches loose paraphrases |
| Filler removal | Yes |
| Typical reduction | **55–75%** |
| Best for | RAG context, pasted articles, conversation history |

### Compare both on your text

```python
from tokenmesh import TokenMeshLite, TokenMeshAggressive

text = your_long_text

r1 = TokenMeshLite().optimize(text)
r2 = TokenMeshAggressive().optimize(text, query="your question")

print(f"Lite:       {r1.reduction_percent}%  →  {r1.optimized_tokens} tokens")
print(f"Aggressive: {r2.reduction_percent}%  →  {r2.optimized_tokens} tokens")
```

### Custom — tune every parameter

```python
from tokenmesh import TokenMesh

tm = TokenMesh(
    chunk_size=40,           # words per chunk — smaller = finer dedup
    dedup_threshold=0.85,    # 0.70–0.95 range. Lower = more aggressive
    min_relevance=0.30,      # drop chunks scoring below this (0–1)
    token_budget=1500,       # hard cap — never exceed N output tokens
    normalize=True,          # strip filler phrases before chunking
)
result = tm.optimize(text, query="optional — activates Stage 3 scoring")
```

---

## Claude Integration

### Method 1 — Drop-in client (zero changes to your existing code)

```python
from tokenmesh.integrations.claude import TokenMeshClaude

# Before: client = anthropic.Anthropic()
# After:
client = TokenMeshClaude(
    optimize_system=True,     # compress system prompt  (default: True)
    optimize_user=False,      # leave user query intact (default: False)
    mesh_kwargs={
        "chunk_size": 40,
        "dedup_threshold": 0.82,
        "token_budget": 2000,
    }
)

# Single turn — same API as anthropic.Anthropic
response = client.chat(
    system=long_system_prompt,
    user="your question",
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
)

# Multi-turn with history
response = client.chat(
    system=long_system_prompt,
    user="follow-up",
    messages=conversation_history,   # list of {role, content} dicts
)

# Streaming
for chunk in client.stream(system=long_system_prompt, user="explain this"):
    print(chunk, end="", flush=True)
```

### Method 2 — Compress then pass to your own client

```python
import anthropic
from tokenmesh import TokenMeshLite

result = TokenMeshLite().optimize(long_system_prompt, query=user_question)

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=result.optimized_text,    # compressed
    messages=[{"role": "user", "content": user_question}],
)
```

### Method 3 — claude.ai Projects (no code)

Run once, paste into your Project's system prompt in claude.ai.

```bash
# Mac
python -c "
from tokenmesh import TokenMeshLite
print(TokenMeshLite().optimize(open('prompt.txt').read()).optimized_text)
" | pbcopy

# Windows
python -c "
from tokenmesh import TokenMeshLite
print(TokenMeshLite().optimize(open('prompt.txt').read()).optimized_text)
" | clip
```

---

## Result Object

```python
result = tm.optimize(text, query="...")

result.optimized_text              # str   — compressed text, ready to send
result.original_tokens             # int   — tokens before
result.optimized_tokens            # int   — tokens after
result.reduction_percent           # float — e.g. 54.2
result.saved_tokens                # int   — tokens removed
result.estimated_savings_usd       # float — cost saved (Sonnet pricing)
result.elapsed_ms                  # float — pipeline latency in ms
result.chunks_original             # int   — chunks before dedup
result.chunks_kept                 # int   — chunks after dedup + scoring
result.normalization_tokens_saved  # int   — tokens removed by Stage 0
result.summary()                   # str   — one-line report
```

---

## 💡 Use Cases

Real production prompts — not cherry-picked FAQ text.

| Content | Original | Lite | Aggressive |
|---|---|---|---|
| Trading system prompt | 684 | 430 (37%) | 290 (57%) |
| Repetitive FAQ doc | 184 | 120 (35%) | 77 (58%) |
| Long pasted article | 1,200 | 820 (32%) | 480 (60%) |
| RAG context chunks | 3,000 | 1,900 (37%) | 1,050 (65%) |

### vs. alternatives (real-world avg)

```
TokenMesh Aggressive   ████████████████████████████████  ~65%
LLMLingua              ████████████████████              ~44%
Graphify               ████████████████                  ~38%  
LangChain trim         ████████████                      ~25%
Naive truncation       ██████████████████████████  50% but destroys content
```

| | TokenMesh | Graphify | LLMLingua | LangChain |
|---|---|---|---|---|
| Query-aware | ✅ | ❌ | ⚠️ | ❌ |
| Full sentence preserved | ✅ | ⚠️ | ❌ | ⚠️ |
| CPU-only | ✅ | ❌ | ❌ | ✅ |
| Latency / 1K tokens | **4–8ms** | 40–80ms | 20–40ms | ~2ms |

Compress long conversations without losing meaning

## How It Works

```
Your text (8,000 tokens)
    │
    ▼  Stage 0 — Normalize         no model, regex only, ~0.1ms
    │  Removes: "It is important to note that", "Furthermore,",
    │  "under any circumstances", ### headers, **bold**, <html>
    │  Saves 6–12% before anything else runs
    │
    ▼  Stage 1 — Chunk             splits at sentence boundaries
    │  40-word windows, overlap=0 (overlap causes output inflation)
    │
    ▼  Stage 2 — Semantic Dedup    all-MiniLM-L6-v2, 22MB, CPU
    │  "Risk management is critical" ≈ "Never skip stop losses"
    │  → same meaning, different words → one gets dropped
    │  → longer chunk always kept
    │
    ▼  Stage 3 — Importance Score  only runs if query is provided
       Reuses Stage 2 embeddings (no second model call)
       Drops chunks irrelevant to the specific user question
    │
    ▼
Optimized text → send to Claude
```

---

## Monthly Savings

```
500 calls/day  ×  400 tokens saved/call  =  6M tokens/month

Claude Sonnet ($3/1M input tokens)       =  $18/month saved
Claude Opus   ($15/1M input tokens)      =  $90/month saved
```

---

## 🛣 Roadmap

* [ ] Async support (`await client.achat(...)`)
* [ ] OpenAI / Gemini integrations
* [ ] Retrieval-Augmented Pruning (FAISS)
* [ ] CLI tool

**All contributions welcome** — from fixing a typo to building a new integration.

### Setup (2 minutes)

```bash
git clone https://github.com/AnkitSingh36/tokenmesh
cd tokenmesh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v          # all 10 should pass
```

### Project map

```
tokenmesh/core/normalizer.py    ← add filler patterns here (easiest contribution)
tokenmesh/core/chunker.py       ← sentence splitter + sliding window
tokenmesh/core/deduplicator.py  ← neural cosine dedup
tokenmesh/core/scorer.py        ← query-aware relevance filter
tokenmesh/pipeline.py           ← main API + Lite/Aggressive presets
tokenmesh/integrations/         ← add OpenAI/Gemini/LangChain here
tests/test_pipeline.py          ← add your test cases here
```

### Good first issues

| Task | File | Effort |
|---|---|---|
| Add filler patterns | `core/normalizer.py` | 5 min |
| Add edge case tests | `tests/test_pipeline.py` | 15 min |
| Write OpenAI integration | `integrations/openai.py` | 1 hr |
| Add async `aoptimize()` | `pipeline.py` | 2 hr |
| Build CLI tool | new `cli.py` | half day |

### Add a filler pattern — the 5-minute contribution

Open `tokenmesh/core/normalizer.py` and add to `_FILLER_RULES`:

```python
(r"\bAs you can see,?\s*", ""),
(r"\bWith that out of the way,?\s*", ""),
(r"\bAt the end of the day,?\s*", ""),
```

Run `pytest tests/ -v`. If green, open a PR. Done.

### Add an OpenAI integration — the 1-hour contribution

Create `tokenmesh/integrations/openai.py`:

```python
from openai import OpenAI
from tokenmesh.pipeline import TokenMesh

class TokenMeshOpenAI:
    def __init__(self, api_key=None, mesh_kwargs=None):
        self._client = OpenAI(api_key=api_key)
        self._mesh = TokenMesh(**(mesh_kwargs or {}))

    def chat(self, system, user, model="gpt-4o", max_tokens=1024):
        result = self._mesh.optimize(system, query=user)
        return self._client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": result.optimized_text},
                {"role": "user",   "content": user},
            ],
        )
```

Export from `integrations/__init__.py`, add one test, open a PR.

### PR checklist

- `pytest tests/ -v` all green
- New code has docstrings
- New features have at least one test
- No new dependencies without an issue first

### Bug report

Open an issue with: Python version, OS, `pip show tokenmesh` output, minimal reproduction code, expected vs actual behavior.

### Feature request

Open a Discussion (not an Issue). Describe the problem it solves and whether you'd build it.

---

## Roadmap

- [ ] `v0.3` — Async `aoptimize()` + `achat()`
- [ ] `v0.3` — OpenAI / Gemini integration
- [ ] `v0.4` — FAISS retrieval for 10K+ token documents
- [ ] `v0.4` — CLI: `tokenmesh optimize input.txt --query "..."`
- [ ] `v0.5` — Flutter/Dart SDK

---

## 📜 License

MIT — use it, fork it, ship it.

---

*If TokenMesh saved you tokens — leave a ⭐*
