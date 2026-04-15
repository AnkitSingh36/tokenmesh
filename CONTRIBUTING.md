# Contributing to TokenMesh

Thank you for taking the time. Every contribution — from a one-line filler pattern to a full new integration — makes TokenMesh more useful for everyone building on Claude and other LLMs.

---

## Quick setup

```bash
git clone https://github.com/yourusername/tokenmesh
cd tokenmesh

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # installs package + pytest + ruff + mypy

pytest tests/ -v                 # 10/10 should pass before you change anything
```

---

## How the codebase is structured

```
tokenmesh/
├── tokenmesh/
│   ├── __init__.py            exports TokenMesh, TokenMeshLite, TokenMeshAggressive
│   ├── pipeline.py            main API — TokenMesh class + preset factories
│   ├── core/
│   │   ├── normalizer.py      Stage 0 — regex filler/markdown stripping
│   │   ├── chunker.py         Stage 1 — sentence-aware sliding window
│   │   ├── deduplicator.py    Stage 2 — neural cosine dedup (all-MiniLM-L6-v2)
│   │   └── scorer.py          Stage 3 — query-aware importance scoring
│   ├── integrations/
│   │   └── claude.py          TokenMeshClaude drop-in client
│   └── utils/
│       └── tokencount.py      tiktoken-based counting + cost estimation
├── examples/
│   ├── basic_usage.py
│   ├── claude_integration.py
│   └── visual_demo.py
└── tests/
    └── test_pipeline.py
```

The pipeline stages run in order: `text → normalize → chunk → dedup → score → output`. Each stage is self-contained. To improve one stage, you only need to read that file.

---

## Types of contributions

### Type 1 — Add a filler pattern (5 minutes)

This is the easiest and most impactful contribution. Open `tokenmesh/core/normalizer.py` and add your pattern to the `_FILLER_RULES` list:

```python
_FILLER_RULES: list[tuple[str, str]] = [
    # existing patterns ...
    (r"\bAs you can see,?\s*", ""),
    (r"\bWith that out of the way,?\s*", ""),
    (r"\bAt the end of the day,?\s*", ""),
    (r"\bNeedless to say,?\s*", ""),   # wait, this one is already there!
]
```

Rules:
- Pattern must be case-insensitive (the compiler applies `re.IGNORECASE` automatically)
- Replacement must be `""` (empty string) — the normalizer adds space cleanup after
- Add only phrases that carry zero semantic value — phrases that exist to fill sentences, not inform them
- Run `pytest tests/ -v` to confirm nothing breaks

---

### Type 2 — Add an integration (1–2 hours)

Create a new file in `tokenmesh/integrations/`. Model it after `claude.py`.

**OpenAI example skeleton:**

```python
# tokenmesh/integrations/openai.py
from __future__ import annotations
from tokenmesh.pipeline import TokenMesh

class TokenMeshOpenAI:
    def __init__(self, api_key=None, optimize_system=True, mesh_kwargs=None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        self._client = OpenAI(api_key=api_key)
        self._optimize_system = optimize_system
        self._mesh = TokenMesh(**(mesh_kwargs or {}))

    def chat(self, user, system="", model="gpt-4o", max_tokens=1024, **kwargs):
        optimized_system = system
        if system and self._optimize_system:
            result = self._mesh.optimize(system, query=user)
            optimized_system = result.optimized_text

        messages = []
        if optimized_system:
            messages.append({"role": "system", "content": optimized_system})
        messages.append({"role": "user", "content": user})

        return self._client.chat.completions.create(
            model=model, max_tokens=max_tokens, messages=messages, **kwargs
        )
```

Then:
1. Export from `tokenmesh/integrations/__init__.py`
2. Add at least one test in `tests/test_pipeline.py`
3. Add a usage example in `examples/`

---

### Type 3 — Add a test case (15 minutes)

Open `tests/test_pipeline.py` and add a function:

```python
def test_normalizer_removes_filler():
    """Normalizer must strip known filler phrases."""
    from tokenmesh.core.normalizer import TextNormalizer
    norm = TextNormalizer()
    result = norm.normalize("It is important to note that risk matters.")
    assert "It is important to note that" not in result
    assert "risk matters" in result

def test_chunker_handles_unicode():
    """Chunker must not crash on non-ASCII input."""
    chunker = SlidingWindowChunker(chunk_size=20)
    chunks = chunker.split("Nifty ₹50 index. SEBI नियामक. 股票市场。")
    assert len(chunks) >= 1
```

Run `pytest tests/ -v -k test_normalizer_removes_filler` to verify your test.

---

### Type 4 — Improve the pipeline (half day+)

Open an issue first describing the change. Main areas open for improvement:

**Async support** — `pipeline.py` needs an `aoptimize()` method that awaits the embedding call. The deduplicator's `_embed()` call is the bottleneck. Wrapping it with `asyncio.to_thread()` in Python 3.9+ is the simplest path.

**Token budget loop** — `pipeline._enforce_budget()` currently removes chunks by lowest relevance score. A smarter strategy would cluster remaining chunks and remove the least-distinct cluster rather than the least-relevant single chunk.

**CLI** — add `tokenmesh/cli.py` using `argparse` or `click`:
```bash
tokenmesh optimize input.txt --query "key risks" --mode lite
tokenmesh optimize input.txt --budget 1500 --output compressed.txt
```

**FAISS retrieval** — for 10K+ token documents, a FAISS index retrieval before dedup would make Stage 3 significantly faster. Implementation lives in a new `tokenmesh/core/retriever.py`.

---

## Running tests

```bash
pytest tests/ -v                          # all tests
pytest tests/ -v -k test_chunker          # specific test
pytest tests/ -v --tb=short              # short traceback
```

To test against the real Claude API (requires `ANTHROPIC_API_KEY`):

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/claude_integration.py
```

---

## Code style

```bash
ruff check tokenmesh/          # linting
ruff check tokenmesh/ --fix    # auto-fix safe issues
mypy tokenmesh/                # type checking (optional for first PRs)
```

Rules that matter:
- Functions under 50 lines where possible
- Docstrings on every public class and method
- Type hints on all function signatures
- No `print()` in library code — use `logging.debug()` / `logging.info()`

---

## PR process

1. Fork the repo, create a branch: `git checkout -b feat/openai-integration`
2. Make your changes
3. Run `pytest tests/ -v` — all green
4. Run `ruff check tokenmesh/` — no errors
5. Open a PR with a clear title and description of what changes and why
6. A maintainer will review within a few days

Small PRs (< 100 lines changed) get reviewed and merged faster than large ones. If you're building something large, open a Discussion first so we can agree on the approach before you write the code.

---

## Getting help

- **Question about the code** → open a Discussion
- **Found a bug** → open an Issue with reproduction steps
- **Want to build something large** → open a Discussion first

---

Thank you for contributing.
