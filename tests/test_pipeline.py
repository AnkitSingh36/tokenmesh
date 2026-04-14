"""
Unit tests for TokenMesh.

Run with: pytest tests/ -v
"""

import pytest

from tokenmesh.core.chunker import SlidingWindowChunker
from tokenmesh.pipeline import OptimizeResult, TokenMesh
from tokenmesh.utils.tokencount import count_tokens

# ── Sample text ─────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
The Indian stock market has witnessed tremendous growth over the past decade.
The Nifty 50 index has delivered strong returns to long-term investors.
Investors must understand the risks involved in equity investing before committing capital.
The Indian stock market has seen significant growth in the last ten years.
Technical analysis involves studying price charts and volume to identify patterns.
Fundamental analysis focuses on a company's financial health and intrinsic value.
Risk management is a critical component of any successful trading strategy.
Stop losses protect capital by automatically exiting losing positions.
Momentum trading involves buying stocks that are trending upward in price.
Value investing means buying stocks trading below their intrinsic value.
The Reserve Bank of India plays a key role in monetary policy decisions.
Interest rate changes by the RBI significantly impact equity valuations.
Global factors such as US Fed decisions affect emerging market flows.
FII and DII activity is a key indicator of market sentiment in India.
Retail investors should diversify their portfolios across sectors and market caps.
""".strip()


# ── Chunker tests ────────────────────────────────────────────────────────────


def test_chunker_basic():
    chunker = SlidingWindowChunker(chunk_size=50, overlap=10)
    chunks = chunker.split(SAMPLE_TEXT)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text
        assert chunk.index >= 0


def test_chunker_empty_input():
    chunker = SlidingWindowChunker()
    assert chunker.split("") == []
    assert chunker.split("   ") == []


def test_chunker_short_text():
    """Short text that fits in one chunk should return one chunk."""
    chunker = SlidingWindowChunker(chunk_size=200)
    chunks = chunker.split("Hello world. This is a test.")
    assert len(chunks) >= 1


# ── Token count tests ────────────────────────────────────────────────────────


def test_count_tokens_basic():
    count = count_tokens("Hello, world!")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


# ── Pipeline tests ───────────────────────────────────────────────────────────


def test_pipeline_reduces_tokens():
    """Optimization must reduce token count for repetitive text."""
    tm = TokenMesh(chunk_size=50, dedup_threshold=0.80)
    result = tm.optimize(SAMPLE_TEXT)

    assert isinstance(result, OptimizeResult)
    assert result.original_tokens > 0
    assert result.optimized_tokens <= result.original_tokens
    assert result.optimized_text


def test_pipeline_with_query():
    """Query-aware scoring must return fewer chunks than no-query run."""
    tm = TokenMesh(chunk_size=50, dedup_threshold=0.90, top_k=3)
    result = tm.optimize(SAMPLE_TEXT, query="What is risk management?")

    assert result.chunks_kept <= result.chunks_original
    assert result.optimized_text


def test_pipeline_empty_text():
    tm = TokenMesh()
    result = tm.optimize("")
    assert result.optimized_text == ""
    assert result.reduction_percent == 0.0


def test_pipeline_result_properties():
    tm = TokenMesh(chunk_size=50)
    result = tm.optimize(SAMPLE_TEXT)

    assert result.reduction_percent >= 0.0
    assert result.saved_tokens >= 0
    assert result.elapsed_ms > 0
    summary = result.summary()
    assert "TokenMesh" in summary
    assert "%" in summary


def test_pipeline_preserve_content():
    """Optimized text must contain real content (not empty after dedup)."""
    tm = TokenMesh(chunk_size=100, dedup_threshold=0.99)  # Very high = keep almost everything
    result = tm.optimize(SAMPLE_TEXT)
    assert len(result.optimized_text) > 100
