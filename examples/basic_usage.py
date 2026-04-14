"""
examples/basic_usage.py

TokenMesh basic usage — no Claude API key needed.
"""

from tokenmesh import TokenMesh

# A long, repetitive document (simulate a pasted blog post or research paper)
document = """
The Indian equity market has experienced remarkable growth over the past decade.
Nifty 50 has consistently delivered strong compounding returns for patient investors.
The stock market in India has seen impressive performance over the last ten years.
Long-term investors in Nifty 50 have been rewarded with excellent compounding.

Risk management is the foundation of every professional trading system.
Without a stop loss, a single bad trade can wipe out weeks of gains.
Every successful trader prioritizes protecting capital above chasing returns.
Proper risk management prevents one losing trade from destroying your account.

Technical analysis uses price action, volume, and indicators to time entries.
Chart patterns like breakouts, flags, and demand zones help identify opportunities.
Moving averages, MACD, and RSI are core tools in the technical analyst's toolkit.
Price action combined with volume confirmation is the basis of technical trading.

Fundamental analysis examines financial statements, earnings growth, and valuation.
Companies with consistent profit growth and low debt make strong long-term holds.
Strong fundamentals paired with good technicals give the highest probability trades.
Debt-to-equity below 1 and three consecutive quarters of growth are key filters.
"""

# ── Example 1: Basic optimization (no query) ────────────────────────────────
print("=" * 60)
print("Example 1: Basic Optimization")
print("=" * 60)

tm = TokenMesh(chunk_size=60, dedup_threshold=0.82)
result = tm.optimize(document)

print(result.summary())
print(f"\nOriginal ({result.original_tokens} tokens):")
print(document[:200] + "...")
print(f"\nOptimized ({result.optimized_tokens} tokens):")
print(result.optimized_text)

# ── Example 2: Query-aware optimization ─────────────────────────────────────
print("\n" + "=" * 60)
print("Example 2: Query-Aware Optimization")
print("=" * 60)

tm2 = TokenMesh(chunk_size=60, dedup_threshold=0.82, top_k=3)
result2 = tm2.optimize(document, query="How do I manage risk in trading?")

print(result2.summary())
print(f"\nQuery-filtered text ({result2.optimized_tokens} tokens):")
print(result2.optimized_text)

# ── Example 3: Tune aggressiveness ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 3: Aggressive Compression")
print("=" * 60)

tm3 = TokenMesh(
    chunk_size=80,
    dedup_threshold=0.75,  # Lower = more aggressive dedup
    min_relevance=0.3,
)
result3 = tm3.optimize(document, query="stock market returns")
print(result3.summary())
