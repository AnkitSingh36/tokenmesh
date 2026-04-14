"""
examples/claude_integration.py

TokenMesh + Claude API — automatic token optimization on every call.

Set your API key:
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import os

from tokenmesh.integrations.claude import TokenMeshClaude

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# A very long system prompt (simulate a real trading assistant prompt)
LONG_SYSTEM_PROMPT = """
You are an expert swing trading assistant specializing in NSE and BSE equity markets.
You are a knowledgeable financial co-pilot helping retail Indian investors trade profitably.
Your role is to assist with trade analysis, scanner interpretation, and risk management.
You help Indian retail investors make smarter decisions in NSE and BSE equity markets.

Always apply strict risk management: every trade must have a defined stop loss.
Never let a losing trade run beyond the defined stop loss level under any circumstances.
Risk management is non-negotiable — protecting capital is more important than profit.
Never enter a trade without a pre-defined stop loss and target ratio of at least 1:2.

Use technical analysis tools including EMA crossovers, MACD, RSI, and volume analysis.
Identify demand and supply zones using price action and volume at key levels.
Wyckoff accumulation patterns and breakout confirmation are part of your analysis toolkit.
EMA 20/50 crossovers combined with RSI confirmation produce high-probability signals.

Fundamental filters: prefer stocks with debt-to-equity below 1 and consistent profit growth.
Three consecutive quarters of profit growth with low debt is your fundamental filter.
HDFCBANK, MAZDOCK, and WAAREEENER are examples of fundamentally strong companies.
Always check Screener.in for financial health before entering any swing trade.

Monitor India VIX for market-wide risk assessment before taking new positions.
High VIX (above 20) means reduce position size and tighten stop losses.
PCR and Max Pain levels at key expiry dates provide additional market context.
FII and DII flows give insight into institutional money movement in the market.

Your trading routine: pre-market scan 8:45-9:15 AM, no trades in first 15 minutes,
GTT order placement by 9:30 AM, post-market review at 3:30 PM daily.
Use Chartink for free scanning, TradingView for charts, Zerodha GTT for order management.
The daily routine is designed to fit around a full-time employment schedule.

Diversification: index SIPs via Nifty 50, US tech ETF, swing trading, gold ETF, limited crypto.
Long-term wealth building through disciplined systematic investment is the core philosophy.
Swing trading supplements passive index investing — it is not the only strategy.
Crypto exposure should be capped at 5-7% of total portfolio at all times.
"""

# ── Example 1: One-shot chat with auto-optimization ─────────────────────────
print("=" * 60)
print("Example 1: Chat with System Prompt Optimization")
print("=" * 60)

client = TokenMeshClaude(
    api_key=API_KEY,
    optimize_system=True,
    mesh_kwargs={"chunk_size": 60, "dedup_threshold": 0.82, "top_k": 12},
)

if not API_KEY:
    print("[Skipped — set ANTHROPIC_API_KEY to run live]")
    print("\nHow it works:")
    print("  - System prompt fed into TokenMesh pipeline")
    print("  - Semantic duplicates removed automatically")
    print("  - Top-K relevant chunks kept based on your query")
    print("  - Optimized prompt sent to Claude")
    print("  - You see token savings in response.token_savings")
else:
    response = client.chat(
        system=LONG_SYSTEM_PROMPT,
        user="SHRIRAMFIN has broken above 20-EMA with volume. Should I enter?",
        model="claude-sonnet-4-20250514",
        max_tokens=512,
    )
    print(f"Claude says:\n{response.content}")
    print(f"\nToken savings: {response.token_savings} tokens saved")
    print(f"Cost savings:  ${response.savings_usd:.4f} per call")
    print(f"Input tokens used: {response.input_tokens}")


# ── Example 2: Streaming response ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Example 2: Streaming with Optimization")
print("=" * 60)

if not API_KEY:
    print("[Skipped — set ANTHROPIC_API_KEY to run live]")
else:
    print("Streaming Claude response...\n")
    for chunk in client.stream(
        system=LONG_SYSTEM_PROMPT,
        user="Give me a quick checklist for pre-market prep.",
        model="claude-sonnet-4-20250514",
        max_tokens=400,
    ):
        print(chunk, end="", flush=True)
    print()


# ── Example 3: Dry run — show optimization stats without API call ────────────
print("\n" + "=" * 60)
print("Example 3: Dry Run — Optimization Stats")
print("=" * 60)

from tokenmesh import TokenMesh

tm = TokenMesh(chunk_size=60, dedup_threshold=0.82)
result = tm.optimize(LONG_SYSTEM_PROMPT, query="SHRIRAMFIN breakout entry analysis")
print(result.summary())
print(f"\nChunks: {result.chunks_original} → {result.chunks_kept} kept")
print(f"Tokens: {result.original_tokens} → {result.optimized_tokens}")
print(f"Time:   {result.elapsed_ms:.0f}ms")
