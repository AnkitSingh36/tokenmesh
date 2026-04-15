"""
examples/visual_demo.py
=======================
Visual demonstration of TokenMesh token optimization.

Shows before/after token counts, reduction percentage, elapsed time,
and generates matplotlib charts for a clear visual comparison.

Run:
    python examples/visual_demo.py
"""

import time
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Add project root to path so this runs from any working directory ─────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenmesh import TokenMesh

matplotlib.rcParams["font.family"] = "monospace"

# ── Sample long text (hardcoded — intentionally verbose and repetitive) ──────
SAMPLE_TEXT = """
You are an expert swing trading assistant specializing in NSE and BSE equity markets.
You are a knowledgeable financial co-pilot helping retail Indian investors trade profitably.
Your role is to assist with trade analysis, scanner interpretation, and risk management.
You help Indian retail investors make smarter decisions in NSE and BSE equity markets.

Always apply strict risk management: every trade must have a defined stop loss.
Never let a losing trade run beyond the defined stop loss level under any circumstances.
Risk management is non-negotiable — protecting capital is more important than profit.
Never enter a trade without a pre-defined stop loss and target ratio of at least 1:2.
Stop losses are mandatory on every single position without any exception whatsoever.

Use technical analysis tools including EMA crossovers, MACD, RSI, and volume analysis.
Identify demand and supply zones using price action and volume at key price levels.
Wyckoff accumulation patterns and breakout confirmation are part of your analysis toolkit.
EMA 20/50 crossovers combined with RSI confirmation produce high-probability trade signals.
Technical indicators like MACD and RSI help confirm the direction of the price trend.

Fundamental filters: prefer stocks with debt-to-equity below 1 and consistent profit growth.
Three consecutive quarters of profit growth with low debt is your fundamental screening filter.
HDFCBANK, MAZDOCK, and WAAREEENER are examples of fundamentally strong large-cap companies.
Always check Screener.in for financial health before entering any swing trade position.
Fundamental analysis of balance sheets and income statements is important before trading.

Monitor India VIX for market-wide risk assessment before taking any new positions in markets.
High VIX above 20 means reduce your position size and tighten all stop loss levels immediately.
PCR and Max Pain levels at key expiry dates provide additional market context for trading.
FII and DII flows give important insight into institutional money movement across the market.
India VIX is a critical indicator that must be checked before opening any new trade position.

Your trading routine: pre-market scan 8:45-9:15 AM, no trades in first 15 minutes of open.
GTT order placement by 9:30 AM, post-market review at 3:30 PM every single trading day.
Use Chartink for free scanner-based screening, TradingView for chart analysis and study.
The daily trading routine is designed to fit around a full-time day job employment schedule.
Pre-market preparation is essential and must be completed before the market opens each day.

Diversification: index SIPs via Nifty 50, US tech ETF, active swing trading, gold ETF.
Long-term wealth building through disciplined systematic investment is the core philosophy.
Swing trading supplements passive index investing — active trading is not the only strategy.
Crypto exposure should be capped at 5-7 percent of total portfolio value at all times always.
A well-diversified portfolio reduces concentration risk and smooths long-term returns over time.

When analyzing breakouts, always confirm with volume. A breakout without volume is a fake-out.
Look for at least 1.5x average volume on the breakout candle to confirm genuine buying interest.
Volume confirmation is mandatory before entering any breakout trade in NSE or BSE markets.
Low-volume breakouts frequently fail and result in false signals that cause unnecessary losses.
Volume is the most important confirmation signal for any price breakout above resistance levels.

Use trailing stop losses once the trade moves 1R in your favor after entry.
Move your stop loss to breakeven once the trade is up by at least 1x the initial risk taken.
Trail the stop loss by 10-EMA on the daily chart once the position is up by 2R profit target.
Protecting profits with trailing stops is as important as the initial stop loss placement entry.
Never let a winning trade turn into a losing trade — always trail stops to lock in profits.

Demand zones are identified by a strong impulsive move away from a price level with high volume.
Supply zones form when price makes a sharp decline from a price level on above-average volume.
Always mark the most recent and most significant demand and supply zones on the price chart.
Trade entries from the most significant demand zone visible on the daily chart time frame only.
Supply zones act as overhead resistance and demand zones act as strong support price levels.
""".strip()

# ── User query (drives query-aware importance scoring in Stage 3) ─────────────
QUERY = "risk management stop loss rules and trading routine"


def print_bar(label: str, tokens: int, max_tokens: int, bar_width: int = 40) -> None:
    """Print a single ASCII progress bar representing token count."""
    filled = int((tokens / max_tokens) * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {label:<10} {bar}  {tokens:,} tokens")


def print_console_report(result) -> None:
    """Print a clean before/after summary to the terminal."""
    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  TokenMesh v0.2  ·  Optimization Report")
    print(sep)

    # ASCII bar comparison
    max_t = result.original_tokens
    print_bar("Before:", result.original_tokens, max_t)
    print_bar("After: ", result.optimized_tokens, max_t)

    print(sep)

    # Key metrics
    print(f"  Original tokens   : {result.original_tokens:,}")
    print(f"  Optimized tokens  : {result.optimized_tokens:,}")
    print(f"  Tokens saved      : {result.saved_tokens:,}")
    print(f"  Reduction         : {result.reduction_percent:.1f}%")
    print(f"  Normalizer saved  : {result.normalization_tokens_saved} tokens (Stage 0)")
    print(f"  Chunks            : {result.chunks_original} → {result.chunks_kept} kept")
    print(f"  Elapsed           : {result.elapsed_ms:.1f} ms")
    print(f"  Est. cost saved   : ${result.estimated_savings_usd:.5f} / call")
    print(f"  Est. monthly save : ${result.estimated_savings_usd * 500 * 30:.2f}  (500 calls/day)")
    print(sep)


def build_charts(result) -> plt.Figure:
    """
    Build a 2x2 figure with four panels:
      [0,0] Bar chart    — original vs optimised token counts
      [0,1] Donut chart  — reduction vs retained breakdown
      [1,0] Stage breakdown — tokens removed per pipeline stage
      [1,1] Cost projection — monthly savings at different call volumes
    """
    orig = result.original_tokens
    opti = result.optimized_tokens
    saved = result.saved_tokens
    pct = result.reduction_percent
    norm_saved = result.normalization_tokens_saved
    dedup_saved = max(0, saved - norm_saved)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "TokenMesh  ·  Optimization Report",
        fontsize=15, fontweight="bold", y=0.98
    )
    fig.patch.set_facecolor("#f9f9f9")
    for ax in axes.flat:
        ax.set_facecolor("#f9f9f9")

    # ── Panel 1: Bar chart — token counts ─────────────────────────────────
    ax1 = axes[0, 0]
    bars = ax1.bar(
        ["Original", "Optimized"],
        [orig, opti],
        width=0.45,
        edgecolor="white",
        linewidth=1.2,
    )
    # Annotate each bar with its value
    for bar, val in zip(bars, [orig, opti]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + orig * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    # Draw a reduction arrow between the bars
    ax1.annotate(
        f"−{pct:.1f}%",
        xy=(1, opti), xytext=(0.5, (orig + opti) / 2),
        fontsize=12, fontweight="bold", color="darkgreen",
        ha="center",
        arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5),
    )
    ax1.set_title("Token Reduction", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Token count")
    ax1.set_ylim(0, orig * 1.22)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: Donut — what fraction was removed ─────────────────────────
    ax2 = axes[0, 1]
    sizes = [opti, saved]
    labels = [f"Kept\n{opti:,} tokens", f"Removed\n{saved:,} tokens"]
    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    autotexts[1].set_fontweight("bold")
    ax2.set_title("Composition After Optimization", fontsize=12, fontweight="bold", pad=10)
    # Centre label
    ax2.text(
        0, 0, f"{pct:.1f}%\nreduced",
        ha="center", va="center", fontsize=13, fontweight="bold"
    )

    # ── Panel 3: Stage breakdown — where did savings come from ─────────────
    ax3 = axes[1, 0]
    stages = ["Stage 0\nNormalizer", "Stage 2\nDedup + Score", "Remaining\n(kept)"]
    values = [norm_saved, dedup_saved, opti]
    colors_stage = ["#5a9fd4", "#f4a261", "#a8d5a2"]
    b = ax3.barh(stages, values, color=colors_stage, edgecolor="white", linewidth=1.2, height=0.5)
    for rect, val in zip(b, values):
        ax3.text(
            rect.get_width() + orig * 0.01,
            rect.get_y() + rect.get_height() / 2,
            f"{val:,} tokens",
            va="center", fontsize=10
        )
    ax3.set_title("Savings by Pipeline Stage", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Tokens")
    ax3.set_xlim(0, orig * 1.25)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax3.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Monthly cost projection ──────────────────────────────────
    ax4 = axes[1, 1]
    # Claude Sonnet pricing: $3 / 1M input tokens
    call_volumes = [100, 250, 500, 1000, 2000, 5000]
    # Monthly = daily × 30
    savings_original = [
        (orig / 1_000_000) * 3.0 * v * 30 for v in call_volumes
    ]
    savings_optimized = [
        (opti / 1_000_000) * 3.0 * v * 30 for v in call_volumes
    ]
    ax4.plot(call_volumes, savings_original, marker="o", linewidth=2, label="Without TokenMesh")
    ax4.plot(call_volumes, savings_optimized, marker="s", linewidth=2, label="With TokenMesh")
    ax4.fill_between(call_volumes, savings_optimized, savings_original, alpha=0.15)

    # Annotate the gap at 1000 calls/day
    idx = call_volumes.index(1000)
    gap = savings_original[idx] - savings_optimized[idx]
    ax4.annotate(
        f"Save ${gap:.0f}/mo\nat 1K calls/day",
        xy=(1000, (savings_original[idx] + savings_optimized[idx]) / 2),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        xytext=(1200, savings_original[idx] * 0.6),
    )
    ax4.set_title("Monthly Cost Projection (Sonnet)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("API calls per day")
    ax4.set_ylabel("Monthly cost (USD)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax4.legend(fontsize=9)
    ax4.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> None:
    print("\n  TokenMesh  ·  Visual Demo")
    print("  Running optimization pipeline...\n")

    # Step 1: initialise TokenMesh with recommended settings
    tm = TokenMesh(
        chunk_size=40,          # sentence-level granularity
        overlap=0,              # no overlap inflation
        dedup_threshold=0.82,   # balanced deduplication
        normalize=True,         # strip filler phrases first
    )

    # Step 2: run optimization (neural model downloads on first call ~30s)
    result = tm.optimize(SAMPLE_TEXT, query=QUERY)

    # Step 3: print console report
    print_console_report(result)

    # Step 4: build and show charts
    print("\n  Generating charts...")
    fig = build_charts(result)

    # Save to file next to this script
    output_path = os.path.join(os.path.dirname(__file__), "tokenmesh_demo.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Chart saved → {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
