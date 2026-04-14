"""
Token counting utilities.

Uses tiktoken for accurate counts compatible with OpenAI/Anthropic tokenizers.
Falls back to a word-based estimate if tiktoken is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ENCODER = None
_TIKTOKEN_AVAILABLE = False

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")  # Claude/GPT-4 compatible
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning(
        "tiktoken not installed — using word-based token estimate. "
        "Install with: pip install tiktoken"
    )


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base) or word estimate."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Fallback: English averages ~0.75 tokens/word
    return max(1, int(len(text.split()) * 1.33))


def estimate_cost_usd(
    token_count: int,
    model: str = "claude-sonnet-4-20250514",
) -> float:
    """
    Rough cost estimate in USD for input tokens.

    Prices are indicative and may change — check Anthropic pricing page
    for accurate current rates.
    """
    # $/1M input tokens (approximate)
    pricing = {
        "claude-opus-4-20250514": 15.00,
        "claude-sonnet-4-20250514": 3.00,
        "claude-haiku-4-5-20251001": 0.80,
    }
    rate = pricing.get(model, 3.00)
    return (token_count / 1_000_000) * rate
