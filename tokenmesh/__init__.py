"""
TokenMesh — Semantic token optimizer for LLM prompts.

Reduce your Claude/GPT-4 token usage by 40–75% with zero semantic loss
using a three-stage pipeline: sliding window chunking → semantic dedup
→ query-aware importance scoring.

Quickstart::

    from tokenmesh import TokenMesh

    tm = TokenMesh()
    result = tm.optimize(long_document, query="What are the key risks?")

    print(result.optimized_text)       # Feed directly to Claude
    print(result.reduction_percent)    # e.g. 62.4
    print(result.summary())            # One-line stats

Drop-in Claude client::

    from tokenmesh.integrations.claude import TokenMeshClaude

    client = TokenMeshClaude()
    response = client.chat(system=long_prompt, user="Summarize the risks")
    print(response.content)
    print(f"Saved {response.token_savings} tokens")
"""

from tokenmesh.pipeline import OptimizeResult, TokenMesh

__version__ = "0.2.0"
__all__ = ["TokenMesh", "OptimizeResult"]
