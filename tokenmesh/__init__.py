"""
TokenMesh — Semantic token optimizer for LLM prompts.

Reduce Claude / GPT-4 token usage by 40-75% with zero semantic loss
using a four-stage pipeline: normalize → chunk → dedup → score.

Quickstart::

    from tokenmesh import TokenMesh
    tm = TokenMesh()
    result = tm.optimize(text, query="...")
    print(result.optimized_text)

Preset modes::

    from tokenmesh import TokenMeshLite        # safe, 20-40% reduction
    from tokenmesh import TokenMeshAggressive  # max, 55-75% reduction

Drop-in Claude client::

    from tokenmesh.integrations.claude import TokenMeshClaude
    client = TokenMeshClaude()
    response = client.chat(system=long_prompt, user="your question")
"""

from tokenmesh.pipeline import OptimizeResult, TokenMesh, TokenMeshAggressive, TokenMeshLite

__version__ = "0.2.0"
__all__ = ["TokenMesh", "TokenMeshLite", "TokenMeshAggressive", "OptimizeResult"]
