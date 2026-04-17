"""
Query-aware importance scorer.

Ranks chunks by semantic relevance to a user query.

Fix vs v0.1 (GAP #4):
  Accepts pre-computed embeddings from the deduplicator so chunks are
  not re-embedded. This makes the full pipeline ~40-50% faster.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tokenmesh.core.deduplicator import _get_embedder

if TYPE_CHECKING:
    from tokenmesh.core.chunker import Chunk

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """
    Scores each chunk by cosine similarity to a query.

    Scores are written to chunk.relevance_score (0.0-1.0).

    Args:
        top_k:         Keep only top-K chunks. None = use min_relevance floor.
        min_relevance: Drop chunks below this score (default 0.30).
    """

    def __init__(
        self,
        top_k: int | None = None,
        min_relevance: float = 0.30,
    ) -> None:
        self.top_k = top_k
        self.min_relevance = min_relevance

    def score_and_filter(
        self,
        chunks: list[Chunk],
        query: str,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> list[Chunk]:
        """
        Score chunks against query and return filtered list in original order.

        Args:
            chunks:                 Chunks to score.
            query:                  User query string.
            precomputed_embeddings: L2-normalized embeddings from deduplicator
                                    (shape N x D). When provided, only the query
                                    is re-encoded — saves one full encode pass.
        """
        if not query or not query.strip() or not chunks:
            return chunks

        model = _get_embedder()

        if precomputed_embeddings is not None and len(precomputed_embeddings) == len(chunks):
            # FIX #4: reuse dedup embeddings — only encode the query
            query_emb = model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            chunk_embs = precomputed_embeddings  # already L2-normalized
        else:
            # Fallback: encode query + all chunks in one batch
            all_texts = [query] + [c.text for c in chunks]
            all_embs = model.encode(
                all_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            query_emb = all_embs[0]
            chunk_embs = all_embs[1:]

        # Cosine similarity = dot product on L2-normalized vectors
        scores: np.ndarray = chunk_embs @ query_emb

        for chunk, score in zip(chunks, scores):
            chunk.relevance_score = float(np.clip(score, 0.0, 1.0))

        if self.top_k is not None:
            by_score = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
            kept = by_score[: self.top_k]
            result = sorted(kept, key=lambda c: c.index)
        else:
            result = [c for c in chunks if c.relevance_score >= self.min_relevance]

        logger.debug(
            "Scoring: %d -> %d chunks kept (query: %.40s)",
            len(chunks), len(result), query,
        )
        return result
