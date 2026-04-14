"""
Semantic deduplicator.

Embeds text chunks and removes near-duplicates based on cosine similarity,
keeping the most information-dense representative from each cluster.

Fixes vs v0.1:
  BUG #1  Victim was chosen by len(embedding_vector) which is always 384
          for every chunk — now correctly uses len(chunk.text).
  GAP #4  Returns normalized embeddings so scorer can reuse them,
          eliminating a second model.encode() call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tokenmesh.core.chunker import Chunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = None  # Lazy singleton


def _get_embedder():
    """Lazy-load sentence transformer to avoid startup cost."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("Loaded sentence-transformers: all-MiniLM-L6-v2")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
    return _EMBED_MODEL


class SemanticDeduplicator:
    """
    Removes near-duplicate chunks using embedding cosine similarity.

    From each duplicate pair the LONGER (more informative) chunk is kept.

    Args:
        threshold:   Cosine similarity above which chunks are duplicates.
                     0.85 is default. Lower = more aggressive pruning.
        batch_size:  Embedding batch size (tune for GPU if available).
    """

    def __init__(self, threshold: float = 0.85, batch_size: int = 64) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        self.batch_size = batch_size

    def deduplicate(
        self,
        chunks: list[Chunk],
        return_embeddings: bool = False,
    ) -> tuple[list[Chunk], np.ndarray | None]:
        """
        Remove near-duplicate chunks.

        Args:
            chunks:            Chunks to deduplicate.
            return_embeddings: If True, return L2-normalized embeddings of kept
                               chunks so the scorer can reuse them (no re-embed).

        Returns:
            (deduplicated_chunks, normalized_embeddings_or_None)
        """
        if len(chunks) <= 1:
            embs = None
            if return_embeddings and chunks:
                raw = self._embed([chunks[0].text])
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                embs = raw / np.where(norms == 0, 1e-9, norms)
            return chunks, embs

        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)

        # Normalize once — reused for both dedup and (optionally) scoring
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = embeddings / norms

        kept_indices = sorted(self._greedy_dedup(normed, chunks))
        result = [chunks[i] for i in kept_indices]

        logger.debug(
            "Dedup: %d -> %d chunks (%.1f%% removed)",
            len(chunks), len(result),
            (1 - len(result) / len(chunks)) * 100,
        )

        kept_embeddings = normed[kept_indices] if return_embeddings else None
        return result, kept_embeddings

    def _greedy_dedup(self, normed: np.ndarray, chunks: list[Chunk]) -> set[int]:
        """
        Vectorised greedy dedup using a precomputed similarity matrix.

        FIX #1: victim is now chosen by len(chunk.text) — the shorter text
        is dropped, keeping the more detailed chunk. Previously this used
        len(embedding_vector) which is the same value (384) for every chunk,
        causing the wrong chunk to always be dropped.
        """
        sim_matrix = normed @ normed.T   # shape (N, N), values in [-1, 1]
        n = len(chunks)
        removed: set[int] = set()

        for i in range(n):
            if i in removed:
                continue
            for j in range(i + 1, n):
                if j in removed:
                    continue
                if sim_matrix[i, j] >= self.threshold:
                    # Keep the longer (more informative) chunk
                    victim = i if len(chunks[i].text) < len(chunks[j].text) else j
                    removed.add(victim)

        return set(range(n)) - removed

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = _get_embedder()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize manually for reuse
        )
