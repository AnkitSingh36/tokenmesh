"""
Sliding window semantic chunker.

Splits text into overlapping chunks with configurable size and stride,
respecting sentence boundaries to avoid mid-sentence cuts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    index: int
    token_count: int = 0
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class SlidingWindowChunker:
    """
    Splits text into sentence-aware sliding window chunks.

    Unlike character-based splitters, this respects sentence boundaries
    so chunks are always semantically coherent units.

    Args:
        chunk_size:   Target number of words per chunk.
                      Default 40 (~1-2 sentences) gives best dedup granularity.
        overlap:      Word overlap between chunks. Default 0 — non-zero overlap
                      causes overlapping words to appear twice in joined output,
                      inflating token count. Only increase for retrieval tasks.
        min_chunk:    Minimum words to keep a trailing chunk (avoids tiny slivers).
    """

    # Fixed-width lookbehinds only — Python re does not support variable-width
    _SENTENCE_END = re.compile(
        r'(?<!Dr)(?<!Mr)(?<!Ms)(?<!Sr)(?<!Jr)(?<!vs)\.\s+|[!?]\s+'
    )

    def __init__(
        self,
        chunk_size: int = 40,
        overlap: int = 0,
        min_chunk: int = 1,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk = min_chunk

    def split(self, text: str) -> list[Chunk]:
        """Split text into chunks, flushing any remainder at the end."""
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        word_buffer: list[str] = []
        chunk_idx = 0

        for sentence in sentences:
            word_buffer.extend(sentence.split())

            if len(word_buffer) >= self.chunk_size:
                chunk_text = " ".join(word_buffer[: self.chunk_size])
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=chunk_idx,
                        token_count=self._estimate_tokens(chunk_text),
                    )
                )
                chunk_idx += 1
                # Slide forward; overlap=0 by default to avoid output inflation
                step = max(1, self.chunk_size - self.overlap)
                word_buffer = word_buffer[step:]

        # Flush remaining words as final chunk
        if len(word_buffer) >= self.min_chunk:
            chunk_text = " ".join(word_buffer)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_idx,
                    token_count=self._estimate_tokens(chunk_text),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text at sentence boundaries, filtering empty strings."""
        parts = self._SENTENCE_END.split(text.strip())
        return [s.strip() for s in parts if s and s.strip()]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Fast estimate: English averages ~1.33 tokens per word."""
        return max(1, int(len(text.split()) * 1.33))
