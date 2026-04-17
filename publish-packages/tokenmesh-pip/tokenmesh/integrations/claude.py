"""
Claude integration for TokenMesh.

A drop-in wrapper around `anthropic.Anthropic` that automatically
optimizes system prompts and user messages before sending to Claude.

Usage:
    from tokenmesh.integrations.claude import TokenMeshClaude

    client = TokenMeshClaude(api_key="sk-ant-...")
    response = client.chat(
        system="Your very long system prompt...",
        user="What are the key points?",
        model="claude-sonnet-4-20250514",
    )
    print(response.content)
    print(response.token_savings)  # Extra attribute!
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tokenmesh.pipeline import OptimizeResult, TokenMesh

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Response from TokenMeshClaude, extended with token savings info."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    optimization: OptimizeResult | None = None

    @property
    def token_savings(self) -> int:
        return self.optimization.saved_tokens if self.optimization else 0

    @property
    def savings_usd(self) -> float:
        return self.optimization.estimated_savings_usd if self.optimization else 0.0


class TokenMeshClaude:
    """
    Drop-in Claude client with automatic token optimization.

    Wraps `anthropic.Anthropic` — same interface, but automatically
    compresses system prompts and/or user messages before every API call.

    Args:
        api_key:            Anthropic API key. Reads ANTHROPIC_API_KEY env var if omitted.
        optimize_system:    Optimize the system prompt (default True).
        optimize_user:      Optimize user messages (default False — preserve intent).
        mesh_kwargs:        Extra kwargs forwarded to TokenMesh (chunk_size, top_k, etc.)

    Example::

        client = TokenMeshClaude(optimize_system=True, mesh_kwargs={"top_k": 15})
        resp = client.chat(
            system=long_system_prompt,
            user="Summarize the risks",
            model="claude-sonnet-4-20250514",
        )
        print(f"Saved {resp.token_savings} tokens (${resp.savings_usd:.4f})")
    """

    def __init__(
        self,
        api_key: str | None = None,
        optimize_system: bool = True,
        optimize_user: bool = False,
        mesh_kwargs: dict | None = None,
        **anthropic_kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from e

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        client_kwargs.update(anthropic_kwargs)

        self._client = anthropic.Anthropic(**client_kwargs)
        self._optimize_system = optimize_system
        self._optimize_user = optimize_user
        self._mesh = TokenMesh(**(mesh_kwargs or {}))

    def chat(
        self,
        user: str,
        system: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> ClaudeResponse:
        """
        Send a message to Claude with automatic token optimization.

        Args:
            user:        User message content.
            system:      System prompt (will be optimized if optimize_system=True).
            model:       Claude model string.
            max_tokens:  Max tokens for Claude response.
            messages:    Full conversation history (list of role/content dicts).
                         If provided, `user` is appended as the latest message.
            **kwargs:    Extra kwargs forwarded to anthropic messages.create().

        Returns:
            ClaudeResponse with .content, .token_savings, and .savings_usd.
        """
        optimization: OptimizeResult | None = None

        # Optimize system prompt
        if system and self._optimize_system:
            optimization = self._mesh.optimize(system, query=user)
            optimized_system = optimization.optimized_text
            logger.info(
                "System prompt: %d → %d tokens (%.1f%% reduction)",
                optimization.original_tokens,
                optimization.optimized_tokens,
                optimization.reduction_percent,
            )
        else:
            optimized_system = system

        # Build message list
        if messages is None:
            messages = []

        # Optionally optimize user message
        user_content = user
        if user and self._optimize_user:
            user_opt = self._mesh.optimize(user)
            user_content = user_opt.optimized_text

        messages = list(messages) + [{"role": "user", "content": user_content}]

        # Call Claude
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if optimized_system:
            create_kwargs["system"] = optimized_system

        response = self._client.messages.create(**create_kwargs)

        content = response.content[0].text if response.content else ""
        return ClaudeResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            optimization=optimization,
        )

    def stream(
        self,
        user: str,
        system: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        """
        Stream Claude response with optimization applied before call.

        Yields text chunks as they arrive. Use in a for loop:

            for chunk in client.stream(system=..., user=...):
                print(chunk, end="", flush=True)
        """
        if system and self._optimize_system:
            opt = self._mesh.optimize(system, query=user)
            optimized_system = opt.optimized_text
        else:
            optimized_system = system

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            **kwargs,
        }
        if optimized_system:
            create_kwargs["system"] = optimized_system

        with self._client.messages.stream(**create_kwargs) as stream:
            for text in stream.text_stream:
                yield text
