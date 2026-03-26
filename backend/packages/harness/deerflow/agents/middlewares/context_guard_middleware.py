"""Middleware to guard against context window overflow.

When many tool calls (especially web searches) accumulate large amounts of
content, the conversation can exceed the model's context window.  This
middleware estimates the token count of all messages before each model call
and, if it exceeds a configurable threshold, truncates the content of older
tool result messages to keep the context manageable.

This acts as a safety net *in addition to* the optional SummarizationMiddleware
which operates at a coarser granularity.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio for estimation (conservative: ~3.5 chars/token)
_CHARS_PER_TOKEN = 3.5

# Default context budget: 80% of a 128k context window
DEFAULT_MAX_CONTEXT_TOKENS = 100_000

# When truncating old tool results, cap each at this many characters
TRUNCATED_TOOL_RESULT_MAX_CHARS = 500

# Keep the most recent N messages untouched (never truncate recent context)
KEEP_RECENT_MESSAGES = 20


def _estimate_tokens(text: str) -> int:
    """Fast token estimate without requiring tiktoken."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _estimate_message_tokens(msg) -> int:
    """Estimate tokens for a single message."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += _estimate_tokens(block)
            elif isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str):
                    total += _estimate_tokens(text)
        return total
    return _estimate_tokens(str(content))


def _truncate_tool_messages(messages: list, max_tokens: int) -> list | None:
    """Truncate old tool result messages if total context exceeds max_tokens.

    Returns a new message list with old ToolMessage content truncated,
    or None if no truncation was needed.
    """
    total_tokens = sum(_estimate_message_tokens(m) for m in messages)

    if total_tokens <= max_tokens:
        return None

    logger.warning(
        "Context guard: estimated %d tokens exceeds limit %d, truncating old tool results",
        total_tokens, max_tokens,
    )

    # Work through messages oldest-first, truncating ToolMessages
    # but preserving the most recent KEEP_RECENT_MESSAGES
    cutoff = max(0, len(messages) - KEEP_RECENT_MESSAGES)
    new_messages = list(messages)
    tokens_saved = 0
    needed_savings = total_tokens - max_tokens

    for i in range(cutoff):
        msg = new_messages[i]
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        original_tokens = _estimate_tokens(content)

        if original_tokens <= TRUNCATED_TOOL_RESULT_MAX_CHARS // _CHARS_PER_TOKEN:
            continue

        # Truncate the content
        truncated = content[:TRUNCATED_TOOL_RESULT_MAX_CHARS] + "\n... [content truncated to save context space]"
        new_msg = ToolMessage(
            content=truncated,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", "unknown"),
            status=getattr(msg, "status", None),
        )
        new_messages[i] = new_msg
        tokens_saved += original_tokens - _estimate_tokens(truncated)

        if tokens_saved >= needed_savings:
            break

    if tokens_saved > 0:
        new_total = total_tokens - tokens_saved
        logger.info(
            "Context guard: truncated old tool results, saved ~%d tokens (now ~%d tokens)",
            tokens_saved, new_total,
        )
        return new_messages

    return None


class ContextGuardMiddleware(AgentMiddleware[AgentState]):
    """Truncates old tool result messages when context approaches the model limit.

    This prevents context overflow errors that occur when many web searches
    or large tool results accumulate in the conversation history.
    """

    def __init__(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS):
        super().__init__()
        self.max_context_tokens = max_context_tokens

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        patched = _truncate_tool_messages(request.messages, self.max_context_tokens)
        if patched is not None:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        patched = _truncate_tool_messages(request.messages, self.max_context_tokens)
        if patched is not None:
            request = request.override(messages=patched)
        return await handler(request)
