"""Context Window Manager for ResearchCrew

Provides token counting, context compression, and sliding window management
to handle long research sessions without context overflow.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Supported model families with their token limits."""

    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE = "claude"


@dataclass
class ModelConfig:
    """Configuration for a specific model's context limits."""

    name: str
    max_context_tokens: int
    max_output_tokens: int
    encoding_name: str = "cl100k_base"  # Default to GPT-4/Claude encoding
    warning_threshold: float = 0.8  # Warn at 80% capacity
    compression_threshold: float = 0.7  # Compress at 70% capacity


# Model configurations with token limits
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "gemini-2.0-flash": ModelConfig(
        name="gemini-2.0-flash",
        max_context_tokens=1_000_000,  # 1M context window
        max_output_tokens=8_192,
    ),
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        max_context_tokens=2_000_000,  # 2M context window
        max_output_tokens=8_192,
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        max_context_tokens=8_192,
        max_output_tokens=4_096,
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        max_context_tokens=16_385,
        max_output_tokens=4_096,
    ),
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        max_context_tokens=200_000,
        max_output_tokens=4_096,
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        max_context_tokens=200_000,
        max_output_tokens=4_096,
    ),
    "claude-opus-4-5-20251101": ModelConfig(
        name="claude-opus-4-5-20251101",
        max_context_tokens=200_000,
        max_output_tokens=32_000,
    ),
    "claude-opus-4.5": ModelConfig(
        name="claude-opus-4.5",
        max_context_tokens=200_000,
        max_output_tokens=32_000,
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        max_context_tokens=200_000,
        max_output_tokens=16_000,
    ),
    "claude-4-sonnet": ModelConfig(
        name="claude-4-sonnet",
        max_context_tokens=200_000,
        max_output_tokens=16_000,
    ),
}

# Default configuration for unknown models
DEFAULT_CONFIG = ModelConfig(
    name="default",
    max_context_tokens=100_000,
    max_output_tokens=4_096,
)


@dataclass
class ContextUsage:
    """Tracks token usage within a context window."""

    total_tokens: int = 0
    system_tokens: int = 0
    history_tokens: int = 0
    facts_tokens: int = 0
    query_tokens: int = 0
    reserved_output_tokens: int = 0

    @property
    def content_tokens(self) -> int:
        """Total tokens used by content (excluding reserved output)."""
        return self.total_tokens - self.reserved_output_tokens


@dataclass
class ContextWarning:
    """Warning about context usage."""

    level: str  # "info", "warning", "critical"
    message: str
    usage_percent: float
    suggested_action: Optional[str] = None


@dataclass
class CompressedContext:
    """Result of context compression."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    content: str
    items_removed: int = 0
    items_summarized: int = 0


class TokenCounter:
    """Counts tokens using tiktoken for accurate estimation."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize token counter for a specific model.

        Args:
            model_name: Name of the model to configure for.
        """
        self.model_name = model_name
        self.config = MODEL_CONFIGS.get(model_name, DEFAULT_CONFIG)

        # Initialize tiktoken encoder
        # tiktoken doesn't support Gemini directly, so we use cl100k_base
        # which provides reasonably accurate estimates for most modern models
        try:
            self._encoder = tiktoken.get_encoding(self.config.encoding_name)
        except Exception:
            # Fallback to cl100k_base if encoding not found
            self._encoder = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"TokenCounter initialized for {model_name} "
            f"(max_context={self.config.max_context_tokens:,})"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts.

        Args:
            texts: List of texts to count.

        Returns:
            List of token counts for each text.
        """
        return [self.count_tokens(text) for text in texts]

    def estimate_context_usage(
        self,
        system_prompt: str = "",
        conversation_history: list[str] | None = None,
        key_facts: list[str] | None = None,
        current_query: str = "",
    ) -> ContextUsage:
        """Estimate total context usage.

        Args:
            system_prompt: The system/instruction prompt.
            conversation_history: Previous conversation turns.
            key_facts: Accumulated research facts.
            current_query: The current user query.

        Returns:
            ContextUsage with detailed token breakdown.
        """
        conversation_history = conversation_history or []
        key_facts = key_facts or []

        system_tokens = self.count_tokens(system_prompt)
        history_tokens = sum(self.count_tokens(h) for h in conversation_history)
        facts_tokens = sum(self.count_tokens(f) for f in key_facts)
        query_tokens = self.count_tokens(current_query)

        total_tokens = system_tokens + history_tokens + facts_tokens + query_tokens

        return ContextUsage(
            total_tokens=total_tokens,
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            facts_tokens=facts_tokens,
            query_tokens=query_tokens,
            reserved_output_tokens=self.config.max_output_tokens,
        )

    def get_remaining_tokens(self, current_usage: int) -> int:
        """Get remaining tokens in the context window.

        Args:
            current_usage: Current token count.

        Returns:
            Number of remaining tokens.
        """
        available = self.config.max_context_tokens - self.config.max_output_tokens
        return max(0, available - current_usage)

    def get_usage_percent(self, current_usage: int) -> float:
        """Get context usage as a percentage.

        Args:
            current_usage: Current token count.

        Returns:
            Usage percentage (0.0 to 1.0+).
        """
        available = self.config.max_context_tokens - self.config.max_output_tokens
        if available <= 0:
            return 1.0
        return current_usage / available

    def check_usage(self, current_usage: int) -> Optional[ContextWarning]:
        """Check context usage and return warning if needed.

        Args:
            current_usage: Current token count.

        Returns:
            ContextWarning if threshold exceeded, None otherwise.
        """
        usage_percent = self.get_usage_percent(current_usage)

        if usage_percent >= 0.95:
            return ContextWarning(
                level="critical",
                message="Context window nearly full - immediate action required",
                usage_percent=usage_percent,
                suggested_action="Compress context or start new session",
            )
        elif usage_percent >= self.config.warning_threshold:
            return ContextWarning(
                level="warning",
                message="Approaching context limit - consider compressing",
                usage_percent=usage_percent,
                suggested_action="Enable automatic context compression",
            )
        elif usage_percent >= self.config.compression_threshold:
            return ContextWarning(
                level="info",
                message="Context usage moderate - monitoring",
                usage_percent=usage_percent,
                suggested_action=None,
            )

        return None


class ContextCompressor:
    """Compresses context to fit within token limits."""

    def __init__(self, token_counter: TokenCounter):
        """Initialize context compressor.

        Args:
            token_counter: TokenCounter instance for measuring compression.
        """
        self.token_counter = token_counter

    def summarize_turn(self, turn_content: str, max_tokens: int = 100) -> str:
        """Summarize a conversation turn to fit within token limit.

        This is a simple truncation-based summarizer. In production,
        this should use an LLM for intelligent summarization.

        Args:
            turn_content: The full turn content.
            max_tokens: Maximum tokens for the summary.

        Returns:
            Summarized or truncated turn.
        """
        current_tokens = self.token_counter.count_tokens(turn_content)

        if current_tokens <= max_tokens:
            return turn_content

        # Simple approach: truncate with ellipsis
        # In production, use LLM summarization
        words = turn_content.split()
        summary = ""
        for word in words:
            test_summary = summary + " " + word if summary else word
            if self.token_counter.count_tokens(test_summary + "...") > max_tokens:
                break
            summary = test_summary

        return summary.strip() + "..." if summary else turn_content[:50] + "..."

    def compress_history(
        self,
        turns: list[str],
        target_tokens: int,
        keep_recent: int = 3,
    ) -> CompressedContext:
        """Compress conversation history using sliding window and summarization.

        Strategy:
        1. Keep the most recent `keep_recent` turns in full detail
        2. Summarize older turns progressively
        3. Remove oldest turns if still over budget

        Args:
            turns: List of conversation turns (oldest first).
            target_tokens: Target token count for compressed history.
            keep_recent: Number of recent turns to keep in full.

        Returns:
            CompressedContext with compression details.
        """
        if not turns:
            return CompressedContext(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                content="",
            )

        original_tokens = sum(self.token_counter.count_tokens(t) for t in turns)

        # If already under budget, return as-is
        if original_tokens <= target_tokens:
            return CompressedContext(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                content="\n\n".join(turns),
            )

        # Split into recent and older turns
        recent_turns = turns[-keep_recent:] if len(turns) > keep_recent else turns
        older_turns = turns[:-keep_recent] if len(turns) > keep_recent else []

        # Calculate budget for older turns
        recent_tokens = sum(self.token_counter.count_tokens(t) for t in recent_turns)
        remaining_budget = target_tokens - recent_tokens

        items_summarized = 0
        items_removed = 0
        compressed_older = []

        if remaining_budget > 0 and older_turns:
            # Budget per older turn
            budget_per_turn = max(50, remaining_budget // len(older_turns))

            for turn in older_turns:
                turn_tokens = self.token_counter.count_tokens(turn)
                if turn_tokens > budget_per_turn:
                    compressed_turn = self.summarize_turn(turn, budget_per_turn)
                    compressed_older.append(compressed_turn)
                    items_summarized += 1
                else:
                    compressed_older.append(turn)

            # Check if still over budget
            total_older_tokens = sum(
                self.token_counter.count_tokens(t) for t in compressed_older
            )
            while total_older_tokens > remaining_budget and compressed_older:
                compressed_older.pop(0)
                items_removed += 1
                total_older_tokens = sum(
                    self.token_counter.count_tokens(t) for t in compressed_older
                )
        else:
            items_removed = len(older_turns)

        # Combine compressed older + recent
        all_turns = compressed_older + recent_turns
        compressed_content = "\n\n".join(all_turns)
        compressed_tokens = self.token_counter.count_tokens(compressed_content)

        return CompressedContext(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            content=compressed_content,
            items_removed=items_removed,
            items_summarized=items_summarized,
        )

    def compress_facts(
        self,
        facts: list[str],
        target_tokens: int,
        keep_recent: int = 10,
    ) -> CompressedContext:
        """Compress accumulated facts to fit within token budget.

        Strategy:
        1. Keep the most recent facts
        2. Deduplicate similar facts
        3. Truncate individual facts if needed

        Args:
            facts: List of facts (oldest first).
            target_tokens: Target token count.
            keep_recent: Number of recent facts to prioritize.

        Returns:
            CompressedContext with compression details.
        """
        if not facts:
            return CompressedContext(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                content="",
            )

        original_tokens = sum(self.token_counter.count_tokens(f) for f in facts)

        if original_tokens <= target_tokens:
            return CompressedContext(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                content="\n".join(f"- {fact}" for fact in facts),
            )

        # Prioritize recent facts
        recent_facts = facts[-keep_recent:] if len(facts) > keep_recent else facts

        # Add older facts if budget allows
        compressed_facts = list(recent_facts)
        remaining_budget = target_tokens - sum(
            self.token_counter.count_tokens(f) for f in compressed_facts
        )

        items_removed = 0
        for fact in reversed(facts[:-keep_recent] if len(facts) > keep_recent else []):
            fact_tokens = self.token_counter.count_tokens(fact)
            if remaining_budget >= fact_tokens:
                compressed_facts.insert(0, fact)
                remaining_budget -= fact_tokens
            else:
                items_removed += 1

        compressed_content = "\n".join(f"- {fact}" for fact in compressed_facts)
        compressed_tokens = self.token_counter.count_tokens(compressed_content)

        return CompressedContext(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            content=compressed_content,
            items_removed=items_removed,
            items_summarized=0,
        )


@dataclass
class ContextWindow:
    """Manages the context window with automatic compression."""

    token_counter: TokenCounter
    compressor: ContextCompressor
    history_budget_percent: float = 0.3  # 30% for history
    facts_budget_percent: float = 0.2  # 20% for facts
    query_budget_percent: float = 0.5  # 50% for query + system

    _warnings: list[ContextWarning] = field(default_factory=list)

    def __post_init__(self):
        """Validate budget percentages."""
        total = self.history_budget_percent + self.facts_budget_percent + self.query_budget_percent
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Budget percentages must sum to 1.0, got {total}")

    @property
    def available_tokens(self) -> int:
        """Get total available tokens for context."""
        return (
            self.token_counter.config.max_context_tokens
            - self.token_counter.config.max_output_tokens
        )

    @property
    def history_budget(self) -> int:
        """Get token budget for conversation history."""
        return int(self.available_tokens * self.history_budget_percent)

    @property
    def facts_budget(self) -> int:
        """Get token budget for accumulated facts."""
        return int(self.available_tokens * self.facts_budget_percent)

    @property
    def query_budget(self) -> int:
        """Get token budget for current query and system prompt."""
        return int(self.available_tokens * self.query_budget_percent)

    def get_warnings(self) -> list[ContextWarning]:
        """Get accumulated warnings."""
        return self._warnings.copy()

    def clear_warnings(self) -> None:
        """Clear accumulated warnings."""
        self._warnings.clear()

    def build_context(
        self,
        system_prompt: str,
        conversation_history: list[str],
        key_facts: list[str],
        current_query: str,
        auto_compress: bool = True,
    ) -> tuple[str, ContextUsage]:
        """Build optimized context within token limits.

        Args:
            system_prompt: The system/instruction prompt.
            conversation_history: Previous conversation turns.
            key_facts: Accumulated research facts.
            current_query: The current user query.
            auto_compress: Whether to automatically compress if over budget.

        Returns:
            Tuple of (formatted_context, usage_stats).
        """
        self._warnings.clear()

        # Estimate initial usage
        initial_usage = self.token_counter.estimate_context_usage(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            key_facts=key_facts,
            current_query=current_query,
        )

        # Check if compression is needed
        warning = self.token_counter.check_usage(initial_usage.total_tokens)
        if warning:
            self._warnings.append(warning)

        processed_history = conversation_history
        processed_facts = key_facts

        if auto_compress:
            # Compress history if over budget
            history_tokens = initial_usage.history_tokens
            if history_tokens > self.history_budget:
                compressed = self.compressor.compress_history(
                    conversation_history,
                    self.history_budget,
                )
                processed_history = (
                    compressed.content.split("\n\n") if compressed.content else []
                )
                logger.info(
                    f"Compressed history: {compressed.original_tokens} -> "
                    f"{compressed.compressed_tokens} tokens "
                    f"(removed {compressed.items_removed}, summarized {compressed.items_summarized})"
                )

            # Compress facts if over budget
            facts_tokens = initial_usage.facts_tokens
            if facts_tokens > self.facts_budget:
                compressed = self.compressor.compress_facts(
                    key_facts,
                    self.facts_budget,
                )
                # Extract facts from formatted string
                processed_facts = [
                    line[2:] for line in compressed.content.split("\n") if line.startswith("- ")
                ]
                logger.info(
                    f"Compressed facts: {compressed.original_tokens} -> "
                    f"{compressed.compressed_tokens} tokens "
                    f"(removed {compressed.items_removed})"
                )

        # Build final context string
        context_parts = []

        if system_prompt:
            context_parts.append(system_prompt)

        if processed_history:
            history_section = "[Session Context]\n" + "\n\n".join(processed_history)
            context_parts.append(history_section)

        if processed_facts:
            facts_section = "[Key Facts]\n" + "\n".join(f"- {f}" for f in processed_facts)
            context_parts.append(facts_section)

        if current_query:
            context_parts.append(f"[Current Query]\n{current_query}")

        final_context = "\n\n".join(context_parts)

        # Calculate final usage
        final_usage = self.token_counter.estimate_context_usage(
            system_prompt="",  # Already included in final_context
            conversation_history=[],
            key_facts=[],
            current_query=final_context,
        )

        # Check final usage
        final_warning = self.token_counter.check_usage(final_usage.query_tokens)
        if final_warning and final_warning not in self._warnings:
            self._warnings.append(final_warning)

        return final_context, final_usage


class ContextManager:
    """High-level context management for research sessions.

    Provides a unified interface for token counting, context compression,
    and window management.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        history_budget_percent: float = 0.3,
        facts_budget_percent: float = 0.2,
        query_budget_percent: float = 0.5,
    ):
        """Initialize the context manager.

        Args:
            model_name: Name of the model to configure for.
            history_budget_percent: Percentage of context for history.
            facts_budget_percent: Percentage of context for facts.
            query_budget_percent: Percentage of context for query/system.
        """
        self.token_counter = TokenCounter(model_name)
        self.compressor = ContextCompressor(self.token_counter)
        self.context_window = ContextWindow(
            token_counter=self.token_counter,
            compressor=self.compressor,
            history_budget_percent=history_budget_percent,
            facts_budget_percent=facts_budget_percent,
            query_budget_percent=query_budget_percent,
        )

        logger.info(f"ContextManager initialized for {model_name}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count.

        Returns:
            Token count.
        """
        return self.token_counter.count_tokens(text)

    def estimate_usage(
        self,
        system_prompt: str = "",
        conversation_history: list[str] | None = None,
        key_facts: list[str] | None = None,
        current_query: str = "",
    ) -> ContextUsage:
        """Estimate context usage.

        Args:
            system_prompt: System prompt.
            conversation_history: Previous turns.
            key_facts: Accumulated facts.
            current_query: Current query.

        Returns:
            ContextUsage with breakdown.
        """
        return self.token_counter.estimate_context_usage(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            key_facts=key_facts,
            current_query=current_query,
        )

    def build_optimized_context(
        self,
        system_prompt: str,
        conversation_history: list[str],
        key_facts: list[str],
        current_query: str,
        auto_compress: bool = True,
    ) -> tuple[str, ContextUsage, list[ContextWarning]]:
        """Build optimized context with automatic compression.

        Args:
            system_prompt: System prompt.
            conversation_history: Previous turns.
            key_facts: Accumulated facts.
            current_query: Current query.
            auto_compress: Whether to auto-compress.

        Returns:
            Tuple of (context, usage, warnings).
        """
        context, usage = self.context_window.build_context(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            key_facts=key_facts,
            current_query=current_query,
            auto_compress=auto_compress,
        )

        return context, usage, self.context_window.get_warnings()

    def get_stats(self) -> dict:
        """Get context manager statistics.

        Returns:
            Dictionary of statistics.
        """
        config = self.token_counter.config
        return {
            "model": config.name,
            "max_context_tokens": config.max_context_tokens,
            "max_output_tokens": config.max_output_tokens,
            "available_tokens": self.context_window.available_tokens,
            "history_budget": self.context_window.history_budget,
            "facts_budget": self.context_window.facts_budget,
            "query_budget": self.context_window.query_budget,
            "warning_threshold": config.warning_threshold,
            "compression_threshold": config.compression_threshold,
        }


# Global instance management
_context_manager: Optional[ContextManager] = None


def get_context_manager(model_name: str = "gemini-2.0-flash") -> ContextManager:
    """Get the global context manager instance.

    Args:
        model_name: Model name (only used on first call).

    Returns:
        The singleton ContextManager instance.
    """
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(model_name=model_name)
    return _context_manager


def reset_context_manager() -> None:
    """Reset the context manager instance (useful for testing)."""
    global _context_manager
    _context_manager = None
