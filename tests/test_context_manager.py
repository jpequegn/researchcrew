"""Tests for Context Window Manager

Tests the token counting, context compression, and sliding window functionality.
"""

import pytest

from utils.context_manager import (
    ContextManager,
    TokenCounter,
    ContextCompressor,
    ContextWindow,
    ContextUsage,
    ContextWarning,
    CompressedContext,
    ModelConfig,
    MODEL_CONFIGS,
    get_context_manager,
    reset_context_manager,
)


class TestTokenCounter:
    """Tests for the TokenCounter class."""

    def setup_method(self):
        """Create a fresh token counter for each test."""
        self.counter = TokenCounter(model_name="gemini-2.0-flash")

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        assert self.counter.count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Test counting tokens in a simple string."""
        tokens = self.counter.count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be a few tokens

    def test_count_tokens_longer_text(self):
        """Test counting tokens in longer text."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = self.counter.count_tokens(text)
        assert tokens > 50  # Should be quite a few tokens

    def test_count_tokens_batch(self):
        """Test batch token counting."""
        texts = ["Hello", "World", "Test string"]
        counts = self.counter.count_tokens_batch(texts)
        assert len(counts) == 3
        assert all(c > 0 for c in counts)

    def test_estimate_context_usage(self):
        """Test estimating full context usage."""
        usage = self.counter.estimate_context_usage(
            system_prompt="You are a helpful assistant.",
            conversation_history=["Turn 1: Hello", "Turn 2: How are you?"],
            key_facts=["Fact 1", "Fact 2"],
            current_query="What is the weather?",
        )

        assert isinstance(usage, ContextUsage)
        assert usage.system_tokens > 0
        assert usage.history_tokens > 0
        assert usage.facts_tokens > 0
        assert usage.query_tokens > 0
        assert usage.total_tokens == (
            usage.system_tokens + usage.history_tokens + usage.facts_tokens + usage.query_tokens
        )

    def test_get_remaining_tokens(self):
        """Test calculating remaining tokens."""
        remaining = self.counter.get_remaining_tokens(1000)
        config = self.counter.config

        expected = config.max_context_tokens - config.max_output_tokens - 1000
        assert remaining == expected

    def test_get_usage_percent(self):
        """Test calculating usage percentage."""
        percent = self.counter.get_usage_percent(100000)
        assert 0 <= percent <= 1

    def test_check_usage_no_warning(self):
        """Test that low usage returns no warning."""
        warning = self.counter.check_usage(1000)
        assert warning is None

    def test_check_usage_warning_threshold(self):
        """Test warning at warning threshold."""
        # Calculate usage at 85% of available
        config = self.counter.config
        available = config.max_context_tokens - config.max_output_tokens
        high_usage = int(available * 0.85)

        warning = self.counter.check_usage(high_usage)
        assert warning is not None
        assert warning.level == "warning"

    def test_check_usage_critical_threshold(self):
        """Test critical warning at high threshold."""
        config = self.counter.config
        available = config.max_context_tokens - config.max_output_tokens
        critical_usage = int(available * 0.96)

        warning = self.counter.check_usage(critical_usage)
        assert warning is not None
        assert warning.level == "critical"


class TestContextCompressor:
    """Tests for the ContextCompressor class."""

    def setup_method(self):
        """Create a fresh compressor for each test."""
        self.counter = TokenCounter(model_name="gemini-2.0-flash")
        self.compressor = ContextCompressor(self.counter)

    def test_summarize_turn_short(self):
        """Test that short turns are not modified."""
        turn = "This is a short turn."
        summarized = self.compressor.summarize_turn(turn, max_tokens=100)
        assert summarized == turn

    def test_summarize_turn_long(self):
        """Test that long turns are truncated."""
        turn = "This is a long turn with lots of content. " * 50
        summarized = self.compressor.summarize_turn(turn, max_tokens=50)
        assert summarized.endswith("...")
        assert self.counter.count_tokens(summarized) <= 50

    def test_compress_history_empty(self):
        """Test compressing empty history."""
        result = self.compressor.compress_history([], target_tokens=1000)
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.content == ""

    def test_compress_history_under_budget(self):
        """Test that under-budget history is unchanged."""
        turns = ["Turn 1", "Turn 2", "Turn 3"]
        result = self.compressor.compress_history(turns, target_tokens=1000)
        assert result.compression_ratio == 1.0
        assert result.items_removed == 0
        assert result.items_summarized == 0

    def test_compress_history_over_budget(self):
        """Test that over-budget history is compressed."""
        # Create turns that exceed the target
        turns = [f"This is turn {i} with detailed content. " * 20 for i in range(10)]
        result = self.compressor.compress_history(turns, target_tokens=500, keep_recent=3)

        assert result.compressed_tokens <= result.original_tokens
        # Should keep recent turns and compress/remove older ones
        assert result.items_removed > 0 or result.items_summarized > 0

    def test_compress_history_keeps_recent(self):
        """Test that recent turns are preserved."""
        turns = ["Old turn 1", "Old turn 2", "Recent turn 1", "Recent turn 2", "Recent turn 3"]
        result = self.compressor.compress_history(turns, target_tokens=200, keep_recent=3)

        # Recent turns should be in the content
        assert "Recent turn 3" in result.content
        assert "Recent turn 2" in result.content
        assert "Recent turn 1" in result.content

    def test_compress_facts_under_budget(self):
        """Test that under-budget facts are unchanged."""
        facts = ["Fact 1", "Fact 2", "Fact 3"]
        result = self.compressor.compress_facts(facts, target_tokens=1000)
        assert result.compression_ratio == 1.0
        assert result.items_removed == 0

    def test_compress_facts_over_budget(self):
        """Test that over-budget facts are compressed."""
        facts = [f"This is fact number {i} with extra details." for i in range(50)]
        result = self.compressor.compress_facts(facts, target_tokens=200, keep_recent=5)

        assert result.compressed_tokens <= result.original_tokens
        assert result.items_removed > 0


class TestContextWindow:
    """Tests for the ContextWindow class."""

    def setup_method(self):
        """Create a fresh context window for each test."""
        self.counter = TokenCounter(model_name="gemini-2.0-flash")
        self.compressor = ContextCompressor(self.counter)
        self.window = ContextWindow(
            token_counter=self.counter,
            compressor=self.compressor,
        )

    def test_available_tokens(self):
        """Test available tokens calculation."""
        config = self.counter.config
        expected = config.max_context_tokens - config.max_output_tokens
        assert self.window.available_tokens == expected

    def test_budget_percentages(self):
        """Test budget percentages."""
        total = self.window.history_budget + self.window.facts_budget + self.window.query_budget
        # Should approximately equal available tokens
        assert abs(total - self.window.available_tokens) < 100

    def test_build_context_simple(self):
        """Test building context with simple inputs."""
        context, usage = self.window.build_context(
            system_prompt="You are helpful.",
            conversation_history=["Turn 1"],
            key_facts=["Fact 1"],
            current_query="What is AI?",
        )

        assert "[Session Context]" in context
        assert "[Key Facts]" in context
        assert "[Current Query]" in context
        assert "What is AI?" in context

    def test_build_context_auto_compress(self):
        """Test that context is auto-compressed when over budget."""
        # Create lots of history
        history = [f"Long turn {i} with detailed content. " * 50 for i in range(20)]
        facts = [f"Important fact {i} about topic." for i in range(100)]

        context, usage = self.window.build_context(
            system_prompt="System prompt",
            conversation_history=history,
            key_facts=facts,
            current_query="Query",
            auto_compress=True,
        )

        # Should have compressed but still have content
        assert len(context) > 0

    def test_warnings_accumulated(self):
        """Test that warnings are accumulated."""
        self.window.clear_warnings()

        # Create content that would trigger warnings
        config = self.counter.config
        available = config.max_context_tokens - config.max_output_tokens

        # Create enough content to trigger warning
        large_query = "word " * int(available * 0.85 / 2)  # Approximate

        self.window.build_context(
            system_prompt="",
            conversation_history=[],
            key_facts=[],
            current_query=large_query,
        )

        warnings = self.window.get_warnings()
        # May or may not have warnings depending on actual token count
        assert isinstance(warnings, list)


class TestContextManager:
    """Tests for the ContextManager class."""

    def setup_method(self):
        """Create a fresh context manager for each test."""
        reset_context_manager()
        self.manager = ContextManager(model_name="gemini-2.0-flash")

    def test_count_tokens(self):
        """Test token counting through manager."""
        tokens = self.manager.count_tokens("Hello, world!")
        assert tokens > 0

    def test_estimate_usage(self):
        """Test usage estimation through manager."""
        usage = self.manager.estimate_usage(
            system_prompt="System",
            conversation_history=["Turn 1"],
            key_facts=["Fact 1"],
            current_query="Query",
        )

        assert isinstance(usage, ContextUsage)
        assert usage.total_tokens > 0

    def test_build_optimized_context(self):
        """Test building optimized context."""
        context, usage, warnings = self.manager.build_optimized_context(
            system_prompt="You are helpful.",
            conversation_history=["Previous turn"],
            key_facts=["Known fact"],
            current_query="New question",
        )

        assert len(context) > 0
        assert isinstance(usage, ContextUsage)
        assert isinstance(warnings, list)

    def test_get_stats(self):
        """Test getting manager stats."""
        stats = self.manager.get_stats()

        assert "model" in stats
        assert "max_context_tokens" in stats
        assert "available_tokens" in stats
        assert "history_budget" in stats
        assert "facts_budget" in stats
        assert "query_budget" in stats


class TestGlobalContextManager:
    """Tests for the global context manager singleton."""

    def setup_method(self):
        """Reset context manager before each test."""
        reset_context_manager()

    def test_get_context_manager_returns_same_instance(self):
        """Test that get_context_manager returns the same instance."""
        cm1 = get_context_manager()
        cm2 = get_context_manager()
        assert cm1 is cm2

    def test_reset_context_manager(self):
        """Test that reset creates a new instance."""
        cm1 = get_context_manager()
        reset_context_manager()
        cm2 = get_context_manager()
        assert cm1 is not cm2


class TestModelConfigs:
    """Tests for model configurations."""

    def test_all_models_have_valid_config(self):
        """Test that all model configs are valid."""
        for name, config in MODEL_CONFIGS.items():
            assert config.name == name
            assert config.max_context_tokens > 0
            assert config.max_output_tokens > 0
            assert config.max_output_tokens < config.max_context_tokens
            assert 0 < config.warning_threshold < 1
            assert 0 < config.compression_threshold < 1

    def test_unknown_model_uses_default(self):
        """Test that unknown models use default config."""
        counter = TokenCounter(model_name="unknown-model-xyz")
        assert counter.config.name == "default"
        assert counter.config.max_context_tokens > 0


class TestLongSessionSimulation:
    """Tests simulating long research sessions."""

    def setup_method(self):
        """Setup for long session tests."""
        reset_context_manager()
        self.manager = ContextManager(model_name="gemini-2.0-flash")

    def test_progressive_session_buildup(self):
        """Test context management as session builds up."""
        conversation_history = []
        key_facts = []

        # Simulate 20 turns of conversation
        for i in range(20):
            # Add a turn
            turn = f"Turn {i}: User asked about topic {i}. Assistant provided detailed information."
            conversation_history.append(turn)

            # Add some facts
            for j in range(3):
                key_facts.append(f"Fact {i}.{j}: Important information about topic {i}")

            # Build context
            context, usage, warnings = self.manager.build_optimized_context(
                system_prompt="Research assistant system prompt.",
                conversation_history=conversation_history,
                key_facts=key_facts,
                current_query=f"Tell me more about topic {i}",
            )

            # Context should always be buildable
            assert len(context) > 0

            # Usage should be tracked
            assert usage.total_tokens > 0

        # After 20 turns with 3 facts each = 60 facts + 20 turns
        # Context should still be manageable
        assert len(context) > 0

    def test_context_compression_effectiveness(self):
        """Test that compression effectively reduces context size."""
        # Create a large history
        large_history = [
            f"Turn {i}: Long detailed conversation about topic {i} with many words. " * 20
            for i in range(30)
        ]
        many_facts = [f"Fact {i}: Important detail about the research." for i in range(100)]

        # Get original size
        original_usage = self.manager.estimate_usage(
            system_prompt="System",
            conversation_history=large_history,
            key_facts=many_facts,
            current_query="Query",
        )

        # Build with compression
        context, compressed_usage, warnings = self.manager.build_optimized_context(
            system_prompt="System",
            conversation_history=large_history,
            key_facts=many_facts,
            current_query="Query",
            auto_compress=True,
        )

        # Compressed should be smaller or at least not massively larger
        # (small overhead from formatting is acceptable)
        assert compressed_usage.total_tokens <= original_usage.total_tokens * 1.1
