"""ResearchCrew Runner with Session Support

Provides a session-aware interface for running research agents.
Wraps the ADK agent execution with session management for multi-turn conversations.
Includes context window management to handle long research sessions.
"""

import logging
from typing import Any, Optional

from utils.session_manager import (
    SessionManager,
    SessionState,
    get_session_manager,
)
from utils.context_manager import (
    ContextManager,
    ContextUsage,
    ContextWarning,
    get_context_manager,
)

logger = logging.getLogger(__name__)


class ResearchCrewRunner:
    """Runner that manages sessions across multiple research interactions.

    This runner:
    - Creates and manages sessions for multi-turn conversations
    - Injects session context into agent prompts
    - Stores research results in session state
    - Supports follow-up questions that reference previous research
    - Manages context window to prevent overflow in long sessions
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        context_manager: Optional[ContextManager] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        """Initialize the runner.

        Args:
            session_manager: Optional custom session manager (uses global if not provided)
            context_manager: Optional custom context manager (uses global if not provided)
            model_name: Model name for context management (only used if no context_manager)
        """
        self.session_manager = session_manager or get_session_manager()
        self.context_manager = context_manager or get_context_manager(model_name)
        self._last_context_warnings: list[ContextWarning] = []
        self._last_context_usage: Optional[ContextUsage] = None
        logger.info(f"ResearchCrewRunner initialized with model {model_name}")

    def create_session(self, user_id: str = "default_user") -> str:
        """Create a new session and return its ID.

        Args:
            user_id: Identifier for the user

        Returns:
            The session ID
        """
        session = self.session_manager.create_session(user_id=user_id)
        return session.session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session by ID.

        Args:
            session_id: The session identifier

        Returns:
            The session state if found
        """
        return self.session_manager.get_session(session_id)

    def build_prompt_with_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        auto_compress: bool = True,
    ) -> str:
        """Build a prompt that includes session context with token management.

        Args:
            query: The user's current query
            session_id: Optional session ID for context
            auto_compress: Whether to automatically compress context if over budget

        Returns:
            The augmented prompt with session context
        """
        if not session_id:
            self._last_context_usage = self.context_manager.estimate_usage(
                current_query=query
            )
            self._last_context_warnings = []
            return query

        session = self.session_manager.get_session(session_id)
        if not session:
            return query

        # Extract conversation history as strings
        conversation_history = [
            f"Turn {turn.turn_id}: Q: {turn.query}\nSummary: {turn.summary or 'No summary'}"
            for turn in session.conversation_history
        ]

        # Build system prompt
        system_prompt = """You are a research assistant with access to session context.
Use the provided context from previous research to inform your response.
If the query references previous findings, expand on them coherently."""

        # Use context manager to build optimized context
        optimized_context, usage, warnings = self.context_manager.build_optimized_context(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            key_facts=session.key_facts,
            current_query=query,
            auto_compress=auto_compress,
        )

        # Store for inspection
        self._last_context_usage = usage
        self._last_context_warnings = warnings

        # Log any warnings
        for warning in warnings:
            if warning.level == "critical":
                logger.warning(
                    f"Context critical: {warning.message} ({warning.usage_percent:.1%} used)"
                )
            elif warning.level == "warning":
                logger.info(
                    f"Context warning: {warning.message} ({warning.usage_percent:.1%} used)"
                )

        # Augment the query with optimized context
        augmented_prompt = f"""
{optimized_context}

Note: You have access to the context above from previous research in this session.
If the current query references previous findings (e.g., "tell me more about X",
"what did you find about Y"), use the context above to provide a coherent response.
"""
        return augmented_prompt.strip()

    def get_context_usage(self) -> Optional[ContextUsage]:
        """Get the context usage from the last query.

        Returns:
            ContextUsage from last build_prompt_with_context call, or None.
        """
        return self._last_context_usage

    def get_context_warnings(self) -> list[ContextWarning]:
        """Get any context warnings from the last query.

        Returns:
            List of ContextWarning objects.
        """
        return self._last_context_warnings.copy()

    def estimate_context_tokens(self, session_id: str, query: str) -> dict[str, Any]:
        """Estimate token usage for a query without running it.

        Args:
            session_id: The session identifier
            query: The query to estimate for

        Returns:
            Dictionary with token estimates and warnings.
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            usage = self.context_manager.estimate_usage(current_query=query)
            return {
                "total_tokens": usage.total_tokens,
                "query_tokens": usage.query_tokens,
                "history_tokens": 0,
                "facts_tokens": 0,
                "remaining_tokens": self.context_manager.token_counter.get_remaining_tokens(
                    usage.total_tokens
                ),
                "usage_percent": self.context_manager.token_counter.get_usage_percent(
                    usage.total_tokens
                ),
                "warnings": [],
            }

        conversation_history = [
            f"Turn {turn.turn_id}: Q: {turn.query}\nSummary: {turn.summary or 'No summary'}"
            for turn in session.conversation_history
        ]

        usage = self.context_manager.estimate_usage(
            system_prompt="Research assistant system prompt",
            conversation_history=conversation_history,
            key_facts=session.key_facts,
            current_query=query,
        )

        warning = self.context_manager.token_counter.check_usage(usage.total_tokens)

        return {
            "total_tokens": usage.total_tokens,
            "query_tokens": usage.query_tokens,
            "history_tokens": usage.history_tokens,
            "facts_tokens": usage.facts_tokens,
            "remaining_tokens": self.context_manager.token_counter.get_remaining_tokens(
                usage.total_tokens
            ),
            "usage_percent": self.context_manager.token_counter.get_usage_percent(
                usage.total_tokens
            ),
            "warnings": [warning.message] if warning else [],
        }

    def record_research_result(
        self,
        session_id: str,
        query: str,
        result: dict[str, Any],
    ) -> None:
        """Record a research result in the session.

        Args:
            session_id: The session identifier
            query: The original query
            result: The research result dictionary
        """
        # Extract information from result
        summary = None
        findings_count = 0
        sources_count = 0
        key_facts = []
        sources = []
        topics = []

        # Try to extract from report state
        if "report" in result and result["report"]:
            report = result["report"]
            summary = report.get("content", "")[:500] if report.get("content") else None
            sources_count = report.get("sources_cited", 0)

        # Try to extract from research state
        if "research" in result and result["research"]:
            research = result["research"]
            findings = research.get("findings", [])
            findings_count = len(findings)

            # Extract key claims as facts
            for finding in findings[:5]:  # Limit to first 5
                if isinstance(finding, dict) and "claim" in finding:
                    key_facts.append(finding["claim"])

            # Extract topics from sub-queries
            if "sub_queries" in research:
                topics = research["sub_queries"][:3]

        # Try to extract sources
        if "research" in result and result["research"]:
            research = result["research"]
            for finding in research.get("findings", []):
                if isinstance(finding, dict) and "sources" in finding:
                    for source in finding["sources"]:
                        if isinstance(source, dict) and "url" in source:
                            sources.append(source["url"])

        # Add conversation turn
        self.session_manager.add_conversation_turn(
            session_id=session_id,
            query=query,
            summary=summary,
            findings_count=findings_count,
            sources_count=sources_count,
        )

        # Update session context
        self.session_manager.update_session_context(
            session_id=session_id,
            topics=topics,
            key_facts=key_facts,
            sources=sources[:10],  # Limit sources
            findings_summary=summary,
        )

        logger.info(
            f"Recorded result for session {session_id}: "
            f"{findings_count} findings, {sources_count} sources"
        )

    async def run_async(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
        auto_compress: bool = True,
    ) -> dict[str, Any]:
        """Run a research query asynchronously with session support.

        This is an async wrapper that would integrate with ADK's async runner.
        For now, it demonstrates the session management pattern.

        Args:
            query: The research query
            session_id: Optional session ID for multi-turn (creates new if not provided)
            user_id: User identifier for new sessions
            auto_compress: Whether to automatically compress context if over budget

        Returns:
            Dictionary containing the research result and session info
        """
        # Get or create session
        if session_id:
            session = self.session_manager.get_or_create_session(
                session_id=session_id, user_id=user_id
            )
        else:
            session = self.session_manager.create_session(user_id=user_id)

        session_id = session.session_id
        turn_number = len(session.conversation_history) + 1

        logger.info(f"Running query in session {session_id}, turn {turn_number}")

        # Build prompt with context from previous turns (with token management)
        augmented_query = self.build_prompt_with_context(
            query, session_id, auto_compress=auto_compress
        )

        # Get context info for result
        context_info = None
        if self._last_context_usage:
            usage = self._last_context_usage
            context_info = {
                "total_tokens": usage.total_tokens,
                "history_tokens": usage.history_tokens,
                "facts_tokens": usage.facts_tokens,
                "query_tokens": usage.query_tokens,
                "warnings": [
                    {"level": w.level, "message": w.message}
                    for w in self._last_context_warnings
                ],
            }

        # Here we would invoke the actual ADK agent
        # For now, return a placeholder structure
        # In production, this would be:
        #   from agents.orchestrator import orchestrator_agent
        #   result = await orchestrator_agent.run(augmented_query)

        result = {
            "session_id": session_id,
            "turn_number": turn_number,
            "query": query,
            "augmented_query": augmented_query if augmented_query != query else None,
            "context": context_info,
            "status": "pending_implementation",
            "message": (
                "Session and context management ready. "
                "Agent execution integration pending ADK runner setup."
            ),
        }

        return result

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
        auto_compress: bool = True,
    ) -> dict[str, Any]:
        """Run a research query synchronously with session support.

        Args:
            query: The research query
            session_id: Optional session ID for multi-turn
            user_id: User identifier for new sessions
            auto_compress: Whether to automatically compress context if over budget

        Returns:
            Dictionary containing the research result and session info
        """
        import asyncio

        return asyncio.run(
            self.run_async(
                query=query,
                session_id=session_id,
                user_id=user_id,
                auto_compress=auto_compress,
            )
        )

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the conversation history for a session.

        Args:
            session_id: The session identifier

        Returns:
            List of conversation turns as dictionaries
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return []

        return [turn.model_dump() for turn in session.conversation_history]

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: The session identifier

        Returns:
            Session statistics dictionary
        """
        return self.session_manager.get_session_stats(session_id)


# Convenience function for quick usage
def run_research(
    query: str,
    session_id: Optional[str] = None,
    user_id: str = "default_user",
) -> dict[str, Any]:
    """Run a research query with session support.

    This is a convenience function for quick usage.

    Args:
        query: The research query
        session_id: Optional session ID for multi-turn conversations
        user_id: User identifier

    Returns:
        Research result with session information
    """
    runner = ResearchCrewRunner()
    return runner.run(query=query, session_id=session_id, user_id=user_id)
