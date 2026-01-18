"""Session Manager for ResearchCrew

Manages session state for multi-turn conversations, allowing the agent
to remember previous research within the same session.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    turn_id: int = Field(description="Turn number in the conversation")
    timestamp: str = Field(description="ISO timestamp of the turn")
    query: str = Field(description="User's query")
    summary: Optional[str] = Field(default=None, description="Summary of research results")
    findings_count: int = Field(default=0, description="Number of findings produced")
    sources_count: int = Field(default=0, description="Number of sources cited")


class SessionState(BaseModel):
    """State stored for a session."""

    session_id: str = Field(description="Unique session identifier")
    app_name: str = Field(default="researchcrew", description="Application name")
    user_id: str = Field(default="default_user", description="User identifier")
    created_at: str = Field(description="ISO timestamp when session was created")
    last_updated: str = Field(description="ISO timestamp of last update")

    # Conversation history
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list, description="History of conversation turns"
    )

    # Research context for follow-ups
    last_query: Optional[str] = Field(default=None, description="Most recent query")
    last_findings_summary: Optional[str] = Field(
        default=None, description="Summary of most recent findings"
    )
    topics_researched: list[str] = Field(
        default_factory=list, description="List of topics researched in this session"
    )

    # Accumulated knowledge within session
    key_facts: list[str] = Field(
        default_factory=list, description="Key facts discovered in this session"
    )
    sources_used: list[str] = Field(
        default_factory=list, description="URLs of sources used in this session"
    )


class SessionManager:
    """Manages session state for multi-turn research conversations.

    This is an in-memory implementation suitable for development.
    For production, replace with a persistent storage backend.
    """

    def __init__(self):
        """Initialize the session manager."""
        self._sessions: dict[str, SessionState] = {}
        logger.info("SessionManager initialized (in-memory storage)")

    def create_session(
        self,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
    ) -> SessionState:
        """Create a new session.

        Args:
            user_id: Identifier for the user
            session_id: Optional custom session ID (auto-generated if not provided)

        Returns:
            The newly created session state
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_updated=now,
        )

        self._sessions[session_id] = session
        logger.info(f"Created new session: {session_id} for user: {user_id}")

        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve an existing session.

        Args:
            session_id: The session identifier

        Returns:
            The session state if found, None otherwise
        """
        session = self._sessions.get(session_id)
        if session:
            logger.debug(f"Retrieved session: {session_id}")
        else:
            logger.debug(f"Session not found: {session_id}")
        return session

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
    ) -> SessionState:
        """Get an existing session or create a new one.

        Args:
            session_id: Optional session ID to look up
            user_id: User ID for new session creation

        Returns:
            Existing or newly created session state
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session(user_id=user_id, session_id=session_id)

    def add_conversation_turn(
        self,
        session_id: str,
        query: str,
        summary: Optional[str] = None,
        findings_count: int = 0,
        sources_count: int = 0,
    ) -> Optional[ConversationTurn]:
        """Add a conversation turn to the session history.

        Args:
            session_id: The session identifier
            query: The user's query
            summary: Summary of the research results
            findings_count: Number of findings produced
            sources_count: Number of sources cited

        Returns:
            The created conversation turn, or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot add turn: session {session_id} not found")
            return None

        turn = ConversationTurn(
            turn_id=len(session.conversation_history) + 1,
            timestamp=datetime.now().isoformat(),
            query=query,
            summary=summary,
            findings_count=findings_count,
            sources_count=sources_count,
        )

        session.conversation_history.append(turn)
        session.last_query = query
        session.last_updated = datetime.now().isoformat()

        if summary:
            session.last_findings_summary = summary

        logger.info(f"Added turn {turn.turn_id} to session {session_id}")
        return turn

    def update_session_context(
        self,
        session_id: str,
        topics: Optional[list[str]] = None,
        key_facts: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        findings_summary: Optional[str] = None,
    ) -> bool:
        """Update the session's accumulated context.

        Args:
            session_id: The session identifier
            topics: New topics to add
            key_facts: New key facts to add
            sources: New source URLs to add
            findings_summary: Summary to update

        Returns:
            True if update successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Cannot update: session {session_id} not found")
            return False

        if topics:
            # Add unique topics
            for topic in topics:
                if topic not in session.topics_researched:
                    session.topics_researched.append(topic)

        if key_facts:
            # Add unique facts (limit to avoid unbounded growth)
            for fact in key_facts:
                if fact not in session.key_facts and len(session.key_facts) < 50:
                    session.key_facts.append(fact)

        if sources:
            # Add unique sources (limit to avoid unbounded growth)
            for source in sources:
                if source not in session.sources_used and len(session.sources_used) < 100:
                    session.sources_used.append(source)

        if findings_summary:
            session.last_findings_summary = findings_summary

        session.last_updated = datetime.now().isoformat()
        logger.debug(f"Updated context for session {session_id}")
        return True

    def get_session_context_for_prompt(self, session_id: str) -> str:
        """Generate a context string for including in agent prompts.

        This provides the agent with awareness of previous research
        in the current session.

        Args:
            session_id: The session identifier

        Returns:
            A formatted context string, or empty string if no context
        """
        session = self.get_session(session_id)
        if not session or not session.conversation_history:
            return ""

        context_parts = []

        # Add conversation history summary
        if session.conversation_history:
            context_parts.append("## Previous Research in This Session\n")
            for turn in session.conversation_history[-5:]:  # Last 5 turns
                context_parts.append(f"- **Turn {turn.turn_id}**: {turn.query}")
                if turn.summary:
                    context_parts.append(f"  - Summary: {turn.summary[:200]}...")
            context_parts.append("")

        # Add key facts
        if session.key_facts:
            context_parts.append("## Key Facts Discovered\n")
            for fact in session.key_facts[-10:]:  # Last 10 facts
                context_parts.append(f"- {fact}")
            context_parts.append("")

        # Add topics researched
        if session.topics_researched:
            context_parts.append(
                f"## Topics Researched: {', '.join(session.topics_researched[-10:])}\n"
            )

        return "\n".join(context_parts)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self, user_id: Optional[str] = None) -> list[SessionState]:
        """List all sessions, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of session states
        """
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        return sessions

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: The session identifier

        Returns:
            Dictionary of session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "total_turns": len(session.conversation_history),
            "topics_count": len(session.topics_researched),
            "facts_count": len(session.key_facts),
            "sources_count": len(session.sources_used),
        }


# Global session manager instance (singleton pattern for the application)
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance.

    Returns:
        The singleton SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the session manager (useful for testing)."""
    global _session_manager
    _session_manager = None
