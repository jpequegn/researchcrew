"""ResearchCrew Utilities

Helper functions for workflow execution.
"""

from utils.handoffs import (
    create_handoff,
    log_handoff,
    safe_handoff,
    HandoffError,
    create_session_aware_handoff,
    extract_key_findings,
    extract_sources,
)
from utils.session_manager import (
    SessionManager,
    SessionState,
    ConversationTurn,
    get_session_manager,
    reset_session_manager,
)
from utils.knowledge_base import (
    KnowledgeBaseManager,
    KnowledgeEntry,
    SearchResult,
    get_knowledge_base,
    reset_knowledge_base,
)

__all__ = [
    # Handoff utilities
    "create_handoff",
    "log_handoff",
    "safe_handoff",
    "HandoffError",
    "create_session_aware_handoff",
    "extract_key_findings",
    "extract_sources",
    # Session management
    "SessionManager",
    "SessionState",
    "ConversationTurn",
    "get_session_manager",
    "reset_session_manager",
    # Knowledge base
    "KnowledgeBaseManager",
    "KnowledgeEntry",
    "SearchResult",
    "get_knowledge_base",
    "reset_knowledge_base",
]
