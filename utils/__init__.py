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
    # Context management
    "ContextManager",
    "TokenCounter",
    "ContextCompressor",
    "ContextWindow",
    "ContextUsage",
    "ContextWarning",
    "CompressedContext",
    "ModelConfig",
    "MODEL_CONFIGS",
    "get_context_manager",
    "reset_context_manager",
]
