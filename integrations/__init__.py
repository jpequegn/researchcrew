"""ResearchCrew External Agent Integrations

This package provides adapters for integrating ResearchCrew agents
with external agent frameworks like LangGraph, CrewAI, and others.

Supported integrations:
- LangGraph: Bidirectional integration (ADK-to-LangGraph and LangGraph-to-ADK)
"""

from integrations.base import (
    AdapterConfig,
    AdapterError,
    AdapterResult,
    ExternalAgentAdapter,
)
from integrations.base import (
    ValidationError as AdapterValidationError,
)

# Import specific integrations only if dependencies are available
try:
    from integrations.langgraph_adapter import (
        LangGraphAdapter,
        adk_to_langgraph_node,
        langgraph_to_adk_tool,
    )

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

__all__ = [
    # Base classes
    "ExternalAgentAdapter",
    "AdapterConfig",
    "AdapterResult",
    "AdapterError",
    "AdapterValidationError",
    # LangGraph integration
    "LangGraphAdapter",
    "langgraph_to_adk_tool",
    "adk_to_langgraph_node",
    # Availability flags
    "LANGGRAPH_AVAILABLE",
]
