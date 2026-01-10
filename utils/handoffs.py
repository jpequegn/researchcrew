"""Agent Handoff Utilities

Functions for managing state transitions between agents.
"""

import logging
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def create_handoff(
    from_agent: str,
    to_agent: str,
    state: T,
    transform: dict[str, Any] | None = None,
) -> T:
    """Create a handoff between agents with optional state transformation.

    Args:
        from_agent: Name of the source agent
        to_agent: Name of the target agent
        state: Current state object
        transform: Optional dict of field updates to apply

    Returns:
        The (possibly transformed) state for the next agent
    """
    log_handoff(from_agent, to_agent, state)

    if transform:
        # Create a copy with updated fields
        state_dict = state.model_dump()
        state_dict.update(transform)
        return type(state)(**state_dict)

    return state


def log_handoff(
    from_agent: str,
    to_agent: str,
    state: BaseModel,
) -> None:
    """Log a handoff event for debugging and observability.

    Args:
        from_agent: Name of the source agent
        to_agent: Name of the target agent
        state: State being passed
    """
    timestamp = datetime.now().isoformat()

    # Extract key metrics from state for logging
    state_summary = _summarize_state(state)

    logger.info(
        f"[{timestamp}] HANDOFF: {from_agent} → {to_agent} | {state_summary}"
    )


def _summarize_state(state: BaseModel) -> str:
    """Create a brief summary of state for logging."""
    state_dict = state.model_dump()

    # Extract counts and key fields
    summary_parts = []

    if "findings" in state_dict:
        summary_parts.append(f"findings={len(state_dict['findings'])}")

    if "themes" in state_dict:
        summary_parts.append(f"themes={len(state_dict['themes'])}")

    if "verdicts" in state_dict:
        summary_parts.append(f"verdicts={len(state_dict['verdicts'])}")

    if "confidence" in state_dict:
        summary_parts.append(f"confidence={state_dict['confidence']}")

    if "overall_confidence" in state_dict:
        summary_parts.append(f"confidence={state_dict['overall_confidence']}")

    if "status" in state_dict:
        summary_parts.append(f"status={state_dict['status']}")

    return ", ".join(summary_parts) if summary_parts else "state_passed"


class HandoffError(Exception):
    """Raised when a handoff between agents fails."""

    def __init__(self, from_agent: str, to_agent: str, reason: str):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.reason = reason
        super().__init__(f"Handoff failed: {from_agent} → {to_agent}: {reason}")


def safe_handoff(
    from_agent: str,
    to_agent: str,
    state: T,
    required_fields: list[str] | None = None,
) -> T:
    """Perform a handoff with validation.

    Args:
        from_agent: Name of the source agent
        to_agent: Name of the target agent
        state: Current state object
        required_fields: Fields that must be present and non-empty

    Returns:
        The validated state

    Raises:
        HandoffError: If validation fails
    """
    if required_fields:
        state_dict = state.model_dump()
        for field in required_fields:
            if field not in state_dict:
                raise HandoffError(from_agent, to_agent, f"Missing field: {field}")
            if state_dict[field] is None or state_dict[field] == []:
                raise HandoffError(from_agent, to_agent, f"Empty field: {field}")

    return create_handoff(from_agent, to_agent, state)
