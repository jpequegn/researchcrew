"""ResearchCrew - Main ADK Entry Point

This file exports the main agent for ADK to discover.
Run with: adk web

The orchestrator coordinates the full research workflow:
  Research (parallel) → Synthesize → Fact-Check → Write

For multi-turn conversations with session memory, use the ResearchCrewRunner:
  from runner import ResearchCrewRunner
  runner = ResearchCrewRunner()
  result = runner.run("your query", session_id="optional-session-id")
"""

import os

from agents.orchestrator import orchestrator_agent
from agents.researcher import researcher_agent

# Determine which agent to use based on environment
# Set RESEARCHCREW_MODE=simple to use just the researcher
mode = os.getenv("RESEARCHCREW_MODE", "full")

if mode == "simple":
    # Simple mode: just the researcher agent (for testing/debugging)
    root_agent = researcher_agent
else:
    # Full mode: orchestrated multi-agent workflow
    root_agent = orchestrator_agent

# ADK looks for 'root_agent' or 'agent' by convention
agent = root_agent

# Also export individual agents and session runner for direct access
__all__ = [
    "agent",
    "root_agent",
    "orchestrator_agent",
    "researcher_agent",
]
