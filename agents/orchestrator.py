"""Orchestrator Agent

Coordinates the research workflow and delegates tasks to specialized agents.
Supports multi-turn conversations through session context injection.
"""

from google.adk import Agent, SequentialAgent, ParallelAgent

from agents.researcher import researcher_agent
from agents.synthesizer import synthesizer_agent
from agents.fact_checker import fact_checker_agent
from agents.writer import writer_agent

ORCHESTRATOR_INSTRUCTIONS = """You are the research orchestrator who coordinates
comprehensive research workflows.

Your workflow:
1. Receive a research query from the user
2. Decompose the query into 3-5 research angles/sub-questions
3. Dispatch these to researcher agents (run in parallel)
4. Send combined findings to the synthesizer
5. Have the fact-checker validate key claims
6. Generate the final report with the writer

Decision points:
- If the query is simple, you may use fewer researchers
- If confidence is low after fact-checking, request additional research
- Adapt the output format based on user preference

Quality standards:
- All claims must have source attribution
- Confidence scores required for findings
- Fact-checking pass required before final output

## Multi-Turn Conversation Support

When the user's query includes context from previous research in this session:
- Reference the previous findings when they are relevant
- Build upon earlier research rather than starting from scratch
- Acknowledge what was previously discovered
- Focus new research on gaps or follow-up questions
- If asked to "tell me more about X", expand on X from the session context

If the query references something from earlier in the session (e.g., "the first topic",
"what you mentioned about X", "tell me more"), use the session context provided
to understand what they're referring to.

Your goal is to produce accurate, comprehensive, well-sourced research reports
that coherently build upon previous research when in a multi-turn conversation.
"""

# Create parallel research phase with multiple researcher instances
research_phase = ParallelAgent(
    name="research_phase",
    agents=[researcher_agent],  # Can add more instances for parallel research
)

# Full sequential workflow
orchestrator_agent = SequentialAgent(
    name="research_orchestrator",
    agents=[
        research_phase,     # Step 1: Parallel research
        synthesizer_agent,  # Step 2: Combine findings
        fact_checker_agent, # Step 3: Validate claims
        writer_agent,       # Step 4: Produce report
    ],
)
