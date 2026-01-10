# RC-2: Create hello world researcher agent

**Status:** closed
**Epic:** EPIC-1
**Priority:** high

## Description

Create a basic researcher agent using Google ADK that can answer questions about AI agents.

## Tasks

- [x] Create agents/researcher.py
- [x] Define agent with name, model, instructions
- [x] Use gemini-2.0-flash model
- [x] Write clear instructions for research behavior

## Acceptance Criteria

- [x] Agent defined with proper ADK structure
- [x] Instructions specify research responsibilities
- [x] Agent configured with gemini-2.0-flash

## Notes

Implemented in agents/researcher.py with:
- RESEARCHER_INSTRUCTIONS constant
- researcher_agent = Agent(...) definition
- Tools wired (web_search, read_url)
