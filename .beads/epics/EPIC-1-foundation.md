# EPIC-1: Foundation (Phase 1)

**Status:** closed
**Phase:** 1
**Goal:** Set up ADK basics, tools, and configuration management

## Description

Build the foundational infrastructure for ResearchCrew:
- Development environment with Google ADK
- Basic researcher agent that can answer questions
- Web search and URL reading tools
- Configuration management (environments, prompts, permissions)

## Success Criteria

- [x] `adk web` starts the Developer UI
- [x] Agent responds coherently to "What is an AI agent?"
- [x] Tools are functional (web_search, read_url)
- [x] Environment configs work for dev/staging/prod
- [x] Prompts are versioned in /prompts directory

## Issues

- RC-1: Set up development environment
- RC-2: Create hello world researcher agent
- RC-3: Run agent in ADK Developer UI
- RC-4: Implement web_search tool
- RC-5: Implement read_url tool
- RC-6: Wire tools to researcher agent
- RC-7: Create AGENTS.md specification
- RC-8: Set up environment configs
- RC-9: Create prompt versioning structure
- RC-10: Create tool permissions config

## Learning Objectives

- Understand ADK installation and project structure
- Learn basic agent definition with Gemini
- Understand tool creation with @tool decorator
- Learn configuration management patterns
