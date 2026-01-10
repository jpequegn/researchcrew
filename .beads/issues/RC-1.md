# RC-1: Set up development environment

**Status:** closed
**Epic:** EPIC-1
**Priority:** high

## Description

Set up the Python development environment with Google ADK and all required dependencies.

## Tasks

- [x] Create pyproject.toml with google-adk dependency
- [x] Add httpx, beautifulsoup4, tenacity, pydantic dependencies
- [x] Add dev dependencies (pytest, deepeval, ruff, mypy)
- [x] Configure ruff and mypy

## Acceptance Criteria

- [x] `pip install -e .` works without errors
- [x] All dependencies resolve correctly

## Notes

Scaffolded in initial commit. Dependencies include:
- google-adk>=0.1.0
- httpx, beautifulsoup4 for web tools
- tenacity for retry logic
- pydantic for schemas
- pytest/deepeval for testing
