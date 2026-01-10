# RC-6: Wire tools to researcher agent

**Status:** closed
**Epic:** EPIC-1
**Priority:** medium

## Description

Connect the web_search and read_url tools to the researcher agent.

## Tasks

- [x] Import tools in researcher.py
- [x] Add tools to Agent constructor
- [x] Verify tools are accessible

## Acceptance Criteria

- [x] Agent has tools=[web_search, read_url]
- [x] Import statement correct

## Notes

Done in agents/researcher.py:
```python
from tools.search import web_search, read_url
researcher_agent = Agent(..., tools=[web_search, read_url])
```
