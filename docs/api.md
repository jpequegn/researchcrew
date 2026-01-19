# ResearchCrew API Documentation

This document describes the programmatic API for interacting with ResearchCrew.

## Overview

ResearchCrew provides two main interfaces:
1. **Python API**: Direct programmatic access via the `ResearchCrewRunner` class
2. **HTTP API**: REST endpoints when deployed to Vertex AI Agent Engine

## Python API

### ResearchCrewRunner

The main class for running research queries with session support.

#### Import

```python
from runner import ResearchCrewRunner, run_research
```

#### Initialization

```python
runner = ResearchCrewRunner(
    session_manager=None,  # Optional: custom session manager
    context_manager=None,  # Optional: custom context manager
    model_name="gemini-2.0-flash",  # Model for token counting
)
```

#### Methods

##### `create_session(user_id: str = "default_user") -> str`

Create a new session for multi-turn conversations.

**Parameters:**
- `user_id`: Identifier for the user

**Returns:**
- Session ID string

**Example:**
```python
session_id = runner.create_session(user_id="user123")
```

##### `run(query: str, session_id: str = None, user_id: str = "default_user", auto_compress: bool = True) -> dict`

Run a research query synchronously.

**Parameters:**
- `query`: The research question
- `session_id`: Optional session ID for multi-turn conversations
- `user_id`: User identifier for new sessions
- `auto_compress`: Automatically compress context if over budget

**Returns:**
```python
{
    "session_id": str,
    "turn_number": int,
    "query": str,
    "augmented_query": str | None,
    "context": {
        "total_tokens": int,
        "history_tokens": int,
        "facts_tokens": int,
        "query_tokens": int,
        "warnings": list[dict]
    },
    "trace_id": str,
    "status": str,
    "result": dict  # Research findings
}
```

**Example:**
```python
# Single query
result = runner.run("What are the latest developments in AI?")

# Multi-turn conversation
session_id = runner.create_session()
result1 = runner.run("Research quantum computing", session_id=session_id)
result2 = runner.run("Tell me more about quantum supremacy", session_id=session_id)
```

##### `run_async(query: str, ...) -> dict`

Async version of `run()`. Same parameters and return type.

```python
import asyncio

async def main():
    result = await runner.run_async("Research topic")

asyncio.run(main())
```

##### `get_session(session_id: str) -> SessionState | None`

Retrieve a session by ID.

**Returns:**
```python
SessionState(
    session_id=str,
    user_id=str,
    created_at=datetime,
    last_active=datetime,
    conversation_history=list[ConversationTurn],
    context=SessionContext
)
```

##### `get_session_history(session_id: str) -> list[dict]`

Get conversation history for a session.

**Returns:**
```python
[
    {
        "turn_id": int,
        "query": str,
        "timestamp": str,
        "summary": str | None,
        "findings_count": int,
        "sources_count": int
    },
    ...
]
```

##### `get_session_stats(session_id: str) -> dict`

Get statistics for a session.

**Returns:**
```python
{
    "session_id": str,
    "user_id": str,
    "total_turns": int,
    "total_findings": int,
    "total_sources": int,
    "unique_topics": int,
    "duration_seconds": float
}
```

##### `estimate_context_tokens(session_id: str, query: str) -> dict`

Estimate token usage before running a query.

**Returns:**
```python
{
    "total_tokens": int,
    "query_tokens": int,
    "history_tokens": int,
    "facts_tokens": int,
    "remaining_tokens": int,
    "usage_percent": float,
    "warnings": list[str]
}
```

### Convenience Function

```python
from runner import run_research

result = run_research(
    query="Research AI agents",
    session_id=None,  # Creates new session
    user_id="default_user"
)
```

## HTTP API

When deployed to Vertex AI Agent Engine, the following endpoints are available.

### Base URL

```
https://<region>-<project>.cloudfunctions.net/researchcrew
```

Or via ADK development server:
```
http://localhost:8080
```

### Endpoints

#### POST /research

Submit a research query.

**Request:**
```json
{
    "query": "string",
    "session_id": "string (optional)",
    "user_id": "string (optional, default: 'default_user')"
}
```

**Response:**
```json
{
    "session_id": "abc123",
    "turn_number": 1,
    "query": "What are AI agents?",
    "result": {
        "report": {
            "content": "markdown content...",
            "sources_cited": 5,
            "confidence": 0.85
        },
        "research": {
            "findings": [...],
            "sub_queries": [...]
        }
    },
    "trace_id": "trace-xyz"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `429`: Rate limited
- `500`: Internal error

#### GET /sessions/{session_id}

Get session information.

**Response:**
```json
{
    "session_id": "abc123",
    "user_id": "user123",
    "created_at": "2024-01-15T10:30:00Z",
    "total_turns": 3,
    "conversation_history": [...]
}
```

#### GET /sessions/{session_id}/history

Get conversation history for a session.

**Response:**
```json
{
    "session_id": "abc123",
    "history": [
        {
            "turn_id": 1,
            "query": "Research AI",
            "timestamp": "2024-01-15T10:30:00Z",
            "summary": "...",
            "findings_count": 5
        }
    ]
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling

### Error Response Format

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human-readable message",
        "details": {}
    },
    "trace_id": "trace-xyz"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body |
| `VALIDATION_ERROR` | 400 | Invalid parameters |
| `SESSION_NOT_FOUND` | 404 | Session ID not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `TOOL_EXECUTION_ERROR` | 500 | External tool failed |
| `AGENT_ERROR` | 500 | Agent processing error |
| `TOKEN_LIMIT_EXCEEDED` | 500 | Context too long |

### Python Exception Handling

```python
from utils.resilience import TransientError, PermanentError, RateLimitError
from utils.circuit_breaker import CircuitOpenError

try:
    result = runner.run("query")
except CircuitOpenError as e:
    print(f"Service unavailable, retry after {e.retry_after}s")
except RateLimitError as e:
    print(f"Rate limited: {e}")
except TransientError as e:
    print(f"Temporary error, retrying: {e}")
except PermanentError as e:
    print(f"Permanent error: {e}")
```

## Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| POST /research | 10 | Per minute |
| GET /sessions/* | 60 | Per minute |
| GET /health | 120 | Per minute |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1705315860
```

## Authentication

### Local Development

Set the `GOOGLE_API_KEY` environment variable:
```bash
export GOOGLE_API_KEY=your-api-key
```

### Production (Vertex AI)

Uses Google Cloud IAM for authentication:

```python
from google.auth import default

credentials, project = default()
# Credentials automatically used for API calls
```

Or via service account:
```bash
gcloud auth activate-service-account --key-file=service-account.json
```

## SDK Examples

### Python

```python
from runner import ResearchCrewRunner

# Initialize
runner = ResearchCrewRunner()

# Create session
session_id = runner.create_session(user_id="user123")

# Run research
result = runner.run(
    query="What are the latest AI trends?",
    session_id=session_id
)

print(f"Trace ID: {result['trace_id']}")
print(f"Findings: {result.get('result', {})}")

# Follow-up question
result2 = runner.run(
    query="Tell me more about multi-agent systems",
    session_id=session_id
)

# Get session stats
stats = runner.get_session_stats(session_id)
print(f"Total turns: {stats['total_turns']}")
```

### HTTP (curl)

```bash
# Create research query
curl -X POST http://localhost:8080/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Research AI agents", "user_id": "user123"}'

# Get session history
curl http://localhost:8080/sessions/abc123/history

# Health check
curl http://localhost:8080/health
```

## Related Documentation

- [Architecture](./architecture.md) - System architecture
- [Deployment Guide](./deployment.md) - Deployment instructions
- [AGENTS.md](../AGENTS.md) - Agent specifications
