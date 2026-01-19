# ResearchCrew Debugging Runbook

This runbook documents common failure patterns and debugging procedures for the ResearchCrew multi-agent system.

## Prerequisites

Before debugging, ensure observability tools are configured:

```python
from utils import init_tracing, init_metrics, configure_logging

# Initialize observability stack
init_tracing(exporter_type="console")  # or "otlp" for production
init_metrics()
configure_logging(level="DEBUG", json_format=True)
```

## Quick Reference

| Symptom | First Check | Tool |
|---------|-------------|------|
| Slow responses | Request duration histogram | Metrics |
| Missing data | Trace spans | Tracing |
| Errors in output | Error logs with trace IDs | Logging |
| High costs | Token usage counter | Metrics |
| Cascading failures | Circuit breaker state | Metrics |

---

## Failure Pattern 1: Tool Execution Error

### Symptoms
- Error messages containing "ToolExecutionError" or similar
- `tool_calls_total{status="failure"}` metric increasing
- Incomplete research results

### Investigation Steps

1. **Check traces for the failing tool call:**
   ```python
   from utils import get_debug_report, diagnose_failure

   # Get recent failures
   report = diagnose_failure(component="web_search", time_range_minutes=5)
   print(report)
   ```

2. **Look at logs for error context:**
   ```bash
   # Filter logs by trace ID
   grep "trace_id.*abc123" logs/app.log | jq '.'
   ```

3. **Verify external service availability:**
   ```python
   import httpx

   # Check if DuckDuckGo is reachable
   response = httpx.get("https://duckduckgo.com")
   print(f"Status: {response.status_code}")
   ```

### Resolution
- If external service is down: Enable fallback mechanism or retry later
- If parsing error: Check if the API response format changed
- If timeout: Increase timeout value or add retry logic

### Prevention
- Add circuit breaker to prevent cascading failures
- Implement retry logic with exponential backoff
- Monitor external service health

---

## Failure Pattern 2: Token Limit Exceeded

### Symptoms
- Errors containing "token limit" or "context length exceeded"
- `agent_error_total{error_type="token_limit"}` metric increasing
- Truncated responses or incomplete summaries

### Investigation Steps

1. **Check context size:**
   ```python
   from utils import get_context_manager

   ctx_mgr = get_context_manager()
   session_id = "your-session-id"

   usage = ctx_mgr.get_context_usage(session_id)
   print(f"Current tokens: {usage.current_tokens}")
   print(f"Max tokens: {usage.max_tokens}")
   print(f"Usage percent: {usage.usage_percent}%")
   ```

2. **Look at conversation history size:**
   ```python
   from utils import get_session_manager

   session_mgr = get_session_manager()
   history = session_mgr.get_conversation_history("session-id")

   for turn in history:
       print(f"Turn {turn.turn_number}: {len(turn.summary)} chars")
   ```

3. **Analyze which components use most tokens:**
   ```python
   # Check token usage per agent
   from utils import get_metrics_text

   metrics = get_metrics_text()
   # Look for agent_token_usage_total
   ```

### Resolution
- Enable context compression for long sessions
- Reduce the number of previous turns included in prompts
- Summarize intermediate results more aggressively

### Prevention
- Set up alerts when context usage exceeds 80%
- Implement automatic compression when approaching limits
- Use streaming for large responses

---

## Failure Pattern 3: Timeout / Slow Response

### Symptoms
- Requests taking > 30 seconds
- `agent_request_duration_seconds` histogram showing high p99
- Users experiencing timeouts

### Investigation Steps

1. **Check request duration distribution:**
   ```python
   from utils import get_metrics_text

   metrics = get_metrics_text()
   # Look for agent_request_duration_seconds buckets
   ```

2. **Identify slow spans in traces:**
   ```python
   from utils import trace_span

   # Spans will show duration for each component
   # Look for spans with unusually high duration
   ```

3. **Check for resource contention:**
   ```bash
   # Check if ChromaDB queries are slow
   grep "knowledge_base" logs/app.log | jq '.duration_seconds'
   ```

### Resolution
- Identify and optimize the slowest component
- Add caching for repeated queries
- Parallelize independent operations

### Prevention
- Set appropriate timeouts for each component
- Monitor latency percentiles (p50, p95, p99)
- Add performance regression tests

---

## Failure Pattern 4: Agent Handoff Failure

### Symptoms
- `HandoffError` in logs
- Research results missing expected sections
- Agent responses containing "I don't have access to..."

### Investigation Steps

1. **Check handoff traces:**
   ```python
   from utils import trace_span

   # Look for spans named "handoff.*"
   # Check if source_agent and target_agent attributes are set correctly
   ```

2. **Verify session state:**
   ```python
   from utils import get_session_manager

   session_mgr = get_session_manager()
   state = session_mgr.get_session_state("session-id")

   print(f"Active agent: {state.current_agent}")
   print(f"Handoff history: {state.handoff_history}")
   ```

3. **Check for state corruption:**
   ```python
   # Compare expected vs actual state
   print(f"Expected context keys: {expected_keys}")
   print(f"Actual context keys: {list(state.context.keys())}")
   ```

### Resolution
- Restore session state from last known good checkpoint
- Retry the handoff with fresh context
- If persistent, restart from the beginning of the workflow

### Prevention
- Validate context before each handoff
- Implement checkpointing for long workflows
- Add integration tests for handoff sequences

---

## Failure Pattern 5: Rate Limiting

### Symptoms
- HTTP 429 responses from external APIs
- `tool_calls_total{status="rate_limited"}` increasing
- Sporadic failures that succeed on retry

### Investigation Steps

1. **Check rate limit metrics:**
   ```python
   from utils import get_metrics_text

   metrics = get_metrics_text()
   # Look for rate_limit related metrics
   ```

2. **Review request patterns:**
   ```bash
   # Count requests per minute to external APIs
   grep "web_search" logs/app.log | \
       jq -r '.timestamp' | \
       cut -d: -f1,2 | \
       sort | uniq -c
   ```

3. **Check retry behavior:**
   ```python
   # Verify exponential backoff is working
   from utils import diagnose_failure

   report = diagnose_failure(component="web_search")
   # Check if failures are clustered or spread out
   ```

### Resolution
- Implement exponential backoff if not present
- Add request queuing with rate limiting
- Consider caching to reduce API calls

### Prevention
- Pre-calculate expected API usage
- Implement client-side rate limiting
- Set up alerts for approaching rate limits

---

## Failure Pattern 6: Hallucination / Invalid Output

### Symptoms
- Research results containing factually incorrect information
- Sources that don't exist or don't support claims
- Agent outputs that don't match the requested format

### Investigation Steps

1. **Review the prompt and context:**
   ```python
   from utils import get_session_manager

   session_mgr = get_session_manager()
   history = session_mgr.get_conversation_history("session-id")

   # Check what context was provided to the agent
   for turn in history:
       print(f"Turn {turn.turn_number} context:")
       print(turn.context[:500])
   ```

2. **Check source verification:**
   ```python
   from utils import get_knowledge_base

   kb = get_knowledge_base()
   # Verify sources were actually retrieved
   results = kb.search("relevant query")
   print(f"Found {len(results)} sources")
   ```

3. **Compare with fact-checker output:**
   ```python
   # The fact_checker agent should have validated claims
   # Check if it was invoked and what it found
   ```

### Resolution
- Re-run with more specific prompts
- Add explicit instructions for source citation
- Invoke fact-checker on suspicious claims

### Prevention
- Always include source verification in workflows
- Add output validation before returning to user
- Use structured output formats with required fields

---

## Debugging Tools

### Generate Debug Report

```python
from utils import get_debug_report

report = get_debug_report(
    include_metrics=True,
    include_failure_history=True,
)
print(report)
```

### Inject Test Failures

```python
from utils import FailureInjector, FailureType

# Test error handling by injecting failures
with FailureInjector.tool_failure("web_search", error_rate=0.5):
    # 50% of web_search calls will fail
    result = run_research("test query")

# Test timeout handling
with FailureInjector.timeout_failure("orchestrator", delay_seconds=35):
    result = run_research("test query")
```

### Diagnose Specific Failure

```python
from utils import diagnose_failure

# Diagnose failures for a specific component
report = diagnose_failure(
    component="web_search",
    time_range_minutes=10,
)

print(f"Findings: {report['findings']}")
print(f"Recommendations: {report['recommendations']}")
```

### Debug Span for Code Sections

```python
from utils import debug_span

with debug_span("process_query") as ctx:
    ctx["query"] = query
    result = process(query)
    ctx["result_size"] = len(result)

# Automatically captures:
# - Trace ID and span ID
# - Duration
# - Errors (if any)
# - Custom context
```

---

## Escalation Procedures

### Level 1: Self-Service
- Use this runbook
- Check recent failures with `diagnose_failure()`
- Review logs and traces

### Level 2: Team Investigation
- Gather debug report
- Document reproduction steps
- Check for similar past incidents

### Level 3: Architecture Review
- Persistent failures after L2 investigation
- Systemic issues affecting multiple components
- Performance degradation requiring redesign

---

## Metrics Reference

| Metric | Type | Description |
|--------|------|-------------|
| `agent_request_duration_seconds` | Histogram | Time spent processing requests |
| `agent_token_usage_total` | Counter | Total tokens used by agent and type |
| `agent_error_total` | Counter | Total errors by agent and type |
| `agent_error_rate` | Gauge | Current error rate (0-1) |
| `agent_cost_per_request_dollars` | Histogram | Cost per request in dollars |
| `tool_calls_total` | Counter | Total tool calls by tool and status |
| `tool_success_rate` | Gauge | Current success rate per tool (0-1) |
| `knowledge_base_queries_total` | Counter | Total KB operations by type and status |
| `active_sessions_total` | Gauge | Number of active sessions |

---

## Appendix: Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `ToolExecutionError: Connection timeout` | External API slow/down | Retry or use fallback |
| `TokenLimitExceeded: Context too long` | Too much context | Enable compression |
| `HandoffError: Invalid target agent` | Misconfigured workflow | Check agent registry |
| `RateLimitError: Too many requests` | API quota exhausted | Implement backoff |
| `ValidationError: Missing required field` | Malformed output | Check prompt format |
