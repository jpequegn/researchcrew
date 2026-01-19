# Observability Guide

This guide covers monitoring, tracing, and metrics for ResearchCrew in development and production environments.

## Overview

ResearchCrew uses industry-standard observability tools:

| Component | Tool | Purpose |
|-----------|------|---------|
| **Tracing** | OpenTelemetry | Distributed request tracing |
| **Metrics** | Prometheus | System and business metrics |
| **Logging** | Python logging + JSON | Structured log output |

## Tracing

### Setup

Install the observability dependencies:

```bash
pip install -e ".[observability]"
```

This includes:
- `opentelemetry-api`
- `opentelemetry-sdk`
- `opentelemetry-exporter-otlp`
- `prometheus-client`

### Configuration

Initialize tracing in your application:

```python
from utils.tracing import init_tracing

# Console exporter (development)
init_tracing(exporter_type="console")

# OTLP exporter (production)
init_tracing(
    exporter_type="otlp",
    otlp_endpoint="http://localhost:4317"  # or your collector URL
)

# Disable tracing
init_tracing(exporter_type="none")
```

Environment variables:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENVIRONMENT=production
```

### Using Traces

#### Context Manager

```python
from utils.tracing import trace_span

with trace_span("my_operation", attributes={"query": "test"}) as span:
    result = do_work()
    span.set_attribute("result_count", len(result))
```

#### Decorators

For tools:
```python
from utils.tracing import trace_tool

@trace_tool()
def web_search(query: str) -> list[dict]:
    """Search will be automatically traced."""
    return perform_search(query)
```

For agents:
```python
from utils.tracing import trace_agent

@trace_agent("researcher")
def run_researcher(query: str) -> dict:
    """Agent execution will be traced."""
    return gather_information(query)
```

#### Recording LLM Calls

```python
from utils.tracing import trace_llm_call

# Record an LLM interaction
trace_llm_call(
    model_name="gemini-2.0-flash",
    prompt="What is AI?",
    response="AI is...",
    token_count=150
)
```

#### Adding Context

```python
from utils.tracing import add_trace_context

add_trace_context(
    query="Research AI agents",
    session_id="session-123",
    user_id="user-456",
    turn_number=3
)
```

### Trace Structure

A typical research workflow generates this span hierarchy:

```
research_workflow (root)
├── session.resolve
├── context.build
├── agent.orchestrator
│   ├── task.decompose
│   ├── task.delegate (to researcher)
│   └── task.delegate (to researcher)
├── agent.researcher (parallel)
│   ├── tool.web_search
│   │   └── llm.call (event)
│   └── tool.read_url
├── agent.researcher (parallel)
│   └── tool.web_search
├── agent.synthesizer
│   └── llm.call (event)
├── agent.fact_checker
│   └── tool.web_search
└── agent.writer
    └── llm.call (event)
```

### Viewing Traces

#### Local Development with Jaeger

```bash
# Start Jaeger all-in-one
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Configure application
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Run with OTLP exporter
TRACING_ENABLED=true python runner.py
```

View traces at `http://localhost:16686`.

#### Production with Google Cloud Trace

When deployed to Vertex AI, traces are automatically exported to Cloud Trace:

```python
# In production config
init_tracing(
    exporter_type="otlp",
    otlp_endpoint="https://cloudtrace.googleapis.com"
)
```

View in Google Cloud Console under **Trace** > **Trace List**.

### Trace IDs

Get current trace/span IDs for logging:

```python
from utils.tracing import get_trace_id, get_span_id

trace_id = get_trace_id()  # "abc123def456..."
span_id = get_span_id()    # "789xyz..."
```

## Metrics

### Setup

Initialize metrics collection:

```python
from utils.metrics import init_metrics

init_metrics()  # Uses default registry
```

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `agent_request_duration_seconds` | Histogram | agent, status | Time spent processing requests |
| `agent_token_usage_total` | Counter | agent, token_type | Tokens used (input/output) |
| `agent_error_total` | Counter | agent, error_type | Error counts |
| `agent_error_rate` | Gauge | agent | Current error rate (0-1) |
| `agent_cost_per_request_dollars` | Histogram | agent | Request cost in dollars |
| `tool_calls_total` | Counter | tool, status | Tool call counts |
| `tool_success_rate` | Gauge | tool | Tool success rate (0-1) |
| `knowledge_base_queries_total` | Counter | operation, status | KB operation counts |
| `active_sessions_total` | Gauge | - | Number of active sessions |

### Recording Metrics

#### Request Duration

```python
from utils.metrics import record_request_duration

with record_request_duration("orchestrator") as ctx:
    result = await run_agent(query)
    ctx["status"] = "success"  # or "error"
```

#### Token Usage

```python
from utils.metrics import record_token_usage

usage = record_token_usage(
    agent="researcher",
    input_tokens=1500,
    output_tokens=500
)
print(f"Cost: ${usage['total_cost']:.4f}")
```

#### Errors

```python
from utils.metrics import record_error

record_error(
    agent="researcher",
    error_type="RateLimitError",
    error_message="API quota exceeded"
)
```

#### Tool Calls

```python
from utils.metrics import record_tool_call

record_tool_call(
    tool="web_search",
    success=True,
    duration_seconds=1.5
)
```

### Decorators

Automatically instrument functions:

```python
from utils.metrics import metric_tool, metric_agent

@metric_tool()
def web_search(query: str) -> list:
    """Automatically records tool_calls_total metric."""
    return search(query)

@metric_agent("researcher")
async def run_researcher(query: str) -> dict:
    """Records request duration and errors."""
    return await process(query)
```

### Exposing Metrics

#### HTTP Endpoint

Add a `/metrics` endpoint to your application:

```python
from utils.metrics import get_metrics_text

@app.get("/metrics")
def metrics():
    return Response(
        content=get_metrics_text(),
        media_type="text/plain"
    )
```

#### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'researchcrew'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 15s
```

### Running Prometheus Locally

```bash
# Start Prometheus
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana for visualization
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

Access:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

## Logging

### Configuration

ResearchCrew uses structured JSON logging in production:

```python
from utils.logging_config import setup_logging

# Development (human-readable)
setup_logging(level="DEBUG", format_type="console")

# Production (JSON)
setup_logging(level="INFO", format_type="json")
```

### Log Levels

| Level | Use Case |
|-------|----------|
| `DEBUG` | Detailed debugging information |
| `INFO` | General operational events |
| `WARNING` | Unexpected but handled situations |
| `ERROR` | Errors that need attention |
| `CRITICAL` | System failures |

### Structured Logging

Logs include trace context for correlation:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(
    "Research completed",
    extra={
        "query": "AI agents",
        "findings_count": 5,
        "duration_seconds": 3.2,
        "trace_id": get_trace_id(),
    }
)
```

Output (JSON format):
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Research completed",
  "query": "AI agents",
  "findings_count": 5,
  "duration_seconds": 3.2,
  "trace_id": "abc123def456"
}
```

## Alerting

### Example Alert Rules (Prometheus)

```yaml
# alerts.yml
groups:
  - name: researchcrew
    rules:
      - alert: HighErrorRate
        expr: agent_error_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for {{ $labels.agent }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, agent_request_duration_seconds_bucket) > 60
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency > 60s for {{ $labels.agent }}"

      - alert: ToolFailures
        expr: increase(tool_calls_total{status="failure"}[5m]) > 10
        labels:
          severity: warning
        annotations:
          summary: "Tool {{ $labels.tool }} failing frequently"
```

### Google Cloud Alerting

When deployed to Vertex AI, configure alerts in Cloud Monitoring:

1. Go to **Monitoring** > **Alerting**
2. Create policy with conditions based on custom metrics
3. Configure notification channels (email, Slack, PagerDuty)

## Dashboard Examples

### Grafana Dashboard Queries

**Request Rate:**
```promql
rate(agent_request_duration_seconds_count[5m])
```

**P95 Latency:**
```promql
histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m]))
```

**Error Rate:**
```promql
rate(agent_error_total[5m]) / rate(agent_request_duration_seconds_count[5m])
```

**Token Usage per Hour:**
```promql
increase(agent_token_usage_total[1h])
```

**Cost per Hour:**
```promql
sum(increase(agent_cost_per_request_dollars_sum[1h]))
```

## Production Recommendations

### Tracing

- Sample traces in production (e.g., 10% of requests)
- Always include trace IDs in error logs
- Set reasonable span limits to avoid memory issues

### Metrics

- Use appropriate histogram buckets for your latency profile
- Alert on rates, not raw counts
- Monitor resource usage (CPU, memory) alongside business metrics

### Logging

- Use JSON format in production
- Include correlation IDs (trace_id, session_id)
- Log at INFO level by default, DEBUG only when troubleshooting
- Avoid logging sensitive data (API keys, PII)

## Troubleshooting

### Traces Not Appearing

1. Verify exporter is configured correctly
2. Check OTLP endpoint is reachable
3. Ensure `init_tracing()` is called before any spans are created
4. Check collector logs for ingestion errors

### Metrics Not Updating

1. Verify `init_metrics()` is called
2. Check Prometheus can reach `/metrics` endpoint
3. Verify scrape interval is appropriate
4. Check for metric name collisions

### High Cardinality Issues

Avoid using unbounded values as metric labels:
- Bad: `user_id`, `query_text`
- Good: `agent_name`, `error_type`, `status`

## Related Documentation

- [Architecture](../architecture.md) - System overview
- [Deployment](../deployment.md) - Production deployment
- [Debugging Runbook](../debugging-runbook.md) - Troubleshooting guide
