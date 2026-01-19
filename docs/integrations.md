# External Agent Integrations

ResearchCrew provides adapters for integrating with external agent frameworks, enabling bidirectional interoperability with systems like LangGraph, CrewAI, and others.

## Overview

The integration layer provides:
- **Pattern A**: Use external agents as ADK tools
- **Pattern B**: Use ADK agents as nodes in external workflows
- Unified error handling and tracing
- Performance metrics and monitoring
- State translation between frameworks

## LangGraph Integration

### Pattern A: LangGraph as ADK Tool

Use LangGraph agents, graphs, or runnables as tools within your ADK agents.

```python
from integrations.langgraph_adapter import langgraph_to_adk_tool

# Decorate any LangGraph-style function
@langgraph_to_adk_tool(
    "data_analyzer",
    description="Analyze data using LangGraph pipeline"
)
async def analyze_with_langgraph(state: dict) -> dict:
    """LangGraph-style analysis function."""
    input_data = state["input"]

    # Your LangGraph logic here
    # Can use LangGraph graphs, chains, etc.
    analysis_result = await my_langgraph_graph.ainvoke({"input": input_data})

    return {"output": analysis_result["output"]}

# Use as ADK tool
result = await analyze_with_langgraph.adk_invoke("Research topic")
```

#### With LangGraph StateGraph

```python
from langgraph.graph import StateGraph
from integrations.langgraph_adapter import LangGraphToADKAdapter, AdapterConfig

# Create your LangGraph workflow
def process_node(state):
    return {"processed": state["input"].upper()}

def analyze_node(state):
    return {"analysis": f"Analysis of: {state['processed']}"}

# Build the graph
builder = StateGraph(dict)
builder.add_node("process", process_node)
builder.add_node("analyze", analyze_node)
builder.add_edge("process", "analyze")
builder.set_entry_point("process")
builder.set_finish_point("analyze")
graph = builder.compile()

# Wrap as ADK adapter
adapter = LangGraphToADKAdapter(
    name="analysis_pipeline",
    langgraph_runnable=graph,
    config=AdapterConfig(name="analysis_pipeline", timeout=60.0),
    input_key="input",
    output_key="analysis",
)

# Use in ADK
result = await adapter.execute({"input": "some data"})
print(result.value["analysis"])
```

### Pattern B: ADK as LangGraph Node

Use ADK agents as nodes within a LangGraph workflow.

```python
from google.adk import Agent
from integrations.langgraph_adapter import adk_to_langgraph_node
from langgraph.graph import StateGraph

# Your ADK agent
researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    tools=[web_search, read_url],
    system_prompt="You are a research assistant..."
)

# Convert to LangGraph node
research_node = adk_to_langgraph_node(
    researcher,
    state_key="query",      # Key to read input from state
    result_key="research",  # Key to store result in state
)

# Use in LangGraph workflow
builder = StateGraph(dict)
builder.add_node("research", research_node)
builder.add_node("process", process_research)
builder.add_edge("research", "process")
builder.set_entry_point("research")
builder.set_finish_point("process")

graph = builder.compile()

# Run the workflow
result = await graph.ainvoke({"query": "What are the latest AI trends?"})
print(result["research"])
```

### High-Level LangGraphAdapter

For managing multiple integrations:

```python
from integrations.langgraph_adapter import LangGraphAdapter

# Create adapter manager
adapter = LangGraphAdapter(name="research_crew")

# Register tools (LangGraph -> ADK)
analyzer_tool = adapter.register_tool(
    langgraph_runnable=my_analysis_graph,
    tool_name="deep_analyzer",
    description="Perform deep analysis",
)

# Register nodes (ADK -> LangGraph)
research_node = adapter.register_node(
    adk_agent=researcher_agent,
    node_name="researcher",
)

# List registered components
print(adapter.list_tools())   # ["deep_analyzer"]
print(adapter.list_nodes())   # ["researcher"]
```

## Architecture

### Adapter Base Class

All adapters inherit from `ExternalAgentAdapter`:

```python
from integrations.base import ExternalAgentAdapter, AdapterConfig, AdapterResult

class MyCustomAdapter(ExternalAgentAdapter[InputType, OutputType]):
    @property
    def source_framework(self) -> str:
        return "Source"

    @property
    def target_framework(self) -> str:
        return "Target"

    async def _execute(self, input_data: InputType) -> OutputType:
        # Your integration logic
        return result
```

### Configuration

```python
from integrations.base import AdapterConfig

config = AdapterConfig(
    name="my_adapter",
    timeout=60.0,                    # Timeout in seconds
    validate_input=True,             # Validate inputs
    validate_output=True,            # Validate outputs
    max_retries=2,                   # Retry failed calls
    enable_tracing=True,             # Enable OpenTelemetry tracing
    enable_metrics=True,             # Record Prometheus metrics
    state_translator=my_translator,  # Custom state translation
    result_translator=my_result_fn,  # Custom result translation
)
```

### Results

All adapter calls return `AdapterResult`:

```python
result = await adapter.execute(input_data)

print(result.value)              # The actual result
print(result.success)            # True if successful
print(result.execution_time)     # Time taken
print(result.source_framework)   # e.g., "ADK"
print(result.target_framework)   # e.g., "LangGraph"
print(result.error_message)      # Error if failed
```

### Statistics

Track adapter performance:

```python
from integrations.base import get_all_adapter_stats

stats = get_all_adapter_stats()
for name, stat in stats.items():
    print(f"{name}: {stat.total_calls} calls, {stat.success_rate:.2%} success rate")
```

## Error Handling

### Custom Exceptions

```python
from integrations.base import (
    AdapterError,           # Base adapter error
    ValidationError,        # Input/output validation failed
    StateTranslationError,  # State translation failed
    TimeoutError,           # Adapter call timed out
)

try:
    result = await adapter.execute(input_data)
except ValidationError as e:
    print(f"Validation failed for field: {e.field}")
except TimeoutError as e:
    print(f"Adapter {e.adapter_name} timed out")
except AdapterError as e:
    print(f"Adapter error: {e.original_error}")
```

### Automatic Retries

Adapters automatically retry failed calls based on configuration:

```python
config = AdapterConfig(
    name="resilient_adapter",
    max_retries=3,  # Retry up to 3 times
)
```

## Tracing and Metrics

### Tracing

Adapter calls create OpenTelemetry spans:

```
adapter.my_adapter
├── adapter.name: my_adapter
├── adapter.source: ADK
├── adapter.target: LangGraph
├── trace_id: abc123
└── adapter.success: true
```

### Metrics

Prometheus metrics are recorded automatically:

- `request_duration_seconds{agent="adapter_my_adapter"}` - Call duration
- `errors_total{agent="adapter_my_adapter"}` - Error count

## Best Practices

### 1. State Translation

Always define explicit state translation for complex integrations:

```python
def translate_to_langgraph(adk_state):
    return {
        "input": adk_state.get("query", ""),
        "context": adk_state.get("context", {}),
    }

def translate_from_langgraph(lg_state):
    return {
        "result": lg_state.get("output", ""),
        "metadata": lg_state.get("metadata", {}),
    }

config = AdapterConfig(
    name="translator_adapter",
    state_translator=translate_to_langgraph,
    result_translator=translate_from_langgraph,
)
```

### 2. Timeouts

Set appropriate timeouts for external calls:

```python
# LLM calls may take longer
llm_adapter = LangGraphToADKAdapter(
    name="llm_pipeline",
    langgraph_runnable=llm_graph,
    config=AdapterConfig(name="llm_pipeline", timeout=120.0),
)

# Simple processing should be fast
process_adapter = LangGraphToADKAdapter(
    name="processor",
    langgraph_runnable=process_graph,
    config=AdapterConfig(name="processor", timeout=30.0),
)
```

### 3. Error Recovery

Handle adapter errors gracefully:

```python
async def safe_external_call(input_data):
    try:
        result = await adapter.execute(input_data)
        return result.value
    except AdapterError as e:
        logger.warning(f"External call failed: {e}, using fallback")
        return fallback_result()
```

### 4. Testing

Test integrations with mock runnables:

```python
class MockLangGraphRunnable:
    async def ainvoke(self, state):
        return {"output": f"Mock result for: {state['input']}"}

adapter = LangGraphToADKAdapter(
    name="test_adapter",
    langgraph_runnable=MockLangGraphRunnable(),
    config=AdapterConfig(name="test_adapter"),
)

result = await adapter.execute({"input": "test"})
assert result.success
```

## Supported Frameworks

| Framework | Pattern A (as Tool) | Pattern B (as Node) | Status |
|-----------|---------------------|---------------------|--------|
| LangGraph | Yes | Yes | Implemented |
| CrewAI | Planned | Planned | Coming Soon |
| AutoGen | Planned | Planned | Coming Soon |

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Google ADK Documentation](https://ai.google.dev/adk)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
