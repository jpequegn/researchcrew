# Development Setup Guide

This guide walks you through setting up ResearchCrew for local development.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime |
| pip/uv | Latest | Package management |
| Git | 2.x+ | Version control |
| gcloud CLI | Latest | Google Cloud authentication |

### Google Cloud Requirements

- Google Cloud account with billing enabled
- Vertex AI API enabled
- Application Default Credentials configured

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/researchcrew.git
cd researchcrew
```

### 2. Set Up Python Environment

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

Development installation with all extras:
```bash
pip install -e ".[dev,observability]"
```

Or with uv:
```bash
uv pip install -e ".[dev,observability]"
```

### 4. Configure Google Cloud

```bash
# Log in to Google Cloud
gcloud auth login

# Set up Application Default Credentials
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 5. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your-gemini-api-key

# Optional - for local development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
KNOWLEDGE_BASE_PATH=./knowledge_base

# Optional - for web search (mock data without)
SERPER_API_KEY=your-serper-api-key
```

### 6. Verify Installation

```bash
# Run tests
pytest tests/ -v

# Start the ADK developer UI
adk web

# Run a simple query
adk run "What is the current date?"
```

## Project Structure

```
researchcrew/
├── agents/                 # Agent implementations
│   ├── orchestrator.py     # Main coordinator
│   ├── researcher.py       # Information gatherer
│   ├── synthesizer.py      # Finding combiner
│   ├── fact_checker.py     # Claim validator
│   └── writer.py           # Report generator
├── tools/                  # Tool implementations
│   ├── search.py           # Web search tool
│   ├── web_fetch.py        # URL content extractor
│   └── knowledge_base.py   # Vector DB operations
├── schemas/                # Pydantic models
│   ├── research.py         # Research data structures
│   └── agent_io.py         # Agent input/output schemas
├── utils/                  # Shared utilities
│   ├── resilience.py       # Retry and circuit breaker
│   ├── tracing.py          # OpenTelemetry integration
│   └── config.py           # Configuration loader
├── prompts/                # Versioned prompt templates
│   └── v1/                 # Version 1 prompts
├── config/                 # Environment configurations
│   ├── dev.yaml            # Development settings
│   ├── staging.yaml        # Staging settings
│   └── prod.yaml           # Production settings
├── evals/                  # Evaluation framework
│   ├── golden_dataset.jsonl
│   ├── metrics.py          # Custom metrics
│   └── run_evals.py        # Evaluation runner
├── mcp-servers/            # MCP tool servers
│   ├── web-research-server/
│   └── knowledge-base-server/
├── tests/                  # Test suite
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_integration.py
└── docs/                   # Documentation
```

## Configuration

### Environment Configurations

The system uses YAML configuration files for different environments:

| File | Purpose |
|------|---------|
| `config/dev.yaml` | Local development (debug logging, mock services) |
| `config/staging.yaml` | Pre-production testing |
| `config/prod.yaml` | Production deployment |

### Configuration Priority

1. Environment variables (highest priority)
2. `.env` file
3. YAML config file
4. Default values (lowest priority)

### Key Configuration Options

```yaml
# config/dev.yaml example
environment: development

model:
  name: gemini-2.0-flash
  temperature: 0.7
  max_tokens: 8192

agents:
  max_parallel_researchers: 3
  timeout_seconds: 120

tools:
  web_search:
    max_results: 5
    provider: duckduckgo  # or serper
  knowledge_base:
    path: ./knowledge_base
    collection_name: research_findings

logging:
  level: DEBUG
  format: json

observability:
  tracing_enabled: false
  metrics_enabled: false
```

## Running Locally

### ADK Developer UI

The easiest way to interact with ResearchCrew:

```bash
adk web
```

This starts a web interface at `http://localhost:8080` where you can:
- Submit research queries
- View agent interactions
- Inspect tool calls
- See execution traces

### Command Line

Run single queries:
```bash
adk run "Research the latest developments in AI agents"
```

### Python API

```python
from runner import ResearchCrewRunner

runner = ResearchCrewRunner()
session_id = runner.create_session(user_id="dev-user")

result = runner.run(
    query="What are the best practices for multi-agent systems?",
    session_id=session_id
)

print(result["result"])
```

### Running with Different Configurations

```bash
# Use development config (default)
ENVIRONMENT=development adk web

# Use staging config
ENVIRONMENT=staging adk web

# Override specific settings
LOG_LEVEL=DEBUG ENVIRONMENT=development adk web
```

## IDE Setup

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- YAML

Settings (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },
  "yaml.schemas": {
    "https://json-schema.org/draft/2020-12/schema": ["config/*.yaml"]
  }
}
```

### PyCharm

1. Open project and configure Python interpreter to use `.venv`
2. Enable Ruff plugin for linting
3. Configure pytest as test runner
4. Mark `tests/` as Test Sources Root

## Common Tasks

### Adding a New Agent

1. Create agent file in `agents/`:
```python
# agents/my_agent.py
from google.adk import Agent

class MyAgent(Agent):
    name = "my_agent"
    description = "Does something specific"
    tools = []  # List of allowed tools

    async def process(self, input_data):
        # Agent logic here
        pass
```

2. Register in orchestrator if needed
3. Add tests in `tests/test_agents.py`
4. Update AGENTS.md specification

### Adding a New Tool

1. Create tool in `tools/`:
```python
# tools/my_tool.py
from google.adk import tool

@tool
def my_tool(param: str) -> dict:
    """Tool description for the LLM.

    Args:
        param: Description of parameter

    Returns:
        dict: Result of the tool operation
    """
    # Tool implementation
    return {"result": "..."}
```

2. Add to appropriate agent's tool list
3. Add tests in `tests/test_tools.py`
4. Document in MCP server if applicable

### Updating Prompts

1. Create new version directory: `prompts/v2/`
2. Copy and modify prompts
3. Update config to use new version
4. Run evaluations to verify quality

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependencies | Run `pip install -e ".[dev]"` |
| `GOOGLE_API_KEY not set` | Missing env variable | Add to `.env` file |
| `Permission denied` | Auth not configured | Run `gcloud auth application-default login` |
| `Connection refused` | Service not running | Check if `adk web` is running |
| `Rate limit exceeded` | Too many API calls | Wait or use mock services |

### Debug Mode

Enable verbose logging:
```bash
LOG_LEVEL=DEBUG python -m pytest tests/ -v -s
```

### Viewing Traces

When running with tracing enabled:
```bash
# Start local Jaeger for trace viewing
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Run with tracing
TRACING_ENABLED=true adk web
```

View traces at `http://localhost:16686`.

## Next Steps

- [Testing Guide](./testing.md) - How to write and run tests
- [Contributing Guide](./contributing.md) - How to contribute
- [Architecture Overview](../architecture.md) - System design
- [API Reference](../api.md) - Detailed API documentation
