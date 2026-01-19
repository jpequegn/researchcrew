# ResearchCrew

A multi-agent research assistant built with **Google ADK** (Agent Development Kit).

This is a hands-on learning project for mastering autonomous AI agents, covering:
- Multi-agent orchestration
- Agent development lifecycle (build, test, deploy, monitor)
- Configuration management
- Memory & RAG
- Evaluation frameworks
- Resiliency patterns
- Production deployment to Vertex AI

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                    │
│            (coordinates the research workflow)           │
└─────────────────┬───────────────────────┬───────────────┘
                  │                       │
    ┌─────────────▼─────────┐   ┌────────▼────────────┐
    │   Researcher Agent    │   │  Synthesizer Agent  │
    │  (finds information)  │   │ (combines findings) │
    └─────────────┬─────────┘   └─────────┬───────────┘
                  │                       │
    ┌─────────────▼─────────┐   ┌────────▼────────────┐
    │  Fact-Checker Agent   │   │   Writer Agent      │
    │ (validates claims)    │   │ (produces report)   │
    └───────────────────────┘   └─────────────────────┘
```

## Project Structure

```
researchcrew/
├── agents/              # Agent implementations
│   ├── orchestrator.py  # Workflow coordinator
│   ├── researcher.py    # Information gatherer
│   ├── synthesizer.py   # Finding combiner
│   ├── fact_checker.py  # Claim validator
│   └── writer.py        # Report generator
├── tools/               # Tool implementations
│   ├── search.py        # Web search tool
│   ├── web_fetch.py     # URL content extractor
│   └── knowledge_base.py # Vector DB operations
├── schemas/             # Pydantic data models
├── utils/               # Shared utilities
│   ├── resilience.py    # Retry & circuit breaker
│   ├── tracing.py       # OpenTelemetry tracing
│   └── metrics.py       # Prometheus metrics
├── prompts/             # Versioned prompt templates
│   └── v1/
├── config/              # Environment configurations
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── evals/               # Evaluation framework
│   ├── golden_dataset.jsonl
│   ├── metrics.py
│   └── run_evals.py
├── mcp-servers/         # MCP tool servers
│   ├── web-research-server/
│   └── knowledge-base-server/
├── tests/               # Test suite
├── docs/                # Documentation
├── AGENTS.md            # Agent behavior specification
├── pyproject.toml       # Project dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud account with Vertex AI enabled
- `gcloud` CLI configured

### Installation

```bash
# Install Google ADK
pip install google-adk

# Install project dependencies
pip install -e .

# Set up Google Cloud credentials
gcloud auth application-default login
```

### Run Locally

```bash
# Start the ADK Developer UI
adk web

# Or run from command line
adk run "Research the latest AI agent frameworks"
```

## Learning Phases

| Phase | Focus | Duration |
|-------|-------|----------|
| 1. Foundation | ADK basics, tools, config | Days 1-3 |
| 2. Multi-Agent | Orchestration, delegation | Days 4-7 |
| 3. Memory | RAG, context management | Days 8-10 |
| 4. Evaluation | Evals, CI/CD | Days 11-14 |
| 5. Observability | Tracing, monitoring | Days 15-17 |
| 6. Resiliency | Fault tolerance | Days 18-21 |
| 7. Deployment | Vertex AI, MCP | Days 22-25 |
| 8. Capstone | Full integration | Days 26-30 |

See [docs/LEARNING_PROJECT.md](docs/LEARNING_PROJECT.md) for detailed instructions.

## Documentation

### User Guide

Start here if you want to use ResearchCrew for research:

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide/index.md) | Complete user documentation |
| [Getting Started](docs/user-guide/getting-started.md) | Your first query in 5 minutes |
| [Best Practices](docs/user-guide/best-practices.md) | Tips for better research |
| [Troubleshooting](docs/user-guide/troubleshooting.md) | Common issues and solutions |

### Developer Guide

For developers extending or contributing to ResearchCrew:

| Document | Description |
|----------|-------------|
| [Development Setup](docs/development/setup.md) | Local development environment setup |
| [Testing Guide](docs/development/testing.md) | How to write and run tests |
| [Contributing Guide](docs/development/contributing.md) | Contribution guidelines |

### Architecture & Design

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | System architecture with diagrams |
| [AGENTS.md](AGENTS.md) | Agent specifications and capabilities |
| [API Reference](docs/api.md) | Python and HTTP API documentation |

### Operations

| Document | Description |
|----------|-------------|
| [Deployment Guide](docs/deployment.md) | Deploying to Vertex AI |
| [Observability Guide](docs/infrastructure/observability.md) | Tracing, metrics, and logging |
| [Debugging Runbook](docs/debugging-runbook.md) | Troubleshooting common issues |

### Integrations

| Document | Description |
|----------|-------------|
| [MCP Servers](docs/mcp-servers.md) | Model Context Protocol servers |
| [Integrations](docs/integrations.md) | External framework integration |

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Vertex AI Agent Engine](https://docs.cloud.google.com/agent-builder/agent-engine/develop/adk)
- [DeepEval](https://docs.deepeval.com/)
- [OpenTelemetry](https://opentelemetry.io/)

## License

MIT
