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
├── agents/              # Agent definitions
│   ├── orchestrator.py
│   ├── researcher.py
│   ├── synthesizer.py
│   ├── fact_checker.py
│   └── writer.py
├── tools/               # Tool implementations
│   ├── search.py
│   ├── web_fetch.py
│   └── knowledge_base.py
├── prompts/             # Versioned prompts
│   └── v1/
├── config/              # Environment configs
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── evals/               # Evaluation datasets & scripts
│   ├── golden_dataset.jsonl
│   └── run_evals.py
├── mcp-servers/         # MCP tool servers
├── docs/                # Documentation
│   └── LEARNING_PROJECT.md
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

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Vertex AI Agent Engine](https://docs.cloud.google.com/agent-builder/agent-engine/develop/adk)
- [DeepEval](https://docs.deepeval.com/)
- [OpenTelemetry](https://opentelemetry.io/)

## License

MIT
