# Learning Project: Multi-Agent Research Assistant

**Goal:** Build a production-ready multi-agent system that researches topics, synthesizes findings, and produces reports - learning the full agent development lifecycle along the way.

**Why This Project:**
- Immediately useful (you can actually use it)
- Covers all key concepts: orchestration, memory, tools, evals, deployment
- Progressive complexity - start simple, add sophistication
- Uses Google ADK (your company's framework)
- Meta-learning: building a research agent teaches you about agents

---

## Project Overview

You'll build **"ResearchCrew"** - a multi-agent system with:

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

---

## Phase 1: Foundation (Days 1-3)
**Concepts:** Basic agent setup, tool use, Google ADK fundamentals

### Milestone 1.1: Hello World Agent
**Learn:** ADK installation, basic agent structure, Gemini integration

```bash
# Setup
pip install google-adk
adk init research-crew
cd research-crew
```

**Tasks:**
1. Create a simple agent that answers questions using Gemini
2. Run it locally with the ADK Developer UI
3. Experiment with different prompts and observe behavior

**Success Criteria:** Agent responds coherently to "What is an AI agent?"

### Milestone 1.2: Add Your First Tool
**Learn:** Tool definition, function calling, structured outputs

**Tasks:**
1. Create a `web_search` tool that searches the web
2. Create a `read_url` tool that fetches and summarizes web pages
3. Test tool calling through the Developer UI

**Code Pattern:**
```python
from google.adk import Agent, tool

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    # Implement using Google Search API or similar
    pass

@tool
def read_url(url: str) -> str:
    """Fetch and extract content from a URL."""
    pass

researcher = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instructions="You are a research assistant. Use tools to find information.",
    tools=[web_search, read_url]
)
```

**Success Criteria:** Agent can search web and summarize a URL when asked

### Milestone 1.3: Configuration Management
**Learn:** AGENTS.md, environment configs, prompt versioning

**Tasks:**
1. Create `AGENTS.md` file defining your agent's behavior
2. Set up environment-specific configs (dev, staging, prod)
3. Version your prompts in a `/prompts` directory
4. Add a `config.yaml` for tool permissions

**File Structure:**
```
research-crew/
├── AGENTS.md                 # Agent behavior specification
├── config/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── prompts/
│   ├── researcher_v1.md
│   └── researcher_v2.md
├── tools/
│   └── search_tools.py
└── agents/
    └── researcher.py
```

**Success Criteria:** Can switch configs between environments; prompts are versioned

---

## Phase 2: Multi-Agent Orchestration (Days 4-7)
**Concepts:** Hierarchical agents, delegation, workflow patterns

### Milestone 2.1: Add Specialized Agents
**Learn:** Agent specialization, separation of concerns

**Tasks:**
1. Create `SynthesizerAgent` - combines multiple research findings
2. Create `FactCheckerAgent` - validates claims against sources
3. Create `WriterAgent` - produces formatted reports

**Each agent should have:**
- Clear, focused instructions
- Specific tools for its role
- Defined input/output schemas

### Milestone 2.2: Build the Orchestrator
**Learn:** ADK workflow agents, delegation patterns

**Tasks:**
1. Create `OrchestratorAgent` using ADK's Sequential/Parallel workflow agents
2. Implement the research workflow:
   ```
   Research (parallel) → Synthesize → Fact-Check → Write
   ```
3. Handle handoffs between agents

**Code Pattern:**
```python
from google.adk import Agent, SequentialAgent, ParallelAgent

# Parallel research phase
research_phase = ParallelAgent(
    name="research_phase",
    agents=[researcher_1, researcher_2, researcher_3]
)

# Full workflow
orchestrator = SequentialAgent(
    name="research_orchestrator",
    agents=[
        research_phase,      # Step 1: Parallel research
        synthesizer,         # Step 2: Combine findings
        fact_checker,        # Step 3: Validate
        writer               # Step 4: Produce report
    ]
)
```

**Success Criteria:** Full workflow executes end-to-end for a research query

### Milestone 2.3: Add Agent-to-Agent Communication
**Learn:** State passing, context sharing, handoff patterns

**Tasks:**
1. Define a shared state schema for the research workflow
2. Pass context between agents (findings, sources, confidence scores)
3. Implement graceful handoffs with error handling

**Success Criteria:** Agents share context; later agents can reference earlier findings

---

## Phase 3: Memory & Context (Days 8-10)
**Concepts:** Short-term memory, long-term memory, RAG

### Milestone 3.1: Add Session Memory
**Learn:** ADK session management, conversation context

**Tasks:**
1. Enable session memory for multi-turn conversations
2. Agent remembers previous research in the same session
3. Test with follow-up questions

### Milestone 3.2: Add Long-Term Knowledge Base
**Learn:** Vector databases, RAG, knowledge persistence

**Tasks:**
1. Set up a vector database (Vertex AI Vector Search or Pinecone)
2. Store research findings as embeddings
3. Create a `knowledge_search` tool that queries past research
4. Agent can reference previous research sessions

**Code Pattern:**
```python
@tool
def knowledge_search(query: str) -> str:
    """Search the knowledge base for previously researched information."""
    # Query vector DB for similar past research
    pass

@tool
def save_to_knowledge(content: str, metadata: dict) -> str:
    """Save research findings to the knowledge base."""
    # Embed and store in vector DB
    pass
```

**Success Criteria:** Agent can recall and reference past research sessions

### Milestone 3.3: Context Window Management
**Learn:** Summarization, compression, token optimization

**Tasks:**
1. Implement automatic summarization for long research sessions
2. Add token counting and context warnings
3. Create a "compact context" strategy for efficiency

**Success Criteria:** System handles long research sessions without context overflow

---

## Phase 4: Evaluation & Testing (Days 11-14)
**Concepts:** Agent evals, quality metrics, CI/CD for agents

### Milestone 4.1: Build Evaluation Dataset
**Learn:** Golden datasets, test cases, evaluation criteria

**Tasks:**
1. Create 20+ test research queries with expected outputs
2. Define quality metrics:
   - Factual accuracy (% claims verified)
   - Source quality (authoritative sources used)
   - Completeness (key aspects covered)
   - Coherence (logical flow)

**File:** `evals/golden_dataset.jsonl`
```json
{"query": "What are the main AI agent frameworks in 2025?", "expected_topics": ["LangGraph", "CrewAI", "Google ADK"], "min_sources": 3}
{"query": "Explain circuit breaker patterns for agents", "expected_topics": ["fault tolerance", "retry", "fallback"], "min_sources": 2}
```

### Milestone 4.2: Implement Automated Evals
**Learn:** LLM-as-judge, RAGAS metrics, evaluation pipelines

**Tasks:**
1. Set up DeepEval or RAGAS for evaluation
2. Create custom metrics for research quality
3. Run evals on your golden dataset
4. Generate evaluation reports

**Code Pattern:**
```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

metrics = [
    AnswerRelevancyMetric(),
    FaithfulnessMetric(),
    # Custom: SourceQualityMetric()
]

results = evaluate(test_cases, metrics)
```

**Success Criteria:** Automated eval pipeline runs; baseline scores established

### Milestone 4.3: CI/CD for Agents
**Learn:** Agent testing in CI, quality gates, regression detection

**Tasks:**
1. Add pytest tests for individual agent behaviors
2. Create GitHub Action that runs evals on PR
3. Set up quality gates (fail if accuracy drops below threshold)
4. Add regression tests for critical behaviors

**Success Criteria:** PRs automatically run evals; quality gates block bad changes

---

## Phase 5: Observability & Debugging (Days 15-17)
**Concepts:** Tracing, monitoring, debugging agents

### Milestone 5.1: Add Tracing
**Learn:** OpenTelemetry, trace visualization, debugging

**Tasks:**
1. Enable ADK's built-in OpenTelemetry tracing
2. Set up trace visualization (Jaeger, Cloud Trace, or Phoenix)
3. Trace a full research workflow - inspect each step

**Success Criteria:** Can visualize full agent execution trace with timings

### Milestone 5.2: Add Production Monitoring
**Learn:** Metrics, alerts, dashboards

**Tasks:**
1. Track key metrics: latency, token usage, error rate, cost
2. Create a monitoring dashboard (Grafana, Cloud Monitoring)
3. Set up alerts for anomalies (high error rate, cost spikes)

**Metrics to Track:**
```
- agent_request_duration_seconds
- agent_token_usage_total
- agent_error_rate
- agent_cost_per_request
- tool_call_success_rate
```

**Success Criteria:** Dashboard shows real-time agent health; alerts fire on issues

### Milestone 5.3: Debug a Failure
**Learn:** Root cause analysis, trace inspection, replay

**Tasks:**
1. Intentionally break something (bad prompt, tool failure)
2. Use traces to identify the failure point
3. Fix the issue and verify with traces
4. Document the debugging process

**Success Criteria:** Can identify and fix issues using observability tools

---

## Phase 6: Resiliency & Production Hardening (Days 18-21)
**Concepts:** Fault tolerance, circuit breakers, graceful degradation

### Milestone 6.1: Add Retry Logic
**Learn:** Exponential backoff, jitter, transient vs permanent errors

**Tasks:**
1. Implement retry with exponential backoff for tool calls
2. Classify errors (transient: retry, permanent: fail fast)
3. Add jitter to prevent thundering herd

**Code Pattern:**
```python
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type(TransientError)
)
def call_with_retry(func, *args):
    return func(*args)
```

### Milestone 6.2: Add Circuit Breakers
**Learn:** Circuit breaker pattern, failure isolation

**Tasks:**
1. Implement circuit breaker for external API calls
2. Track failure rates per tool/service
3. Open circuit when failure rate exceeds threshold
4. Implement fallback behavior when circuit is open

**Success Criteria:** System gracefully degrades when a tool/service fails

### Milestone 6.3: Add Fallback Strategies
**Learn:** Graceful degradation, alternative paths

**Tasks:**
1. Define fallback for each tool (e.g., if web search fails, use cached results)
2. Implement model fallback (if Gemini fails, try backup model)
3. Add human escalation path for critical failures

**Success Criteria:** System continues operating (degraded) even with failures

---

## Phase 7: Deployment & MCP (Days 22-25)
**Concepts:** Production deployment, MCP, composability

### Milestone 7.1: Deploy to Vertex AI Agent Engine
**Learn:** Managed deployment, scaling, security

**Tasks:**
1. Containerize your agent with Docker
2. Deploy to Vertex AI Agent Engine
3. Configure memory, scaling, and security settings
4. Test in production environment

**Commands:**
```bash
adk deploy --project=your-project --region=us-central1
```

**Success Criteria:** Agent running in production on Vertex AI

### Milestone 7.2: Create MCP Tools
**Learn:** Model Context Protocol, reusable tools

**Tasks:**
1. Package your tools as MCP servers
2. Create an MCP server for your knowledge base
3. Make tools reusable across different agents/projects

**File Structure:**
```
mcp-servers/
├── knowledge-base-server/
│   ├── server.py
│   └── mcp.json
├── web-research-server/
│   ├── server.py
│   └── mcp.json
```

**Success Criteria:** Tools work as standalone MCP servers; can be reused

### Milestone 7.3: Compose with External Agents
**Learn:** Agent interoperability, hybrid architectures

**Tasks:**
1. Use a LangGraph agent as a tool in your ADK system
2. Or use your ADK agent as a node in a LangGraph workflow
3. Test the hybrid system

**Success Criteria:** Successfully combine frameworks in one system

---

## Phase 8: Capstone - Full Production System (Days 26-30)
**Concepts:** Putting it all together

### Final Project Checklist

**Architecture:**
- [ ] Multi-agent orchestration working
- [ ] Hierarchical delegation implemented
- [ ] Session and long-term memory functional

**Quality:**
- [ ] 20+ test cases in golden dataset
- [ ] Automated evals running in CI
- [ ] Quality gates blocking bad PRs
- [ ] Baseline accuracy > 80%

**Operations:**
- [ ] Tracing enabled and visualizable
- [ ] Monitoring dashboard live
- [ ] Alerts configured
- [ ] Circuit breakers implemented
- [ ] Retry logic with backoff

**Deployment:**
- [ ] Running on Vertex AI Agent Engine
- [ ] MCP tools packaged and reusable
- [ ] Configuration management in place
- [ ] Secrets properly managed

**Documentation:**
- [ ] AGENTS.md complete
- [ ] Architecture diagram
- [ ] Runbook for common issues
- [ ] API documentation

---

## Learning Resources

### Google ADK
- [ADK Documentation](https://google.github.io/adk-docs/)
- [ADK GitHub](https://github.com/google/adk-python)
- [Vertex AI Agent Engine](https://docs.cloud.google.com/agent-builder/agent-engine/develop/adk)

### Agent Patterns
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [LangChain: Agent Architectures](https://blog.langchain.dev/langgraph-multi-agent-workflows/)

### Evaluation
- [DeepEval Documentation](https://docs.deepeval.com/)
- [RAGAS Documentation](https://docs.ragas.io/)

### Observability
- [OpenTelemetry for LLMs](https://opentelemetry.io/)
- [Arize Phoenix](https://docs.arize.com/phoenix)

---

## Estimated Timeline

| Phase | Days | Key Learning |
|-------|------|--------------|
| 1. Foundation | 1-3 | ADK basics, tools, config |
| 2. Multi-Agent | 4-7 | Orchestration, delegation |
| 3. Memory | 8-10 | RAG, context management |
| 4. Evaluation | 11-14 | Evals, CI/CD |
| 5. Observability | 15-17 | Tracing, monitoring |
| 6. Resiliency | 18-21 | Fault tolerance |
| 7. Deployment | 22-25 | Vertex AI, MCP |
| 8. Capstone | 26-30 | Integration |

**Total: ~30 days** (can be compressed to 2 weeks if full-time)

---

## Success Metrics

By the end, you should be able to:

1. **Build** a multi-agent system from scratch using Google ADK
2. **Configure** agents with proper version control and environment management
3. **Test** agents with automated evaluations and quality gates
4. **Monitor** agents in production with tracing and dashboards
5. **Harden** agents with retry logic, circuit breakers, and fallbacks
6. **Deploy** to Vertex AI Agent Engine
7. **Compose** reusable tools with MCP
8. **Debug** agent issues using observability tools

**You'll have built something real that you can actually use and show your team.**

---

## Optional Extensions

Once complete, consider adding:

1. **Voice Interface** - Add speech-to-text/text-to-speech
2. **Slack/Teams Integration** - Deploy as a bot
3. **Scheduled Research** - Automated daily briefings
4. **Multi-Modal** - Research images and videos
5. **Collaborative** - Multiple users, shared knowledge base
