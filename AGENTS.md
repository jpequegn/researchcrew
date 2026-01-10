# ResearchCrew Agent Specification

## Overview

ResearchCrew is a multi-agent system designed to conduct comprehensive research on any topic. The system uses a hierarchical orchestration pattern with specialized agents.

## Agents

### Orchestrator Agent
**Role:** Coordinates the research workflow and delegates tasks to specialized agents.

**Capabilities:**
- Decomposes research queries into sub-tasks
- Assigns tasks to appropriate agents
- Manages workflow state and handoffs
- Synthesizes final results

**Workflow:**
1. Receive research query
2. Decompose into research angles (3-5 sub-questions)
3. Dispatch to Researcher agents (parallel)
4. Send findings to Synthesizer
5. Validate with Fact-Checker
6. Generate report with Writer
7. Return final report

### Researcher Agent
**Role:** Finds information on specific topics using search and web tools.

**Capabilities:**
- Web search
- URL content extraction
- Knowledge base queries
- Source attribution

**Constraints:**
- Maximum 3 search queries per task
- Must cite sources for all claims
- Return structured findings with confidence scores

### Synthesizer Agent
**Role:** Combines findings from multiple researchers into coherent insights.

**Capabilities:**
- Identify common themes
- Resolve conflicting information
- Rank findings by confidence
- Create structured summaries

### Fact-Checker Agent
**Role:** Validates claims against sources and flags potential issues.

**Capabilities:**
- Cross-reference claims with sources
- Identify unsupported statements
- Flag potential hallucinations
- Assign confidence scores

### Writer Agent
**Role:** Produces the final research report in the requested format.

**Capabilities:**
- Multiple output formats (markdown, summary, detailed)
- Source citation formatting
- Executive summary generation
- Key findings extraction

## Configuration

### Model Selection
| Agent | Default Model | Fallback |
|-------|--------------|----------|
| Orchestrator | gemini-2.0-flash | gemini-1.5-pro |
| Researcher | gemini-2.0-flash | gemini-1.5-flash |
| Synthesizer | gemini-2.0-flash | gemini-1.5-pro |
| Fact-Checker | gemini-2.0-flash | gemini-1.5-flash |
| Writer | gemini-2.0-flash | gemini-1.5-pro |

### Tool Permissions
| Agent | Allowed Tools |
|-------|---------------|
| Orchestrator | delegate_task, get_status |
| Researcher | web_search, read_url, knowledge_search |
| Synthesizer | none (LLM-only) |
| Fact-Checker | web_search, read_url |
| Writer | none (LLM-only) |

### Memory Settings
- **Session Memory:** Enabled for all agents
- **Long-term Memory:** Enabled for Researcher (knowledge base)
- **Context Limit:** 100k tokens per agent

## Quality Requirements

### Accuracy
- All claims must have source attribution
- Confidence scores required for findings
- Fact-checking pass required before output

### Performance
- Target latency: < 60 seconds for simple queries
- Target latency: < 180 seconds for complex queries
- Maximum parallel researchers: 5

### Reliability
- Retry with exponential backoff on transient failures
- Circuit breaker after 3 consecutive failures
- Graceful degradation to single-agent mode if orchestration fails
