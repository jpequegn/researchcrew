# Understanding Research Reports

This guide explains how to read and interpret ResearchCrew's research outputs.

## Report Structure

A typical research report contains these sections:

```
┌─────────────────────────────────────┐
│         Executive Summary           │
├─────────────────────────────────────┤
│          Key Findings               │
│  • Finding 1 [Source] [Confidence]  │
│  • Finding 2 [Source] [Confidence]  │
│  • Finding 3 [Source] [Confidence]  │
├─────────────────────────────────────┤
│        Detailed Analysis            │
├─────────────────────────────────────┤
│           Sources                   │
├─────────────────────────────────────┤
│     Confidence & Methodology        │
└─────────────────────────────────────┘
```

## Report Sections Explained

### Executive Summary

A 2-3 sentence overview of the research findings.

**Example:**
```markdown
## Executive Summary

Kubernetes has become the dominant container orchestration platform, used by
78% of organizations running containers in production. While it provides
powerful features for scaling and managing containerized applications, it
requires significant operational expertise and may be overkill for simpler
deployments.
```

**Use this for:** Quick understanding of the main points without reading the full report.

### Key Findings

The most important discoveries, each with source attribution.

**Example:**
```markdown
## Key Findings

### Adoption and Market Position
- **Kubernetes dominates container orchestration** with 78% market share
  among organizations running containers [Source: CNCF Survey 2024]
  *Confidence: 0.92*

- **Managed Kubernetes services** (EKS, GKE, AKS) are preferred by 65%
  of users over self-managed clusters [Source: Datadog Report]
  *Confidence: 0.88*

### Technical Considerations
- **Learning curve is significant** - average time to production readiness
  is 6-12 months for teams new to Kubernetes [Source: Platform Engineering Survey]
  *Confidence: 0.75*
```

**Elements of a finding:**
- **Bold claim**: The main assertion
- **Supporting detail**: Evidence or context
- **Source**: Where the information came from
- **Confidence**: How reliable this finding is (0-1 scale)

### Detailed Analysis

Deeper exploration of findings, including:
- Context and background
- Comparisons and trade-offs
- Implications and recommendations

**Example:**
```markdown
## Detailed Analysis

### When to Use Kubernetes

Kubernetes excels in scenarios requiring:
1. **High availability**: Built-in support for replicas and self-healing
2. **Horizontal scaling**: Automatic scaling based on metrics
3. **Complex deployments**: Microservices with many interdependencies

However, for simpler use cases (single-service applications, development
environments), alternatives like Docker Compose or managed container services
(AWS ECS, Cloud Run) offer lower complexity...
```

### Sources

Complete list of sources used in the research.

**Example:**
```markdown
## Sources

1. CNCF Annual Survey 2024 - https://cncf.io/survey-2024
2. Datadog Container Report - https://datadog.com/container-report
3. Platform Engineering Survey - https://platformengineering.org/survey
4. Kubernetes Documentation - https://kubernetes.io/docs
5. AWS EKS Best Practices - https://aws.github.io/aws-eks-best-practices
```

### Confidence & Methodology

Information about how the research was conducted and how confident the findings are.

**Example:**
```markdown
## Research Methodology

- **Sources consulted**: 12
- **Primary sources**: 5 (official documentation, surveys)
- **Secondary sources**: 7 (articles, reports)
- **Overall confidence**: 0.85

### Confidence Factors
- High source agreement on market share data
- Some variance in complexity assessments (subjective)
- Recent sources (within 12 months) preferred
```

## Understanding Confidence Scores

### Score Ranges

| Score | Meaning | Action |
|-------|---------|--------|
| **0.90-1.00** | Very High | Highly reliable, well-sourced |
| **0.80-0.89** | High | Reliable for most purposes |
| **0.70-0.79** | Moderate | Consider verifying important claims |
| **0.60-0.69** | Low | Treat as preliminary; verify before using |
| **Below 0.60** | Very Low | Significant uncertainty; needs more research |

### What Affects Confidence

**Increases Confidence:**
- Multiple sources agreeing
- Official/authoritative sources
- Recent information
- Verifiable facts

**Decreases Confidence:**
- Conflicting sources
- Opinion-based claims
- Outdated information
- Single source only

## Reading Citations

### Inline Citations

```markdown
Kubernetes adoption has reached 78% among container users [1].
```
The `[1]` refers to the numbered source in the Sources section.

### Source Quality Indicators

Sources are categorized by reliability:

| Type | Examples | Reliability |
|------|----------|-------------|
| **Primary** | Official docs, research papers | Highest |
| **Industry** | Gartner, Forrester, surveys | High |
| **Technical** | Engineering blogs, tutorials | Medium |
| **News** | Tech news sites | Lower |
| **Community** | Forums, Stack Overflow | Verify |

## Report Formats

### Standard Report

Full detailed report (default):
```
Executive Summary
Key Findings (with sources)
Detailed Analysis
Sources
Confidence
```

### Summary Report

Condensed version for quick consumption:
```
Key Points (3-5 bullets)
Top Sources
Overall Confidence
```

### JSON Format

Structured data for programmatic use:
```json
{
  "query": "...",
  "summary": "...",
  "findings": [
    {
      "claim": "...",
      "source": "...",
      "confidence": 0.85,
      "category": "..."
    }
  ],
  "sources": [...],
  "overall_confidence": 0.82
}
```

## Interpreting Different Query Types

### Factual Questions

Look for:
- Clear, direct answers
- High confidence scores
- Multiple corroborating sources

**Example output structure:**
```
Answer: [Direct answer]
Supporting Evidence: [Details]
Confidence: [Usually high if well-established fact]
```

### Comparison Queries

Look for:
- Balanced presentation
- Clear criteria
- Trade-off analysis

**Example output structure:**
```
| Criteria | Option A | Option B |
|----------|----------|----------|
| ...      | ...      | ...      |

Recommendation: [Based on criteria]
```

### Opinion/Recommendation Queries

Look for:
- Acknowledged subjectivity
- Multiple perspectives
- Context-dependent advice

**Example output structure:**
```
Perspectives:
- View 1: ...
- View 2: ...

For your context: [Tailored recommendation]
Note: This involves subjective judgment
```

## Acting on Reports

### For High-Confidence Findings

- Use directly in decision-making
- Cite as reliable sources
- Build on for further research

### For Moderate-Confidence Findings

- Verify critical claims independently
- Look for additional sources
- Consider as directional guidance

### For Low-Confidence Findings

- Treat as hypothesis, not fact
- Conduct additional research
- Ask follow-up questions for clarity

## Exporting Reports

### Markdown

```bash
# Reports are in Markdown by default
# Copy/paste directly into documents
```

### PDF (via command line)

```bash
# Convert with pandoc
adk run "query" | pandoc -o report.pdf
```

### Programmatic Access

```python
result = runner.run("query")
report = result["result"]["report"]

# Access specific fields
summary = report["summary"]
findings = report["findings"]
sources = report["sources"]
confidence = report["confidence"]
```

## Common Questions

### "Why is confidence lower than expected?"

- Conflicting information found
- Limited sources available
- Topic is rapidly changing
- Subjective/opinion-based aspects

### "Why are some sources missing?"

- Paywalled content couldn't be accessed
- Source was down during research
- Content was dynamically loaded

### "How recent is this information?"

- Check source dates in the Sources section
- Ask a follow-up: "Is this information current as of [date]?"

## Next Steps

- [Best Practices](../best-practices.md) - Get better research results
- [Multi-turn Conversations](./conversations.md) - Follow up on findings
- [Tutorials](../tutorials/comprehensive-report.md) - Build comprehensive reports
