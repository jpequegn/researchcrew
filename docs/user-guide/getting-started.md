# Getting Started with ResearchCrew

This guide walks you through your first research query in under 5 minutes.

## Prerequisites

Before starting, ensure you have:
- ResearchCrew installed (see [Installation](./installation.md))
- An active internet connection
- A Google API key configured

## Your First Query

### Option 1: Using the Web Interface

1. **Start the server:**
   ```bash
   adk web
   ```

2. **Open your browser** at `http://localhost:8080`

3. **Enter a research query:**
   ```
   What are the main benefits and drawbacks of electric vehicles?
   ```

4. **Review the results** - ResearchCrew will return a structured report with:
   - Executive summary
   - Key findings with sources
   - Confidence indicators

### Option 2: Using the Command Line

Run a single query directly:

```bash
adk run "What are the main benefits and drawbacks of electric vehicles?"
```

### Option 3: Using Python

```python
from runner import run_research

result = run_research("What are the main benefits and drawbacks of electric vehicles?")
print(result["result"])
```

## Understanding the Output

When ResearchCrew completes your query, you'll receive a research report like this:

```
## Executive Summary
Electric vehicles offer significant environmental and cost benefits but face
challenges in charging infrastructure and upfront costs...

## Key Findings

### Benefits
- **Lower Operating Costs**: EVs cost approximately 50% less to operate than
  gas vehicles [Source: Department of Energy]
- **Environmental Impact**: Zero direct emissions during operation
  [Source: EPA]
- **Reduced Maintenance**: Fewer moving parts mean less maintenance
  [Source: Consumer Reports]

### Drawbacks
- **Higher Purchase Price**: EVs typically cost $10,000+ more than comparable
  gas vehicles [Source: Kelley Blue Book]
- **Charging Infrastructure**: Limited fast-charging stations in rural areas
  [Source: AFDC]
- **Range Anxiety**: Most EVs have 200-300 mile range [Source: Car and Driver]

## Sources
1. Department of Energy - https://energy.gov/...
2. EPA - https://epa.gov/...
3. Consumer Reports - https://consumerreports.org/...
...

## Confidence: 0.87
```

### Report Sections Explained

| Section | What It Contains |
|---------|------------------|
| **Executive Summary** | Quick overview of the main points |
| **Key Findings** | Detailed information organized by theme |
| **Sources** | Links to original source material |
| **Confidence** | Overall reliability score (0-1) |

## Following Up

ResearchCrew remembers your conversation. You can ask follow-up questions:

```
Tell me more about the environmental impact of EV battery production
```

ResearchCrew will understand this relates to your previous query about EVs.

## Try These Example Queries

### Simple Research
```
What is machine learning and how is it different from traditional programming?
```

### Comparison
```
Compare React and Vue.js for building web applications
```

### Fact-Check
```
Is it true that renewable energy is now cheaper than fossil fuels?
```

### Technical Deep-Dive
```
Explain how transformers work in natural language processing
```

## What's Next?

- **Learn query techniques**: [Best Practices](./best-practices.md)
- **Have multi-turn conversations**: [Conversations Guide](./usage/conversations.md)
- **Understand reports better**: [Reports Guide](./usage/reports.md)
- **Try detailed tutorials**: [Tutorials](./tutorials/technical-research.md)

## Quick Tips

1. **Be specific** - "React performance optimization techniques" works better than "React tips"
2. **Ask follow-ups** - Build on previous research instead of starting over
3. **Check sources** - Click through to verify important claims
4. **Note confidence** - Lower confidence means more verification needed

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No results | Check internet connection and API key |
| Slow response | Complex queries take longer; be patient |
| Irrelevant results | Try rephrasing your query more specifically |

For more help, see the [Troubleshooting Guide](./troubleshooting.md).
