# Best Practices for Research

Get the most out of ResearchCrew with these tips and recommendations.

## Query Formulation

### Be Specific, Not Vague

The more specific your query, the better your results.

| Instead of | Try |
|------------|-----|
| "Tell me about databases" | "Compare PostgreSQL and MySQL for a high-traffic e-commerce application" |
| "Python help" | "How do I handle async file I/O in Python 3.11?" |
| "Cloud stuff" | "What are the cost optimization strategies for AWS Lambda?" |

### Include Context

Provide relevant context to get tailored answers.

**Without context:**
```
What's the best caching strategy?
```

**With context:**
```
What's the best caching strategy for a read-heavy REST API with
10,000 requests per minute serving user profile data?
```

### Ask One Thing at a Time

Complex questions work better as a series of focused queries.

**Too broad:**
```
Explain everything about microservices including architecture, deployment,
monitoring, security, and cost considerations.
```

**Better approach:**
```
Query 1: "What are the main microservices architecture patterns?"
Query 2: "How should I approach microservices deployment?"
Query 3: "What monitoring solutions work best for microservices?"
```

### Use Action Words

Start with clear action verbs:

| Verb | Use When |
|------|----------|
| **Compare** | Evaluating options |
| **Explain** | Understanding concepts |
| **How do I** | Learning to do something |
| **What are** | Getting lists or overviews |
| **Is it true** | Fact-checking |
| **Analyze** | Deep examination |

## Building Research Sessions

### Start Broad, Go Deep

Begin with an overview, then drill into specifics:

```
Turn 1: "What are the main approaches to API authentication?"
Turn 2: "Tell me more about OAuth 2.0"
Turn 3: "How do I implement OAuth 2.0 in a Node.js application?"
Turn 4: "What are the security pitfalls to avoid?"
```

### Use Follow-Ups Effectively

Build on previous answers rather than repeating context:

**Good follow-up:**
```
Previous: [Discussion about React state management]
"How does Zustand compare to what we discussed about Redux?"
```

**Unnecessary repetition:**
```
"We were discussing React state management earlier. You mentioned Redux
has a lot of boilerplate. Now I want to know about Zustand which is
another state management library. How does Zustand compare to Redux
for managing state in React applications?"
```

### Know When to Start Fresh

Start a new session when:
- Changing to an unrelated topic
- Previous session is very long (>20 turns)
- You want fresh perspective without prior assumptions

## Interpreting Results

### Check Confidence Scores

| Score | Interpretation |
|-------|----------------|
| 0.9+ | Highly reliable |
| 0.8-0.9 | Generally trustworthy |
| 0.7-0.8 | Verify important claims |
| <0.7 | Treat as preliminary |

### Verify Critical Information

For important decisions:
1. Click through to original sources
2. Cross-reference with official documentation
3. Ask follow-up questions for clarity
4. Consider asking for additional sources

### Recognize Limitations

ResearchCrew may not excel at:
- Real-time information (stock prices, live events)
- Highly specialized niche topics
- Information behind paywalls
- Personal opinions or subjective judgments

## Query Examples

### Technical Research

**Good:**
```
What are the performance characteristics of different message queue systems
(Kafka, RabbitMQ, Redis Streams) for a system processing 50,000 events/second?
```

**Why it works:**
- Specific systems named
- Clear use case (event processing)
- Quantifiable requirement (50k events/second)

### Comparison Research

**Good:**
```
Compare TypeScript and Go for building a CLI tool that processes large files,
focusing on: development speed, runtime performance, and cross-platform distribution.
```

**Why it works:**
- Clear alternatives (TypeScript vs Go)
- Specific use case (CLI for large files)
- Explicit criteria (speed, performance, distribution)

### Fact-Checking

**Good:**
```
Is it true that GraphQL always outperforms REST APIs? Please cite sources
that support or refute this claim.
```

**Why it works:**
- Specific claim to verify
- Requests evidence
- Open to either outcome

### Learning/Tutorial

**Good:**
```
Explain how connection pooling works in PostgreSQL, with a practical example
of configuring it in a Node.js application using pg-pool.
```

**Why it works:**
- Specific topic (connection pooling)
- Specific technology (PostgreSQL, Node.js)
- Requests practical example

## Common Mistakes to Avoid

### 1. Being Too Vague

```
❌ "Help with my code"
✓ "Why does my React useEffect run twice in development mode?"
```

### 2. Asking Multiple Unrelated Questions

```
❌ "What is Kubernetes and also how do I make a website?"
✓ "What is Kubernetes and when should I use it for deploying web applications?"
```

### 3. Assuming Context

```
❌ "Fix the error" (what error?)
✓ "How do I fix 'ECONNREFUSED' when connecting to PostgreSQL locally?"
```

### 4. Ignoring Confidence Scores

```
❌ Treating all findings as equally reliable
✓ Verifying low-confidence claims before acting on them
```

### 5. Not Following Up

```
❌ Accepting a partial answer
✓ Asking "Can you elaborate on the security implications?"
```

## Advanced Techniques

### Request Specific Formats

```
"Compare React and Vue in a table format covering: learning curve,
performance, ecosystem size, and enterprise adoption."
```

### Ask for Trade-offs

```
"What are the trade-offs of using serverless vs containers,
including costs, scaling, and operational complexity?"
```

### Request Sources with Criteria

```
"What do industry reports say about AI adoption in healthcare?
Please prioritize sources from 2023-2024."
```

### Build Compound Knowledge

```
Turn 1: "What are microservices?"
Turn 2: "What are the common communication patterns?"
Turn 3: "How does service mesh fit into this?"
Turn 4: "Compare Istio and Linkerd for a 20-service application"
Turn 5: "Summarize what we've learned as an architecture decision guide"
```

## Quality Checklist

Before accepting research results, verify:

- [ ] Confidence scores are acceptable for your use case
- [ ] Sources are authoritative and recent
- [ ] Findings make logical sense
- [ ] Critical claims have multiple sources
- [ ] Any conflicting information is addressed
- [ ] The scope matches your original question

## Getting Help

When results aren't helpful:

1. **Rephrase** your query with more specifics
2. **Break down** complex questions into simpler ones
3. **Add context** about your specific situation
4. **Ask for clarification** on confusing findings
5. **Request different sources** if current ones seem inadequate

## Summary: The CLEAR Method

**C**ontext - Provide relevant background
**L**imited - Ask one thing at a time
**E**xplicit - Be specific about what you want
**A**ctionable - Use clear action verbs
**R**efinable - Build on previous answers

## Next Steps

- [Troubleshooting](./troubleshooting.md) - When things don't work
- [Tutorials](./tutorials/technical-research.md) - See best practices in action
- [Understanding Reports](./usage/reports.md) - Interpret results effectively
