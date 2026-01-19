# Multi-Turn Research Conversations

ResearchCrew remembers context from previous queries, allowing you to have natural research conversations that build on prior findings.

## How Sessions Work

When you start researching, ResearchCrew creates a session that tracks:

- Your conversation history
- Key facts discovered
- Sources used
- Topics explored

This context carries forward, making follow-up questions more effective.

## Starting a Session

### Web Interface
Sessions start automatically when you submit your first query.

### Command Line
```bash
# Start interactive session
adk web

# Or programmatically
python -c "
from runner import ResearchCrewRunner
runner = ResearchCrewRunner()
session_id = runner.create_session()
print(f'Session: {session_id}')
"
```

### Python API
```python
from runner import ResearchCrewRunner

runner = ResearchCrewRunner()
session_id = runner.create_session(user_id="my-user")

# First query
result1 = runner.run("What is Kubernetes?", session_id=session_id)

# Follow-up (remembers context)
result2 = runner.run("How does it compare to Docker Swarm?", session_id=session_id)
```

## Follow-Up Query Patterns

### Dig Deeper

Start broad, then explore specific aspects:

```
Query 1: "What are the main cloud providers?"
Query 2: "Tell me more about AWS's serverless offerings"
Query 3: "What are the pricing considerations for Lambda?"
```

### Compare Options

After learning about options, compare them:

```
Query 1: "What are the popular JavaScript frameworks for frontend?"
Query 2: "Compare React and Vue for my use case: a dashboard with real-time data"
```

### Clarify and Refine

Ask for clarification or different perspectives:

```
Query 1: "Explain microservices architecture"
Query 2: "Can you give me a concrete example with an e-commerce application?"
Query 3: "What about the downsides and when NOT to use microservices?"
```

### Apply to Your Context

Take general findings and apply to specifics:

```
Query 1: "What are database indexing best practices?"
Query 2: "How would I apply these to my PostgreSQL tables with millions of user records?"
```

## Effective Follow-Up Phrasing

### Reference Previous Context

| Phrase | Use When |
|--------|----------|
| "Tell me more about..." | Expanding on a specific point |
| "You mentioned..." | Referencing a previous finding |
| "Going back to..." | Returning to an earlier topic |
| "Based on that..." | Building on previous conclusions |

### Request Different Angles

| Phrase | Use When |
|--------|----------|
| "What about..." | Exploring related topics |
| "How does this compare to..." | Comparing options |
| "What are the downsides of..." | Getting balanced view |
| "Can you give an example of..." | Getting practical examples |

### Refine Scope

| Phrase | Use When |
|--------|----------|
| "Focusing specifically on..." | Narrowing down |
| "In the context of..." | Adding constraints |
| "For a [specific use case]..." | Applying to your situation |

## Session Memory in Action

### Example Conversation

**Turn 1:**
```
You: "What are the main approaches to caching in web applications?"

ResearchCrew: [Explains browser caching, CDN caching, server-side caching,
database caching, with pros/cons of each]
```

**Turn 2:**
```
You: "Which would be best for an e-commerce product catalog?"

ResearchCrew: [Remembers the caching discussion, recommends CDN + Redis
combination specifically for product catalogs, explains why]
```

**Turn 3:**
```
You: "How do I handle cache invalidation for this?"

ResearchCrew: [Builds on Redis/CDN recommendation, explains invalidation
strategies specific to product catalogs: TTL, event-based, tag-based]
```

Notice how each response builds on previous context without needing to repeat information.

## When to Start a New Session

Start fresh when:
- Researching a completely different topic
- Previous session is very old
- You want a clean slate without prior assumptions

Continue the session when:
- Following up on previous findings
- Exploring related aspects
- Refining or comparing earlier options

## Viewing Session History

### Python API
```python
history = runner.get_session_history(session_id)
for turn in history:
    print(f"Turn {turn['turn_id']}: {turn['query']}")
    print(f"  Findings: {turn['findings_count']}")
```

### Session Stats
```python
stats = runner.get_session_stats(session_id)
print(f"Total turns: {stats['total_turns']}")
print(f"Total findings: {stats['total_findings']}")
print(f"Unique topics: {stats['unique_topics']}")
```

## Context Window Management

ResearchCrew automatically manages context to stay within limits:

- **Summarization**: Older turns are summarized to save space
- **Prioritization**: Key facts are retained longer than details
- **Compression**: When context is full, less relevant info is compressed

You don't need to manage this manually, but be aware that very old conversation details may be summarized.

## Tips for Multi-Turn Research

### Do

- Build naturally on previous queries
- Reference specific findings you want to explore
- Ask clarifying questions when needed
- Use follow-ups to go deeper rather than repeating

### Don't

- Repeat the full context in every query
- Ask completely unrelated questions in the same session
- Assume ResearchCrew remembers every detail from many turns ago
- Include unnecessary information in follow-ups

## Example: Complete Research Session

Here's a full example of a productive research session:

```
Turn 1: "I'm building a real-time collaborative document editor. What are the
        main technical challenges?"

Turn 2: "Tell me more about operational transformation vs CRDTs"

Turn 3: "Which approach would be better for my use case where I expect
        100-1000 concurrent users per document?"

Turn 4: "What libraries or frameworks implement CRDTs that I could use?"

Turn 5: "Compare Yjs and Automerge for a React-based editor"

Turn 6: "What about the backend? How do I persist the CRDT state?"

Turn 7: "Give me a high-level architecture for this system"
```

Each query builds on the previous, creating a comprehensive research journey.

## Next Steps

- [Knowledge Base](./knowledge-base.md) - Save and retrieve past research
- [Understanding Reports](./reports.md) - Interpret research results
- [Best Practices](../best-practices.md) - More tips for effective research
