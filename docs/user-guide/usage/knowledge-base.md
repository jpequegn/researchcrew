# Knowledge Base Guide

ResearchCrew can save your research findings to a knowledge base, allowing you to build on past research and avoid repeating work.

## What is the Knowledge Base?

The knowledge base is a searchable storage system that:

- Stores research findings from your sessions
- Enables semantic search across past research
- Links findings to original sources
- Organizes information by topic

## How It Works

```
Research Session
      ↓
Findings Discovered
      ↓
Saved to Knowledge Base (with sources, topics, timestamps)
      ↓
Future Queries Search Knowledge Base First
      ↓
Past Research Enhances New Research
```

## Automatic Saving

By default, ResearchCrew automatically saves significant findings:

- Key facts with high confidence
- Important conclusions
- Well-sourced information

You can also manually save specific findings.

## Searching the Knowledge Base

### Automatic Search

When you submit a query, ResearchCrew automatically checks if relevant past research exists:

```
You: "What are the best practices for API authentication?"

ResearchCrew: [Searches knowledge base first]
"I found previous research on this topic from [date]. Here's what we
learned before, plus new findings..."
```

### Manual Search

Query your knowledge base directly:

```python
from utils.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
results = kb.search("API authentication", limit=5)

for result in results:
    print(f"Topic: {result['topic']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['source']}")
```

## Saving Research

### Automatic Saving

High-quality findings are saved automatically during research sessions.

### Manual Saving

Save specific information:

```python
kb.save(
    content="OAuth 2.0 is the industry standard for API authentication...",
    topic="API Security",
    source="https://oauth.net/2/"
)
```

### Via MCP (Claude Code)

If using the MCP server:

```
Save to knowledge base:
- Topic: "API Security"
- Content: "OAuth 2.0 provides secure delegated access..."
- Source: "https://oauth.net/2/"
```

## Organizing by Topic

The knowledge base organizes information by topic:

```
Knowledge Base
├── API Security
│   ├── OAuth 2.0 overview
│   ├── JWT best practices
│   └── API key management
├── Database Performance
│   ├── Indexing strategies
│   └── Query optimization
└── Cloud Architecture
    ├── Microservices patterns
    └── Serverless considerations
```

### Listing Topics

```python
topics = kb.list_topics()
for topic in topics:
    print(f"- {topic}")
```

### Filtering by Topic

```python
results = kb.search("authentication", topic="API Security")
```

## Benefits of Using the Knowledge Base

### 1. Build on Past Research

Don't start from scratch. Your previous research enhances new queries:

```
Previous Research: "Learned about React state management"
New Query: "How do I handle global state?"
Result: "Building on your previous research about React, here are
        specific global state solutions..."
```

### 2. Cross-Reference Information

Connect findings across different research sessions:

```
Session 1: Research on database choices
Session 2: Research on API design
Knowledge Base: Links "database performance" insights to "API response time"
               discussions automatically
```

### 3. Maintain Consistency

Ensure your research stays consistent over time:

```
Previous Finding: "Chose PostgreSQL for our use case"
New Query: "How should I handle data migrations?"
Result: "For your PostgreSQL database, here are migration strategies..."
```

### 4. Team Knowledge Sharing

When configured for team use, share research across team members:

```
Team Member A: Researches authentication approaches
Team Member B: Query references authentication
Result: B benefits from A's research automatically
```

## Knowledge Base Queries

### Semantic Search

Find conceptually related information, not just keyword matches:

```
Query: "How to make database faster"
Finds: Entries about "query optimization", "indexing", "caching strategies"
       (even if they don't contain "faster")
```

### Time-Based Search

Find recent research:

```python
# Research from last week
results = kb.search("authentication", days=7)
```

### Confidence Filtering

Get only high-confidence findings:

```python
results = kb.search("best practices", min_confidence=0.8)
```

## Managing the Knowledge Base

### Viewing Statistics

```python
stats = kb.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Topics: {stats['topic_count']}")
print(f"Sources: {stats['unique_sources']}")
```

### Deleting Entries

Remove outdated or incorrect information:

```python
kb.delete(entry_id="entry-123")
```

### Exporting Research

Export your knowledge base for backup or sharing:

```python
kb.export("my_research.json")
```

### Importing Research

Import previously exported research:

```python
kb.import_from("shared_research.json")
```

## Configuration

### Storage Location

Set where the knowledge base stores data:

```bash
export KNOWLEDGE_BASE_PATH=/path/to/knowledge_base
```

Or in configuration:
```yaml
knowledge_base:
  path: ./knowledge_base
  collection_name: research_findings
```

### Retention Policy

Configure how long to keep entries:

```yaml
knowledge_base:
  retention_days: 365  # Keep for one year
  max_entries: 10000   # Maximum entries to store
```

## Best Practices

### Do

- Let the knowledge base build naturally through research
- Review saved findings periodically for accuracy
- Organize with meaningful topic names
- Link related research through follow-up queries

### Don't

- Save every trivial finding manually
- Let outdated information accumulate
- Ignore low-confidence warnings on old findings
- Assume all past research is still accurate

## Example Workflow

1. **Initial Research**
   ```
   "What are the best practices for microservices authentication?"
   → Findings saved to knowledge base under "Microservices Security"
   ```

2. **Later Research** (days/weeks later)
   ```
   "How should I secure service-to-service communication?"
   → Knowledge base provides relevant past findings
   → New research builds on existing foundation
   ```

3. **Review and Update**
   ```
   "Have authentication best practices changed since my last research?"
   → Compares current information with knowledge base
   → Updates outdated entries
   ```

## Next Steps

- [Understanding Reports](./reports.md) - Interpret research results
- [Best Practices](../best-practices.md) - Tips for better research
- [Tutorials](../tutorials/comprehensive-report.md) - Build comprehensive reports
