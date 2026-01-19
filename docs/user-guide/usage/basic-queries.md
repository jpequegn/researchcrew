# Basic Queries Guide

Learn how to formulate effective research queries to get the best results from ResearchCrew.

## Query Anatomy

A good research query has these components:

```
[Topic] + [Specific Aspect] + [Context/Constraints]
```

**Example:**
```
Machine learning (topic) + deployment best practices (aspect) + for production web applications (context)
```

## Query Types

### Information Gathering

Get comprehensive information on a topic.

**Pattern:** "What is/are [topic]?"

**Examples:**
```
What is quantum computing and how does it differ from classical computing?

What are the main machine learning frameworks available today?

What is the current state of renewable energy adoption globally?
```

### Comparison

Compare multiple options, technologies, or approaches.

**Pattern:** "Compare [A] and [B] for [use case]"

**Examples:**
```
Compare PostgreSQL and MongoDB for a social media application

Compare React, Vue, and Angular for enterprise web development

Compare AWS, Google Cloud, and Azure for machine learning workloads
```

### How-To

Learn how to accomplish something.

**Pattern:** "How do I/to [action] [context]?"

**Examples:**
```
How do I implement authentication in a Node.js application?

How to optimize PostgreSQL queries for large datasets?

How do I set up CI/CD for a Python project?
```

### Analysis

Understand implications, trends, or impacts.

**Pattern:** "[Analyze/Explain] [topic] [aspect]"

**Examples:**
```
Explain the security implications of using third-party APIs

Analyze the performance characteristics of different caching strategies

What are the trade-offs between microservices and monolithic architecture?
```

### Fact-Checking

Verify claims or statements.

**Pattern:** "Is it true that [claim]? / Verify: [statement]"

**Examples:**
```
Is it true that Python is slower than compiled languages for all use cases?

Verify: Electric vehicles have a lower total cost of ownership than gas vehicles

Is the claim that 5G causes health problems supported by scientific evidence?
```

## Good vs Bad Queries

### Be Specific

| Bad | Good |
|-----|------|
| "Tell me about AI" | "What are the main approaches to natural language processing in 2024?" |
| "Database help" | "How do I optimize slow queries in PostgreSQL with millions of rows?" |
| "Cloud computing" | "Compare serverless vs containers for a high-traffic e-commerce site" |

### Include Context

| Bad | Good |
|-----|------|
| "Best programming language" | "Best programming language for building REST APIs with high performance" |
| "How to deploy" | "How to deploy a Django application to AWS with automatic scaling" |
| "Security tips" | "Security best practices for handling user authentication in mobile apps" |

### Be Clear About What You Want

| Bad | Good |
|-----|------|
| "React stuff" | "React state management options: compare Redux, Zustand, and Context API" |
| "Help with errors" | "What causes 'CORS' errors and how do I fix them in a Node.js/React app?" |
| "ML model" | "How do I train and deploy a text classification model using transformers?" |

## Query Modifiers

Enhance your queries with these modifiers:

### Scope

- **"briefly"** - Get a concise summary
- **"in detail"** - Get comprehensive information
- **"with examples"** - Include practical examples
- **"for beginners"** - Explain concepts simply

### Focus

- **"focusing on [aspect]"** - Narrow to specific area
- **"especially [topic]"** - Emphasize particular points
- **"excluding [topic]"** - Omit certain areas

### Format

- **"as a comparison"** - Structure as comparison
- **"step by step"** - Sequential instructions
- **"with pros and cons"** - Balanced analysis

**Example with modifiers:**
```
Compare React and Vue for building dashboards, focusing on performance
and developer experience, with specific examples of when to choose each
```

## Query Length

- **Too short**: Lacks context, vague results
- **Just right**: 10-30 words, specific and clear
- **Too long**: May confuse focus; break into multiple queries

## Multi-Part Questions

For complex questions, you have two options:

### Option 1: Single Comprehensive Query

```
What are the best practices for building a production-ready REST API,
including authentication, rate limiting, and documentation?
```

### Option 2: Sequential Queries (Recommended)

```
Query 1: What are authentication best practices for REST APIs?
Query 2: How should I implement rate limiting? (follow-up)
Query 3: What tools are best for API documentation? (follow-up)
```

The sequential approach allows deeper exploration of each aspect.

## What ResearchCrew Does With Your Query

1. **Analyzes** the query to understand intent
2. **Decomposes** into 3-5 research angles
3. **Searches** multiple sources for each angle
4. **Synthesizes** findings into coherent insights
5. **Verifies** key claims against sources
6. **Formats** results with citations

## Common Mistakes

### Mistake 1: Too Vague
```
Bad:  "Help with Python"
Good: "How do I handle exceptions properly in Python with logging?"
```

### Mistake 2: Multiple Unrelated Topics
```
Bad:  "Tell me about React and also explain blockchain and what's good for breakfast"
Good: "What are React best practices for state management?" (single focused topic)
```

### Mistake 3: Assuming Context
```
Bad:  "How do I fix the error?" (what error?)
Good: "How do I fix 'TypeError: Cannot read property of undefined' in JavaScript?"
```

### Mistake 4: Opinion Questions
```
Bad:  "What's the best framework?" (best for what?)
Good: "What framework is best for building a real-time chat application at scale?"
```

## Next Steps

- [Multi-turn Conversations](./conversations.md) - Follow up on your research
- [Understanding Reports](./reports.md) - Interpret research results
- [Best Practices](../best-practices.md) - More tips for better results
