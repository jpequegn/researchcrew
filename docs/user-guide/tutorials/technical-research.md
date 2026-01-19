# Tutorial: Technical Research

Learn how to research a technical topic effectively with ResearchCrew.

## Scenario

You're evaluating message queue systems for a new project. You need to understand the options, compare them, and make a recommendation.

## Step 1: Start with an Overview

Begin with a broad question to understand the landscape.

**Query:**
```
What are the main message queue systems used in production today,
and what are their primary use cases?
```

**Expected Output:**

ResearchCrew will return:
- Overview of major systems (Kafka, RabbitMQ, Redis Streams, AWS SQS, etc.)
- Primary use cases for each
- Market adoption data
- Sources from official documentation and industry reports

**What to Look For:**
- Which systems are most widely used
- General categorization (streaming vs traditional queues)
- Initial sense of which might fit your needs

## Step 2: Narrow to Your Use Case

Now focus on your specific requirements.

**Query:**
```
For a system that needs to process 100,000 events per second with
at-least-once delivery guarantees, which message queue would you recommend?
Focus on Kafka, RabbitMQ, and AWS SQS.
```

**Expected Output:**

- Performance comparisons at scale
- Delivery guarantee mechanisms
- Operational complexity trade-offs
- Specific recommendations with reasoning

**What to Look For:**
- How each handles your throughput requirements
- Delivery guarantee implementations
- Operational considerations

## Step 3: Deep Dive on the Top Candidate

Based on the recommendation, explore further.

**Query:**
```
Tell me more about Kafka's architecture for handling 100k events/second.
What are the key configuration parameters and potential bottlenecks?
```

**Expected Output:**

- Kafka architecture explanation (partitions, brokers, consumers)
- Key configuration parameters
- Common bottlenecks and how to address them
- Scaling considerations

**What to Look For:**
- Specific technical details you'll need for implementation
- Configuration best practices
- Warning signs and pitfalls

## Step 4: Understand Trade-offs

Get a balanced view of challenges.

**Query:**
```
What are the main challenges and downsides of running Kafka in production?
Include operational complexity, cost considerations, and common pitfalls.
```

**Expected Output:**

- Operational challenges (ZooKeeper dependency, partition management)
- Cost considerations (infrastructure, expertise)
- Common pitfalls teams encounter
- Mitigation strategies

**What to Look For:**
- Realistic expectations
- Hidden costs
- Skills required for your team

## Step 5: Explore Alternatives

Ensure you're not missing a better option.

**Query:**
```
Given the challenges with Kafka, would a managed service like
Confluent Cloud or AWS MSK be a better choice for a team
new to Kafka?
```

**Expected Output:**

- Comparison of self-managed vs managed Kafka
- Cost comparison analysis
- Feature differences
- Recommendations based on team experience

## Step 6: Get Actionable Guidance

End with practical next steps.

**Query:**
```
If we decide to go with AWS MSK, what are the first steps
to set up a production-ready cluster? Include security
and monitoring considerations.
```

**Expected Output:**

- Step-by-step setup guidance
- Security best practices
- Monitoring setup recommendations
- Links to relevant documentation

## Complete Conversation Summary

```
Turn 1: Overview of message queue landscape
Turn 2: Narrowed comparison for your requirements
Turn 3: Deep dive on Kafka architecture
Turn 4: Understanding challenges and trade-offs
Turn 5: Exploring managed alternatives
Turn 6: Actionable implementation guidance
```

## Tips for Technical Research

### Start Broad, Go Deep

```
1. What options exist? (landscape)
2. Which fit my needs? (filtering)
3. How does it work? (understanding)
4. What can go wrong? (risk assessment)
5. How do I implement it? (action)
```

### Include Your Constraints

Mention:
- Scale requirements (events/sec, data volume)
- Team experience level
- Budget constraints
- Existing infrastructure

### Verify Critical Claims

For important decisions:
- Click through to source documentation
- Test claims in a proof of concept
- Cross-reference with other research

### Build on Previous Answers

Use context from earlier in the conversation:
```
"Based on the Kafka challenges you mentioned, how does Pulsar address those issues?"
```

## Common Mistakes

### Too Broad Initial Query

```
❌ "Tell me everything about message queues"
✓ "What are the main message queue systems for high-throughput event processing?"
```

### Skipping Context

```
❌ "Which is better, Kafka or RabbitMQ?"
✓ "Which is better for 100k events/sec with at-least-once delivery, Kafka or RabbitMQ?"
```

### Not Following Up on Concerns

```
❌ Accept "Kafka is operationally complex" at face value
✓ "What specifically makes Kafka operationally complex? How do teams address this?"
```

## Sample Query Patterns

### Architecture Understanding

```
"Explain how [technology] handles [specific concern] at scale"
```

### Performance Comparison

```
"Compare [A] and [B] performance for [specific workload] with [specific metrics]"
```

### Best Practices

```
"What are the production best practices for [technology] including [specific areas]?"
```

### Risk Assessment

```
"What are the main risks and challenges of using [technology] for [use case]?"
```

## Next Steps

- [Comparison Research](./comparison-research.md) - Compare solutions systematically
- [Fact-Checking](./fact-checking.md) - Verify technical claims
- [Best Practices](../best-practices.md) - More query tips
