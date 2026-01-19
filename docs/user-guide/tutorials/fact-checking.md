# Tutorial: Fact-Checking

Learn how to verify claims and statements using ResearchCrew.

## Scenario

You've heard several claims about technology and software development that you want to verify before making decisions or sharing with your team.

## Step 1: Simple Fact Verification

Start with a straightforward claim.

**Claim to verify:**
> "Python is always slower than compiled languages"

**Query:**
```
Is it true that Python is always slower than compiled languages?
Please cite specific benchmarks and explain any nuances.
```

**Expected Output:**

- Verdict: Partially true with important nuances
- Performance comparison data
- Cases where Python matches compiled languages
- Explanation of when the claim holds and when it doesn't
- Source citations

**What to Look For:**
- Specific benchmark data
- Nuanced analysis (not just yes/no)
- Sources you can verify

## Step 2: Technology Claim Verification

Verify a claim about a specific technology.

**Claim to verify:**
> "GraphQL always outperforms REST APIs"

**Query:**
```
Verify this claim: "GraphQL always outperforms REST APIs"
Include evidence that supports or refutes this claim.
```

**Expected Output:**

- Verdict: False (oversimplified)
- Scenarios where GraphQL excels
- Scenarios where REST is better
- Performance comparison factors
- Real-world benchmarks and case studies

## Step 3: Statistics Verification

Verify a statistical claim.

**Claim to verify:**
> "90% of startups fail within the first year"

**Query:**
```
Is the claim that "90% of startups fail within the first year" accurate?
Please cite authoritative sources with actual statistics.
```

**Expected Output:**

- Verdict with actual statistics
- Sources (Bureau of Labor Statistics, etc.)
- Clarification on definitions (what counts as "fail")
- Time-frame accuracy (first year vs first five years)
- Context that's often missing from the claim

## Step 4: Best Practice Verification

Verify a commonly stated best practice.

**Claim to verify:**
> "You should always use microservices architecture for scalable applications"

**Query:**
```
Is it true that microservices architecture should always be used for
scalable applications? What does the evidence say about when to use
microservices vs monoliths?
```

**Expected Output:**

- Verdict: False (context-dependent)
- When microservices are appropriate
- When monoliths are better
- Evidence from real companies
- Decision criteria for choosing

## Step 5: Security Claim Verification

Verify a security-related claim.

**Claim to verify:**
> "HTTPS is sufficient to protect your web application from all attacks"

**Query:**
```
Fact-check: "HTTPS alone is sufficient to protect a web application
from all security threats." What does HTTPS actually protect against,
and what doesn't it cover?
```

**Expected Output:**

- Verdict: False (incomplete protection)
- What HTTPS protects against
- What HTTPS doesn't protect against
- Additional security measures needed
- Common misconceptions

## Fact-Checking Framework

Use this structure for verifying claims:

```
1. State the exact claim
2. Ask for evidence (supporting AND refuting)
3. Request source citations
4. Ask for nuance and context
5. Verify critical sources independently
```

## Query Patterns for Fact-Checking

### Direct Verification

```
"Is it true that [claim]? Please cite sources."
```

### Evidence-Based

```
"What does the evidence say about [topic]? I've heard that [claim]."
```

### Balanced Request

```
"Evaluate this claim: [claim]. Include evidence both supporting
and refuting it."
```

### Source Request

```
"Verify [claim] and provide primary sources I can check myself."
```

### Statistical Verification

```
"Check the accuracy of this statistic: [statistic].
What do authoritative sources say?"
```

## How to Read Fact-Check Results

### Verdicts

| Verdict | Meaning |
|---------|---------|
| **True** | Claim is accurate as stated |
| **Mostly True** | Claim is accurate but needs minor clarification |
| **Half True** | Partially accurate but missing important context |
| **Mostly False** | Contains some truth but misleads |
| **False** | Claim is inaccurate |

### Confidence Indicators

- **High confidence**: Multiple authoritative sources agree
- **Medium confidence**: Sources exist but some disagreement
- **Low confidence**: Limited sources or significant uncertainty

## Tips for Effective Fact-Checking

### Be Precise About the Claim

```
❌ "Is AI dangerous?"
✓ "Is the claim that 'AI will replace 50% of jobs by 2030' supported by research?"
```

### Request Primary Sources

```
"Please provide primary sources (research papers, official statistics)
rather than secondary reporting"
```

### Ask for Context

```
"What context is often missing when this claim is made?"
```

### Verify Independently

For critical claims:
1. Click through to the cited sources
2. Cross-reference with other authoritative sources
3. Check the date of the information
4. Consider the source's credibility

### Consider Source Quality

| Source Type | Reliability |
|-------------|-------------|
| Peer-reviewed research | Highest |
| Government statistics | High |
| Industry reports | Medium-High |
| News articles | Medium |
| Blog posts | Lower |
| Social media | Verify elsewhere |

## Common Fact-Checking Scenarios

### Technical Claims

```
"Verify: 'MongoDB is better for large-scale applications than PostgreSQL'"
```

### Performance Claims

```
"Is it true that Rust is 10x faster than Go for all use cases?"
```

### Trend Claims

```
"Verify: 'Most companies are moving away from cloud to on-premises infrastructure'"
```

### Security Claims

```
"Fact-check: 'Two-factor authentication prevents 99.9% of account breaches'"
```

### Cost Claims

```
"Is the claim that 'serverless is always cheaper than containers' accurate?"
```

## Red Flags in Claims

Be skeptical of claims that:
- Use absolute language ("always", "never", "all")
- Cite round numbers (50%, 90%, 10x)
- Come without sources
- Sound too good/bad to be true
- Are from biased sources (vendor marketing)

## After Fact-Checking

### If Claim is True
- Note the sources for future reference
- Understand any conditions or caveats
- Share with appropriate context

### If Claim is False
- Understand why it's false
- Learn the correct information
- Consider where the misconception came from

### If Claim is Nuanced
- Understand the conditions when it's true
- Know when it doesn't apply
- Communicate the full context

## Practice Exercises

Try fact-checking these common claims:

1. "React is faster than Angular"
2. "NoSQL databases don't support ACID transactions"
3. "Agile teams are 25% more productive than waterfall teams"
4. "Test coverage of 80% guarantees bug-free software"
5. "Kubernetes is necessary for any production deployment"

## Next Steps

- [Technical Research](./technical-research.md) - Deep dive on verified topics
- [Comparison Research](./comparison-research.md) - Compare options fairly
- [Best Practices](../best-practices.md) - Improve your research skills
