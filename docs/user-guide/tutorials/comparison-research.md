# Tutorial: Comparison Research

Learn how to systematically compare multiple options using ResearchCrew.

## Scenario

Your team needs to choose a frontend framework for a new enterprise dashboard application. You need to compare React, Vue, and Angular to make an informed decision.

## Step 1: Define Your Criteria

Before comparing, establish what matters.

**Query:**
```
What criteria should I consider when choosing a frontend framework
for an enterprise dashboard with complex data visualizations,
real-time updates, and a team of 5 developers?
```

**Expected Output:**

- Key decision criteria (learning curve, performance, ecosystem, etc.)
- Enterprise-specific considerations
- Team-size considerations
- Dashboard-specific requirements

**From this response, extract your criteria:**
- Performance with data visualizations
- Learning curve for the team
- Enterprise support and longevity
- Ecosystem for dashboard components
- Real-time data handling

## Step 2: High-Level Comparison

Get an overview comparison on all criteria.

**Query:**
```
Compare React, Vue, and Angular for an enterprise dashboard application.
Create a comparison covering: performance with data visualizations,
learning curve, enterprise support, component ecosystem, and real-time
data handling.
```

**Expected Output:**

| Criteria | React | Vue | Angular |
|----------|-------|-----|---------|
| Performance | ... | ... | ... |
| Learning curve | ... | ... | ... |
| Enterprise support | ... | ... | ... |
| Component ecosystem | ... | ... | ... |
| Real-time handling | ... | ... | ... |

Plus detailed analysis for each criterion.

## Step 3: Deep Dive on Top Concerns

Explore your most important criteria in detail.

**Query:**
```
For complex data visualizations (charts, graphs, large tables),
compare the performance and library ecosystem of React and Vue.
Include specific visualization libraries and their capabilities.
```

**Expected Output:**

- Specific visualization libraries (D3, Chart.js, ECharts integrations)
- Performance benchmarks
- Community support and maintenance status
- Integration complexity

## Step 4: Address Team Considerations

Consider your team's specific situation.

**Query:**
```
Our team has experience with JavaScript but limited framework experience.
How long would it typically take to become productive in React vs Vue
for building dashboards? Include learning resources and common challenges.
```

**Expected Output:**

- Learning curve comparison with timelines
- Recommended learning paths
- Common challenges for each framework
- Resources (official docs, courses, tutorials)

## Step 5: Real-World Validation

Check how others have made similar decisions.

**Query:**
```
What are some case studies of companies using React or Vue for
enterprise dashboards? What were their experiences and any
lessons learned?
```

**Expected Output:**

- Case studies from real companies
- Challenges they faced
- Successes they achieved
- Lessons learned

## Step 6: Make a Recommendation

Get a synthesized recommendation.

**Query:**
```
Based on our discussion, which framework would you recommend for our
enterprise dashboard (5-person team, JavaScript experience, complex
visualizations, real-time updates)? Summarize the key reasons.
```

**Expected Output:**

- Clear recommendation with reasoning
- Key advantages for your use case
- Potential challenges to prepare for
- Next steps for getting started

## Comparison Framework

Use this structure for any comparison:

```
1. Define criteria (what matters for YOUR situation)
2. High-level comparison (quick overview)
3. Deep dive on critical criteria (detailed analysis)
4. Consider your context (team, timeline, constraints)
5. Validate with real-world examples
6. Synthesize and recommend
```

## Tips for Effective Comparisons

### Be Specific About Context

| Vague | Specific |
|-------|----------|
| "Compare React and Vue" | "Compare React and Vue for an e-commerce site with high SEO requirements" |
| "Which is faster?" | "Which handles rendering 10,000 table rows more efficiently?" |

### Define Criteria First

Don't compare everything—compare what matters:

```
"Compare A and B focusing on:
1. [Most important criterion]
2. [Second most important]
3. [Third most important]"
```

### Ask for Trade-offs

```
"What would I give up by choosing React over Vue for this use case?"
```

### Consider Long-term

```
"How do React and Vue compare in terms of long-term maintenance
and community support trends?"
```

## Sample Comparison Queries

### Quick Decision

```
"Should I use [A] or [B] for [specific use case]? Give me the
main deciding factors."
```

### Detailed Analysis

```
"Create a detailed comparison of [A], [B], and [C] for [use case],
covering: [criterion 1], [criterion 2], [criterion 3]. Include
specific examples and data where available."
```

### Trade-off Analysis

```
"What are the trade-offs between [A] and [B] for [use case]?
When would each be the better choice?"
```

### Migration Consideration

```
"We're currently using [A] and considering moving to [B].
What would we gain and lose? Is it worth the migration cost?"
```

## Comparison Output Formats

### Request a Table

```
"Compare these options in a table format with rows for each criterion"
```

### Request Pros/Cons

```
"Give me pros and cons lists for each option"
```

### Request a Recommendation

```
"Based on the comparison, which would you recommend and why?"
```

### Request a Decision Matrix

```
"Create a weighted decision matrix for these options based on my criteria"
```

## Common Comparison Mistakes

### Comparing Incomparables

```
❌ "Compare React to PostgreSQL"
✓ "Compare React to Vue" (same category)
```

### Too Many Options

```
❌ "Compare 10 different databases"
✓ "Compare the top 3 databases for [specific use case]"
```

### No Context

```
❌ "Which is better, Python or Java?"
✓ "Which is better for building REST APIs with high concurrency requirements?"
```

### Ignoring Constraints

```
❌ "Compare all the options"
✓ "Compare options that work with our existing PostgreSQL database and have good TypeScript support"
```

## Real-World Decision Template

After your research, you should have:

1. **Clear criteria** - What mattered most
2. **Comparison data** - How options stack up
3. **Context analysis** - How it fits your situation
4. **Recommendation** - What to choose
5. **Next steps** - How to proceed
6. **Risk mitigation** - How to address weaknesses

## Next Steps

- [Technical Research](./technical-research.md) - Deep dive on your chosen option
- [Comprehensive Reports](./comprehensive-report.md) - Document your decision
- [Best Practices](../best-practices.md) - More research tips
