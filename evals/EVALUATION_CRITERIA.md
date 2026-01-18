# Evaluation Criteria for ResearchCrew

This document defines the evaluation criteria for assessing ResearchCrew agent outputs against the golden dataset.

## Quality Metrics

Each test case in `golden_dataset.jsonl` includes target quality metrics. Evaluators should score outputs on these dimensions:

### 1. Factual Accuracy (0.0 - 1.0)

Measures whether the information provided is correct and verifiable.

| Score | Criteria |
|-------|----------|
| 0.9 - 1.0 | All claims are accurate and verifiable |
| 0.7 - 0.9 | Minor inaccuracies that don't affect conclusions |
| 0.5 - 0.7 | Some factual errors present |
| 0.3 - 0.5 | Significant factual errors |
| 0.0 - 0.3 | Mostly incorrect or fabricated information |

**Evaluation Process:**
1. Identify each factual claim in the response
2. Verify against authoritative sources
3. Calculate: (correct claims) / (total claims)

### 2. Source Quality (0.0 - 1.0)

Measures the reliability and relevance of cited sources.

| Score | Criteria |
|-------|----------|
| 0.9 - 1.0 | All sources are authoritative, recent, and directly relevant |
| 0.7 - 0.9 | Mostly high-quality sources with minor gaps |
| 0.5 - 0.7 | Mix of good and questionable sources |
| 0.3 - 0.5 | Mostly low-quality or outdated sources |
| 0.0 - 0.3 | No sources or unreliable sources only |

**Source Quality Indicators:**
- Official documentation (highest)
- Peer-reviewed papers
- Reputable tech publications
- Personal blogs (context-dependent)
- Outdated content (>2 years for fast-moving topics)

### 3. Completeness (0.0 - 1.0)

Measures whether all expected topics and aspects are covered.

| Score | Criteria |
|-------|----------|
| 0.9 - 1.0 | All expected topics covered with depth |
| 0.7 - 0.9 | Most topics covered, minor gaps |
| 0.5 - 0.7 | Key topics covered but lacking depth |
| 0.3 - 0.5 | Several expected topics missing |
| 0.0 - 0.3 | Most expected topics not addressed |

**Evaluation Process:**
1. Compare response against `expected_topics` in test case
2. Check for additional relevant topics discovered
3. Assess depth of coverage for each topic

### 4. Coherence (0.0 - 1.0)

Measures logical flow, organization, and readability.

| Score | Criteria |
|-------|----------|
| 0.9 - 1.0 | Clear structure, logical flow, easy to follow |
| 0.7 - 0.9 | Well-organized with minor flow issues |
| 0.5 - 0.7 | Understandable but could be better organized |
| 0.3 - 0.5 | Disjointed or hard to follow |
| 0.0 - 0.3 | Incoherent or contradictory |

**Coherence Indicators:**
- Clear introduction and conclusion
- Logical transitions between sections
- Consistent terminology
- No contradictions within response

## Test Case Categories

### Research
Standard research queries requiring information gathering and synthesis.
- Expected: Comprehensive answers with multiple sources
- Focus: Factual accuracy and source quality

### Implementation
Technical how-to queries requiring practical guidance.
- Expected: Working code examples or step-by-step instructions
- Focus: Accuracy and completeness

### Comparison
Queries requiring analysis of multiple options.
- Expected: Balanced analysis with pros/cons
- Focus: Completeness and coherence

### Architecture
System design and pattern queries.
- Expected: Conceptual explanations with practical applications
- Focus: Coherence and completeness

### Security
Security-focused queries requiring careful handling.
- Expected: Accurate, cautious responses
- Focus: Factual accuracy (critical)

### Adversarial
Edge cases designed to test robustness.
- Expected: Appropriate handling (clarification, scoping, balanced response)
- Focus: Coherence and appropriate response behavior

## Difficulty Levels

### Easy
- Clear, well-defined queries
- Readily available information
- Target: High scores across all metrics

### Medium
- Some ambiguity or complexity
- Requires synthesis from multiple sources
- Target: Good scores with acceptable variation

### Hard
- Complex, multi-faceted queries
- May require reasoning or comparison
- Target: Acceptable scores with understanding of challenges

## Agent Type Expectations

### Researcher
Primary research agent for information gathering.
- Strengths: Source discovery, fact compilation
- Evaluation focus: Source quality, factual accuracy

### Synthesizer
Combines information from multiple sources.
- Strengths: Integration, comparison, analysis
- Evaluation focus: Completeness, coherence

### Fact-Checker
Verifies claims and identifies conflicts.
- Strengths: Accuracy verification, source validation
- Evaluation focus: Factual accuracy (critical)

### Writer
Produces final reports and summaries.
- Strengths: Clear communication, organization
- Evaluation focus: Coherence, completeness

## Scoring Process

### Automated Evaluation
1. Parse response for expected topics (keyword/semantic matching)
2. Count and validate source citations
3. Check response length and structure
4. Compare against `min_sources` requirement

### Human Evaluation
1. Read response completely
2. Score each metric independently (0.0 - 1.0)
3. Document specific issues found
4. Compare against target metrics in test case

### Pass/Fail Criteria
A test case passes if:
- All four metrics meet or exceed targets in `quality_metrics`
- `min_sources` requirement is met
- No critical factual errors (for security/accuracy-critical queries)

### Aggregate Scoring
- Per-category average across test cases
- Per-difficulty average
- Overall weighted average (harder tests weighted more)

## Adversarial Test Guidelines

Adversarial test cases (tc-023 through tc-026) require special handling:

| ID | Challenge | Expected Behavior |
|----|-----------|------------------|
| tc-023 | Sensationalist claim | Nuanced, balanced response |
| tc-024 | Vague temporal reference | Clarification or limitation acknowledgment |
| tc-025 | Overly broad query | Appropriate scoping |
| tc-026 | Subjective question | Balanced analysis, no single "best" answer |

For adversarial tests, coherence and appropriate response behavior are more important than strict topic coverage.

## Running Evaluations

```bash
# Run full evaluation suite
python evals/run_evals.py

# Run specific category
python evals/run_evals.py --category research

# Run specific difficulty
python evals/run_evals.py --difficulty hard

# Generate evaluation report
python evals/run_evals.py --output report.json
```

## Updating the Dataset

When adding new test cases:
1. Assign unique ID (tc-XXX format)
2. Define expected_topics (3-5 key concepts)
3. Set min_sources based on query complexity
4. Choose appropriate difficulty, category, agent_type
5. Set realistic quality_metrics targets
6. Add notes field for special handling requirements
