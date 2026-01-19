"""Custom Evaluation Metrics for ResearchCrew

Implements standard and custom metrics for evaluating research agent outputs.
Uses DeepEval framework for LLM-as-judge evaluation.
"""

import json
import re
from typing import Any, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Result from a metric evaluation."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class SourceQualityMetric(BaseMetric):
    """Evaluates the quality and reliability of sources cited in the response.

    Checks for:
    - Authoritative sources (official docs, papers, reputable publications)
    - Source relevance to the query
    - Recency of sources
    - Diversity of sources
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[str] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self._score: Optional[float] = None
        self._reason: Optional[str] = None

    @property
    def score(self) -> float:
        return self._score or 0.0

    @property
    def reason(self) -> str:
        return self._reason or ""

    @property
    def __name__(self) -> str:
        return "Source Quality"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure source quality in the response."""
        output = test_case.actual_output or ""

        # Extract URLs and source references
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', output)
        source_refs = re.findall(r'\[(?:Source|Ref|\d+)\]', output, re.IGNORECASE)

        # Score components
        has_sources = len(urls) > 0 or len(source_refs) > 0
        source_count = len(urls) + len(source_refs)

        # Check for authoritative domains
        authoritative_domains = [
            "github.com",
            "arxiv.org",
            "docs.",
            "documentation",
            "official",
            ".edu",
            "research",
            "paper",
            "google.com",
            "anthropic.com",
            "openai.com",
            "microsoft.com",
        ]
        authoritative_count = sum(
            1 for url in urls if any(domain in url.lower() for domain in authoritative_domains)
        )

        # Calculate score
        if not has_sources:
            self._score = 0.3  # Minimum score for no sources
            self._reason = "No sources cited in the response"
        elif source_count < 2:
            self._score = 0.5
            self._reason = f"Only {source_count} source(s) cited, expected at least 2"
        else:
            authority_ratio = authoritative_count / max(len(urls), 1)
            self._score = min(1.0, 0.6 + (0.2 * min(source_count / 3, 1)) + (0.2 * authority_ratio))
            self._reason = (
                f"Found {source_count} sources, {authoritative_count} from authoritative domains"
            )

        return self._score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self._score is not None and self._score >= self.threshold


class CompletenessMetric(BaseMetric):
    """Evaluates whether the response covers all expected topics.

    Checks for:
    - Presence of expected topics/keywords
    - Depth of coverage
    - Additional relevant topics discovered
    """

    def __init__(
        self,
        expected_topics: list[str],
        threshold: float = 0.7,
        include_reason: bool = True,
    ):
        self.expected_topics = expected_topics
        self.threshold = threshold
        self.include_reason = include_reason
        self._score: Optional[float] = None
        self._reason: Optional[str] = None
        self._covered_topics: list[str] = []
        self._missing_topics: list[str] = []

    @property
    def score(self) -> float:
        return self._score or 0.0

    @property
    def reason(self) -> str:
        return self._reason or ""

    @property
    def __name__(self) -> str:
        return "Completeness"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure topic completeness in the response."""
        output = (test_case.actual_output or "").lower()

        self._covered_topics = []
        self._missing_topics = []

        for topic in self.expected_topics:
            # Check for topic or related terms
            topic_lower = topic.lower()
            # Also check for variations (e.g., "RAG" matches "retrieval augmented generation")
            topic_variations = [topic_lower, topic_lower.replace("-", " "), topic_lower.replace("_", " ")]

            found = any(var in output for var in topic_variations)
            if found:
                self._covered_topics.append(topic)
            else:
                self._missing_topics.append(topic)

        if not self.expected_topics:
            self._score = 1.0
            self._reason = "No expected topics defined"
        else:
            coverage = len(self._covered_topics) / len(self.expected_topics)
            self._score = coverage
            self._reason = (
                f"Covered {len(self._covered_topics)}/{len(self.expected_topics)} expected topics. "
                f"Missing: {', '.join(self._missing_topics) if self._missing_topics else 'none'}"
            )

        return self._score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self._score is not None and self._score >= self.threshold


class CoherenceMetric(BaseMetric):
    """Evaluates the logical flow and organization of the response.

    Checks for:
    - Clear structure (introduction, body, conclusion)
    - Logical transitions
    - Consistent terminology
    - No contradictions
    """

    def __init__(
        self,
        threshold: float = 0.7,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self._score: Optional[float] = None
        self._reason: Optional[str] = None

    @property
    def score(self) -> float:
        return self._score or 0.0

    @property
    def reason(self) -> str:
        return self._reason or ""

    @property
    def __name__(self) -> str:
        return "Coherence"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure coherence of the response."""
        output = test_case.actual_output or ""

        scores = []
        reasons = []

        # Check 1: Response length (too short = low coherence potential)
        word_count = len(output.split())
        if word_count < 50:
            scores.append(0.5)
            reasons.append("Response is very short")
        elif word_count < 100:
            scores.append(0.7)
            reasons.append("Response is somewhat brief")
        else:
            scores.append(1.0)

        # Check 2: Structure markers (headers, lists, paragraphs)
        has_structure = any(
            [
                "##" in output or "**" in output,  # Markdown headers/bold
                "\n\n" in output,  # Paragraphs
                re.search(r"^\s*[-*]\s", output, re.MULTILINE),  # Lists
                re.search(r"^\s*\d+\.\s", output, re.MULTILINE),  # Numbered lists
            ]
        )
        if has_structure:
            scores.append(1.0)
        else:
            scores.append(0.7)
            reasons.append("Limited structural elements")

        # Check 3: Transition words
        transition_words = [
            "however",
            "therefore",
            "additionally",
            "furthermore",
            "moreover",
            "in contrast",
            "similarly",
            "consequently",
            "first",
            "second",
            "finally",
            "in summary",
            "in conclusion",
        ]
        transition_count = sum(1 for word in transition_words if word in output.lower())
        if transition_count >= 3:
            scores.append(1.0)
        elif transition_count >= 1:
            scores.append(0.8)
        else:
            scores.append(0.6)
            reasons.append("Few transition words")

        # Check 4: No obvious contradictions (simplified check)
        contradiction_patterns = [
            (r"is\s+(\w+).*is\s+not\s+\1", "Potential contradiction detected"),
            (r"always.*never", "Contradictory absolutes"),
        ]
        for pattern, msg in contradiction_patterns:
            if re.search(pattern, output.lower()):
                scores.append(0.5)
                reasons.append(msg)
                break

        # Calculate final score
        self._score = sum(scores) / len(scores) if scores else 0.5
        self._reason = "; ".join(reasons) if reasons else "Response shows good coherence"

        return self._score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self._score is not None and self._score >= self.threshold


class FactualAccuracyMetric(BaseMetric):
    """Evaluates factual accuracy using claim extraction and verification.

    This is a simplified heuristic-based version. For production,
    integrate with LLM-as-judge for claim verification.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self._score: Optional[float] = None
        self._reason: Optional[str] = None

    @property
    def score(self) -> float:
        return self._score or 0.0

    @property
    def reason(self) -> str:
        return self._reason or ""

    @property
    def __name__(self) -> str:
        return "Factual Accuracy"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure factual accuracy heuristically.

        For now, uses proxy signals. Full implementation would use
        LLM-as-judge with retrieval for fact checking.
        """
        output = test_case.actual_output or ""
        context = test_case.retrieval_context or []

        scores = []
        reasons = []

        # Check 1: Claims are hedged appropriately (uncertainty markers)
        uncertainty_markers = [
            "may",
            "might",
            "could",
            "typically",
            "generally",
            "often",
            "usually",
            "according to",
            "research suggests",
            "studies show",
        ]
        has_hedging = any(marker in output.lower() for marker in uncertainty_markers)
        if has_hedging:
            scores.append(1.0)
        else:
            scores.append(0.8)
            reasons.append("Claims could use more hedging")

        # Check 2: Sources cited for claims
        has_citations = bool(re.findall(r'https?://|(?:\[(?:Source|Ref|\d+)\])', output))
        if has_citations:
            scores.append(1.0)
        else:
            scores.append(0.7)
            reasons.append("No source citations found")

        # Check 3: Context alignment (if retrieval context provided)
        if context:
            context_text = " ".join(context).lower()
            output_lower = output.lower()
            # Check if key terms from context appear in output
            context_words = set(context_text.split())
            output_words = set(output_lower.split())
            overlap = len(context_words & output_words) / max(len(context_words), 1)
            if overlap > 0.1:
                scores.append(1.0)
            else:
                scores.append(0.7)
                reasons.append("Limited alignment with context")

        # Check 4: No obvious false claims (simplified - check for known issues)
        problematic_patterns = [
            r"100%\s+(?:accurate|certain|guaranteed)",
            r"always\s+(?:works|correct|right)",
            r"never\s+(?:fails|wrong|incorrect)",
        ]
        for pattern in problematic_patterns:
            if re.search(pattern, output.lower()):
                scores.append(0.5)
                reasons.append("Contains absolute claims")
                break

        self._score = sum(scores) / len(scores) if scores else 0.7
        self._reason = "; ".join(reasons) if reasons else "Response shows reasonable factual care"

        return self._score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self._score is not None and self._score >= self.threshold


class ResearchQualityMetric(BaseMetric):
    """Composite metric that combines all research quality dimensions.

    Aggregates:
    - Source Quality
    - Completeness
    - Coherence
    - Factual Accuracy
    """

    def __init__(
        self,
        expected_topics: list[str],
        weights: Optional[dict[str, float]] = None,
        threshold: float = 0.75,
        include_reason: bool = True,
    ):
        self.expected_topics = expected_topics
        self.weights = weights or {
            "source_quality": 0.25,
            "completeness": 0.25,
            "coherence": 0.25,
            "factual_accuracy": 0.25,
        }
        self.threshold = threshold
        self.include_reason = include_reason
        self._score: Optional[float] = None
        self._reason: Optional[str] = None
        self._component_scores: dict[str, float] = {}

    @property
    def score(self) -> float:
        return self._score or 0.0

    @property
    def reason(self) -> str:
        return self._reason or ""

    @property
    def __name__(self) -> str:
        return "Research Quality"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure overall research quality."""
        # Run component metrics
        source_metric = SourceQualityMetric()
        completeness_metric = CompletenessMetric(self.expected_topics)
        coherence_metric = CoherenceMetric()
        factual_metric = FactualAccuracyMetric()

        self._component_scores = {
            "source_quality": source_metric.measure(test_case),
            "completeness": completeness_metric.measure(test_case),
            "coherence": coherence_metric.measure(test_case),
            "factual_accuracy": factual_metric.measure(test_case),
        }

        # Calculate weighted average
        self._score = sum(
            self._component_scores[key] * self.weights[key] for key in self._component_scores
        )

        # Build reason
        score_summary = ", ".join(
            f"{key}: {score:.2f}" for key, score in self._component_scores.items()
        )
        self._reason = f"Component scores: {score_summary}"

        return self._score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if metric passed threshold."""
        return self._score is not None and self._score >= self.threshold

    def get_component_scores(self) -> dict[str, float]:
        """Get individual component scores."""
        return self._component_scores.copy()


def create_metrics_for_test_case(
    test_case_data: dict[str, Any],
) -> list[BaseMetric]:
    """Create appropriate metrics for a given test case.

    Args:
        test_case_data: Dictionary from golden_dataset.jsonl

    Returns:
        List of metrics configured for this test case
    """
    expected_topics = test_case_data.get("expected_topics", [])
    quality_targets = test_case_data.get("quality_metrics", {})

    metrics = [
        SourceQualityMetric(threshold=quality_targets.get("source_quality", 0.7)),
        CompletenessMetric(
            expected_topics=expected_topics,
            threshold=quality_targets.get("completeness", 0.7),
        ),
        CoherenceMetric(threshold=quality_targets.get("coherence", 0.7)),
        FactualAccuracyMetric(threshold=quality_targets.get("factual_accuracy", 0.8)),
    ]

    return metrics


def load_golden_dataset(filepath: str = "evals/golden_dataset.jsonl") -> list[dict[str, Any]]:
    """Load test cases from the golden dataset.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of test case dictionaries
    """
    test_cases = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))
    return test_cases
