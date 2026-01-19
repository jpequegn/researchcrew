"""Tests for Evaluation Metrics and Runner

Tests the custom evaluation metrics and the eval runner pipeline.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Skip this entire module if deepeval is not installed
pytest.importorskip("deepeval", reason="deepeval not installed")

from deepeval.test_case import LLMTestCase

from evals.metrics import (
    CoherenceMetric,
    CompletenessMetric,
    FactualAccuracyMetric,
    ResearchQualityMetric,
    SourceQualityMetric,
    create_metrics_for_test_case,
    load_golden_dataset,
)
from evals.run_evals import EvalRunner, TestCaseResult


class TestSourceQualityMetric:
    """Tests for the SourceQualityMetric."""

    def test_no_sources_low_score(self):
        """Test that responses without sources get low scores."""
        metric = SourceQualityMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence. It's used in many applications.",
        )

        score = metric.measure(test_case)
        assert score <= 0.5
        assert not metric.is_successful()

    def test_with_urls_higher_score(self):
        """Test that responses with URLs get higher scores."""
        metric = SourceQualityMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output=(
                "AI is artificial intelligence. "
                "According to https://docs.example.com/ai-guide, "
                "it involves machine learning. "
                "See also https://arxiv.org/abs/2024.12345 for more details."
            ),
        )

        score = metric.measure(test_case)
        assert score >= 0.6

    def test_authoritative_sources_boost_score(self):
        """Test that authoritative sources boost the score."""
        metric = SourceQualityMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="What is LangGraph?",
            actual_output=(
                "LangGraph is a framework for building agents. "
                "Source: https://docs.langchain.com/langgraph "
                "Additional info: https://github.com/langchain-ai/langgraph "
                "Research: https://arxiv.org/abs/2024.00001"
            ),
        )

        score = metric.measure(test_case)
        assert score >= 0.7
        assert metric.is_successful()


class TestCompletenessMetric:
    """Tests for the CompletenessMetric."""

    def test_all_topics_covered(self):
        """Test perfect score when all topics are covered."""
        metric = CompletenessMetric(
            expected_topics=["LangGraph", "CrewAI", "agents"],
            threshold=0.8,
        )
        test_case = LLMTestCase(
            input="What are AI agent frameworks?",
            actual_output=(
                "The main AI agent frameworks include LangGraph for graph-based workflows, "
                "CrewAI for role-based agents, and various other agents tools."
            ),
        )

        score = metric.measure(test_case)
        assert score == 1.0
        assert metric.is_successful()

    def test_partial_coverage(self):
        """Test partial score when some topics are missing."""
        metric = CompletenessMetric(
            expected_topics=["LangGraph", "CrewAI", "Google ADK", "AutoGen"],
            threshold=0.8,
        )
        test_case = LLMTestCase(
            input="What are AI agent frameworks?",
            actual_output="LangGraph and CrewAI are popular frameworks for building agents.",
        )

        score = metric.measure(test_case)
        assert score == 0.5  # 2 out of 4 topics
        assert not metric.is_successful()

    def test_no_topics_covered(self):
        """Test zero score when no topics are covered."""
        metric = CompletenessMetric(
            expected_topics=["specific_topic_1", "specific_topic_2"],
            threshold=0.5,
        )
        test_case = LLMTestCase(
            input="Tell me about something",
            actual_output="This is a generic response without any specific topics.",
        )

        score = metric.measure(test_case)
        assert score == 0.0

    def test_empty_expected_topics(self):
        """Test that empty expected topics gives full score."""
        metric = CompletenessMetric(expected_topics=[], threshold=0.5)
        test_case = LLMTestCase(
            input="What happened yesterday?",
            actual_output="I cannot provide information about yesterday.",
        )

        score = metric.measure(test_case)
        assert score == 1.0


class TestCoherenceMetric:
    """Tests for the CoherenceMetric."""

    def test_well_structured_response(self):
        """Test that well-structured responses score high."""
        metric = CoherenceMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="Explain agents",
            actual_output="""
## Introduction

AI agents are autonomous systems that can perform tasks.

## Key Features

- **Autonomy**: Agents can make decisions independently
- **Tool Use**: They can use external tools

Furthermore, agents can be combined into multi-agent systems.

## Conclusion

In summary, agents represent an important advancement in AI capabilities.
            """,
        )

        score = metric.measure(test_case)
        assert score >= 0.7
        assert metric.is_successful()

    def test_short_response_lower_score(self):
        """Test that very short responses score lower."""
        metric = CoherenceMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="What is AI?",
            actual_output="AI is artificial intelligence.",
        )

        score = metric.measure(test_case)
        assert score < 0.8

    def test_response_with_transitions(self):
        """Test that responses with transition words score better."""
        metric = CoherenceMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="Compare frameworks",
            actual_output=(
                "First, let's consider LangGraph which uses a graph-based approach. "
                "However, CrewAI takes a different approach with role-based agents. "
                "Furthermore, both frameworks support tool use. "
                "In conclusion, the choice depends on your specific needs."
            ),
        )

        score = metric.measure(test_case)
        assert score >= 0.7


class TestFactualAccuracyMetric:
    """Tests for the FactualAccuracyMetric."""

    def test_hedged_claims_score_well(self):
        """Test that properly hedged claims score well."""
        metric = FactualAccuracyMetric(threshold=0.8)
        test_case = LLMTestCase(
            input="Is LangGraph better than CrewAI?",
            actual_output=(
                "Research suggests that LangGraph may be better suited for complex workflows, "
                "while CrewAI typically excels in role-based scenarios. "
                "According to benchmarks (https://example.com/benchmarks), "
                "performance depends on the specific use case."
            ),
        )

        score = metric.measure(test_case)
        assert score >= 0.8

    def test_absolute_claims_penalized(self):
        """Test that absolute claims are penalized."""
        metric = FactualAccuracyMetric(threshold=0.8)
        test_case = LLMTestCase(
            input="Does this always work?",
            actual_output="This approach is 100% accurate and always works correctly.",
        )

        score = metric.measure(test_case)
        assert score < 0.8

    def test_cited_claims_score_better(self):
        """Test that cited claims score better."""
        metric = FactualAccuracyMetric(threshold=0.7)
        test_case = LLMTestCase(
            input="What is MCP?",
            actual_output=(
                "The Model Context Protocol (MCP) is a standard developed by Anthropic "
                "[Source: https://anthropic.com/mcp] for tool interoperability."
            ),
        )

        score = metric.measure(test_case)
        assert score >= 0.7


class TestResearchQualityMetric:
    """Tests for the composite ResearchQualityMetric."""

    def test_composite_score_calculation(self):
        """Test that composite score is calculated correctly."""
        metric = ResearchQualityMetric(
            expected_topics=["LangGraph", "agents"],
            threshold=0.7,
        )
        test_case = LLMTestCase(
            input="Tell me about LangGraph",
            actual_output=(
                "## LangGraph Overview\n\n"
                "LangGraph is a framework for building AI agents using a graph-based approach. "
                "According to the documentation (https://docs.langchain.com), "
                "it provides state management and workflow orchestration.\n\n"
                "Furthermore, LangGraph supports multi-agent systems.\n\n"
                "In conclusion, LangGraph is well-suited for complex agent workflows."
            ),
        )

        score = metric.measure(test_case)
        assert 0 <= score <= 1

        component_scores = metric.get_component_scores()
        assert "source_quality" in component_scores
        assert "completeness" in component_scores
        assert "coherence" in component_scores
        assert "factual_accuracy" in component_scores


class TestMetricFactory:
    """Tests for the metric factory function."""

    def test_create_metrics_for_test_case(self):
        """Test creating metrics from test case data."""
        test_case_data = {
            "id": "tc-001",
            "query": "What are AI agents?",
            "expected_topics": ["autonomy", "tools", "reasoning"],
            "quality_metrics": {
                "factual_accuracy": 0.9,
                "source_quality": 0.8,
                "completeness": 0.85,
                "coherence": 0.9,
            },
        }

        metrics = create_metrics_for_test_case(test_case_data)
        assert len(metrics) == 4

        # Check metric types
        metric_types = {type(m).__name__ for m in metrics}
        assert "SourceQualityMetric" in metric_types
        assert "CompletenessMetric" in metric_types
        assert "CoherenceMetric" in metric_types
        assert "FactualAccuracyMetric" in metric_types


class TestGoldenDatasetLoader:
    """Tests for the golden dataset loader."""

    def test_load_golden_dataset(self):
        """Test loading the actual golden dataset."""
        dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"
        if not dataset_path.exists():
            pytest.skip("Golden dataset not found")

        test_cases = load_golden_dataset(str(dataset_path))
        assert len(test_cases) > 0

        # Check structure of first test case
        tc = test_cases[0]
        assert "id" in tc
        assert "query" in tc
        assert "expected_topics" in tc
        assert "quality_metrics" in tc

    def test_load_custom_dataset(self):
        """Test loading a custom JSONL dataset."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            test_cases = [
                {
                    "id": "test-1",
                    "query": "Test query",
                    "expected_topics": ["topic1"],
                    "quality_metrics": {"factual_accuracy": 0.9},
                },
                {
                    "id": "test-2",
                    "query": "Another query",
                    "expected_topics": [],
                    "quality_metrics": {},
                },
            ]
            for tc in test_cases:
                f.write(json.dumps(tc) + "\n")
            f.flush()

            loaded = load_golden_dataset(f.name)
            assert len(loaded) == 2
            assert loaded[0]["id"] == "test-1"


class TestEvalRunner:
    """Tests for the EvalRunner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"

    def test_runner_initialization(self):
        """Test runner initializes correctly."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        assert len(runner.test_cases) > 0

    def test_filter_by_category(self):
        """Test filtering test cases by category."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        research_cases = runner.filter_test_cases(category="research")

        assert all(tc.get("category") == "research" for tc in research_cases)

    def test_filter_by_difficulty(self):
        """Test filtering test cases by difficulty."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        hard_cases = runner.filter_test_cases(difficulty="hard")

        assert all(tc.get("difficulty") == "hard" for tc in hard_cases)

    def test_filter_by_multiple_criteria(self):
        """Test filtering by multiple criteria."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        filtered = runner.filter_test_cases(category="comparison", difficulty="hard")

        assert all(tc.get("category") == "comparison" and tc.get("difficulty") == "hard" for tc in filtered)

    def test_evaluate_single_test_case(self):
        """Test evaluating a single test case."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        test_case = runner.test_cases[0]

        result = runner.evaluate_single(test_case)

        assert isinstance(result, TestCaseResult)
        assert result.test_case_id == test_case["id"]
        assert 0 <= result.overall_score <= 1
        assert len(result.metric_scores) > 0

    def test_run_full_evaluation(self):
        """Test running full evaluation on subset."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))

        # Run on easy difficulty only for speed
        report = runner.run(difficulty="easy")

        assert report.total_test_cases > 0
        assert 0 <= report.overall_pass_rate <= 1
        assert len(report.test_case_results) > 0
        assert "Source Quality" in report.average_scores


class TestEvalRunnerWithCustomProvider:
    """Tests for EvalRunner with custom response provider."""

    def test_custom_response_provider(self):
        """Test using a custom response provider."""
        dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"
        if not dataset_path.exists():
            pytest.skip("Golden dataset not found")

        def custom_provider(query: str, test_case: dict) -> str:
            return f"Custom response for: {query}"

        runner = EvalRunner(str(dataset_path), response_provider=custom_provider)
        test_case = runner.test_cases[0]
        result = runner.evaluate_single(test_case)

        # Score will be lower since custom response doesn't include expected topics
        assert result.overall_score < 0.9
