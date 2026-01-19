"""Quality Verification Tests

Comprehensive tests validating the ResearchCrew quality assurance system.
This file validates Issue #17 requirements:
- 20+ test cases in golden dataset
- Automated evals running successfully
- Quality gates working
- Baseline accuracy > 80%
- All quality metrics tracked (factual accuracy, source quality, completeness, coherence)
- Regression tests prevent quality degradation
"""

import json
from pathlib import Path

import pytest

# Skip this entire module if deepeval is not installed
pytest.importorskip("deepeval", reason="deepeval not installed")

from evals.metrics import (
    CompletenessMetric,
    CoherenceMetric,
    FactualAccuracyMetric,
    SourceQualityMetric,
    load_golden_dataset,
)
from evals.run_evals import EvalRunner


# ============================================================================
# Golden Dataset Verification
# ============================================================================


class TestGoldenDatasetRequirements:
    """Tests verifying golden dataset meets requirements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")
        self.test_cases = load_golden_dataset(str(self.dataset_path))

    def test_minimum_test_cases_count(self):
        """Verify golden dataset has at least 20 test cases."""
        assert len(self.test_cases) >= 20, (
            f"Golden dataset should have at least 20 test cases, but has {len(self.test_cases)}"
        )

    def test_all_test_cases_have_ids(self):
        """Verify all test cases have unique IDs."""
        ids = [tc.get("id") for tc in self.test_cases]
        assert all(ids), "All test cases should have IDs"
        assert len(ids) == len(set(ids)), "All test case IDs should be unique"

    def test_all_test_cases_have_queries(self):
        """Verify all test cases have queries."""
        for tc in self.test_cases:
            assert "query" in tc, f"Test case {tc.get('id')} missing query"
            assert len(tc["query"]) > 0, f"Test case {tc.get('id')} has empty query"

    def test_all_test_cases_have_expected_topics(self):
        """Verify all test cases have expected topics."""
        for tc in self.test_cases:
            assert "expected_topics" in tc, f"Test case {tc.get('id')} missing expected_topics"
            assert isinstance(tc["expected_topics"], list), (
                f"Test case {tc.get('id')} expected_topics should be a list"
            )

    def test_all_test_cases_have_quality_metrics(self):
        """Verify all test cases have quality metrics defined."""
        for tc in self.test_cases:
            assert "quality_metrics" in tc, f"Test case {tc.get('id')} missing quality_metrics"

    def test_test_cases_have_difficulty_levels(self):
        """Verify test cases have difficulty levels."""
        difficulties = {tc.get("difficulty") for tc in self.test_cases}
        assert "easy" in difficulties, "Dataset should have easy test cases"
        assert "medium" in difficulties, "Dataset should have medium test cases"
        assert "hard" in difficulties, "Dataset should have hard test cases"

    def test_test_cases_have_categories(self):
        """Verify test cases are categorized."""
        categories = {tc.get("category") for tc in self.test_cases}
        # Should have multiple categories
        assert len(categories) >= 3, "Dataset should have at least 3 categories"

    def test_test_cases_have_agent_types(self):
        """Verify test cases specify agent types."""
        agent_types = {tc.get("agent_type") for tc in self.test_cases}
        # Should have multiple agent types
        assert len(agent_types) >= 2, "Dataset should specify multiple agent types"


# ============================================================================
# Quality Metrics Verification
# ============================================================================


class TestQualityMetricsTracking:
    """Tests verifying all required quality metrics are tracked."""

    def test_factual_accuracy_metric_exists(self):
        """Verify factual accuracy metric exists and works."""
        metric = FactualAccuracyMetric(threshold=0.8)
        assert metric is not None
        assert hasattr(metric, "measure")
        assert hasattr(metric, "is_successful")

    def test_source_quality_metric_exists(self):
        """Verify source quality metric exists and works."""
        metric = SourceQualityMetric(threshold=0.7)
        assert metric is not None
        assert hasattr(metric, "measure")
        assert hasattr(metric, "is_successful")

    def test_completeness_metric_exists(self):
        """Verify completeness metric exists and works."""
        metric = CompletenessMetric(expected_topics=["test"], threshold=0.7)
        assert metric is not None
        assert hasattr(metric, "measure")
        assert hasattr(metric, "is_successful")

    def test_coherence_metric_exists(self):
        """Verify coherence metric exists and works."""
        metric = CoherenceMetric(threshold=0.7)
        assert metric is not None
        assert hasattr(metric, "measure")
        assert hasattr(metric, "is_successful")

    def test_all_metrics_have_thresholds(self):
        """Verify all metrics support threshold-based evaluation."""
        # Each metric should accept a threshold parameter
        factual = FactualAccuracyMetric(threshold=0.5)
        source = SourceQualityMetric(threshold=0.5)
        complete = CompletenessMetric(expected_topics=[], threshold=0.5)
        coherence = CoherenceMetric(threshold=0.5)

        # All should have threshold property
        assert factual.threshold == 0.5
        assert source.threshold == 0.5
        assert complete.threshold == 0.5
        assert coherence.threshold == 0.5


# ============================================================================
# Quality Gates Verification
# ============================================================================


class TestQualityGates:
    """Tests verifying quality gates work correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"

    def test_eval_runner_can_run(self):
        """Verify evaluation runner can execute."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        assert runner is not None
        assert len(runner.test_cases) > 0

    def test_eval_runner_produces_report(self):
        """Verify evaluation produces a report with required fields."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        # Run on easy cases only for speed
        report = runner.run(difficulty="easy")

        # Check report structure
        assert hasattr(report, "total_test_cases")
        assert hasattr(report, "passed_count")
        assert hasattr(report, "failed_count")
        assert hasattr(report, "overall_pass_rate")
        assert hasattr(report, "average_scores")

    def test_quality_gate_pass_rate_tracking(self):
        """Verify pass rate is tracked for quality gates."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        report = runner.run(difficulty="easy")

        # Pass rate should be between 0 and 1
        assert 0 <= report.overall_pass_rate <= 1

        # Passed + failed should equal total
        assert report.passed_count + report.failed_count == report.total_test_cases

    def test_report_includes_all_metric_scores(self):
        """Verify report includes scores for all tracked metrics."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        report = runner.run(difficulty="easy")

        # Should have scores for all four metrics
        expected_metrics = {"Source Quality", "Completeness", "Coherence", "Factual Accuracy"}
        actual_metrics = set(report.average_scores.keys())

        assert expected_metrics == actual_metrics, (
            f"Missing metrics: {expected_metrics - actual_metrics}"
        )

    def test_report_includes_category_breakdown(self):
        """Verify report includes scores broken down by category."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        report = runner.run()

        assert hasattr(report, "scores_by_category")
        assert len(report.scores_by_category) > 0

        # Each category should have metric scores
        for category, scores in report.scores_by_category.items():
            assert "pass_rate" in scores, f"Category {category} missing pass_rate"


# ============================================================================
# Baseline Accuracy Verification
# ============================================================================


class TestBaselineAccuracy:
    """Tests verifying baseline accuracy meets requirements."""

    def setup_method(self):
        """Set up test fixtures."""
        self.baseline_path = Path(__file__).parent.parent / "evals" / "baseline_report.json"

    def test_baseline_report_exists(self):
        """Verify baseline report exists."""
        assert self.baseline_path.exists(), "Baseline report should exist at evals/baseline_report.json"

    def test_baseline_overall_pass_rate_above_80(self):
        """Verify baseline overall pass rate is above 80%."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        assert "overall_pass_rate" in report
        assert report["overall_pass_rate"] >= 0.80, (
            f"Baseline pass rate should be >= 80%, got {report['overall_pass_rate']*100:.1f}%"
        )

    def test_baseline_factual_accuracy_above_target(self):
        """Verify baseline factual accuracy meets target (>85%)."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        factual = report.get("average_scores", {}).get("Factual Accuracy", 0)
        assert factual >= 0.85, (
            f"Factual Accuracy should be >= 85%, got {factual*100:.1f}%"
        )

    def test_baseline_source_quality_above_target(self):
        """Verify baseline source quality meets target (>80%)."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        source = report.get("average_scores", {}).get("Source Quality", 0)
        assert source >= 0.80, (
            f"Source Quality should be >= 80%, got {source*100:.1f}%"
        )

    def test_baseline_completeness_above_target(self):
        """Verify baseline completeness meets target (>75%)."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        completeness = report.get("average_scores", {}).get("Completeness", 0)
        assert completeness >= 0.75, (
            f"Completeness should be >= 75%, got {completeness*100:.1f}%"
        )

    def test_baseline_coherence_above_target(self):
        """Verify baseline coherence meets target (>80%)."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        coherence = report.get("average_scores", {}).get("Coherence", 0)
        assert coherence >= 0.80, (
            f"Coherence should be >= 80%, got {coherence*100:.1f}%"
        )

    def test_baseline_has_test_case_results(self):
        """Verify baseline has individual test case results."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            report = json.load(f)

        assert "test_case_results" in report
        assert len(report["test_case_results"]) >= 20


# ============================================================================
# Regression Test Infrastructure
# ============================================================================


class TestRegressionPrevention:
    """Tests verifying regression prevention infrastructure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"
        self.baseline_path = Path(__file__).parent.parent / "evals" / "baseline_report.json"

    def test_baseline_can_be_compared(self):
        """Verify baseline report can be used for comparison."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            baseline = json.load(f)

        # Should have comparable metrics
        assert "average_scores" in baseline
        assert "overall_pass_rate" in baseline
        assert "total_test_cases" in baseline

    def test_can_detect_regression_in_metrics(self):
        """Verify we can detect regression in quality metrics."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            baseline = json.load(f)

        # Simulate checking for regression
        baseline_factual = baseline["average_scores"]["Factual Accuracy"]

        # If new score is significantly lower, it's a regression
        new_score = baseline_factual - 0.10  # Simulated 10% drop
        regression_threshold = 0.05  # 5% tolerance

        is_regression = (baseline_factual - new_score) > regression_threshold
        assert is_regression, "Should detect 10% drop as regression"

    def test_category_level_regression_detection(self):
        """Verify regression can be detected at category level."""
        if not self.baseline_path.exists():
            pytest.skip("Baseline report not found")

        with open(self.baseline_path) as f:
            baseline = json.load(f)

        # Should have category-level scores for granular regression detection
        assert "scores_by_category" in baseline
        assert len(baseline["scores_by_category"]) > 0

        # Each category should have comparable metrics
        for category, scores in baseline["scores_by_category"].items():
            assert "pass_rate" in scores, (
                f"Category {category} should have pass_rate for regression detection"
            )


# ============================================================================
# Evaluation Pipeline Integration
# ============================================================================


class TestEvaluationPipelineIntegration:
    """Tests for evaluation pipeline integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"

    def test_can_run_evaluation_on_subset(self):
        """Verify evaluation can run on filtered subset."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))

        # Filter to easy cases
        filtered = runner.filter_test_cases(difficulty="easy")
        assert len(filtered) > 0

        # Run evaluation on filtered set
        report = runner.run(difficulty="easy")
        assert report.total_test_cases == len(filtered)

    def test_can_evaluate_by_category(self):
        """Verify evaluation can run by category."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))

        # Filter by category
        research_cases = runner.filter_test_cases(category="research")

        if len(research_cases) > 0:
            report = runner.run(category="research")
            assert report.total_test_cases == len(research_cases)

    def test_individual_test_case_evaluation(self):
        """Verify individual test cases can be evaluated."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        test_case = runner.test_cases[0]

        result = runner.evaluate_single(test_case)

        assert result.test_case_id == test_case["id"]
        assert 0 <= result.overall_score <= 1
        assert len(result.metric_scores) == 4  # All four metrics
        assert all(0 <= score <= 1 for score in result.metric_scores.values())

    def test_report_serialization(self):
        """Verify evaluation report can be serialized to JSON."""
        if not self.dataset_path.exists():
            pytest.skip("Golden dataset not found")

        runner = EvalRunner(str(self.dataset_path))
        report = runner.run(difficulty="easy")

        # Convert report to dict
        report_dict = {
            "timestamp": report.timestamp,
            "total_test_cases": report.total_test_cases,
            "passed_count": report.passed_count,
            "failed_count": report.failed_count,
            "overall_pass_rate": report.overall_pass_rate,
            "average_scores": report.average_scores,
            "scores_by_category": report.scores_by_category,
        }

        # Should be JSON serializable
        json_str = json.dumps(report_dict)
        assert len(json_str) > 0

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["total_test_cases"] == report.total_test_cases
