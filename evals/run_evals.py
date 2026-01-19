#!/usr/bin/env python3
"""Evaluation Runner for ResearchCrew

Runs automated evaluations against the golden dataset using custom metrics.
Supports filtering by category, difficulty, and agent type.
Generates detailed reports with scores and breakdowns.

Usage:
    python evals/run_evals.py                      # Run all evaluations
    python evals/run_evals.py --category research  # Filter by category
    python evals/run_evals.py --difficulty hard    # Filter by difficulty
    python evals/run_evals.py --output report.json # Save report to file
    python evals/run_evals.py --dry-run            # Show test cases without running
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from deepeval.test_case import LLMTestCase

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.metrics import (
    CompletenessMetric,
    CoherenceMetric,
    FactualAccuracyMetric,
    ResearchQualityMetric,
    SourceQualityMetric,
    create_metrics_for_test_case,
    load_golden_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Result from evaluating a single test case."""

    test_case_id: str
    query: str
    category: str
    difficulty: str
    agent_type: str
    passed: bool
    overall_score: float
    metric_scores: dict[str, float]
    metric_reasons: dict[str, str]
    target_metrics: dict[str, float]
    execution_time_ms: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    timestamp: str
    total_test_cases: int
    passed_count: int
    failed_count: int
    overall_pass_rate: float
    average_scores: dict[str, float]
    scores_by_category: dict[str, dict[str, float]]
    scores_by_difficulty: dict[str, dict[str, float]]
    scores_by_agent_type: dict[str, dict[str, float]]
    test_case_results: list[dict[str, Any]]
    filters_applied: dict[str, Optional[str]] = field(default_factory=dict)


class EvalRunner:
    """Runner for evaluation pipeline."""

    def __init__(
        self,
        dataset_path: str = "evals/golden_dataset.jsonl",
        response_provider: Optional[callable] = None,
    ):
        """Initialize the eval runner.

        Args:
            dataset_path: Path to golden dataset JSONL file
            response_provider: Optional callable that takes a query and returns a response.
                              If not provided, uses mock responses for testing.
        """
        self.dataset_path = dataset_path
        self.response_provider = response_provider or self._mock_response_provider
        self.test_cases = load_golden_dataset(dataset_path)
        logger.info(f"Loaded {len(self.test_cases)} test cases from {dataset_path}")

    def _mock_response_provider(self, query: str, test_case: dict[str, Any]) -> str:
        """Generate a mock response for testing the evaluation pipeline.

        In production, this would be replaced with actual agent execution.
        """
        expected_topics = test_case.get("expected_topics", [])
        category = test_case.get("category", "research")

        # Generate a mock response that includes expected topics
        response_parts = [
            f"## Response to: {query}\n",
            "Based on my research, here are the key findings:\n\n",
        ]

        # Add topics
        for i, topic in enumerate(expected_topics, 1):
            response_parts.append(f"{i}. **{topic}**: This is an important aspect to consider. ")
            response_parts.append(
                "Research suggests that understanding this concept is crucial for success.\n\n"
            )

        # Add some sources
        response_parts.append("\n### Sources\n")
        response_parts.append("- https://docs.example.com/topic-guide\n")
        response_parts.append("- https://github.com/example/project\n")
        response_parts.append("- https://arxiv.org/abs/2024.12345\n")

        # Add conclusion with transition words
        response_parts.append("\n### Conclusion\n")
        response_parts.append(
            "In summary, the key takeaways are that these concepts work together "
            "to provide a comprehensive solution. Furthermore, understanding the trade-offs "
            "is essential for making informed decisions.\n"
        )

        return "".join(response_parts)

    def filter_test_cases(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        agent_type: Optional[str] = None,
        test_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Filter test cases by criteria.

        Args:
            category: Filter by category (e.g., "research", "comparison")
            difficulty: Filter by difficulty (e.g., "easy", "medium", "hard")
            agent_type: Filter by agent type (e.g., "researcher", "synthesizer")
            test_ids: Filter by specific test case IDs

        Returns:
            Filtered list of test cases
        """
        filtered = self.test_cases.copy()

        if category:
            filtered = [tc for tc in filtered if tc.get("category") == category]

        if difficulty:
            filtered = [tc for tc in filtered if tc.get("difficulty") == difficulty]

        if agent_type:
            filtered = [tc for tc in filtered if tc.get("agent_type") == agent_type]

        if test_ids:
            filtered = [tc for tc in filtered if tc.get("id") in test_ids]

        return filtered

    def evaluate_single(self, test_case: dict[str, Any]) -> TestCaseResult:
        """Evaluate a single test case.

        Args:
            test_case: Test case dictionary from golden dataset

        Returns:
            TestCaseResult with scores and details
        """
        import time

        start_time = time.time()

        test_id = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        expected_topics = test_case.get("expected_topics", [])
        target_metrics = test_case.get("quality_metrics", {})

        logger.info(f"Evaluating {test_id}: {query[:50]}...")

        # Get response from provider
        response = self.response_provider(query, test_case)

        # Create DeepEval test case
        llm_test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=None,  # We use custom metrics instead
            retrieval_context=[],  # Would be populated from actual retrieval
        )

        # Run metrics
        metrics = create_metrics_for_test_case(test_case)
        metric_scores = {}
        metric_reasons = {}

        for metric in metrics:
            score = metric.measure(llm_test_case)
            metric_scores[metric.__name__] = score
            metric_reasons[metric.__name__] = metric.reason

        # Run composite metric
        composite = ResearchQualityMetric(expected_topics=expected_topics)
        overall_score = composite.measure(llm_test_case)

        # Determine pass/fail based on target metrics
        passed = True
        for metric_name, target in target_metrics.items():
            # Map target metric names to our metric names
            metric_name_map = {
                "factual_accuracy": "Factual Accuracy",
                "source_quality": "Source Quality",
                "completeness": "Completeness",
                "coherence": "Coherence",
            }
            our_metric_name = metric_name_map.get(metric_name)
            if our_metric_name and our_metric_name in metric_scores:
                if metric_scores[our_metric_name] < target:
                    passed = False

        execution_time = (time.time() - start_time) * 1000

        return TestCaseResult(
            test_case_id=test_id,
            query=query,
            category=test_case.get("category", "unknown"),
            difficulty=test_case.get("difficulty", "unknown"),
            agent_type=test_case.get("agent_type", "unknown"),
            passed=passed,
            overall_score=overall_score,
            metric_scores=metric_scores,
            metric_reasons=metric_reasons,
            target_metrics=target_metrics,
            execution_time_ms=execution_time,
        )

    def run(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        agent_type: Optional[str] = None,
        test_ids: Optional[list[str]] = None,
    ) -> EvaluationReport:
        """Run evaluation on filtered test cases.

        Args:
            category: Filter by category
            difficulty: Filter by difficulty
            agent_type: Filter by agent type
            test_ids: Filter by specific test IDs

        Returns:
            EvaluationReport with all results
        """
        filtered_cases = self.filter_test_cases(category, difficulty, agent_type, test_ids)
        logger.info(f"Running evaluation on {len(filtered_cases)} test cases")

        results: list[TestCaseResult] = []
        for test_case in filtered_cases:
            result = self.evaluate_single(test_case)
            results.append(result)

        # Calculate aggregates
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        # Average scores by metric
        metric_names = ["Source Quality", "Completeness", "Coherence", "Factual Accuracy"]
        average_scores = {}
        for metric_name in metric_names:
            scores = [r.metric_scores.get(metric_name, 0) for r in results]
            average_scores[metric_name] = sum(scores) / len(scores) if scores else 0

        # Scores by category
        scores_by_category = self._aggregate_by_field(results, "category", metric_names)

        # Scores by difficulty
        scores_by_difficulty = self._aggregate_by_field(results, "difficulty", metric_names)

        # Scores by agent type
        scores_by_agent_type = self._aggregate_by_field(results, "agent_type", metric_names)

        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_test_cases=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            overall_pass_rate=passed_count / len(results) if results else 0,
            average_scores=average_scores,
            scores_by_category=scores_by_category,
            scores_by_difficulty=scores_by_difficulty,
            scores_by_agent_type=scores_by_agent_type,
            test_case_results=[self._result_to_dict(r) for r in results],
            filters_applied={
                "category": category,
                "difficulty": difficulty,
                "agent_type": agent_type,
            },
        )

    def _aggregate_by_field(
        self,
        results: list[TestCaseResult],
        field: str,
        metric_names: list[str],
    ) -> dict[str, dict[str, float]]:
        """Aggregate scores by a grouping field."""
        from collections import defaultdict

        groups: dict[str, list[TestCaseResult]] = defaultdict(list)
        for r in results:
            key = getattr(r, field, "unknown")
            groups[key].append(r)

        aggregated = {}
        for group_key, group_results in groups.items():
            group_scores = {}
            for metric_name in metric_names:
                scores = [r.metric_scores.get(metric_name, 0) for r in group_results]
                group_scores[metric_name] = sum(scores) / len(scores) if scores else 0
            group_scores["pass_rate"] = (
                sum(1 for r in group_results if r.passed) / len(group_results)
                if group_results
                else 0
            )
            aggregated[group_key] = group_scores

        return aggregated

    def _result_to_dict(self, result: TestCaseResult) -> dict[str, Any]:
        """Convert TestCaseResult to dictionary."""
        return {
            "test_case_id": result.test_case_id,
            "query": result.query,
            "category": result.category,
            "difficulty": result.difficulty,
            "agent_type": result.agent_type,
            "passed": result.passed,
            "overall_score": result.overall_score,
            "metric_scores": result.metric_scores,
            "metric_reasons": result.metric_reasons,
            "target_metrics": result.target_metrics,
            "execution_time_ms": result.execution_time_ms,
        }


def print_report(report: EvaluationReport) -> None:
    """Print a formatted evaluation report to console."""
    print("\n" + "=" * 60)
    print("RESEARCHCREW EVALUATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Test Cases: {report.total_test_cases}")
    print(f"Passed: {report.passed_count} | Failed: {report.failed_count}")
    print(f"Overall Pass Rate: {report.overall_pass_rate:.1%}")

    print("\n" + "-" * 40)
    print("AVERAGE SCORES")
    print("-" * 40)
    for metric, score in report.average_scores.items():
        print(f"  {metric}: {score:.2f}")

    print("\n" + "-" * 40)
    print("SCORES BY DIFFICULTY")
    print("-" * 40)
    for difficulty, scores in report.scores_by_difficulty.items():
        print(f"\n  {difficulty.upper()}:")
        for metric, score in scores.items():
            if metric == "pass_rate":
                print(f"    Pass Rate: {score:.1%}")
            else:
                print(f"    {metric}: {score:.2f}")

    print("\n" + "-" * 40)
    print("SCORES BY CATEGORY")
    print("-" * 40)
    for category, scores in report.scores_by_category.items():
        print(f"\n  {category}:")
        print(f"    Pass Rate: {scores.get('pass_rate', 0):.1%}")

    print("\n" + "-" * 40)
    print("INDIVIDUAL RESULTS")
    print("-" * 40)
    for result in report.test_case_results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n  [{status}] {result['test_case_id']}: {result['query'][:50]}...")
        print(f"    Overall: {result['overall_score']:.2f}")
        for metric, score in result["metric_scores"].items():
            target = result["target_metrics"].get(metric.lower().replace(" ", "_"), "N/A")
            target_str = f" (target: {target})" if isinstance(target, float) else ""
            print(f"    {metric}: {score:.2f}{target_str}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for the eval runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluations on ResearchCrew golden dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evals/run_evals.py                         # Run all evaluations
  python evals/run_evals.py --category research     # Filter by category
  python evals/run_evals.py --difficulty hard       # Filter by difficulty
  python evals/run_evals.py --output report.json    # Save report to file
  python evals/run_evals.py --dry-run               # Show test cases without running
        """,
    )
    parser.add_argument("--category", help="Filter by category (e.g., research, comparison)")
    parser.add_argument("--difficulty", help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--agent-type", help="Filter by agent type (researcher, synthesizer)")
    parser.add_argument("--test-ids", nargs="+", help="Run specific test case IDs")
    parser.add_argument("--output", "-o", help="Output file for JSON report")
    parser.add_argument("--dry-run", action="store_true", help="Show filtered test cases only")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Initialize runner
    runner = EvalRunner()

    if args.dry_run:
        # Show filtered test cases without running
        filtered = runner.filter_test_cases(
            category=args.category,
            difficulty=args.difficulty,
            agent_type=args.agent_type,
            test_ids=args.test_ids,
        )
        print(f"\nFound {len(filtered)} test cases matching filters:\n")
        for tc in filtered:
            print(f"  {tc['id']}: [{tc['difficulty']}] [{tc['category']}] {tc['query'][:60]}...")
        return

    # Run evaluation
    report = runner.run(
        category=args.category,
        difficulty=args.difficulty,
        agent_type=args.agent_type,
        test_ids=args.test_ids,
    )

    # Print report
    if not args.quiet:
        print_report(report)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "total_test_cases": report.total_test_cases,
                    "passed_count": report.passed_count,
                    "failed_count": report.failed_count,
                    "overall_pass_rate": report.overall_pass_rate,
                    "average_scores": report.average_scores,
                    "scores_by_category": report.scores_by_category,
                    "scores_by_difficulty": report.scores_by_difficulty,
                    "scores_by_agent_type": report.scores_by_agent_type,
                    "test_case_results": report.test_case_results,
                    "filters_applied": report.filters_applied,
                },
                f,
                indent=2,
            )
        print(f"\nReport saved to {output_path}")

    # Exit with appropriate code
    sys.exit(0 if report.overall_pass_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
