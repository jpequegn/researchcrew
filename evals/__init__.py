"""ResearchCrew Evaluation Module

Provides automated evaluation capabilities for research agent outputs.
"""

from evals.metrics import (
    CompletenessMetric,
    CoherenceMetric,
    FactualAccuracyMetric,
    ResearchQualityMetric,
    SourceQualityMetric,
    create_metrics_for_test_case,
    load_golden_dataset,
)
from evals.run_evals import EvalRunner, EvaluationReport, TestCaseResult

__all__ = [
    "SourceQualityMetric",
    "CompletenessMetric",
    "CoherenceMetric",
    "FactualAccuracyMetric",
    "ResearchQualityMetric",
    "create_metrics_for_test_case",
    "load_golden_dataset",
    "EvalRunner",
    "EvaluationReport",
    "TestCaseResult",
]
