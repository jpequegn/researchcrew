# Testing Guide

This guide covers testing practices and tools for ResearchCrew development.

## Overview

ResearchCrew uses a comprehensive testing strategy with three levels:

| Level | Purpose | Tools | Location |
|-------|---------|-------|----------|
| **Unit Tests** | Test individual functions/classes | pytest | `tests/test_*.py` |
| **Integration Tests** | Test agent/tool interactions | pytest + ADK | `tests/test_integration*.py` |
| **Evaluations** | Test output quality | DeepEval | `evals/` |

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=agents --cov=tools --cov=utils --cov-report=html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/test_agents.py tests/test_tools.py

# Integration tests only
pytest tests/test_integration*.py

# Tests by marker
pytest -m "not slow"
pytest -m "integration"

# Single test file
pytest tests/test_agents.py -v

# Single test function
pytest tests/test_agents.py::test_orchestrator_decomposition -v
```

### Watch Mode

```bash
# Rerun tests on file changes
pytest-watch

# Or with specific path
ptw tests/ -- -v
```

## Test Structure

### Directory Layout

```
tests/
├── conftest.py              # Shared fixtures
├── test_agents.py           # Agent unit tests
├── test_tools.py            # Tool unit tests
├── test_schemas.py          # Schema validation tests
├── test_utils.py            # Utility function tests
├── test_integration.py      # Component integration tests
├── test_session.py          # Session management tests
├── test_resilience.py       # Resilience pattern tests
├── test_quality_verification.py    # Quality system tests
└── test_deployment_verification.py # Deployment tests
```

### Test Naming Conventions

```python
# Pattern: test_<unit>_<behavior>
def test_researcher_returns_findings():
    """Tests should have descriptive names."""
    pass

def test_orchestrator_handles_empty_query():
    """Include edge case in name."""
    pass

def test_circuit_breaker_opens_after_failures():
    """Describe expected behavior."""
    pass
```

## Writing Tests

### Unit Test Example

```python
import pytest
from agents.researcher import ResearcherAgent
from schemas.research import Finding

class TestResearcherAgent:
    """Tests for the Researcher agent."""

    @pytest.fixture
    def researcher(self):
        """Create a researcher agent for testing."""
        return ResearcherAgent()

    def test_processes_search_results(self, researcher):
        """Researcher should convert search results to findings."""
        search_results = [
            {"title": "AI News", "url": "https://example.com", "snippet": "Latest AI..."}
        ]

        findings = researcher.process_results(search_results)

        assert len(findings) == 1
        assert isinstance(findings[0], Finding)
        assert findings[0].source == "https://example.com"

    def test_handles_empty_results(self, researcher):
        """Researcher should handle empty search results gracefully."""
        findings = researcher.process_results([])

        assert findings == []

    def test_validates_url_format(self, researcher):
        """Researcher should validate URLs in results."""
        with pytest.raises(ValueError):
            researcher.process_results([{"url": "not-a-url"}])
```

### Integration Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch
from runner import ResearchCrewRunner

class TestResearchWorkflow:
    """Integration tests for the research workflow."""

    @pytest.fixture
    def runner(self):
        """Create a runner with mocked external services."""
        return ResearchCrewRunner()

    @pytest.fixture
    def mock_search(self):
        """Mock web search responses."""
        with patch("tools.search.web_search") as mock:
            mock.return_value = [
                {"title": "Test", "url": "https://test.com", "snippet": "Test content"}
            ]
            yield mock

    @pytest.mark.asyncio
    async def test_full_research_workflow(self, runner, mock_search):
        """Test complete research workflow from query to report."""
        result = await runner.run_async("Test query")

        assert result["status"] == "completed"
        assert "result" in result
        mock_search.assert_called()

    @pytest.mark.asyncio
    async def test_session_continuity(self, runner, mock_search):
        """Test that follow-up queries maintain context."""
        session_id = runner.create_session()

        result1 = await runner.run_async("Research AI", session_id=session_id)
        result2 = await runner.run_async("Tell me more", session_id=session_id)

        assert result2["turn_number"] == 2
        assert result2["session_id"] == session_id
```

### Async Test Example

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_tool_execution():
    """Test async tool execution."""
    from tools.web_fetch import read_url

    result = await read_url("https://example.com")

    assert "content" in result
    assert len(result["content"]) > 0

@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Test parallel tool calls."""
    from tools.search import web_search

    queries = ["AI", "ML", "NLP"]
    tasks = [web_search(q) for q in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(r is not None for r in results)
```

## Fixtures

### Common Fixtures (conftest.py)

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_knowledge_base(tmp_path):
    """Create a temporary knowledge base directory."""
    kb_path = tmp_path / "knowledge_base"
    kb_path.mkdir()
    return kb_path

@pytest.fixture
def sample_findings():
    """Sample findings for testing."""
    return [
        {
            "content": "AI is transforming industries",
            "source": "https://example.com/ai",
            "confidence": 0.85
        },
        {
            "content": "Machine learning requires data",
            "source": "https://example.com/ml",
            "confidence": 0.90
        }
    ]

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a test response from the LLM.",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }

@pytest.fixture(scope="session")
def golden_dataset():
    """Load the golden evaluation dataset."""
    dataset_path = Path(__file__).parent.parent / "evals" / "golden_dataset.jsonl"
    if not dataset_path.exists():
        pytest.skip("Golden dataset not found")
    # Load and return dataset
    import json
    with open(dataset_path) as f:
        return [json.loads(line) for line in f]
```

### Fixture Scopes

| Scope | Lifecycle | Use Case |
|-------|-----------|----------|
| `function` | Per test function | Most tests (default) |
| `class` | Per test class | Shared setup within class |
| `module` | Per module | Expensive setup shared across file |
| `session` | Entire test run | Very expensive setup (DB connections) |

## Mocking

### Mocking External Services

```python
from unittest.mock import patch, MagicMock, AsyncMock

def test_with_mocked_api():
    """Test with mocked external API."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": "test"}
        )

        # Your test here
        pass

@pytest.fixture
def mock_gemini():
    """Mock Gemini API responses."""
    with patch("google.generativeai.GenerativeModel") as mock:
        mock_instance = MagicMock()
        mock_instance.generate_content_async = AsyncMock(
            return_value=MagicMock(text="Generated response")
        )
        mock.return_value = mock_instance
        yield mock
```

### Mocking File Operations

```python
def test_config_loading(tmp_path):
    """Test configuration file loading."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    environment: test
    model:
      name: gemini-2.0-flash
    """)

    from utils.config import load_config
    config = load_config(str(config_file))

    assert config["environment"] == "test"
```

## Evaluations

### Running Evaluations

```bash
# Run all evaluations
python evals/run_evals.py

# Run specific difficulty
python evals/run_evals.py --difficulty easy

# Run specific category
python evals/run_evals.py --category research

# Generate baseline report
python evals/run_evals.py --save-baseline
```

### Evaluation Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Factual Accuracy | ≥ 85% | Claims supported by sources |
| Source Quality | ≥ 80% | Credible, relevant sources |
| Completeness | ≥ 75% | Coverage of expected topics |
| Coherence | ≥ 80% | Logical structure and flow |

### Writing Evaluation Test Cases

```jsonl
{"id": "test-001", "query": "What is machine learning?", "expected_topics": ["algorithms", "data", "training"], "difficulty": "easy", "category": "research", "agent_type": "researcher"}
{"id": "test-002", "query": "Compare supervised vs unsupervised learning", "expected_topics": ["labeled data", "clustering", "classification"], "difficulty": "medium", "category": "comparison", "agent_type": "synthesizer"}
```

### Custom Evaluation Metrics

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    """Custom evaluation metric."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """Compute the metric score."""
        # Your scoring logic here
        score = 0.85
        self.score = score
        return score

    def is_successful(self) -> bool:
        """Check if metric passes threshold."""
        return self.score >= self.threshold
```

## Test Markers

### Available Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_large_dataset():
    pass

# Mark integration tests
@pytest.mark.integration
def test_full_workflow():
    pass

# Mark tests requiring network
@pytest.mark.network
def test_external_api():
    pass

# Skip under certain conditions
@pytest.mark.skipif(condition, reason="Reason")
def test_conditional():
    pass
```

### Running by Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# Combine markers
pytest -m "integration and not network"
```

## Coverage

### Generating Coverage Reports

```bash
# HTML report
pytest --cov=agents --cov=tools --cov=utils --cov-report=html
# View at htmlcov/index.html

# Terminal report
pytest --cov=agents --cov-report=term-missing

# XML for CI
pytest --cov=agents --cov-report=xml
```

### Coverage Requirements

The project requires minimum 60% coverage (configured in `pyproject.toml`):

```toml
[tool.coverage.report]
fail_under = 60
```

### Excluding from Coverage

```python
# Exclude specific lines
if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias

# Exclude entire functions
def debug_only():  # pragma: no cover
    """Only used in debugging."""
    pass
```

## CI Integration

### GitHub Actions Test Job

Tests run automatically on push and pull requests via `.github/workflows/ci.yml`:

```yaml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: pip install -e ".[dev]"
    - run: pytest --cov --cov-report=xml
    - uses: codecov/codecov-action@v4
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest
        entry: pytest tests/ -v --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

## Best Practices

### Do

- Write tests before or alongside code
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies
- Keep tests independent
- Use fixtures for common setup
- Maintain test coverage above threshold

### Don't

- Test implementation details
- Write tests that depend on order
- Use real API keys in tests
- Leave flaky tests in the suite
- Skip tests without documentation
- Write overly complex test setups

## Next Steps

- [Development Setup](./setup.md) - Environment setup
- [Contributing Guide](./contributing.md) - Contribution process
- [Debugging Guide](../debugging-runbook.md) - Troubleshooting
