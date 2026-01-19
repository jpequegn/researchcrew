# Contributing Guide

Thank you for your interest in contributing to ResearchCrew! This guide explains our development process and conventions.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Set up development environment (see [Development Setup](./setup.md))
4. Create a feature branch
5. Make your changes
6. Submit a pull request

## Development Workflow

### Branch Naming

Use descriptive branch names:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<description>` | `feature/add-pdf-tool` |
| Bug fix | `fix/<description>` | `fix/search-timeout` |
| Docs | `docs/<description>` | `docs/api-reference` |
| Refactor | `refactor/<description>` | `refactor/agent-base-class` |

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add PDF extraction tool to researcher

fix(search): handle rate limit errors gracefully

docs(api): add session management examples

test(evals): add edge case for empty queries
```

### Pull Request Process

1. **Create PR** with descriptive title and body
2. **Link issue** if applicable (e.g., "Closes #123")
3. **Ensure CI passes** - all tests must pass
4. **Request review** from maintainers
5. **Address feedback** with additional commits
6. **Squash and merge** when approved

### PR Template

```markdown
## Summary
Brief description of changes

## Changes
- Added X
- Modified Y
- Fixed Z

## Test Plan
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Related Issues
Closes #123
```

## Code Style

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Line length: 100 characters
# Use double quotes for strings
# Use type hints for function signatures

def process_findings(
    findings: list[Finding],
    min_confidence: float = 0.7,
) -> ProcessedResult:
    """Process research findings with confidence filtering.

    Args:
        findings: List of findings to process
        min_confidence: Minimum confidence threshold (default: 0.7)

    Returns:
        ProcessedResult containing filtered and ranked findings

    Raises:
        ValueError: If findings list is empty
    """
    if not findings:
        raise ValueError("Findings list cannot be empty")

    filtered = [f for f in findings if f.confidence >= min_confidence]
    return ProcessedResult(findings=filtered)
```

### Ruff Configuration

Linting is enforced via Ruff:

```bash
# Check for issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

### Import Order

```python
# Standard library
import json
from pathlib import Path

# Third-party
import httpx
from pydantic import BaseModel

# Local
from agents.base import BaseAgent
from schemas.research import Finding
```

### Type Hints

Use type hints for all public functions:

```python
# Good
def search(query: str, limit: int = 5) -> list[dict[str, str]]:
    ...

# Also good - complex types
from typing import TypeAlias

FindingList: TypeAlias = list[dict[str, str | float]]

def process(findings: FindingList) -> FindingList:
    ...
```

## Documentation

### Code Documentation

All public modules, classes, and functions need docstrings:

```python
"""Module for research synthesis operations.

This module provides the Synthesizer agent and related utilities
for combining research findings from multiple sources.
"""

class Synthesizer:
    """Combines findings from multiple researchers into coherent insights.

    The Synthesizer identifies common themes, resolves conflicts between
    sources, and ranks findings by confidence.

    Attributes:
        model: The LLM model used for synthesis
        max_findings: Maximum findings to process per synthesis

    Example:
        >>> synth = Synthesizer()
        >>> result = synth.combine(findings)
        >>> print(result.themes)
    """

    def combine(self, findings: list[Finding]) -> SynthesisResult:
        """Combine multiple findings into a synthesis.

        Args:
            findings: List of findings to synthesize

        Returns:
            SynthesisResult with identified themes and ranked findings

        Raises:
            SynthesisError: If synthesis fails after retries
        """
        pass
```

### Documentation Files

When adding features, update relevant docs:

| Change Type | Update |
|-------------|--------|
| New agent | `AGENTS.md`, `docs/architecture.md` |
| New tool | `docs/api.md`, agent tool permissions |
| New config option | `docs/development/setup.md` |
| New API endpoint | `docs/api.md` |
| Infrastructure | `docs/infrastructure/*.md` |

## Testing Requirements

### Test Coverage

- All new code must have tests
- Maintain minimum 60% coverage
- Critical paths require higher coverage

### Test Structure

```python
class TestNewFeature:
    """Tests for the new feature."""

    @pytest.fixture
    def feature(self):
        """Set up feature for testing."""
        return NewFeature()

    def test_basic_functionality(self, feature):
        """Test the happy path."""
        result = feature.do_thing()
        assert result.success

    def test_handles_edge_case(self, feature):
        """Test edge case handling."""
        result = feature.do_thing(edge_case=True)
        assert result.handled_gracefully

    def test_raises_on_invalid_input(self, feature):
        """Test error handling."""
        with pytest.raises(ValueError):
            feature.do_thing(invalid=True)
```

### Running Tests Before PR

```bash
# Full test suite
pytest

# With coverage
pytest --cov

# Linting
ruff check .

# Type checking
mypy agents tools utils
```

## Adding New Features

### Adding a New Agent

1. **Define in AGENTS.md** - Document capabilities and constraints
2. **Create implementation** in `agents/`:
   ```python
   # agents/my_agent.py
   from google.adk import Agent

   class MyAgent(Agent):
       name = "my_agent"
       description = "Agent description"
       tools = ["allowed_tool"]
   ```
3. **Add tests** in `tests/test_agents.py`
4. **Update orchestrator** if it needs to delegate to this agent
5. **Add evaluation cases** to golden dataset

### Adding a New Tool

1. **Create implementation** in `tools/`:
   ```python
   # tools/my_tool.py
   from google.adk import tool

   @tool
   def my_tool(param: str) -> dict:
       """Tool description."""
       return {"result": "..."}
   ```
2. **Add to agent permissions** in `AGENTS.md`
3. **Add tests** in `tests/test_tools.py`
4. **Document** in `docs/api.md`
5. **Add to MCP server** if applicable

### Adding Configuration Options

1. **Add to schema** in `schemas/config.py`
2. **Update YAML files** (`config/dev.yaml`, etc.)
3. **Document** in `docs/development/setup.md`
4. **Add validation tests**

## Review Checklist

Before submitting PR, verify:

- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description is complete
- [ ] Changes are focused and minimal

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email security@example.com (do not open public issue)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project conventions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Next Steps

- [Development Setup](./setup.md) - Set up your environment
- [Testing Guide](./testing.md) - Learn testing practices
- [Architecture](../architecture.md) - Understand the system
