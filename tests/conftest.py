"""Pytest configuration and fixtures for ResearchCrew tests.

Provides mocks for external dependencies like google.adk.
"""

import sys
from unittest.mock import MagicMock

# Mock only google.adk, not the entire google package
# This preserves google.protobuf needed by chromadb
mock_adk = MagicMock()
mock_adk.Agent = MagicMock()
mock_adk.tool = lambda func: func  # Pass-through decorator

sys.modules["google.adk"] = mock_adk


# Now we can safely import the project modules
import pytest

from utils.context_manager import reset_context_manager
from utils.session_manager import reset_session_manager


@pytest.fixture(autouse=True)
def reset_managers():
    """Reset session and context managers before each test."""
    reset_session_manager()
    reset_context_manager()
    yield
    reset_session_manager()
    reset_context_manager()


@pytest.fixture
def mock_httpx_response():
    """Fixture for mocking httpx responses."""

    def _make_response(text: str, status_code: int = 200):
        mock = MagicMock()
        mock.text = text
        mock.status_code = status_code
        mock.raise_for_status = MagicMock()
        return mock

    return _make_response


@pytest.fixture
def sample_search_html():
    """Sample HTML for search results."""
    return """
    <html>
    <body>
        <div class="result">
            <a class="result__title" href="https://example.com/1">Result One</a>
            <div class="result__snippet">First result snippet.</div>
        </div>
        <div class="result">
            <a class="result__title" href="https://example.com/2">Result Two</a>
            <div class="result__snippet">Second result snippet.</div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_page_html():
    """Sample HTML for page content."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Main Content</h1>
        <p>This is the main content of the page with useful information.</p>
        <p>Additional paragraph with more details.</p>
    </body>
    </html>
    """
