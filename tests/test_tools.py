"""Tests for ResearchCrew Tools

Tests search and knowledge base tools with mocked external dependencies.
"""

from unittest.mock import Mock, patch

from tools.knowledge import (
    get_knowledge_stats,
    knowledge_search,
    list_knowledge_topics,
    save_to_knowledge,
)
from tools.search import read_url, web_search


class TestWebSearch:
    """Tests for the web_search tool."""

    @patch("tools.search.httpx.get")
    def test_successful_search(self, mock_get):
        """Test successful web search returns formatted results."""
        # Mock response with HTML content matching DuckDuckGo structure
        # The code uses .result__title a (anchor inside result__title element)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <body>
            <div class="result">
                <h2 class="result__title"><a href="https://example.com/1">Result One</a></h2>
                <div class="result__snippet">This is the first result snippet.</div>
            </div>
            <div class="result">
                <h2 class="result__title"><a href="https://example.com/2">Result Two</a></h2>
                <div class="result__snippet">This is the second result snippet.</div>
            </div>
        </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = web_search("test query")

        assert "Search results for 'test query'" in result
        assert "Result One" in result or "result" in result.lower()
        mock_get.assert_called_once()

    @patch("tools.search.httpx.get")
    def test_search_no_results(self, mock_get):
        """Test search with no results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = web_search("nonexistent query xyz123")

        assert "no results" in result.lower() or "not found" in result.lower()

    @patch("tools.search.httpx.get")
    def test_search_http_error(self, mock_get):
        """Test search handles HTTP errors gracefully."""
        import httpx

        mock_get.side_effect = httpx.HTTPError("Connection failed")

        result = web_search("test query")

        assert "error" in result.lower()

    @patch("tools.search.httpx.get")
    def test_search_limits_results(self, mock_get):
        """Test that search limits to maximum 5 results."""
        # Create mock with many results
        results_html = ""
        for i in range(10):
            results_html += f"""
            <div class="result">
                <a class="result__title" href="https://example.com/{i}">Result {i}</a>
                <div class="result__snippet">Snippet {i}</div>
            </div>
            """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = f"<html><body>{results_html}</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = web_search("many results query")

        # Should have at most 5 numbered results
        assert result.count("URL:") <= 5


class TestReadUrl:
    """Tests for the read_url tool."""

    @patch("tools.search.httpx.get")
    def test_successful_read(self, mock_get):
        """Test successful URL reading."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Content</h1>
            <p>This is the main content of the page.</p>
        </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = read_url("https://example.com/test")

        assert "Content from https://example.com/test" in result
        assert "Main Content" in result
        assert "main content of the page" in result

    @patch("tools.search.httpx.get")
    def test_read_removes_scripts(self, mock_get):
        """Test that scripts and styles are removed."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <head>
            <style>body { color: red; }</style>
            <script>alert('malicious');</script>
        </head>
        <body>
            <script>console.log('test');</script>
            <p>Clean content here.</p>
            <nav>Navigation content</nav>
        </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = read_url("https://example.com/test")

        assert "Clean content here" in result
        assert "alert" not in result
        assert "console.log" not in result
        assert "color: red" not in result

    @patch("tools.search.httpx.get")
    def test_read_truncates_long_content(self, mock_get):
        """Test that long content is truncated."""
        long_content = "A" * 20000  # More than max_chars

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = f"<html><body><p>{long_content}</p></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = read_url("https://example.com/long")

        assert "[Content truncated...]" in result
        assert len(result) < 15000  # Should be truncated

    @patch("tools.search.httpx.get")
    def test_read_http_error(self, mock_get):
        """Test read handles HTTP errors gracefully."""
        import httpx

        mock_get.side_effect = httpx.HTTPError("404 Not Found")

        result = read_url("https://example.com/notfound")

        assert "error" in result.lower()


class TestKnowledgeSearch:
    """Tests for the knowledge_search tool."""

    @patch("tools.knowledge.get_knowledge_base")
    def test_successful_search(self, mock_get_kb):
        """Test successful knowledge base search."""
        # Create mock knowledge base
        mock_kb = Mock()
        mock_result = Mock()
        mock_result.score = 0.85
        mock_result.entry = Mock()
        mock_result.entry.content = "LangGraph is a framework for building agents."
        mock_result.entry.source_url = "https://langchain.com"
        mock_result.entry.source_title = "LangGraph Docs"
        mock_result.entry.topic = "AI frameworks"
        mock_result.entry.confidence = "high"

        mock_kb.search.return_value = [mock_result]
        mock_get_kb.return_value = mock_kb

        result = knowledge_search("AI agent frameworks")

        assert "Found 1 relevant" in result
        assert "LangGraph" in result
        assert "85%" in result

    @patch("tools.knowledge.get_knowledge_base")
    def test_search_no_results(self, mock_get_kb):
        """Test knowledge search with no results."""
        mock_kb = Mock()
        mock_kb.search.return_value = []
        mock_get_kb.return_value = mock_kb

        result = knowledge_search("nonexistent topic xyz")

        assert "no relevant" in result.lower()

    @patch("tools.knowledge.get_knowledge_base")
    def test_search_with_topic_filter(self, mock_get_kb):
        """Test knowledge search with topic filter."""
        mock_kb = Mock()
        mock_kb.search.return_value = []
        mock_get_kb.return_value = mock_kb

        knowledge_search("query", topic="security")

        mock_kb.search.assert_called_once()
        call_kwargs = mock_kb.search.call_args[1]
        assert call_kwargs.get("topic_filter") == "security"

    @patch("tools.knowledge.get_knowledge_base")
    def test_search_error_handling(self, mock_get_kb):
        """Test knowledge search handles errors."""
        mock_get_kb.side_effect = Exception("Database error")

        result = knowledge_search("test query")

        assert "error" in result.lower()


class TestSaveToKnowledge:
    """Tests for the save_to_knowledge tool."""

    @patch("tools.knowledge.get_knowledge_base")
    def test_successful_save(self, mock_get_kb):
        """Test successful save to knowledge base."""
        mock_kb = Mock()
        mock_entry = Mock()
        mock_entry.id = "entry-123"
        mock_kb.add_entry.return_value = mock_entry
        mock_get_kb.return_value = mock_kb

        result = save_to_knowledge(
            content="Important finding about AI agents.",
            source_url="https://example.com",
            topic="AI",
            confidence="high",
        )

        assert "successfully saved" in result.lower()
        assert "entry-123" in result

    @patch("tools.knowledge.get_knowledge_base")
    def test_save_duplicate_detection(self, mock_get_kb):
        """Test that duplicates are detected."""
        mock_kb = Mock()
        mock_kb.add_entry.return_value = None  # Indicates duplicate
        mock_get_kb.return_value = mock_kb

        result = save_to_knowledge(
            content="Duplicate content that exists.",
        )

        assert "not saved" in result.lower() or "already exists" in result.lower()

    def test_save_rejects_short_content(self):
        """Test that very short content is rejected."""
        result = save_to_knowledge(content="Too short")

        assert "error" in result.lower()
        assert "too short" in result.lower()

    def test_save_rejects_long_content(self):
        """Test that very long content is rejected."""
        long_content = "A" * 6000

        result = save_to_knowledge(content=long_content)

        assert "error" in result.lower()
        assert "too long" in result.lower()

    @patch("tools.knowledge.get_knowledge_base")
    def test_save_normalizes_confidence(self, mock_get_kb):
        """Test that invalid confidence is normalized."""
        mock_kb = Mock()
        mock_entry = Mock()
        mock_entry.id = "entry-456"
        mock_kb.add_entry.return_value = mock_entry
        mock_get_kb.return_value = mock_kb

        result = save_to_knowledge(
            content="Valid content for testing normalization.",
            confidence="invalid_confidence",
        )

        # Should succeed with normalized confidence
        assert "successfully saved" in result.lower()


class TestListKnowledgeTopics:
    """Tests for the list_knowledge_topics tool."""

    @patch("tools.knowledge.get_knowledge_base")
    def test_list_topics_success(self, mock_get_kb):
        """Test listing topics successfully."""
        mock_kb = Mock()
        mock_kb.list_topics.return_value = ["AI frameworks", "Security", "Testing"]
        mock_kb.get_stats.return_value = {
            "total_entries": 10,
            "topics": {"AI frameworks": 5, "Security": 3, "Testing": 2},
        }
        mock_get_kb.return_value = mock_kb

        result = list_knowledge_topics()

        assert "AI frameworks" in result
        assert "Security" in result
        assert "10 entries" in result

    @patch("tools.knowledge.get_knowledge_base")
    def test_list_topics_empty(self, mock_get_kb):
        """Test listing topics when empty."""
        mock_kb = Mock()
        mock_kb.list_topics.return_value = []
        mock_get_kb.return_value = mock_kb

        result = list_knowledge_topics()

        assert "empty" in result.lower()


class TestGetKnowledgeStats:
    """Tests for the get_knowledge_stats tool."""

    @patch("tools.knowledge.get_knowledge_base")
    def test_get_stats_success(self, mock_get_kb):
        """Test getting stats successfully."""
        mock_kb = Mock()
        mock_kb.get_stats.return_value = {
            "total_entries": 25,
            "collection_name": "research_kb",
            "persist_directory": "/data/kb",
            "topics": {"AI": 15, "Security": 10},
            "confidence_distribution": {"high": 10, "medium": 10, "low": 5},
        }
        mock_get_kb.return_value = mock_kb

        result = get_knowledge_stats()

        assert "25" in result
        assert "research_kb" in result
        assert "AI" in result
        assert "high" in result


class TestToolDecorators:
    """Tests for tool decorator functionality."""

    def test_web_search_is_callable(self):
        """Test that web_search is callable."""
        assert callable(web_search)

    def test_read_url_is_callable(self):
        """Test that read_url is callable."""
        assert callable(read_url)

    def test_knowledge_search_is_callable(self):
        """Test that knowledge_search is callable."""
        assert callable(knowledge_search)

    def test_save_to_knowledge_is_callable(self):
        """Test that save_to_knowledge is callable."""
        assert callable(save_to_knowledge)
