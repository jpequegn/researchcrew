"""Search Tools

Web search and URL reading tools for research agents.
"""

from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from google.adk import tool

from utils.tracing import get_trace_id, trace_span


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query to execute.

    Returns:
        Search results with titles, snippets, and URLs.
    """
    with trace_span(
        "tool.web_search",
        attributes={
            "tool.name": "web_search",
            "tool.query": query[:500],
            "trace.id": get_trace_id(),
        },
    ) as span:
        try:
            # Use DuckDuckGo HTML search (no API key required)
            encoded_query = quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

            response = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            # DuckDuckGo HTML results are in divs with class "result"
            for i, result_div in enumerate(soup.select(".result"), 1):
                if i > 5:  # Limit to 5 results
                    break

                title_elem = result_div.select_one(".result__title a")
                snippet_elem = result_div.select_one(".result__snippet")

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else "No description"

                    results.append(f"{i}. **{title}**\n   URL: {href}\n   {snippet}")

            span.set_attribute("tool.result_count", len(results))

            if results:
                return f"Search results for '{query}':\n\n" + "\n\n".join(results)
            else:
                return f"No results found for '{query}'. Try a different search term."

        except httpx.HTTPError as e:
            span.set_attribute("tool.error", str(e))
            return f"Search error: {e}. Please try again."
        except Exception as e:
            span.set_attribute("tool.error", str(e))
            return f"Unexpected error during search: {e}"


@tool
def read_url(url: str) -> str:
    """Fetch and extract content from a URL.

    Args:
        url: The URL to fetch and read.

    Returns:
        Extracted text content from the page.
    """
    with trace_span(
        "tool.read_url",
        attributes={
            "tool.name": "read_url",
            "tool.url": url[:500],
            "trace.id": get_trace_id(),
        },
    ) as span:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchCrew/1.0; +research-assistant)"}
            response = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Limit content length
            max_chars = 10000
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars] + "\n\n[Content truncated...]"

            span.set_attribute("tool.content_length", len(text))
            span.set_attribute("tool.truncated", truncated)

            return f"Content from {url}:\n\n{text}"

        except httpx.HTTPError as e:
            span.set_attribute("tool.error", str(e))
            return f"Error fetching URL {url}: {e}"
        except Exception as e:
            span.set_attribute("tool.error", str(e))
            return f"Error processing URL {url}: {e}"
