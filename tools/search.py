"""Search Tools

Web search and URL reading tools for research agents.
"""

import httpx
from bs4 import BeautifulSoup
from google.adk import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query to execute.

    Returns:
        Search results with titles, snippets, and URLs.
    """
    # TODO: Implement with Google Custom Search API or similar
    # For now, return a placeholder that indicates implementation needed
    return f"""[Web Search Placeholder]
Query: {query}

To implement this tool:
1. Set up Google Custom Search API credentials
2. Add GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID to .env
3. Replace this placeholder with actual search implementation

Example implementation:
    import os
    from googleapiclient.discovery import build

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=query, cx=engine_id, num=5).execute()
    return format_results(result.get("items", []))
"""


@tool
def read_url(url: str) -> str:
    """Fetch and extract content from a URL.

    Args:
        url: The URL to fetch and read.

    Returns:
        Extracted text content from the page.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchCrew/1.0; +research-assistant)"
        }
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
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated...]"

        return f"Content from {url}:\n\n{text}"

    except httpx.HTTPError as e:
        return f"Error fetching URL {url}: {e}"
    except Exception as e:
        return f"Error processing URL {url}: {e}"
