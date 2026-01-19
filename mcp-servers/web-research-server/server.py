#!/usr/bin/env python3
"""Web Research MCP Server

An MCP server that provides web research tools for searching the web
and reading URL content.

Tools:
- web_search: Search the web for information
- read_url: Fetch and extract content from URLs

Usage:
    # Run the server
    python server.py

    # With API key
    SERPER_API_KEY=xxx python server.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import (
    MCPError,
    ToolError,
    ValidationError,
    get_logger,
    setup_logging,
    truncate_text,
    clean_text,
    validate_optional_param,
    validate_required_param,
)

# Set up logging
logger = setup_logging("web-research-server")

# ============================================================================
# Web Search Implementation
# ============================================================================

# Check for optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, web search will be limited")

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not available, URL reading will be limited")


async def search_web(
    query: str,
    num_results: int = 10,
    search_type: str = "search",
) -> dict[str, Any]:
    """Search the web using Serper API or fallback.

    Args:
        query: Search query.
        num_results: Number of results to return.
        search_type: Type of search (search, news, images).

    Returns:
        Search results dictionary.
    """
    api_key = os.environ.get("SERPER_API_KEY")

    if api_key and HTTPX_AVAILABLE:
        return await _search_serper(query, num_results, search_type, api_key)
    else:
        # Return a mock response for development/testing
        logger.warning("No SERPER_API_KEY set, returning mock results")
        return _mock_search_results(query)


async def _search_serper(
    query: str,
    num_results: int,
    search_type: str,
    api_key: str,
) -> dict[str, Any]:
    """Search using Serper API."""
    endpoint = f"https://google.serper.dev/{search_type}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            endpoint,
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={
                "q": query,
                "num": num_results,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    # Format results
    results = []
    for item in data.get("organic", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "position": item.get("position"),
            }
        )

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
        "knowledge_graph": data.get("knowledgeGraph"),
        "answer_box": data.get("answerBox"),
    }


def _mock_search_results(query: str) -> dict[str, Any]:
    """Return mock search results for testing."""
    return {
        "query": query,
        "results": [
            {
                "title": f"Example Result for: {query}",
                "url": "https://example.com/1",
                "snippet": f"This is a mock result for the query: {query}. In production, this would be real search results from the Serper API.",
                "position": 1,
            },
            {
                "title": f"Another Result about {query}",
                "url": "https://example.com/2",
                "snippet": f"More information about {query}. Set SERPER_API_KEY to get real results.",
                "position": 2,
            },
        ],
        "result_count": 2,
        "mock": True,
    }


# ============================================================================
# URL Reading Implementation
# ============================================================================


async def read_url(
    url: str,
    max_length: int = 50000,
    extract_links: bool = False,
) -> dict[str, Any]:
    """Fetch and extract content from a URL.

    Args:
        url: URL to fetch.
        max_length: Maximum content length.
        extract_links: Whether to extract links from the page.

    Returns:
        Extracted content dictionary.
    """
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValidationError("Invalid URL format", field="url")

    if not HTTPX_AVAILABLE:
        return {
            "url": url,
            "error": "httpx not available",
            "content": "",
        }

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchCrew/1.0; +https://github.com/jpequegn/researchcrew)"
                },
                timeout=30.0,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "text/html" in content_type:
                return _extract_html_content(
                    url,
                    response.text,
                    max_length,
                    extract_links,
                )
            elif "application/json" in content_type:
                return {
                    "url": url,
                    "content_type": "json",
                    "content": truncate_text(response.text, max_length),
                }
            else:
                return {
                    "url": url,
                    "content_type": content_type,
                    "content": truncate_text(response.text, max_length),
                }

    except httpx.HTTPStatusError as e:
        return {
            "url": url,
            "error": f"HTTP {e.response.status_code}",
            "content": "",
        }
    except httpx.RequestError as e:
        return {
            "url": url,
            "error": str(e),
            "content": "",
        }


def _extract_html_content(
    url: str,
    html: str,
    max_length: int,
    extract_links: bool,
) -> dict[str, Any]:
    """Extract content from HTML."""
    if not BS4_AVAILABLE:
        # Simple regex-based extraction
        text = re.sub(r"<[^>]+>", " ", html)
        text = clean_text(text)
        return {
            "url": url,
            "content_type": "html",
            "content": truncate_text(text, max_length),
        }

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    # Extract title
    title = ""
    if soup.title:
        title = soup.title.get_text().strip()

    # Extract main content
    main_content = soup.find("main") or soup.find("article") or soup.body
    if main_content:
        text = main_content.get_text(separator=" ", strip=True)
    else:
        text = soup.get_text(separator=" ", strip=True)

    text = clean_text(text)

    result = {
        "url": url,
        "content_type": "html",
        "title": title,
        "content": truncate_text(text, max_length),
        "word_count": len(text.split()),
    }

    # Extract links if requested
    if extract_links:
        links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if href.startswith("http"):
                links.append(
                    {
                        "text": link.get_text().strip()[:100],
                        "url": href,
                    }
                )
        result["links"] = links[:20]  # Limit to 20 links

    return result


# ============================================================================
# MCP Server Implementation
# ============================================================================

# Tool definitions for MCP
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information on any topic. Returns search results with titles, URLs, and snippets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)",
                    "default": 10,
                },
                "search_type": {
                    "type": "string",
                    "description": "Type of search: 'search', 'news', or 'images'",
                    "enum": ["search", "news", "images"],
                    "default": "search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Fetch and extract content from a URL. Useful for reading articles, documentation, or any web page.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch and read",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length (default: 50000)",
                    "default": 50000,
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Whether to extract links from the page",
                    "default": False,
                },
            },
            "required": ["url"],
        },
    },
]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Handle a tool call.

    Args:
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        Tool result.

    Raises:
        ToolError: If tool execution fails.
    """
    try:
        if name == "web_search":
            query = validate_required_param(arguments, "query", str)
            num_results = validate_optional_param(
                arguments, "num_results", default=10, param_type=int
            )
            search_type = validate_optional_param(
                arguments, "search_type", default="search", param_type=str
            )

            result = await search_web(
                query=query,
                num_results=num_results,
                search_type=search_type,
            )
            return result

        elif name == "read_url":
            url = validate_required_param(arguments, "url", str)
            max_length = validate_optional_param(
                arguments, "max_length", default=50000, param_type=int
            )
            extract_links = validate_optional_param(
                arguments, "extract_links", default=False, param_type=bool
            )

            result = await read_url(
                url=url,
                max_length=max_length,
                extract_links=extract_links,
            )
            return result

        else:
            raise ToolError(
                message=f"Unknown tool: {name}",
                tool_name=name,
            )

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Tool '{name}' failed: {e}")
        raise ToolError(
            message=str(e),
            tool_name=name,
            original_error=e,
        )


# ============================================================================
# MCP Protocol Implementation (stdio)
# ============================================================================


async def handle_message(message: dict[str, Any]) -> dict[str, Any]:
    """Handle an MCP message.

    Args:
        message: The MCP message.

    Returns:
        Response message.
    """
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params", {})

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                    },
                    "serverInfo": {
                        "name": "web-research-server",
                        "version": "1.0.0",
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": TOOLS,
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            result = await handle_tool_call(tool_name, tool_args)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }
                    ],
                },
            }

        elif method == "notifications/initialized":
            # This is a notification, no response needed
            return None

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    except MCPError as e:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32000,
                "message": e.message,
                "data": e.to_dict(),
            },
        }
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": str(e),
            },
        }


async def run_server():
    """Run the MCP server using stdio transport."""
    logger.info("Starting Web Research MCP Server")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

    while True:
        try:
            # Read header
            header = await reader.readline()
            if not header:
                break

            # Parse content length
            if header.startswith(b"Content-Length:"):
                content_length = int(header.decode().split(":")[1].strip())

                # Read empty line
                await reader.readline()

                # Read content
                content = await reader.read(content_length)
                message = json.loads(content.decode())

                logger.debug(f"Received: {message}")

                # Handle message
                response = await handle_message(message)

                if response is not None:
                    response_bytes = json.dumps(response).encode()
                    writer.write(f"Content-Length: {len(response_bytes)}\r\n\r\n".encode())
                    writer.write(response_bytes)
                    await writer.drain()

                    logger.debug(f"Sent: {response}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Server error: {e}")
            break

    logger.info("Server stopped")


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
