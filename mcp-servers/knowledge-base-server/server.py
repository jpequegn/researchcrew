#!/usr/bin/env python3
"""Knowledge Base MCP Server

An MCP server that provides access to a local knowledge base for storing
and retrieving research findings.

Tools:
- knowledge_search: Search the knowledge base
- save_to_knowledge: Save new entries
- list_topics: List available topics
- get_entry: Get a specific entry by ID

Usage:
    # Run the server
    python server.py

    # Or with custom storage path
    KNOWLEDGE_BASE_PATH=/path/to/kb python server.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import (
    MCPError,
    ToolError,
    ValidationError,
    format_error_response,
    get_logger,
    setup_logging,
    truncate_text,
    validate_optional_param,
    validate_required_param,
)

# Set up logging
logger = setup_logging("knowledge-base-server")

# ============================================================================
# Knowledge Base Implementation
# ============================================================================


class SimpleKnowledgeBase:
    """Simple file-based knowledge base for MCP server."""

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize knowledge base.

        Args:
            storage_path: Path to storage directory. Defaults to ./knowledge_base
        """
        self.storage_path = Path(
            storage_path or os.environ.get("KNOWLEDGE_BASE_PATH", "./knowledge_base")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.storage_path / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load or create the index."""
        if self._index_file.exists():
            with open(self._index_file) as f:
                self._index = json.load(f)
        else:
            self._index = {"entries": {}, "topics": {}}
            self._save_index()

    def _save_index(self) -> None:
        """Save the index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def search(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Args:
            query: Search query.
            topic: Optional topic filter.
            limit: Maximum results to return.

        Returns:
            List of matching entries.
        """
        results = []
        query_lower = query.lower()

        for entry_id, entry in self._index["entries"].items():
            # Filter by topic if specified
            if topic and entry.get("topic") != topic:
                continue

            # Simple keyword search
            content = entry.get("content", "").lower()
            title = entry.get("title", "").lower()
            tags = " ".join(entry.get("tags", [])).lower()

            if (
                query_lower in content
                or query_lower in title
                or query_lower in tags
            ):
                results.append(
                    {
                        "id": entry_id,
                        "title": entry.get("title", ""),
                        "topic": entry.get("topic", ""),
                        "snippet": truncate_text(entry.get("content", ""), 200),
                        "score": self._calculate_score(query_lower, entry),
                        "created_at": entry.get("created_at"),
                    }
                )

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _calculate_score(self, query: str, entry: dict[str, Any]) -> float:
        """Calculate relevance score for an entry."""
        score = 0.0
        content = entry.get("content", "").lower()
        title = entry.get("title", "").lower()

        # Title match is worth more
        if query in title:
            score += 2.0

        # Content match
        if query in content:
            score += 1.0
            # Bonus for multiple occurrences
            score += content.count(query) * 0.1

        return score

    def save(
        self,
        content: str,
        title: str,
        topic: str,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Save a new entry to the knowledge base.

        Args:
            content: The content to save.
            title: Title for the entry.
            topic: Topic category.
            source: Optional source URL or reference.
            tags: Optional tags for the entry.

        Returns:
            Entry ID.
        """
        entry_id = str(uuid4())[:8]

        entry = {
            "title": title,
            "content": content,
            "topic": topic,
            "source": source,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save entry
        self._index["entries"][entry_id] = entry

        # Update topic index
        if topic not in self._index["topics"]:
            self._index["topics"][topic] = []
        self._index["topics"][topic].append(entry_id)

        self._save_index()

        logger.info(f"Saved entry {entry_id}: {title}")
        return entry_id

    def get_entry(self, entry_id: str) -> Optional[dict[str, Any]]:
        """Get a specific entry by ID.

        Args:
            entry_id: Entry ID.

        Returns:
            Entry data or None if not found.
        """
        entry = self._index["entries"].get(entry_id)
        if entry:
            return {"id": entry_id, **entry}
        return None

    def list_topics(self) -> list[dict[str, Any]]:
        """List all topics with entry counts.

        Returns:
            List of topics with metadata.
        """
        topics = []
        for topic, entry_ids in self._index["topics"].items():
            topics.append(
                {
                    "topic": topic,
                    "entry_count": len(entry_ids),
                }
            )
        return sorted(topics, key=lambda x: x["entry_count"], reverse=True)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry.

        Args:
            entry_id: Entry ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        if entry_id not in self._index["entries"]:
            return False

        entry = self._index["entries"].pop(entry_id)
        topic = entry.get("topic")
        if topic and topic in self._index["topics"]:
            self._index["topics"][topic].remove(entry_id)
            if not self._index["topics"][topic]:
                del self._index["topics"][topic]

        self._save_index()
        logger.info(f"Deleted entry {entry_id}")
        return True


# ============================================================================
# MCP Server Implementation
# ============================================================================

# Global knowledge base instance
_kb: Optional[SimpleKnowledgeBase] = None


def get_knowledge_base() -> SimpleKnowledgeBase:
    """Get or create the knowledge base instance."""
    global _kb
    if _kb is None:
        _kb = SimpleKnowledgeBase()
    return _kb


# Tool definitions for MCP
TOOLS = [
    {
        "name": "knowledge_search",
        "description": "Search the knowledge base for relevant information on a topic. Returns matching entries with titles, snippets, and relevance scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant entries",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic to filter results",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_to_knowledge",
        "description": "Save new information to the knowledge base for future reference. Useful for storing research findings, summaries, and important facts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to save",
                },
                "title": {
                    "type": "string",
                    "description": "A descriptive title for the entry",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic category (e.g., 'AI', 'Science', 'Technology')",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source URL or reference",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for the entry",
                },
            },
            "required": ["content", "title", "topic"],
        },
    },
    {
        "name": "list_topics",
        "description": "List all topics in the knowledge base with entry counts. Useful for discovering what information is available.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_entry",
        "description": "Get a specific knowledge base entry by its ID. Returns the full content of the entry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The ID of the entry to retrieve",
                },
            },
            "required": ["entry_id"],
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
    kb = get_knowledge_base()

    try:
        if name == "knowledge_search":
            query = validate_required_param(arguments, "query", str)
            topic = validate_optional_param(arguments, "topic", param_type=str)
            limit = validate_optional_param(arguments, "limit", default=10, param_type=int)

            results = kb.search(query=query, topic=topic, limit=limit)
            return {
                "results": results,
                "query": query,
                "result_count": len(results),
            }

        elif name == "save_to_knowledge":
            content = validate_required_param(arguments, "content", str)
            title = validate_required_param(arguments, "title", str)
            topic = validate_required_param(arguments, "topic", str)
            source = validate_optional_param(arguments, "source", param_type=str)
            tags = validate_optional_param(arguments, "tags", default=[], param_type=list)

            entry_id = kb.save(
                content=content,
                title=title,
                topic=topic,
                source=source,
                tags=tags,
            )
            return {
                "success": True,
                "entry_id": entry_id,
                "message": f"Entry saved with ID: {entry_id}",
            }

        elif name == "list_topics":
            topics = kb.list_topics()
            return {
                "topics": topics,
                "total_topics": len(topics),
            }

        elif name == "get_entry":
            entry_id = validate_required_param(arguments, "entry_id", str)
            entry = kb.get_entry(entry_id)
            if entry is None:
                return {
                    "error": True,
                    "message": f"Entry not found: {entry_id}",
                }
            return entry

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
                        "name": "knowledge-base-server",
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
    logger.info("Starting Knowledge Base MCP Server")

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
