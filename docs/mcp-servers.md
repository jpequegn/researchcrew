# MCP Server Documentation

Model Context Protocol (MCP) servers expose ResearchCrew tools for use with any MCP-compatible client.

## Overview

ResearchCrew provides two MCP servers that package the research tools for reuse:

| Server | Purpose | Tools |
|--------|---------|-------|
| **Web Research Server** | Web searching and content extraction | `web_search`, `read_url` |
| **Knowledge Base Server** | Local knowledge storage and retrieval | `knowledge_search`, `save_to_knowledge`, `list_topics`, `get_entry` |

## Quick Start

### With Claude Code

Add to your Claude Code configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "researchcrew-web": {
      "command": "python",
      "args": ["/path/to/researchcrew/mcp-servers/web-research-server/server.py"],
      "env": {
        "SERPER_API_KEY": "your-api-key"
      }
    },
    "researchcrew-kb": {
      "command": "python",
      "args": ["/path/to/researchcrew/mcp-servers/knowledge-base-server/server.py"],
      "env": {
        "KNOWLEDGE_BASE_PATH": "/path/to/knowledge_base"
      }
    }
  }
}
```

### Standalone

```bash
# Web Research Server
SERPER_API_KEY=xxx python mcp-servers/web-research-server/server.py

# Knowledge Base Server
python mcp-servers/knowledge-base-server/server.py
```

## Available Tools

### Web Research Server

#### `web_search`

Search the web for information.

**Input:**
```json
{
  "query": "string (required)",
  "num_results": "integer (optional, default: 5)"
}
```

**Output:**
```json
{
  "results": [
    {
      "title": "string",
      "url": "string",
      "snippet": "string"
    }
  ]
}
```

#### `read_url`

Fetch and extract content from a URL.

**Input:**
```json
{
  "url": "string (required)",
  "max_length": "integer (optional, default: 10000)"
}
```

**Output:**
```json
{
  "content": "string",
  "title": "string",
  "url": "string"
}
```

### Knowledge Base Server

#### `knowledge_search`

Search the knowledge base for relevant entries.

**Input:**
```json
{
  "query": "string (required)",
  "limit": "integer (optional, default: 5)"
}
```

**Output:**
```json
{
  "results": [
    {
      "id": "string",
      "content": "string",
      "topic": "string",
      "score": "number"
    }
  ]
}
```

#### `save_to_knowledge`

Save a new entry to the knowledge base.

**Input:**
```json
{
  "content": "string (required)",
  "topic": "string (required)",
  "source": "string (optional)"
}
```

**Output:**
```json
{
  "id": "string",
  "success": "boolean"
}
```

#### `list_topics`

List all topics in the knowledge base.

**Output:**
```json
{
  "topics": ["string"]
}
```

#### `get_entry`

Get a specific entry by ID.

**Input:**
```json
{
  "id": "string (required)"
}
```

**Output:**
```json
{
  "id": "string",
  "content": "string",
  "topic": "string",
  "created_at": "string"
}
```

## Configuration

### Environment Variables

| Variable | Server | Description | Default |
|----------|--------|-------------|---------|
| `SERPER_API_KEY` | Web Research | API key for search (mock data without) | None |
| `KNOWLEDGE_BASE_PATH` | Knowledge Base | Storage directory | `./knowledge_base` |
| `LOG_LEVEL` | Both | Logging verbosity | `INFO` |

## Architecture

```
mcp-servers/
├── README.md                     # Detailed documentation
├── shared/                       # Shared utilities
│   ├── __init__.py
│   └── utils.py                  # Error classes, validation, logging
├── knowledge-base-server/
│   ├── server.py                 # MCP server implementation
│   └── mcp.json                  # Tool definitions
└── web-research-server/
    ├── server.py                 # MCP server implementation
    └── mcp.json                  # Tool definitions
```

## Creating Custom Servers

### Step 1: Create Directory Structure

```bash
mkdir mcp-servers/my-server
touch mcp-servers/my-server/server.py
touch mcp-servers/my-server/mcp.json
```

### Step 2: Implement Server

```python
# server.py
import json
import sys

TOOLS = [
    {
        "name": "my_tool",
        "description": "What the tool does",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Parameter"}
            },
            "required": ["param"]
        }
    }
]

async def handle_tool_call(name: str, arguments: dict) -> str:
    if name == "my_tool":
        return json.dumps({"result": arguments["param"]})
    return json.dumps({"error": f"Unknown tool: {name}"})

# MCP server loop implementation...
```

### Step 3: Create mcp.json

```json
{
  "name": "my-server",
  "version": "1.0.0",
  "description": "My custom MCP server",
  "tools": [
    {
      "name": "my_tool",
      "description": "What the tool does"
    }
  ]
}
```

## Testing

### With MCP Inspector

```bash
npm install -g @modelcontextprotocol/inspector
mcp-inspector python mcp-servers/web-research-server/server.py
```

### Programmatically

```python
import asyncio
from mcp_servers.web_research_server.server import handle_tool_call

async def test():
    result = await handle_tool_call("web_search", {"query": "AI"})
    print(result)

asyncio.run(test())
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Server won't start | Python version | Requires Python 3.11+ |
| Search returns mock data | Missing API key | Set `SERPER_API_KEY` |
| Connection refused | Path incorrect | Use absolute paths in config |
| Tool not found | Tool name mismatch | Check mcp.json definitions |

### Debug Mode

```bash
LOG_LEVEL=DEBUG python mcp-servers/web-research-server/server.py
```

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Full MCP Server Documentation](../mcp-servers/README.md)
- [Claude Code MCP Guide](https://docs.anthropic.com/claude-code/mcp)

## Related Documentation

- [Architecture](./architecture.md) - System architecture
- [API Documentation](./api.md) - API reference
- [Integrations](./integrations.md) - External framework integration
