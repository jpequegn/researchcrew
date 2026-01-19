# ResearchCrew MCP Servers

This directory contains Model Context Protocol (MCP) servers that expose ResearchCrew tools for use with any MCP-compatible client, including Claude Code, VS Code extensions, and custom integrations.

## Available Servers

### 1. Knowledge Base Server

A local knowledge base for storing and retrieving research findings.

**Tools:**
- `knowledge_search` - Search the knowledge base
- `save_to_knowledge` - Save new entries
- `list_topics` - List available topics
- `get_entry` - Get a specific entry by ID

**Configuration:**
- `KNOWLEDGE_BASE_PATH` - Storage directory (default: `./knowledge_base`)

### 2. Web Research Server

Web research tools for searching and reading web content.

**Tools:**
- `web_search` - Search the web for information
- `read_url` - Fetch and extract content from URLs

**Configuration:**
- `SERPER_API_KEY` - API key for Serper search (optional, uses mock data without it)

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Install Dependencies

```bash
# From the project root
pip install httpx beautifulsoup4

# Or using uv
uv pip install httpx beautifulsoup4
```

## Usage

### With Claude Code

Add the servers to your Claude Code configuration (`~/.claude.json` or project `.claude.json`):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "python",
      "args": ["/path/to/researchcrew/mcp-servers/knowledge-base-server/server.py"],
      "env": {
        "KNOWLEDGE_BASE_PATH": "/path/to/knowledge_base"
      }
    },
    "web-research": {
      "command": "python",
      "args": ["/path/to/researchcrew/mcp-servers/web-research-server/server.py"],
      "env": {
        "SERPER_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Standalone

Run servers directly:

```bash
# Knowledge Base Server
python mcp-servers/knowledge-base-server/server.py

# Web Research Server (with API key)
SERPER_API_KEY=xxx python mcp-servers/web-research-server/server.py
```

### With MCP Inspector (Testing)

Use the MCP Inspector to test servers interactively:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test a server
mcp-inspector python mcp-servers/knowledge-base-server/server.py
```

## Protocol

The servers implement the MCP specification (version 2024-11-05):

- **Transport**: stdio (standard input/output)
- **Protocol**: JSON-RPC 2.0 over MCP message format
- **Capabilities**: `tools` with `listChanged` support

### Message Format

```
Content-Length: <length>\r\n
\r\n
<JSON-RPC message>
```

### Supported Methods

| Method | Description |
|--------|-------------|
| `initialize` | Initialize the server connection |
| `tools/list` | List available tools |
| `tools/call` | Call a specific tool |
| `notifications/initialized` | Client initialization complete |

## Development

### Adding New Tools

1. Define the tool schema in the `TOOLS` list
2. Implement the handler in `handle_tool_call()`
3. Update the mcp.json configuration

Example tool definition:

```python
TOOLS = [
    {
        "name": "my_tool",
        "description": "Description of what the tool does",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description",
                },
            },
            "required": ["param1"],
        },
    },
]
```

### Creating a New Server

1. Create a new directory under `mcp-servers/`
2. Copy the structure from an existing server
3. Implement your tools in `server.py`
4. Create `mcp.json` with tool definitions
5. Add documentation

### Testing

Test tools using the shared utilities:

```python
import asyncio
from server import handle_tool_call

async def test():
    result = await handle_tool_call("my_tool", {"param1": "value"})
    print(result)

asyncio.run(test())
```

## Architecture

```
mcp-servers/
├── README.md                     # This file
├── shared/                       # Shared utilities
│   ├── __init__.py
│   └── utils.py                  # Common functions
├── knowledge-base-server/
│   ├── server.py                 # MCP server implementation
│   └── mcp.json                  # Tool definitions
└── web-research-server/
    ├── server.py
    └── mcp.json
```

### Shared Utilities

The `shared/` directory contains:

- **Error classes**: `MCPError`, `ValidationError`, `ToolError`
- **Validation**: `validate_required_param()`, `validate_optional_param()`
- **Logging**: `setup_logging()`, `get_logger()`
- **Text processing**: `truncate_text()`, `clean_text()`

## Troubleshooting

### Server won't start

1. Check Python version: `python --version` (needs 3.11+)
2. Verify dependencies are installed
3. Check for port conflicts

### Tools return errors

1. Enable debug logging: `LOG_LEVEL=DEBUG python server.py`
2. Check API keys are set correctly
3. Verify network connectivity

### Connection issues with Claude Code

1. Verify the path in configuration is absolute
2. Check the server process is running
3. Review Claude Code logs for errors

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
