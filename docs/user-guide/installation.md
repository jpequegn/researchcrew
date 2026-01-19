# Installation Guide

This guide covers the different ways to install and run ResearchCrew.

## Installation Options

| Option | Best For | Difficulty |
|--------|----------|------------|
| **Local Install** | Developers, full control | Medium |
| **Docker** | Easy setup, isolated environment | Easy |
| **Claude Code MCP** | Claude Code users | Easy |
| **Vertex AI** | Production deployment | Advanced |

## Option 1: Local Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Google Cloud account (for API access)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/researchcrew.git
   cd researchcrew
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Configure your API key:**

   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your-api-key-here
   ```

5. **Verify installation:**
   ```bash
   adk run "Hello, can you hear me?"
   ```

### Getting a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Navigate to "Get API Key"
4. Create a new API key
5. Copy and save it securely

## Option 2: Docker Installation

Docker provides an isolated environment with all dependencies included.

### Prerequisites

- Docker installed and running

### Steps

1. **Pull the image:**
   ```bash
   docker pull your-org/researchcrew:latest
   ```

2. **Run with your API key:**
   ```bash
   docker run -e GOOGLE_API_KEY=your-api-key -p 8080:8080 your-org/researchcrew
   ```

3. **Access the web interface** at `http://localhost:8080`

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/researchcrew.git
cd researchcrew

# Build the image
docker build -t researchcrew .

# Run the container
docker run -e GOOGLE_API_KEY=your-api-key -p 8080:8080 researchcrew
```

## Option 3: Claude Code MCP Server

Use ResearchCrew tools directly within Claude Code.

### Prerequisites

- Claude Code installed
- Python 3.11+

### Steps

1. **Install ResearchCrew:**
   ```bash
   pip install researchcrew
   ```

2. **Configure Claude Code:**

   Add to your `~/.claude.json`:
   ```json
   {
     "mcpServers": {
       "researchcrew-web": {
         "command": "python",
         "args": ["-m", "researchcrew.mcp.web_research_server"],
         "env": {
           "SERPER_API_KEY": "your-search-api-key"
         }
       },
       "researchcrew-kb": {
         "command": "python",
         "args": ["-m", "researchcrew.mcp.knowledge_base_server"],
         "env": {
           "KNOWLEDGE_BASE_PATH": "~/.researchcrew/knowledge_base"
         }
       }
     }
   }
   ```

3. **Restart Claude Code** to load the new MCP servers

4. **Use the tools** in your Claude Code sessions:
   - `web_search` - Search the web
   - `read_url` - Extract content from URLs
   - `knowledge_search` - Query your knowledge base
   - `save_to_knowledge` - Store research findings

## Option 4: Vertex AI Deployment

For production use with Google Cloud infrastructure.

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- Vertex AI API enabled

### Steps

See the [Deployment Guide](../deployment.md) for detailed instructions.

## Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | Yes | Gemini API key | None |
| `SERPER_API_KEY` | No | Serper search API key | Uses DuckDuckGo |
| `KNOWLEDGE_BASE_PATH` | No | Knowledge storage location | `./knowledge_base` |
| `LOG_LEVEL` | No | Logging verbosity | `INFO` |
| `ENVIRONMENT` | No | Config environment | `development` |

### Configuration File

Create `config/user.yaml` for persistent settings:

```yaml
# User configuration
model:
  temperature: 0.7  # Lower = more focused, higher = more creative

research:
  max_sources: 10   # Maximum sources per query
  min_confidence: 0.6  # Minimum confidence threshold

output:
  format: markdown  # markdown, json, or summary
  include_sources: true
  include_confidence: true
```

## Verifying Your Installation

Run this test to ensure everything is working:

```bash
# Test basic functionality
adk run "What is 2+2?"

# Test web search (requires API key)
adk run "What is the current weather in San Francisco?"

# Start the web UI
adk web
```

If all commands work, your installation is complete!

## Updating

### Local Installation
```bash
git pull
pip install -e .
```

### Docker
```bash
docker pull your-org/researchcrew:latest
```

## Uninstalling

### Local Installation
```bash
pip uninstall researchcrew
rm -rf .venv
```

### Docker
```bash
docker rmi your-org/researchcrew
```

### Claude Code MCP
Remove the entries from `~/.claude.json`.

## Troubleshooting Installation

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated |
| `GOOGLE_API_KEY not set` | Add key to `.env` file or export as environment variable |
| `Permission denied` | Check file permissions or run with appropriate privileges |
| Docker fails to start | Ensure Docker daemon is running |

For more help, see the [Troubleshooting Guide](./troubleshooting.md).

## Next Steps

- [Getting Started](./getting-started.md) - Run your first query
- [Basic Queries](./usage/basic-queries.md) - Learn how to ask questions
- [Best Practices](./best-practices.md) - Get better results
