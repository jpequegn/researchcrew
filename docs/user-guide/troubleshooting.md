# Troubleshooting Guide

Solutions for common issues when using ResearchCrew.

## Quick Diagnosis

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| No response | API key missing | Check `.env` file |
| Slow response | Complex query | Wait or simplify |
| Irrelevant results | Vague query | Be more specific |
| "Rate limited" | Too many requests | Wait and retry |
| Connection error | Network issue | Check internet |

## Common Issues

### Installation Issues

#### "ModuleNotFoundError: No module named 'google.adk'"

**Cause:** Dependencies not installed.

**Solution:**
```bash
# Activate virtual environment first
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
```

#### "GOOGLE_API_KEY not set"

**Cause:** API key not configured.

**Solution:**

1. Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your-api-key-here
   ```

2. Or export directly:
   ```bash
   export GOOGLE_API_KEY=your-api-key-here
   ```

#### "Permission denied" on macOS/Linux

**Cause:** Script not executable or wrong permissions.

**Solution:**
```bash
chmod +x ./script-name
# Or run with Python directly
python -m researchcrew
```

### Connection Issues

#### "Connection refused" or "ECONNREFUSED"

**Cause:** Server not running or wrong port.

**Solutions:**
1. Start the server: `adk web`
2. Check if port 8080 is available
3. Try a different port: `adk web --port 8081`

#### "Timeout" errors

**Cause:** Network issues or API slowness.

**Solutions:**
1. Check your internet connection
2. Try again (may be temporary)
3. Simplify your query
4. Check API status at [Google Cloud Status](https://status.cloud.google.com/)

#### "SSL Certificate" errors

**Cause:** Certificate issues, often corporate proxy.

**Solutions:**
1. Update certificates: `pip install --upgrade certifi`
2. If behind corporate proxy, configure proxy certificates
3. Contact IT for corporate network issues

### Query Issues

#### Getting Irrelevant Results

**Cause:** Query too vague or ambiguous.

**Solutions:**
1. Be more specific:
   - Before: "Tell me about Python"
   - After: "What are Python best practices for async programming?"

2. Add context:
   - Before: "Best database?"
   - After: "Best database for a high-traffic e-commerce application with complex queries?"

3. Clarify what you want:
   - Before: "React help"
   - After: "How do I manage global state in React 18?"

#### Results Are Outdated

**Cause:** Sources from older time period.

**Solutions:**
1. Ask for recent information:
   ```
   "What are the latest developments in [topic] as of 2024?"
   ```

2. Request recent sources:
   ```
   "Please prioritize sources from the last 12 months"
   ```

3. Follow up:
   ```
   "Is this information still current? Are there more recent developments?"
   ```

#### Low Confidence Scores

**Cause:** Limited sources, conflicting information, or subjective topic.

**Solutions:**
1. Ask for more sources:
   ```
   "Can you find additional sources to verify this?"
   ```

2. Narrow the scope:
   ```
   "Focus specifically on [narrower aspect]"
   ```

3. Request clarification on uncertainty:
   ```
   "What makes this finding uncertain? What additional information would help?"
   ```

#### Empty or Minimal Results

**Cause:** Topic too niche, query too restrictive, or technical issues.

**Solutions:**
1. Broaden the query:
   - Before: "Performance of XYZ library version 2.3.1 on ARM64 M2"
   - After: "Performance characteristics of XYZ library on Apple Silicon"

2. Try alternative phrasing
3. Check if the topic exists (very new or obscure topics may have limited information)

### Performance Issues

#### Queries Taking Too Long

**Cause:** Complex query requiring extensive research.

**Expected times:**
- Simple query: 10-30 seconds
- Complex query: 30-90 seconds
- Comprehensive research: 60-180 seconds

**Solutions:**
1. Simplify the query
2. Break into smaller questions
3. Check your internet connection
4. Wait patiently for complex topics

#### System Running Slowly

**Cause:** Resource constraints or too many concurrent operations.

**Solutions:**
1. Close other applications
2. Reduce concurrent queries
3. Check system resources (memory, CPU)
4. Restart the application

### Rate Limiting

#### "Rate limit exceeded" Error

**Cause:** Too many API requests in short period.

**Solutions:**
1. Wait before retrying (usually 1-5 minutes)
2. Space out your queries
3. Use the knowledge base for repeated questions
4. Check rate limit headers for reset time:
   ```
   X-RateLimit-Reset: [timestamp]
   ```

### Knowledge Base Issues

#### Can't Find Previous Research

**Cause:** Search terms don't match, or knowledge base not configured.

**Solutions:**
1. Try different search terms
2. Check knowledge base location:
   ```bash
   echo $KNOWLEDGE_BASE_PATH
   ```
3. List topics to find what's stored:
   ```python
   kb.list_topics()
   ```

#### "Knowledge base not available"

**Cause:** Storage not configured or permissions issue.

**Solutions:**
1. Set the path:
   ```bash
   export KNOWLEDGE_BASE_PATH=./knowledge_base
   ```
2. Ensure directory exists and is writable
3. Check disk space

### Session Issues

#### "Session not found"

**Cause:** Session expired or invalid session ID.

**Solutions:**
1. Create a new session
2. Don't reuse very old session IDs
3. Check if session was explicitly deleted

#### Context Seems Lost

**Cause:** Very long session, context was compressed.

**Solutions:**
1. Briefly remind of previous context:
   ```
   "Earlier we discussed [topic]. Now I want to..."
   ```
2. Start a new session for fresh context
3. Reference specific previous findings

## Error Messages

### API Errors

| Error | Meaning | Solution |
|-------|---------|----------|
| `INVALID_API_KEY` | API key wrong or expired | Get new key from Google AI Studio |
| `QUOTA_EXCEEDED` | Usage limit reached | Wait for quota reset or upgrade |
| `MODEL_NOT_AVAILABLE` | Model temporarily unavailable | Try again later |
| `CONTENT_FILTERED` | Query triggered safety filter | Rephrase query |

### System Errors

| Error | Meaning | Solution |
|-------|---------|----------|
| `TOOL_EXECUTION_ERROR` | External tool failed | Retry, check tool service |
| `TOKEN_LIMIT_EXCEEDED` | Context too long | Start new session, shorter queries |
| `CIRCUIT_OPEN` | Service temporarily unavailable | Wait for recovery (30s-5min) |

## Debug Mode

For more detailed error information:

```bash
LOG_LEVEL=DEBUG adk run "your query"
```

This shows:
- API request/response details
- Tool execution traces
- Timing information
- Error stack traces

## Getting Help

If you can't resolve an issue:

1. **Check documentation** - You might have missed something
2. **Search existing issues** - Someone may have had the same problem
3. **Open a GitHub issue** with:
   - What you tried to do
   - What happened instead
   - Error messages (if any)
   - Debug output (set LOG_LEVEL=DEBUG)
   - Your environment (OS, Python version)

## Preventive Measures

### Avoid Issues

1. **Keep dependencies updated:**
   ```bash
   pip install --upgrade researchcrew
   ```

2. **Use virtual environments** to avoid conflicts

3. **Back up your knowledge base** periodically

4. **Monitor your API usage** to avoid surprise rate limits

### Health Check

Run this periodically to verify everything works:

```bash
# Basic test
adk run "What is 2+2?"

# Web search test
adk run "What is the current date?"

# Knowledge base test
python -c "from utils.knowledge_base import KnowledgeBase; kb = KnowledgeBase(); print(kb.get_stats())"
```

## Next Steps

- [Getting Started](./getting-started.md) - Re-verify your setup
- [Best Practices](./best-practices.md) - Improve your queries
- [Installation](./installation.md) - Reinstall if needed
