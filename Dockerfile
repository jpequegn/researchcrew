# ResearchCrew Multi-Agent Research Assistant
# Dockerfile for Vertex AI Agent Engine Deployment
#
# Build: docker build -t researchcrew:latest .
# Run:   docker run -p 8080:8080 -e GOOGLE_API_KEY=xxx researchcrew:latest

# ============================================================================
# Stage 1: Builder - Install dependencies and prepare the application
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install production dependencies only (no dev deps)
RUN uv pip install --no-cache .

# ============================================================================
# Stage 2: Runtime - Minimal image for production
# ============================================================================
FROM python:3.12-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY prompts/ ./prompts/
COPY config/ ./config/
COPY runner.py ./

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the default port for Vertex AI Agent Engine
EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default command - run the agent server
# The ADK CLI provides a server mode for deployment
CMD ["python", "-m", "google.adk.cli", "run", "--host", "0.0.0.0", "--port", "8080"]
