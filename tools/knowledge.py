"""Knowledge Base Tools for ResearchCrew

Tools for agents to search and store research findings in the long-term knowledge base.
"""

import json
import logging
from typing import Optional

from utils.knowledge_base import get_knowledge_base, SearchResult

logger = logging.getLogger(__name__)


def knowledge_search(
    query: str,
    max_results: int = 5,
    topic: Optional[str] = None,
) -> str:
    """Search the knowledge base for previously researched information.

    Use this tool to find relevant findings from past research sessions.
    This helps avoid repeating research and builds upon prior knowledge.

    Args:
        query: The search query describing what information you're looking for
        max_results: Maximum number of results to return (default: 5)
        topic: Optional topic filter to narrow results

    Returns:
        A formatted string containing relevant past research findings,
        or a message indicating no relevant findings were found.

    Example:
        >>> knowledge_search("AI agent frameworks")
        "Found 3 relevant entries:
        1. [Score: 0.85] LangGraph is a framework for building...
           Source: https://example.com/langgraph
           Topic: AI frameworks
        ..."
    """
    try:
        kb = get_knowledge_base()
        results = kb.search(
            query=query,
            n_results=max_results,
            topic_filter=topic,
        )

        if not results:
            return (
                f"No relevant past research found for: '{query}'. "
                "This appears to be a new topic that hasn't been researched before."
            )

        # Format results for the agent
        output_lines = [f"Found {len(results)} relevant entries from past research:\n"]

        for i, result in enumerate(results, 1):
            entry = result.entry
            score_pct = int(result.score * 100)

            output_lines.append(f"{i}. [Relevance: {score_pct}%] {entry.content[:500]}")

            if entry.source_url:
                output_lines.append(f"   Source: {entry.source_url}")
            if entry.source_title:
                output_lines.append(f"   Title: {entry.source_title}")
            if entry.topic:
                output_lines.append(f"   Topic: {entry.topic}")
            if entry.confidence:
                output_lines.append(f"   Confidence: {entry.confidence}")

            output_lines.append("")  # Empty line between entries

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return f"Error searching knowledge base: {str(e)}"


def save_to_knowledge(
    content: str,
    source_url: Optional[str] = None,
    source_title: Optional[str] = None,
    topic: Optional[str] = None,
    confidence: str = "medium",
) -> str:
    """Save research findings to the long-term knowledge base.

    Use this tool to store important findings that should be remembered
    for future research sessions. This builds the agent's long-term memory.

    Args:
        content: The finding or information to save (should be a complete,
                 self-contained piece of knowledge)
        source_url: URL of the source (highly recommended for citations)
        source_title: Title of the source document/page
        topic: Category/topic for organizing the knowledge
        confidence: Confidence level - 'high', 'medium', or 'low'

    Returns:
        Confirmation message with the entry ID, or error message.

    Example:
        >>> save_to_knowledge(
        ...     content="LangGraph is a library for building stateful, multi-actor applications with LLMs",
        ...     source_url="https://langchain.com/langgraph",
        ...     source_title="LangGraph Documentation",
        ...     topic="AI frameworks",
        ...     confidence="high"
        ... )
        "Successfully saved to knowledge base with ID: abc123..."
    """
    # Validate confidence level
    valid_confidence = ["high", "medium", "low"]
    if confidence not in valid_confidence:
        confidence = "medium"

    # Validate content length
    if len(content) < 10:
        return "Error: Content too short. Please provide a meaningful finding (at least 10 characters)."

    if len(content) > 5000:
        return "Error: Content too long. Please break into smaller, focused findings (max 5000 characters)."

    try:
        kb = get_knowledge_base()
        entry = kb.add_entry(
            content=content,
            source_url=source_url,
            source_title=source_title,
            topic=topic,
            confidence=confidence,
            skip_duplicates=True,
        )

        if entry:
            return (
                f"Successfully saved to knowledge base.\n"
                f"  ID: {entry.id}\n"
                f"  Topic: {topic or 'uncategorized'}\n"
                f"  Confidence: {confidence}\n"
                f"  Source: {source_url or 'none provided'}"
            )
        else:
            return (
                "Content was not saved - similar information already exists in the knowledge base. "
                "This prevents duplicate entries."
            )

    except Exception as e:
        logger.error(f"Error saving to knowledge base: {e}")
        return f"Error saving to knowledge base: {str(e)}"


def list_knowledge_topics() -> str:
    """List all topics in the knowledge base.

    Use this to understand what areas have been researched before.

    Returns:
        A list of all topics, or a message if the knowledge base is empty.
    """
    try:
        kb = get_knowledge_base()
        topics = kb.list_topics()

        if not topics:
            return "The knowledge base is empty - no topics have been researched yet."

        stats = kb.get_stats()
        output_lines = [
            f"Knowledge base contains {stats['total_entries']} entries across {len(topics)} topics:\n"
        ]

        for topic in topics:
            count = stats["topics"].get(topic, 0)
            output_lines.append(f"  - {topic}: {count} entries")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        return f"Error listing topics: {str(e)}"


def get_knowledge_stats() -> str:
    """Get statistics about the knowledge base.

    Returns:
        Statistics including total entries, topic distribution, etc.
    """
    try:
        kb = get_knowledge_base()
        stats = kb.get_stats()

        output_lines = [
            "Knowledge Base Statistics:",
            f"  Total entries: {stats['total_entries']}",
            f"  Collection: {stats['collection_name']}",
            f"  Storage: {stats['persist_directory'] or 'in-memory'}",
            "",
            "Topic distribution:",
        ]

        for topic, count in sorted(stats["topics"].items()):
            output_lines.append(f"  - {topic}: {count}")

        output_lines.append("")
        output_lines.append("Confidence distribution:")
        for conf, count in sorted(stats["confidence_distribution"].items()):
            output_lines.append(f"  - {conf}: {count}")

        return "\n".join(output_lines)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return f"Error getting stats: {str(e)}"
