"""Workflow State Schemas

Pydantic models for state that flows between agents in the research workflow.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level for findings and verdicts."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verdict(str, Enum):
    """Fact-check verdict for claims."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class Source(BaseModel):
    """A source citation."""
    url: str = Field(description="URL of the source")
    title: str = Field(description="Title of the source")
    description: Optional[str] = Field(default=None, description="Brief description")
    credibility: Optional[ConfidenceLevel] = Field(default=None, description="Source credibility")


class ResearchFinding(BaseModel):
    """A single research finding from a researcher agent."""
    claim: str = Field(description="The factual claim or finding")
    sources: list[Source] = Field(default_factory=list, description="Supporting sources")
    confidence: ConfidenceLevel = Field(description="Confidence in this finding")
    notes: Optional[str] = Field(default=None, description="Additional context")


class ResearchState(BaseModel):
    """State output from the research phase."""
    query: str = Field(description="Original research query")
    sub_queries: list[str] = Field(default_factory=list, description="Decomposed sub-queries")
    findings: list[ResearchFinding] = Field(default_factory=list, description="Research findings")
    gaps: list[str] = Field(default_factory=list, description="Areas needing more research")


class Theme(BaseModel):
    """A synthesized theme from multiple findings."""
    title: str = Field(description="Theme title")
    summary: str = Field(description="Theme summary")
    supporting_findings: list[str] = Field(default_factory=list, description="Related finding claims")
    confidence: ConfidenceLevel = Field(description="Confidence in this theme")


class Conflict(BaseModel):
    """A conflict between findings that was resolved."""
    topic: str = Field(description="Topic of conflict")
    positions: list[str] = Field(description="Different positions found")
    resolution: str = Field(description="How the conflict was resolved")


class SynthesisState(BaseModel):
    """State output from the synthesis phase."""
    executive_summary: str = Field(description="2-3 sentence summary")
    themes: list[Theme] = Field(default_factory=list, description="Identified themes")
    conflicts: list[Conflict] = Field(default_factory=list, description="Resolved conflicts")
    overall_confidence: ConfidenceLevel = Field(description="Overall confidence")
    research_gaps: list[str] = Field(default_factory=list, description="Gaps to address")


class FactCheckVerdict(BaseModel):
    """Verdict for a single fact-checked claim."""
    claim: str = Field(description="The claim being checked")
    source_url: str = Field(description="Source URL cited")
    verdict: Verdict = Field(description="Fact-check verdict")
    evidence: str = Field(description="Evidence for verdict")
    confidence: ConfidenceLevel = Field(description="Confidence in verdict")


class FactCheckState(BaseModel):
    """State output from the fact-checking phase."""
    verdicts: list[FactCheckVerdict] = Field(default_factory=list, description="Claim verdicts")
    supported_count: int = Field(default=0, description="Number of supported claims")
    unsupported_count: int = Field(default=0, description="Number of unsupported claims")
    overall_reliability: ConfidenceLevel = Field(description="Overall reliability")
    flagged_issues: list[str] = Field(default_factory=list, description="Issues found")


class ReportFormat(str, Enum):
    """Output format for the final report."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    MARKDOWN = "markdown"


class ReportState(BaseModel):
    """State output from the writing phase (final output)."""
    format: ReportFormat = Field(description="Report format")
    title: str = Field(description="Report title")
    content: str = Field(description="Full report content")
    word_count: int = Field(default=0, description="Word count")
    sources_cited: int = Field(default=0, description="Number of sources cited")


class WorkflowState(BaseModel):
    """Complete workflow state tracking all phases."""
    query: str = Field(description="Original query")
    status: str = Field(default="pending", description="Current workflow status")

    # Session tracking for multi-turn conversations
    session_id: Optional[str] = Field(default=None, description="Session identifier for multi-turn")
    turn_number: int = Field(default=1, description="Current turn number in the session")
    session_context: Optional[str] = Field(
        default=None, description="Context from previous turns in this session"
    )

    # Phase outputs (populated as workflow progresses)
    research: Optional[ResearchState] = Field(default=None)
    synthesis: Optional[SynthesisState] = Field(default=None)
    fact_check: Optional[FactCheckState] = Field(default=None)
    report: Optional[ReportState] = Field(default=None)

    # Metadata
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
    duration_seconds: Optional[float] = Field(default=None, description="Total duration")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp when created")
