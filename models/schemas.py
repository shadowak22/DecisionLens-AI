"""
DecisionLens AI - Pydantic Data Models and Schemas
Defines structured data representations for all agents, RAG outputs, and workflow states.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PerspectiveItem(BaseModel):
    """Represents a dynamically selected perspective to evaluate a decision."""
    name: str = Field(..., description="Name of the perspective, e.g., 'Finance', 'Student', 'Security', 'Compliance'")
    reason_selected: str = Field(..., description="Why this specific perspective is relevant to this decision")
    focus: str = Field(..., description="Core analytical focus and evaluation angle for this perspective")
    priority: Literal["high", "medium", "low"] = Field(default="high", description="Importance priority of this perspective")


class PerspectivePlannerOutput(BaseModel):
    """Structured output from the Perspective Planner Agent."""
    decision_summary: str = Field(..., description="Concise summary of the core decision problem")
    domain: str = Field(..., description="Identified domain category (e.g., Higher Education, Fintech, Cloud Architecture)")
    perspectives: List[PerspectiveItem] = Field(
        ..., 
        min_length=3, 
        max_length=8, 
        description="Dynamic list of 3 to 8 distinct perspectives chosen specifically for this problem"
    )


class PerspectiveAnalysisOutput(BaseModel):
    """Structured output from a single Perspective Analyst Agent."""
    perspective_name: str = Field(..., description="Name of the perspective evaluated")
    viewpoint: str = Field(..., description="Primary stance or philosophical angle of this perspective")
    benefits: List[str] = Field(default_factory=list, description="Key opportunities or positive outcomes identified")
    concerns: List[str] = Field(default_factory=list, description="Major doubts, hesitations, or reservations")
    risks: List[str] = Field(default_factory=list, description="Specific failure scenarios or operational hazards")
    assumptions: List[str] = Field(default_factory=list, description="Underlying assumptions made in this evaluation")
    evidence: List[str] = Field(default_factory=list, description="Data points, frameworks, or facts supporting this view")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in this perspective analysis (0.0 to 1.0)")
    recommendation: str = Field(..., description="Actionable recommendation solely from this stakeholder's viewpoint")


class ResearchFinding(BaseModel):
    """An individual research discovery from the ReAct Research Agent."""
    topic: str = Field(..., description="Topic or query that was researched")
    finding: str = Field(..., description="Synthesized finding, trend, or empirical benchmark")
    source: str = Field(default="Web Search", description="Origin or domain citation")
    relevance: str = Field(..., description="How this finding connects directly to the decision")


class ResearchOutput(BaseModel):
    """Structured result of the ReAct Research Agent."""
    status: Literal["completed", "fallback", "unavailable"] = Field(
        default="completed", 
        description="Status of external web research"
    )
    key_findings: List[ResearchFinding] = Field(default_factory=list, description="Grounded research findings")
    summary: str = Field(..., description="High-level synthesis of external research and real-world benchmarks")


class RAGDocumentResult(BaseModel):
    """A retrieved knowledge base snippet."""
    source_file: str = Field(..., description="Name of the source markdown file")
    content: str = Field(..., description="Relevant passage text")
    relevance_score: Optional[float] = Field(default=None, description="Similarity score if calculated")


class RAGOutput(BaseModel):
    """Structured result of the RAG Knowledge Retrieval Agent."""
    status: str = Field(default="retrieved", description="Retrieval status")
    retrieved_docs: List[RAGDocumentResult] = Field(default_factory=list, description="Top matching document passages")
    summary: str = Field(..., description="Summary of how local frameworks apply to this decision")


class JudgeOutput(BaseModel):
    """Synthesized verdict from the Judge Agent."""
    agreements: List[str] = Field(default_factory=list, description="Areas where perspectives aligned")
    disagreements: List[str] = Field(default_factory=list, description="Areas of tension, conflict, or trade-off")
    trade_offs: List[str] = Field(default_factory=list, description="Critical trade-offs that leadership must weigh")
    major_risks: List[str] = Field(default_factory=list, description="Primary risks and existential failure modes")
    evidence_sufficiency: str = Field(..., description="Assessment of whether available evidence was adequate")
    final_recommendation: str = Field(..., description="Nuanced, actionable final decision verdict")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the final recommendation")
    next_steps: List[str] = Field(default_factory=list, description="Concrete immediate next steps")


class ReviewerOutput(BaseModel):
    """Self-Reflection evaluation from the Reviewer Agent."""
    status: Literal["APPROVED", "NEEDS_REVISION"] = Field(
        ..., 
        description="Whether the Judge's recommendation is approved or requires revision"
    )
    evidence_checked: bool = Field(default=True, description="Verified if recommendation is grounded in evidence")
    contradictions_checked: bool = Field(default=True, description="Verified if conflicting claims were reconciled")
    missing_perspectives_checked: bool = Field(default=True, description="Verified if critical viewpoints were overlooked")
    recommendation_quality_checked: bool = Field(default=True, description="Verified if confidence is well calibrated")
    critique: str = Field(..., description="Analytical review commentary and justification")
    revision_instructions: Optional[str] = Field(
        default=None, 
        description="Specific instructions for the Judge to refine if status is NEEDS_REVISION"
    )


class FullDecisionReport(BaseModel):
    """Complete end-to-end decision intelligence artifact."""
    decision_query: str
    summary: str
    domain: str
    planner: PerspectivePlannerOutput
    research: ResearchOutput
    rag: RAGOutput
    perspective_analyses: List[PerspectiveAnalysisOutput]
    judge: JudgeOutput
    reviewer: ReviewerOutput
    revision_count: int
    human_approved: Optional[bool] = None
