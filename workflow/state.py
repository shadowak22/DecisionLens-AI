"""
DecisionLens AI - Workflow State Definition
Defines the shared typed state across all LangGraph nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional


class DecisionState(TypedDict, total=False):
    """LangGraph shared state for the DecisionLens AI orchestration graph."""
    decision: str
    decision_summary: str
    domain: str
    selected_perspectives: List[Dict[str, Any]]
    research: Dict[str, Any]
    rag_context: Dict[str, Any]
    perspective_results: List[Dict[str, Any]]
    judge_result: Dict[str, Any]
    review_result: Dict[str, Any]
    revision_count: int
    status: str
    logs: List[str]
    human_approved: Optional[bool]
    api_key: Optional[str]
    model: Optional[str]
    is_demo: bool
