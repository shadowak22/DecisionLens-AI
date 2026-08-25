"""
DecisionLens AI - LangGraph Workflow Orchestration Graph
Defines the cyclic multi-agent StateGraph connecting Planner, Research, RAG,
Perspective Analysts, Judge, and Self-Reflection Reviewer with conditional revision loops.
"""

from typing import Dict, Any, Callable, Optional, Generator
from langgraph.graph import StateGraph, START, END
from workflow.state import DecisionState
from models.schemas import (
    PerspectivePlannerOutput,
    PerspectiveItem,
    PerspectiveAnalysisOutput,
    ResearchOutput,
    RAGOutput,
    JudgeOutput,
    ReviewerOutput,
)
from decision_agents.planner import plan_perspectives
from decision_agents.perspective import analyze_all_perspectives_parallel
from decision_agents.research import conduct_research
from decision_agents.judge import synthesize_decision
from decision_agents.reviewer import review_recommendation
from rag.retriever import retrieve_relevant_frameworks


# ==========================================
# LangGraph Node Functions
# ==========================================

def planner_node(state: DecisionState) -> Dict[str, Any]:
    """Node 1: Perspective Planner Agent."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    model = state.get("model")
    is_demo = state.get("is_demo", False)

    planner_output: PerspectivePlannerOutput = plan_perspectives(
        decision=decision,
        api_key=api_key,
        model=model,
        is_demo=is_demo,
    )

    perspectives_data = [p.model_dump() for p in planner_output.perspectives]

    return {
        "decision_summary": planner_output.decision_summary,
        "domain": planner_output.domain,
        "selected_perspectives": perspectives_data,
        "status": "planner_completed",
    }


def research_node(state: DecisionState) -> Dict[str, Any]:
    """Node 2: ReAct External Research Agent."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    model = state.get("model")
    is_demo = state.get("is_demo", False)

    research_output: ResearchOutput = conduct_research(
        decision=decision,
        api_key=api_key,
        model=model,
        is_demo=is_demo,
    )

    return {
        "research": research_output.model_dump(),
        "status": "research_completed",
    }


def rag_node(state: DecisionState) -> Dict[str, Any]:
    """Node 3: RAG Knowledge Retrieval Agent."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    perspectives_raw = state.get("selected_perspectives", [])
    perspective_names = [p.get("name", "") for p in perspectives_raw]

    rag_output: RAGOutput = retrieve_relevant_frameworks(
        query=decision,
        perspectives=perspective_names,
        api_key=api_key,
    )

    return {
        "rag_context": rag_output.model_dump(),
        "status": "rag_completed",
    }


def perspective_analysis_node(state: DecisionState) -> Dict[str, Any]:
    """Node 4: Dynamic Reusable Perspective Analysts (Parallel Execution)."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    model = state.get("model")
    is_demo = state.get("is_demo", False)
    
    perspectives_raw = state.get("selected_perspectives", [])
    perspectives = [PerspectiveItem(**p) for p in perspectives_raw]

    rag_raw = state.get("rag_context")
    rag_context = RAGOutput(**rag_raw) if rag_raw else None

    research_raw = state.get("research")
    research_data = ResearchOutput(**research_raw) if research_raw else None

    analyses = analyze_all_perspectives_parallel(
        decision=decision,
        perspectives=perspectives,
        rag_context=rag_context,
        research_data=research_data,
        api_key=api_key,
        model=model,
        is_demo=is_demo,
    )

    analyses_data = [a.model_dump() for a in analyses]

    return {
        "perspective_results": analyses_data,
        "status": "perspective_analysis_completed",
    }


def judge_node(state: DecisionState) -> Dict[str, Any]:
    """Node 5: Judge Agent Synthesis."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    model = state.get("model")
    is_demo = state.get("is_demo", False)

    perspective_results = [PerspectiveAnalysisOutput(**p) for p in state.get("perspective_results", [])]
    
    rag_raw = state.get("rag_context")
    rag_context = RAGOutput(**rag_raw) if rag_raw else None

    research_raw = state.get("research")
    research_data = ResearchOutput(**research_raw) if research_raw else None

    review_raw = state.get("review_result")
    revision_instructions = review_raw.get("revision_instructions") if review_raw else None

    judge_output: JudgeOutput = synthesize_decision(
        decision=decision,
        perspective_results=perspective_results,
        rag_context=rag_context,
        research_data=research_data,
        revision_instructions=revision_instructions,
        api_key=api_key,
        model=model,
        is_demo=is_demo,
    )

    return {
        "judge_result": judge_output.model_dump(),
        "status": "judge_completed",
    }


def reviewer_node(state: DecisionState) -> Dict[str, Any]:
    """Node 6: Self-Reflection Reviewer Agent."""
    decision = state.get("decision", "")
    api_key = state.get("api_key")
    model = state.get("model")
    is_demo = state.get("is_demo", False)
    revision_count = state.get("revision_count", 0)

    judge_raw = state.get("judge_result", {})
    judge_result = JudgeOutput(**judge_raw) if judge_raw else JudgeOutput(
        evidence_sufficiency="Pending",
        final_recommendation="Pending",
        confidence_score=0.5
    )

    perspective_results = [PerspectiveAnalysisOutput(**p) for p in state.get("perspective_results", [])]

    rag_raw = state.get("rag_context")
    rag_context = RAGOutput(**rag_raw) if rag_raw else None

    research_raw = state.get("research")
    research_data = ResearchOutput(**research_raw) if research_raw else None

    reviewer_output: ReviewerOutput = review_recommendation(
        decision=decision,
        judge_result=judge_result,
        perspective_results=perspective_results,
        rag_context=rag_context,
        research_data=research_data,
        revision_count=revision_count,
        api_key=api_key,
        model=model,
        is_demo=is_demo,
    )

    # If revision requested, increment counter for routing
    new_revision_count = revision_count + (1 if reviewer_output.status == "NEEDS_REVISION" else 0)

    return {
        "review_result": reviewer_output.model_dump(),
        "revision_count": new_revision_count,
        "status": "reviewer_completed",
    }


def should_revise(state: DecisionState) -> str:
    """
    Conditional routing edge evaluating Reviewer verdict.
    Routes back to judge_node if revision requested and revision_count <= 2; otherwise finishes at END.
    """
    review_data = state.get("review_result", {})
    review_status = review_data.get("status", "APPROVED")
    revision_count = state.get("revision_count", 0)

    if review_status == "NEEDS_REVISION" and revision_count <= 2:
        return "judge"
    return END


# ==========================================
# Graph Assembly
# ==========================================

def build_decision_lens_graph():
    """Builds and compiles the LangGraph StateGraph for DecisionLens AI."""
    workflow = StateGraph(DecisionState)

    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("research", research_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("perspective_analysis", perspective_analysis_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("reviewer", reviewer_node)

    # Connect Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "rag")
    workflow.add_edge("rag", "perspective_analysis")
    workflow.add_edge("perspective_analysis", "judge")
    workflow.add_edge("judge", "reviewer")

    # Conditional Reviewer routing
    workflow.add_conditional_edges(
        "reviewer",
        should_revise,
        {
            "judge": "judge",
            END: END,
        }
    )

    return workflow.compile()


# Global compiled graph instance
decision_lens_graph = build_decision_lens_graph()


# ==========================================
# Step-by-Step Workflow Runner for Streamlit
# ==========================================

def run_workflow_stream(
    decision: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    is_demo: bool = False,
) -> Generator[Dict[str, Any], None, DecisionState]:
    """
    Executes the DecisionLens AI LangGraph workflow step by step,
    yielding intermediate state snapshots to update Streamlit status UI in real time.
    """
    initial_state: DecisionState = {
        "decision": decision,
        "decision_summary": "",
        "domain": "",
        "selected_perspectives": [],
        "research": {},
        "rag_context": {},
        "perspective_results": [],
        "judge_result": {},
        "review_result": {},
        "revision_count": 0,
        "status": "started",
        "logs": [],
        "human_approved": None,
        "api_key": api_key,
        "model": model,
        "is_demo": is_demo,
    }

    current_state = dict(initial_state)

    # Step 1: Planner
    yield {"step": "planner", "state": current_state, "message": "Perspective Planner: Dynamically selecting perspectives..."}
    planner_update = planner_node(current_state)
    current_state.update(planner_update)
    yield {"step": "planner_done", "state": current_state, "message": f"Selected {len(current_state['selected_perspectives'])} stakeholder perspectives."}

    # Step 2: Research
    yield {"step": "research", "state": current_state, "message": "Research Agent: ReAct external benchmarking & empirical search..."}
    research_update = research_node(current_state)
    current_state.update(research_update)
    yield {"step": "research_done", "state": current_state, "message": "External empirical evidence synthesized."}

    # Step 3: RAG Knowledge
    yield {"step": "rag", "state": current_state, "message": "RAG Knowledge Agent: Querying local Markdown decision frameworks..."}
    rag_update = rag_node(current_state)
    current_state.update(rag_update)
    yield {"step": "rag_done", "state": current_state, "message": "Local decision frameworks retrieved."}

    # Step 4: Perspective Analyses (Parallel)
    yield {"step": "perspective_analysis", "state": current_state, "message": "Perspective Analysts: Running parallel multi-stakeholder evaluations..."}
    perspective_update = perspective_analysis_node(current_state)
    current_state.update(perspective_update)
    yield {"step": "perspective_analysis_done", "state": current_state, "message": "All perspective evaluations completed."}

    # Step 5 & 6: Judge and Reviewer Loop (with up to 2 revisions)
    while True:
        yield {"step": "judge", "state": current_state, "message": "Judge Agent: Synthesizing trade-offs and decision verdict..."}
        judge_update = judge_node(current_state)
        current_state.update(judge_update)
        yield {"step": "judge_done", "state": current_state, "message": "Recommendation synthesized."}

        yield {"step": "reviewer", "state": current_state, "message": "Reviewer Agent: Metacognitive self-reflection audit..."}
        reviewer_update = reviewer_node(current_state)
        current_state.update(reviewer_update)

        review_status = current_state.get("review_result", {}).get("status", "APPROVED")
        rev_count = current_state.get("revision_count", 0)

        if review_status == "NEEDS_REVISION" and rev_count <= 2:
            yield {
                "step": "revision_triggered", 
                "state": current_state, 
                "message": f"Reviewer requested refinement (Iteration {rev_count}/2). Re-evaluating Judge synthesis..."
            }
            continue
        else:
            yield {"step": "reviewer_done", "state": current_state, "message": f"Self-reflection completed. Final status: {review_status}."}
            break

    current_state["status"] = "completed"
    yield {"step": "finished", "state": current_state, "message": "Decision Intelligence report ready."}
    return current_state
