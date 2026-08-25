"""
DecisionLens AI - Reusable Perspective Analyst Agent
A single specialist agent that dynamically assumes any stakeholder or analytical persona
(Finance, Security, Student, Engineering, Ethics, Healthcare, etc.).
Uses OpenAI Agents SDK and supports concurrent multi-perspective execution.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from models.schemas import (
    PerspectiveItem,
    PerspectiveAnalysisOutput,
    RAGOutput,
    ResearchOutput,
)
from utils.config import (
    OPENAI_API_KEY,
    is_live_ai_available,
    get_active_model,
)

BASE_PERSPECTIVE_INSTRUCTIONS = """You are a specialized stakeholder and domain analyst for DecisionLens AI.
You must adopt the specific perspective persona, priorities, and analytical lens provided to you.

Your task:
Analyze the user's decision dilemma strictly through your assigned perspective's viewpoint.
Weigh all retrieved knowledge base frameworks and external research evidence through this lens.

You must provide:
1. viewpoint: The core philosophical and strategic posture of this perspective.
2. benefits: 2-4 concrete opportunities, upside potentials, or positive outcomes.
3. concerns: 2-4 primary doubts, friction points, or reservations.
4. risks: 2-3 specific failure modes, edge cases, or liabilities.
5. assumptions: 1-2 key assumptions underlying your position.
6. evidence: Specific frameworks or facts cited from retrieved RAG or external research.
7. confidence: A calibrated numerical score between 0.0 and 1.0 representing confidence in this assessment.
8. recommendation: A decisive, actionable recommendation solely from this stakeholder's vantage point.
"""


def _generate_dynamic_demo_perspective_analysis(
    decision: str,
    perspective: PerspectiveItem,
    rag_context: Optional[RAGOutput] = None,
    research_data: Optional[ResearchOutput] = None,
) -> PerspectiveAnalysisOutput:
    """
    Generates a high-quality, realistic perspective analysis for Demo Mode or offline fallback.
    Tailors insights specifically to the perspective's name, focus, and decision context.
    """
    p_name = perspective.name
    p_focus = perspective.focus
    
    viewpoint = (
        f"From the vantage point of {p_name}, the primary mandate is ensuring that {p_focus.lower()} "
        f"is rigorously protected, measurable, and aligned with organizational objectives."
    )
    
    benefits = [
        f"Significant advancement in {p_name.lower()} capabilities and standardized practices.",
        f"Improved operational clarity and efficiency in areas addressing {p_focus}.",
        "Provides proactive readiness for scaling and long-term organizational agility."
    ]
    
    concerns = [
        f"Potential execution friction or resource drain impacting {p_name.lower()} priorities during initial rollout.",
        f"Incomplete stakeholder training and change management regarding {p_focus}.",
        "Risk of unanticipated secondary costs or operational dependencies."
    ]
    
    risks = [
        f"Failure mode where {p_name.lower()} requirements are bypassed due to aggressive implementation timelines.",
        "Over-reliance on unvalidated assumptions leading to performance or compliance degradation."
    ]
    
    assumptions = [
        f"Assumes leadership will allocate dedicated bandwidth to uphold {p_name} standards.",
        "Assumes necessary integration tools and governance guardrails will be maintained."
    ]
    
    evidence_list = []
    if rag_context and rag_context.retrieved_docs:
        evidence_list.append(f"Grounded in local framework: {rag_context.retrieved_docs[0].source_file}")
    else:
        evidence_list.append(f"Multi-Criteria Decision Matrix & Risk Priority Framework ({p_name} domain standards)")
        
    if research_data and research_data.key_findings:
        evidence_list.append(f"Industry benchmark: {research_data.key_findings[0].finding[:80]}...")
    else:
        evidence_list.append("Empirical cross-industry adoption case studies")

    confidence = 0.85 if perspective.priority == "high" else 0.78
    
    recommendation = (
        f"Proceed conditionally: Implement a phased pilot with explicit {p_name} checkpoints, "
        f"ensuring that {p_focus} is continuously monitored against predefined success metrics."
    )

    return PerspectiveAnalysisOutput(
        perspective_name=p_name,
        viewpoint=viewpoint,
        benefits=benefits,
        concerns=concerns,
        risks=risks,
        assumptions=assumptions,
        evidence=evidence_list,
        confidence=confidence,
        recommendation=recommendation,
    )


def analyze_single_perspective(
    decision: str,
    perspective: PerspectiveItem,
    rag_context: Optional[RAGOutput] = None,
    research_data: Optional[ResearchOutput] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    is_demo: bool = False,
) -> PerspectiveAnalysisOutput:
    """
    Executes an individual perspective analysis using the OpenAI Agents SDK Agent + Runner.
    """
    active_key = api_key or OPENAI_API_KEY
    active_model = get_active_model(model)

    if is_demo or not is_live_ai_available(active_key):
        return _generate_dynamic_demo_perspective_analysis(
            decision=decision,
            perspective=perspective,
            rag_context=rag_context,
            research_data=research_data,
        )

    try:
        from agents import Agent, Runner

        os.environ["OPENAI_API_KEY"] = active_key

        instructions = (
            f"{BASE_PERSPECTIVE_INSTRUCTIONS}\n\n"
            f"YOU ARE NOW ACTING AS: {perspective.name.upper()}\n"
            f"YOUR ANALYTICAL FOCUS: {perspective.focus}\n"
            f"SELECTION RATIONALE: {perspective.reason_selected}\n"
            f"PRIORITY LEVEL: {perspective.priority}"
        )

        agent = Agent(
            name=f"Analyst - {perspective.name}",
            instructions=instructions,
            model=active_model,
            output_type=PerspectiveAnalysisOutput,
        )

        rag_summary = rag_context.summary if rag_context else "No local RAG context available."
        rag_docs_text = "\n\n".join([f"[{d.source_file}]: {d.content[:300]}" for d in (rag_context.retrieved_docs if rag_context else [])])
        research_summary = research_data.summary if research_data else "No external research available."

        prompt = (
            f"Decision Problem:\n{decision.strip()}\n\n"
            f"Assigned Perspective: {perspective.name}\n"
            f"Focus Area: {perspective.focus}\n\n"
            f"Local Knowledge Base Evidence (RAG):\n{rag_summary}\n{rag_docs_text}\n\n"
            f"External Research Findings:\n{research_summary}\n\n"
            f"Analyze this decision thoroughly through your perspective persona and return structured output."
        )

        result = Runner.run_sync(agent, prompt)

        if isinstance(result.final_output, PerspectiveAnalysisOutput):
            return result.final_output
        elif isinstance(result.final_output, dict):
            return PerspectiveAnalysisOutput.model_validate(result.final_output)
        elif isinstance(result.final_output, str):
            data = json.loads(result.final_output)
            return PerspectiveAnalysisOutput.model_validate(data)
        else:
            raise ValueError(f"Unexpected perspective output type: {type(result.final_output)}")

    except Exception as e:
        print(f"[Perspective Analyst - {perspective.name}] Live analysis failed ({e}). Using fallback analysis.")
        return _generate_dynamic_demo_perspective_analysis(
            decision=decision,
            perspective=perspective,
            rag_context=rag_context,
            research_data=research_data,
        )


def analyze_all_perspectives_parallel(
    decision: str,
    perspectives: List[PerspectiveItem],
    rag_context: Optional[RAGOutput] = None,
    research_data: Optional[ResearchOutput] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    is_demo: bool = False,
    max_workers: int = 5,
) -> List[PerspectiveAnalysisOutput]:
    """
    Executes multiple independent perspective analyses concurrently in parallel threads
    to minimize user latency and accelerate decision synthesis.
    """
    results: List[PerspectiveAnalysisOutput] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(perspectives) or 1)) as executor:
        futures = {
            executor.submit(
                analyze_single_perspective,
                decision,
                p,
                rag_context,
                research_data,
                api_key,
                model,
                is_demo,
            ): p
            for p in perspectives
        }

        for future in as_completed(futures):
            p = futures[future]
            try:
                analysis = future.result()
                results.append(analysis)
            except Exception as e:
                print(f"[Perspective Batch] Error analyzing {p.name}: {e}")
                results.append(
                    _generate_dynamic_demo_perspective_analysis(
                        decision, p, rag_context, research_data
                    )
                )

    # Maintain original planner order
    ordered_results = []
    p_map = {r.perspective_name: r for r in results}
    for p in perspectives:
        if p.name in p_map:
            ordered_results.append(p_map[p.name])
        else:
            # Match by case-insensitive name if needed
            matched = next((r for r in results if r.perspective_name.lower() == p.name.lower()), None)
            if matched:
                ordered_results.append(matched)

    return ordered_results if len(ordered_results) == len(perspectives) else results
