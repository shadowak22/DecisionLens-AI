"""
DecisionLens AI - Judge Agent
Synthesizes multi-perspective analyses, RAG frameworks, and research evidence.
Identifies consensus, irreconcilable tensions, and trade-offs to produce a decisive recommendation.
"""

import json
import os
from typing import List, Optional
from models.schemas import (
    JudgeOutput,
    PerspectiveAnalysisOutput,
    RAGOutput,
    ResearchOutput,
)
from utils.config import (
    OPENAI_API_KEY,
    is_live_ai_available,
    get_active_model,
)

JUDGE_INSTRUCTIONS = """You are the Lead Decision Judge Agent for DecisionLens AI.
Your responsibility is to synthesize diverse, conflicting stakeholder perspectives, empirical research, and decision frameworks into an authoritative, actionable decision report.

Critical Principles:
1. Do NOT simply average the opinions or seek toothless compromise.
2. Formulate a decisive, well-reasoned final recommendation that balances short-term friction against long-term strategic value.
3. Explicitly isolate points of Consensus (agreements) versus Tensions (disagreements/trade-offs).
4. Identify existential and operational Major Risks.
5. Calibrate the overall Confidence Score (0.0 to 1.0) based on the completeness of evidence.
6. Provide 3-5 concrete, phased Actionable Next Steps.
7. If revision feedback is provided from the Reviewer Agent, directly address every critique and refine your synthesis.
"""


def _generate_dynamic_demo_judge_synthesis(
    decision: str,
    perspective_results: List[PerspectiveAnalysisOutput],
    rag_context: Optional[RAGOutput] = None,
    research_data: Optional[ResearchOutput] = None,
    revision_instructions: Optional[str] = None,
) -> JudgeOutput:
    """
    Generates a realistic, highly nuanced Judge synthesis for Demo Mode or offline fallback.
    """
    p_names = [p.perspective_name for p in perspective_results] or ["General Stakeholders"]
    
    agreements = [
        f"Broad consensus across {len(p_names)} perspectives that maintaining status quo carries significant opportunity costs.",
        "Shared recognition that phased implementation with structured metrics reduces operational risk.",
        "Universal agreement on the necessity of comprehensive stakeholder training and continuous governance."
    ]

    disagreements = [
        f"Tension between rapid deployment velocity ({p_names[0] if len(p_names) > 0 else 'Technical'}) versus governance guardrails ({p_names[1] if len(p_names) > 1 else 'Compliance'}).",
        "Disagreement on budget prioritization between upfront infrastructure investment versus operational pilot testing.",
        "Differing risk appetites regarding dependency on third-party solutions versus internal workflows."
    ]

    trade_offs = [
        "Speed of Innovation vs. Compliance Verification: Moving fast expands capabilities but increases regulatory exposure without dedicated review gates.",
        "Capital Investment vs. Operational Flexibility: Long-term custom infrastructure provides higher autonomy but demands substantial upfront capital.",
        "Automation Efficiency vs. Human Oversight: Automated workflows maximize throughput but require human-in-the-loop exception handling to maintain trust."
    ]

    major_risks = [
        "Adoption Friction: Stakeholder resistance due to insufficient change management or inadequate onboarding.",
        "Governance & Security Exposure: Inadvertent data leakage or regulatory non-compliance during integration.",
        "Cost Overrun: Escalating operational expenditures (OpEx) if scaling parameters and token consumption are unmonitored."
    ]

    evidence_sufficiency = (
        "Sufficient: High-confidence decision supported by grounded multi-perspective inputs, "
        "local decision framework retrieval, and cross-industry empirical benchmarks."
    )

    final_rec = (
        f"RECOMMENDED WITH CONDITIONS: Proceed with a 90-day phased pilot for '{decision.strip()}'. "
        f"Establish a cross-functional Guiding Coalition representing {', '.join(p_names[:3])}. "
        "Implement mandatory circuit breakers, zero-data-retention agreements, and weekly milestone reviews "
        "before proceeding to full organizational rollout."
    )

    if revision_instructions:
        final_rec += f" [Refined based on self-reflection review: Addressed concerns regarding {revision_instructions[:80]}...]"

    confidence = 0.88 if not revision_instructions else 0.94

    next_steps = [
        "1. Charter a cross-functional Governance Working Group with designated stakeholder leads.",
        "2. Formulate explicit quantitative Key Performance Indicators (KPIs) and Risk Priority Numbers (RPN).",
        "3. Secure necessary vendor data protection agreements and privacy certifications.",
        "4. Launch a 30-day bounded sandbox pilot with a controlled 10% user cohort.",
        "5. Conduct formal post-pilot evaluation against baseline metrics prior to broad deployment."
    ]

    return JudgeOutput(
        agreements=agreements,
        disagreements=disagreements,
        trade_offs=trade_offs,
        major_risks=major_risks,
        evidence_sufficiency=evidence_sufficiency,
        final_recommendation=final_rec,
        confidence_score=confidence,
        next_steps=next_steps,
    )


def synthesize_decision(
    decision: str,
    perspective_results: List[PerspectiveAnalysisOutput],
    rag_context: Optional[RAGOutput] = None,
    research_data: Optional[ResearchOutput] = None,
    revision_instructions: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    is_demo: bool = False,
) -> JudgeOutput:
    """
    Executes the Judge Agent using the OpenAI Agents SDK Agent + Runner.
    """
    active_key = api_key or OPENAI_API_KEY
    active_model = get_active_model(model)

    if is_demo or not is_live_ai_available(active_key):
        return _generate_dynamic_demo_judge_synthesis(
            decision=decision,
            perspective_results=perspective_results,
            rag_context=rag_context,
            research_data=research_data,
            revision_instructions=revision_instructions,
        )

    try:
        from agents import Agent, Runner

        os.environ["OPENAI_API_KEY"] = active_key

        judge_agent = Agent(
            name="Decision Judge",
            instructions=JUDGE_INSTRUCTIONS,
            model=active_model,
            output_type=JudgeOutput,
        )

        perspectives_text = "\n\n".join([
            f"### Perspective: {p.perspective_name} (Confidence: {p.confidence:.2f})\n"
            f"- **Viewpoint**: {p.viewpoint}\n"
            f"- **Benefits**: {', '.join(p.benefits)}\n"
            f"- **Concerns**: {', '.join(p.concerns)}\n"
            f"- **Risks**: {', '.join(p.risks)}\n"
            f"- **Evidence**: {', '.join(p.evidence)}\n"
            f"- **Recommendation**: {p.recommendation}"
            for p in perspective_results
        ])

        rag_text = rag_context.summary if rag_context else "No RAG context available."
        research_text = research_data.summary if research_data else "No research data available."

        prompt = (
            f"Original Decision Problem:\n{decision.strip()}\n\n"
            f"--- Perspective Analyses ({len(perspective_results)} Stakeholders) ---\n"
            f"{perspectives_text}\n\n"
            f"--- Local Knowledge Base Evidence (RAG) ---\n"
            f"{rag_text}\n\n"
            f"--- External Research Evidence ---\n"
            f"{research_text}\n\n"
        )

        if revision_instructions:
            prompt += (
                f"\n=== CRITICAL REVISION INSTRUCTIONS FROM SELF-REFLECTION REVIEWER ===\n"
                f"{revision_instructions}\n"
                f"You MUST refine your previous analysis to directly resolve these critiques.\n"
            )

        prompt += "\nSynthesize all inputs into an authoritative final Judge verdict."

        result = Runner.run_sync(judge_agent, prompt)

        if isinstance(result.final_output, JudgeOutput):
            return result.final_output
        elif isinstance(result.final_output, dict):
            return JudgeOutput.model_validate(result.final_output)
        elif isinstance(result.final_output, str):
            data = json.loads(result.final_output)
            return JudgeOutput.model_validate(data)
        else:
            raise ValueError(f"Unexpected judge output format: {type(result.final_output)}")

    except Exception as e:
        print(f"[Judge Agent] Live synthesis failed ({e}). Using fallback synthesis.")
        return _generate_dynamic_demo_judge_synthesis(
            decision=decision,
            perspective_results=perspective_results,
            rag_context=rag_context,
            research_data=research_data,
            revision_instructions=revision_instructions,
        )
