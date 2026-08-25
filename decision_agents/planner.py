"""
DecisionLens AI - Perspective Planner Agent
Analyzes any arbitrary decision dilemma and dynamically selects 3 to 8 relevant perspectives.
Uses OpenAI Agents SDK with structured Pydantic output.
"""

import json
import os
from typing import Optional, List
from pydantic import ValidationError
from models.schemas import PerspectivePlannerOutput, PerspectiveItem
from utils.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    is_live_ai_available,
    get_active_model,
)

PERSPECTIVE_CATALOG = [
    "Student", "Faculty", "Customer", "Employee", "Manager", "Executive",
    "Technical", "Software Engineering", "AI/ML", "Data", "Cloud", "Security",
    "Privacy", "Finance", "Product", "Marketing", "Sales", "Operations",
    "HR", "Legal", "Compliance", "Ethics", "Safety", "Accessibility",
    "Sustainability", "Environmental", "Education", "Healthcare", "Manufacturing",
    "Logistics", "Government", "Research", "Infrastructure", "Architecture",
    "Database", "Performance", "Scalability", "Risk", "Policy", "Regulatory",
    "User Experience", "Cost-Benefit", "Market Analysis", "Competitive Analysis"
]

PLANNER_INSTRUCTIONS = f"""You are the Perspective Planner Agent for DecisionLens AI.
Your purpose is to analyze ANY decision problem and determine the most crucial 3 to 8 stakeholder and analytical perspectives required for comprehensive multi-perspective evaluation.

Perspective Reference Catalog (use these or formulate specialized roles as needed):
{", ".join(PERSPECTIVE_CATALOG)}

Rules:
1. Dynamically select between 3 and 8 distinct, complementary perspectives that represent conflicting or diverse priorities for this specific decision.
2. For each perspective, state clearly why it was chosen (reason_selected), its precise evaluation focus (focus), and its priority ('high', 'medium', or 'low').
3. Categorize the overarching domain (e.g., Higher Education & EdTech, Cloud Infrastructure, Healthcare Operations, Corporate Governance).
4. Provide a crisp 1-2 sentence decision summary.
5. Return your output matching the structured schema.
"""


def _generate_dynamic_demo_perspectives(decision: str) -> PerspectivePlannerOutput:
    """
    Intelligent dynamic fallback generator for Demo Mode or offline evaluation.
    Dynamically deduces relevant perspectives from keywords and domain semantics in the user's prompt.
    """
    dec_lower = decision.lower()
    
    # Heuristic domain detection
    if any(k in dec_lower for k in ["college", "university", "school", "student", "faculty", "assessment", "education", "teacher", "curriculum"]):
        domain = "Higher Education & Academic Governance"
        perspectives = [
            PerspectiveItem(name="Student", reason_selected="Directly impacted by grading fairness, learning outcomes, and assessment workload.", focus="Assessment equity, student privacy, learning autonomy, and accessibility.", priority="high"),
            PerspectiveItem(name="Faculty & Pedagogy", reason_selected="Responsible for instructional integrity, curriculum delivery, and grading workload.", focus="Academic freedom, grading accuracy, workload reduction, and plagiarism detection.", priority="high"),
            PerspectiveItem(name="Ethics & Academic Integrity", reason_selected="Assesses ethical implications of algorithmic grading and potential bias.", focus="Algorithmic bias, appeal procedures, and ethical standards in assessment.", priority="high"),
            PerspectiveItem(name="IT & Security", reason_selected="Responsible for platform integration, data security, and FERPA compliance.", focus="System uptime, integration with LMS, and data protection compliance.", priority="medium"),
            PerspectiveItem(name="Finance & Administration", reason_selected="Evaluates procurement costs, licensing, and return on educational investment.", focus="Subscription licensing, infrastructure costs, and institutional budget impact.", priority="medium"),
        ]
    elif any(k in dec_lower for k in ["microservice", "monolith", "database", "cloud", "aws", "kubernetes", "api", "architecture", "serverless", "devops", "code"]):
        domain = "Software Architecture & Cloud Engineering"
        perspectives = [
            PerspectiveItem(name="Software Engineering", reason_selected="Directly impacts developer velocity, code maintainability, and testing complexity.", focus="Modularity, local development workflows, refactoring effort, and CI/CD pipelines.", priority="high"),
            PerspectiveItem(name="Infrastructure & DevOps", reason_selected="Handles deployment, container orchestration, service discovery, and telemetry.", focus="Deployment complexity, observability, auto-scaling, and cluster overhead.", priority="high"),
            PerspectiveItem(name="Security & Compliance", reason_selected="Assesses network boundaries, authentication across boundaries, and data protection.", focus="Zero-trust networking, secret management, and attack surface expansion.", priority="high"),
            PerspectiveItem(name="Finance & FinOps", reason_selected="Weighs cloud hosting, compute sprawl, and network egress expenditures.", focus="Infrastructure unit economics, operational expenditure (OpEx), and cost predictability.", priority="medium"),
            PerspectiveItem(name="Product & Reliability", reason_selected="Focuses on user latency SLOs, system uptime, and feature delivery cadence.", focus="Time-to-market, fault isolation, blast radius mitigation, and 99.9% uptime SLOs.", priority="high"),
        ]
    elif any(k in dec_lower for k in ["hire", "employee", "remote", "return to office", "salary", "team", "layoff", "workplace", "hr"]):
        domain = "Human Resources & Organizational Strategy"
        perspectives = [
            PerspectiveItem(name="Employee & Morale", reason_selected="Directly affects daily workplace experience, productivity, and retention.", focus="Work-life balance, psychological safety, compensation fairness, and autonomy.", priority="high"),
            PerspectiveItem(name="Operations & Management", reason_selected="Manages team coordination, synchronous collaboration, and output tracking.", focus="Operational velocity, cross-time-zone alignment, and project delivery.", priority="high"),
            PerspectiveItem(name="HR & Talent Acquisition", reason_selected="Responsible for recruitment pipelines, onboarding, and competitive positioning.", focus="Talent pool reach, attrition risks, hiring cycles, and employer branding.", priority="high"),
            PerspectiveItem(name="Finance & Overhead", reason_selected="Analyzes compensation structures, real estate costs, and travel budgets.", focus="Facilities cost savings, localized salary bands, and administrative expenses.", priority="medium"),
            PerspectiveItem(name="Legal & Compliance", reason_selected="Ensures adherence to multi-state labor laws and employment regulations.", focus="Labor law compliance, tax jurisdictions, and data confidentiality agreements.", priority="medium"),
        ]
    elif any(k in dec_lower for k in ["medical", "health", "hospital", "patient", "clinical", "pharma", "treatment", "doctor"]):
        domain = "Healthcare & Clinical Operations"
        perspectives = [
            PerspectiveItem(name="Patient & Family", reason_selected="Direct beneficiaries whose care outcomes and safety are at stake.", focus="Quality of care, patient safety, dignity, and accessibility of services.", priority="high"),
            PerspectiveItem(name="Clinical Staff & Physicians", reason_selected="Frontline practitioners operating the clinical protocols.", focus="Diagnostic efficacy, clinical workflow friction, and professional liability.", priority="high"),
            PerspectiveItem(name="Medical Ethics & Safety", reason_selected="Governs bioethics, informed consent, and risk minimization.", focus="Informed consent, adverse event risk, and evidence-based safety thresholds.", priority="high"),
            PerspectiveItem(name="Regulatory & HIPAA Compliance", reason_selected="Monitors patient data confidentiality and statutory health regulations.", focus="Protected Health Information (PHI) security, FDA/HIPAA compliance, and audits.", priority="high"),
            PerspectiveItem(name="Hospital Administration & Finance", reason_selected="Evaluates hospital capital allocation and reimbursement insurance models.", focus="Reimbursement rates, capital investment, and bed turnover efficiency.", priority="medium"),
        ]
    else:
        # Generic multifaceted decision
        domain = "Strategic Business & Technology Governance"
        perspectives = [
            PerspectiveItem(name="Strategic Leadership & Executive", reason_selected="Aligns decision with overarching organizational mission and market goals.", focus="Long-term strategic moat, market alignment, and corporate reputation.", priority="high"),
            PerspectiveItem(name="Finance & ROI", reason_selected="Assesses capital allocation, operational expenditures, and payback horizons.", focus="Total Cost of Ownership (TCO), ROI modeling, and budget feasibility.", priority="high"),
            PerspectiveItem(name="Operations & Execution", reason_selected="Evaluates implementation friction, staffing capability, and timeline risks.", focus="Resource constraints, change management, and operational bottlenecks.", priority="high"),
            PerspectiveItem(name="Risk & Compliance", reason_selected="Identifies legal liabilities, regulatory hurdles, and governance exposures.", focus="Regulatory exposure, contractual compliance, and risk containment.", priority="medium"),
            PerspectiveItem(name="End-User & Customer Experience", reason_selected="Measures stakeholder reception, usability, and satisfaction metrics.", focus="Adoption friction, customer satisfaction, and usability feedback.", priority="high"),
        ]

    return PerspectivePlannerOutput(
        decision_summary=f"Analysis of strategic proposal: {decision.strip()}",
        domain=domain,
        perspectives=perspectives,
    )


def plan_perspectives(
    decision: str, 
    api_key: Optional[str] = None, 
    model: Optional[str] = None,
    is_demo: bool = False
) -> PerspectivePlannerOutput:
    """
    Executes the Perspective Planner Agent using the OpenAI Agents SDK
    or falls back to the dynamic generator if in demo mode or offline.
    """
    active_key = api_key or OPENAI_API_KEY
    active_model = get_active_model(model)

    if is_demo or not is_live_ai_available(active_key):
        return _generate_dynamic_demo_perspectives(decision)

    try:
        from agents import Agent, Runner

        os.environ["OPENAI_API_KEY"] = active_key

        planner_agent = Agent(
            name="Perspective Planner",
            instructions=PLANNER_INSTRUCTIONS,
            model=active_model,
            output_type=PerspectivePlannerOutput,
        )

        prompt = f"Analyze the following decision problem and dynamically select 3 to 8 relevant perspectives:\n\nDecision:\n{decision.strip()}"
        result = Runner.run_sync(planner_agent, prompt)

        if isinstance(result.final_output, PerspectivePlannerOutput):
            return result.final_output
        elif isinstance(result.final_output, dict):
            return PerspectivePlannerOutput.model_validate(result.final_output)
        elif isinstance(result.final_output, str):
            # Parse JSON string
            data = json.loads(result.final_output)
            return PerspectivePlannerOutput.model_validate(data)
        else:
            raise ValueError(f"Unexpected planner output format: {type(result.final_output)}")

    except Exception as e:
        print(f"[Perspective Planner] Warning: Live agent execution encountered ({e}). Using dynamic planner fallback.")
        return _generate_dynamic_demo_perspectives(decision)
