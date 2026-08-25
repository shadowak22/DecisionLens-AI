"""
DecisionLens AI - Dynamic Multi-Perspective Decision Intelligence
Main Streamlit Application Entrypoint
"""

import os
import json
import time
from typing import Dict, Any, List
import streamlit as st

from utils.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    DEFAULT_DEMO_MODE,
    is_live_ai_available,
    KNOWLEDGE_BASE_DIR,
)
from rag.ingest import load_raw_knowledge_documents
from workflow.graph import run_workflow_stream
from models.schemas import (
    PerspectivePlannerOutput,
    PerspectiveItem,
    PerspectiveAnalysisOutput,
    ResearchOutput,
    RAGOutput,
    JudgeOutput,
    ReviewerOutput,
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="DecisionLens AI | Multi-Perspective Decision Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Aesthetic CSS Styling
CUSTOM_CSS = """
<style>
/* Modern typography and dark theme accents */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hero Title Styling */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 1rem;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
}

.tech-badge {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.tech-badge-emerald {
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.tech-badge-purple {
    background: rgba(192, 132, 252, 0.12);
    color: #c084fc;
    border: 1px solid rgba(192, 132, 252, 0.3);
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Card Containers */
.dl-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.dl-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
}

.dl-card-highlight {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.25) 0%, rgba(17, 24, 39, 0.6) 100%);
    border: 1px solid rgba(96, 165, 250, 0.35);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.perspective-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.tag-high {
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.tag-medium {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.tag-low {
    background: rgba(59, 130, 246, 0.18);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

/* Status Indicator Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-approved {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-revised {
    background: rgba(234, 179, 8, 0.15);
    color: #facc15;
    border: 1px solid rgba(234, 179, 8, 0.3);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "decision_state" not in st.session_state:
    st.session_state.decision_state = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "human_status" not in st.session_state:
    st.session_state.human_status = None

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🧭 DecisionLens AI Control")
    st.caption("Dynamic Multi-Perspective Decision Intelligence")

    # Mode Selector
    live_possible = is_live_ai_available()
    mode_option = st.radio(
        "Execution Mode:",
        ["Live AI Mode (OpenAI API)", "Interactive Demo Mode (Deterministic)"],
        index=0 if live_possible else 1,
        help="Demo mode allows full workflow testing without requiring an active OpenAI key."
    )
    is_demo_mode = "Demo Mode" in mode_option

    # API Key Input
    if not is_demo_mode:
        api_key_input = st.text_input(
            "OpenAI API Key:",
            value=OPENAI_API_KEY,
            type="password",
            placeholder="sk-proj-...",
            help="Your API key is used in-memory for this session only and is never stored on disk."
        )
        effective_api_key = api_key_input.strip() if api_key_input else OPENAI_API_KEY
        if is_live_ai_available(effective_api_key):
            st.success("🟢 OpenAI API Key Configured")
        else:
            st.warning("⚠️ No valid API Key detected. Will fallback to Demo Mode if analyzed.")
    else:
        effective_api_key = None
        st.info("💡 Running in Interactive Demo Mode")

    # Model Configuration
    selected_model = st.selectbox(
        "LLM Model:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="Selected model for OpenAI Agents SDK orchestration."
    )

    st.markdown("---")

    # Sample Questions Quickloader
    st.markdown("#### ⚡ Quick Test Questions")
    st.caption("Click to load an arbitrary decision problem:")
    
    sample_queries = [
        "Should our college adopt AI-assisted assessment in engineering courses?",
        "Should our fintech startup transition our monolithic core to event-driven microservices?",
        "Should our healthcare clinic migrate electronic health records to multi-tenant cloud?",
        "Should our company mandate a 3-day in-office return policy?",
    ]

    for idx, sq in enumerate(sample_queries):
        if st.button(f"📌 {sq[:40]}...", key=f"btn_sample_query_{idx}", use_container_width=True):
            st.session_state.query_input = sq

    st.markdown("---")

    # Knowledge Base Inspector
    with st.expander("📚 Knowledge Base Files (RAG)", expanded=False):
        raw_docs = load_raw_knowledge_documents()
        st.write(f"**Total Indexed Documents:** {len(raw_docs)}")
        for d in raw_docs:
            src = d.metadata.get("source", "doc")
            st.markdown(f"- `{src}` ({len(d.page_content)} chars)")

    # Architecture & Gen AI Badges
    with st.expander("🧠 Gen AI Concepts Implemented", expanded=False):
        st.markdown("""
        - **LangGraph**: StateGraph Cyclic Multi-Agent Orchestration
        - **OpenAI Agents SDK**: Reusable Persona Agent & Runner
        - **ReAct Pattern**: Research Agent with grounded citations
        - **RAG Pipeline**: LangChain + Markdown + ChromaDB
        - **Dynamic Specialization**: Dynamic 3–8 perspective generation
        - **Self-Reflection**: Reviewer Agent with auto-remediation loop
        - **Human-in-the-Loop**: Active review & approval gates
        """)


# Main Application Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">DECISIONLENS AI</div>
    <div class="hero-subtitle">Dynamic Multi-Perspective Decision Intelligence System</div>
    <div class="badge-row">
        <span class="tech-badge">LangGraph Cyclic Graph</span>
        <span class="tech-badge-emerald">OpenAI Agents SDK</span>
        <span class="tech-badge-purple">LangChain ChromaDB RAG</span>
        <span class="tech-badge">ReAct Research</span>
        <span class="tech-badge-emerald">Self-Reflection Reviewer</span>
        <span class="tech-badge-purple">Human-in-the-Loop</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Decision Input Area
st.markdown("### 🎯 What decision would you like to analyze?")
decision_query = st.text_area(
    "Enter any strategic, technical, operational, or policy decision problem:",
    value=st.session_state.query_input,
    height=110,
    placeholder="e.g. Should our university introduce AI-assisted assessment in undergraduate STEM courses?",
    help="Type any decision. DecisionLens AI will dynamically evaluate the problem across tailored stakeholder viewpoints."
)

col_run, col_clear, col_status = st.columns([1.5, 1, 3.5])

with col_run:
    analyze_clicked = st.button("🚀 Analyze Decision", type="primary", use_container_width=True, disabled=st.session_state.is_running, key="btn_analyze_main")

with col_clear:
    if st.button("🧹 Clear State", use_container_width=True, key="btn_clear_main"):
        st.session_state.decision_state = None
        st.session_state.human_status = None
        st.session_state.query_input = ""
        st.rerun()


# Execution Trigger
if analyze_clicked and decision_query.strip():
    st.session_state.is_running = True
    st.session_state.human_status = None
    st.session_state.query_input = decision_query.strip()

    # Create real-time workflow status container
    status_container = st.status("🧭 Initializing Multi-Agent Decision Workflow...", expanded=True)
    
    with status_container:
        st.write("🔄 **Starting LangGraph StateGraph...**")
        
        # Step tracking dictionary
        progress_log = []
        
        # Run workflow generator
        for update in run_workflow_stream(
            decision=decision_query.strip(),
            api_key=effective_api_key,
            model=selected_model,
            is_demo=is_demo_mode or not is_live_ai_available(effective_api_key),
        ):
            step = update["step"]
            msg = update["message"]
            current_st = update["state"]

            if step == "planner":
                st.write("1️⃣ 🤖 **Perspective Planner Agent**: Formulating tailored stakeholder perspectives...")
            elif step == "planner_done":
                st.write(f"   ↳ *Identified Domain:* **{current_st.get('domain', 'General')}**")
                p_names = [p.get("name") for p in current_st.get("selected_perspectives", [])]
                st.write(f"   ↳ *Selected Perspectives ({len(p_names)}):* {', '.join(p_names)}")

            elif step == "research":
                st.write("2️⃣ 🔍 **ReAct Research Agent**: Querying external benchmarks and empirical data...")
            elif step == "research_done":
                findings_count = len(current_st.get("research", {}).get("key_findings", []))
                st.write(f"   ↳ *Synthesized {findings_count} grounded research findings.*")

            elif step == "rag":
                st.write("3️⃣ 📚 **RAG Knowledge Agent**: Performing semantic retrieval on local decision frameworks...")
            elif step == "rag_done":
                docs_count = len(current_st.get("rag_context", {}).get("retrieved_docs", []))
                st.write(f"   ↳ *Retrieved {docs_count} relevant framework passages.*")

            elif step == "perspective_analysis":
                st.write("4️⃣ 👥 **Perspective Analyst Agents**: Executing parallel specialist evaluations...")
            elif step == "perspective_analysis_done":
                st.write(f"   ↳ *Completed deep stakeholder analyses across all {len(current_st.get('perspective_results', []))} perspectives.*")

            elif step == "judge":
                st.write("5️⃣ ⚖️ **Judge Agent**: Synthesizing consensus, trade-offs, and decision verdict...")
            elif step == "judge_done":
                st.write("   ↳ *Decision synthesis and roadmap created.*")

            elif step == "reviewer":
                st.write("6️⃣ 🪞 **Self-Reflection Reviewer**: Auditing evidence, contradictions, and calibration...")
            elif step == "revision_triggered":
                st.warning(f"   ⚠️ *Self-Reflection Triggered Revision Loop: Refined synthesis requested.*")
            elif step == "reviewer_done":
                rev_status = current_st.get("review_result", {}).get("status", "APPROVED")
                st.write(f"   ↳ *Reviewer Audit Complete: **{rev_status}**.*")

            elif step == "finished":
                st.write("7️⃣ ✅ **Decision Intelligence Report Compiled Successfully!**")

        status_container.update(label="🎯 Decision Analysis Complete!", state="complete", expanded=False)
        st.session_state.decision_state = current_st
        st.session_state.is_running = False
        st.rerun()


# ==========================================
# RESULT PRESENTATION SECTION
# ==========================================
state = st.session_state.decision_state

if state and state.get("status") == "completed":
    st.markdown("---")

    # Mode Banner
    if state.get("is_demo"):
        st.info("ℹ️ **Interactive Demo Mode Active**: Displaying grounded multi-agent synthesis generated from structured domain heuristics.")
    else:
        st.success("🟢 **Live AI Analysis**: Generated using OpenAI Agents SDK with OpenAI foundation models and ChromaDB vector retrieval.")

    # A. Decision Summary & Domain
    st.markdown("## 📊 Decision Intelligence Report")
    
    col_summary, col_domain = st.columns([3, 1])
    with col_summary:
        st.markdown(f"**Decision Dilemma:** *{state.get('decision')}*")
        st.markdown(f"**Summary:** {state.get('decision_summary')}")
    with col_domain:
        st.markdown("**Identified Domain:**")
        st.markdown(f"`{state.get('domain', 'Strategic Governance')}`")

    # B. Selected Perspectives
    st.markdown("### 👥 B. Dynamically Selected Perspectives")
    st.caption("The Perspective Planner dynamically chose these perspectives specifically for this decision without hardcoded lists:")

    perspectives = state.get("selected_perspectives", [])
    cols_per_row = 3
    for i in range(0, len(perspectives), cols_per_row):
        row_perspectives = perspectives[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, p in enumerate(row_perspectives):
            with cols[idx]:
                priority = p.get("priority", "high").lower()
                tag_class = f"tag-{priority}"
                st.markdown(f"""
                <div class="dl-card">
                    <span class="perspective-tag {tag_class}">{priority} priority</span>
                    <h4 style="margin: 0.2rem 0 0.5rem 0; color: #38bdf8;">{p.get('name')}</h4>
                    <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Focus:</strong> {p.get('focus')}</p>
                    <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;"><strong>Rationale:</strong> {p.get('reason_selected')}</p>
                </div>
                """, unsafe_allow_html=True)

    # C. Deep Perspective Analyses
    st.markdown("### 🔬 C. Deep Perspective Analyses (Specialist Agents)")
    st.caption("One reusable OpenAI Specialist Agent dynamically assumed each persona and evaluated the problem in parallel:")

    perspective_results = state.get("perspective_results", [])
    tab_titles = [f"👤 {p.get('perspective_name', f'P{i+1}')}" for i, p in enumerate(perspective_results)]
    
    if perspective_results:
        tabs = st.tabs(tab_titles)
        for idx, tab in enumerate(tabs):
            p_data = perspective_results[idx]
            with tab:
                st.markdown(f"#### Stakeholder Viewpoint: **{p_data.get('perspective_name')}**")
                st.info(f"💡 **Core Viewpoint:** {p_data.get('viewpoint')}")
                
                col_b, col_c = st.columns(2)
                with col_b:
                    st.markdown("**✨ Key Benefits & Opportunities:**")
                    for b in p_data.get("benefits", []):
                        st.markdown(f"- {b}")
                with col_c:
                    st.markdown("**⚠️ Concerns & Friction Points:**")
                    for c in p_data.get("concerns", []):
                        st.markdown(f"- {c}")

                col_r, col_a = st.columns(2)
                with col_r:
                    st.markdown("**🚨 Critical Risks:**")
                    for r in p_data.get("risks", []):
                        st.markdown(f"- {r}")
                with col_a:
                    st.markdown("**📌 Key Assumptions:**")
                    for a in p_data.get("assumptions", []):
                        st.markdown(f"- {a}")

                st.markdown(f"**🎯 Stakeholder Recommendation:** *{p_data.get('recommendation')}*")
                
                conf = p_data.get("confidence", 0.8)
                st.progress(conf, text=f"Perspective Analytical Confidence: {conf * 100:.0f}%")

    # D. Evidence Explorer (RAG vs External Research)
    st.markdown("### 📑 D. Evidence Explorer")
    tab_rag, tab_research = st.tabs(["📚 Local RAG Knowledge Retrieval", "🌐 External Research & Industry Benchmarks"])

    with tab_rag:
        rag_data = state.get("rag_context", {})
        st.markdown(f"**Retrieval Pipeline Status:** `{rag_data.get('status', 'Retrieved')}`")
        st.write(rag_data.get("summary", "Knowledge base queried."))
        
        docs = rag_data.get("retrieved_docs", [])
        if docs:
            for d in docs:
                with st.expander(f"📖 Source: {d.get('source_file')}"):
                    st.markdown(d.get("content", ""))
        else:
            st.info("No explicit local markdown passages retrieved.")

    with tab_research:
        res_data = state.get("research", {})
        st.markdown(f"**Research Agent Status:** `{res_data.get('status', 'Completed')}`")
        st.write(res_data.get("summary", "External research performed."))
        
        findings = res_data.get("key_findings", [])
        for f in findings:
            st.markdown(f"""
            <div class="dl-card" style="border-left: 3px solid #38bdf8;">
                <h5 style="margin: 0 0 0.3rem 0; color: #e2e8f0;">{f.get('topic')}</h5>
                <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.4rem;">{f.get('finding')}</p>
                <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;"><strong>Citation:</strong> <em>{f.get('source')}</em> | <strong>Relevance:</strong> {f.get('relevance')}</p>
            </div>
            """, unsafe_allow_html=True)

    # E. Agreement & Disagreement Breakdown
    judge_data = state.get("judge_result", {})
    st.markdown("### ⚖️ E. Consensus and Tension Matrix")
    col_agr, col_disagr = st.columns(2)

    with col_agr:
        st.markdown("#### 🤝 Consensus Points (Agreements)")
        for item in judge_data.get("agreements", []):
            st.markdown(f"- ✅ {item}")

    with col_disagr:
        st.markdown("#### ⚡ Friction Points & Disagreements")
        for item in judge_data.get("disagreements", []):
            st.markdown(f"- ⚠️ {item}")

    # F. Final Recommendation & G. Confidence
    st.markdown("### 🎯 F. Final Synthesized Recommendation")
    
    conf_score = judge_data.get("confidence_score", 0.85)
    st.markdown(f"""
    <div class="dl-card-highlight">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #60a5fa;">Judge Verdict & Strategic Directive</h3>
            <span class="status-pill status-approved" style="font-size: 1rem;">
                Confidence: {conf_score * 100:.0f}%
            </span>
        </div>
        <p style="font-size: 1.15rem; line-height: 1.6; color: #f8fafc; margin-bottom: 1rem;">
            {judge_data.get('final_recommendation', 'Recommendation synthesized.')}
        </p>
        <p style="font-size: 0.85rem; color: #93c5fd; margin: 0;">
            <strong>Evidence Sufficiency Assessment:</strong> {judge_data.get('evidence_sufficiency', 'Validated across multi-source evidence.')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # H. Major Risks & Trade-offs
    col_tr, col_mr = st.columns(2)
    with col_tr:
        st.markdown("### 🔄 Strategic Trade-offs")
        for to in judge_data.get("trade_offs", []):
            st.markdown(f"- ⚖️ {to}")

    with col_mr:
        st.markdown("### 🚨 Major Risks & Vulnerabilities")
        for mr in judge_data.get("major_risks", []):
            st.markdown(f"- 🛑 {mr}")

    # I. Actionable Next Steps
    st.markdown("### 🗺️ I. Recommended Action Roadmap")
    for ns in judge_data.get("next_steps", []):
        st.markdown(f"- {ns}")

    # J. Reviewer Self-Reflection Result
    review_data = state.get("review_result", {})
    st.markdown("### 🪞 J. Self-Reflection Reviewer Audit Trail")
    
    rev_status = review_data.get("status", "APPROVED")
    status_pill_class = "status-approved" if rev_status == "APPROVED" else "status-revised"
    
    st.markdown(f"""
    <div class="dl-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
            <h4 style="margin: 0; color: #c084fc;">Metacognitive Self-Reflection Verification</h4>
            <span class="status-pill {status_pill_class}">{rev_status}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem; font-size: 0.9rem;">
            <div>{'✅' if review_data.get('evidence_checked') else '❌'} <strong>Evidence Grounding Checked</strong></div>
            <div>{'✅' if review_data.get('contradictions_checked') else '❌'} <strong>Contradictions Reconciled</strong></div>
            <div>{'✅' if review_data.get('missing_perspectives_checked') else '❌'} <strong>Missing Angles Audited</strong></div>
            <div>{'✅' if review_data.get('recommendation_quality_checked') else '❌'} <strong>Confidence Calibrated</strong></div>
        </div>
        <p style="font-size: 0.9rem; color: #e2e8f0; margin-bottom: 0.3rem;">
            <strong>Reviewer Critique:</strong> {review_data.get('critique', 'Passed self-reflection checks.')}
        </p>
        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">
            <strong>Revision Iterations Required:</strong> {state.get('revision_count', 0)} / 2
        </p>
    </div>
    """, unsafe_allow_html=True)

    # K. Human-in-the-Loop Approval Section
    st.markdown("### 👤 K. Human Approval & Oversight Gate")
    st.caption("As part of the Human-in-the-Loop (HITL) governance pattern, please review the final decision recommendation:")

    if st.session_state.human_status == "approved":
        st.success("🎉 **Recommendation Approved by Human Decision Maker!** The decision report is finalized.")
    elif st.session_state.human_status == "reanalyze":
        st.info("🔄 **Human Feedback Triggered Re-analysis**: You can re-run the workflow with refined parameters.")

    col_app, col_re, col_exp = st.columns([1.5, 1.5, 2])
    
    with col_app:
        if st.button("✅ Approve Recommendation", use_container_width=True, key="btn_approve_rec"):
            st.session_state.human_status = "approved"
            st.session_state.decision_state["human_approved"] = True
            st.rerun()

    with col_re:
        if st.button("🔄 Re-analyze Decision", use_container_width=True, key="btn_reanalyze_rec"):
            st.session_state.human_status = "reanalyze"
            st.session_state.decision_state["human_approved"] = False
            st.rerun()

    with col_exp:
        # Downloadable Report
        report_json = json.dumps(state, indent=2, default=str)
        st.download_button(
            label="📥 Export Report (JSON)",
            data=report_json,
            file_name=f"DecisionLens_Report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
            key="btn_download_report_json",
        )

# Footer
st.markdown("---")
st.caption("🧭 **DecisionLens AI** — Dynamic Multi-Perspective Decision Intelligence Capstone | Built with LangGraph, OpenAI Agents SDK, LangChain, ChromaDB & Streamlit.")
