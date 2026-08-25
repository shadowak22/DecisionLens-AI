# 🧭 DECISIONLENS AI
### Dynamic Multi-Perspective Decision Intelligence System

> **A Capstone Project in Generative AI Decision Systems**  
> *Autonomous multi-agent orchestration, dynamic stakeholder specialization, localized RAG retrieval, ReAct empirical research, metacognitive self-reflection, and human-in-the-loop governance.*

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Logical Agent Responsibilities](#3-logical-agent-responsibilities)
4. [Dynamic Perspective Selection Mechanism](#4-dynamic-perspective-selection-mechanism)
5. [Local RAG Knowledge Pipeline (LangChain + ChromaDB)](#5-local-rag-knowledge-pipeline)
6. [LangGraph Cyclic Orchestration Workflow](#6-langgraph-cyclic-orchestration-workflow)
7. [ReAct External Research Workflow](#7-react-external-research-workflow)
8. [Self-Reflection and Metacognitive Reviewer](#8-self-reflection-and-metacognitive-reviewer)
9. [Human-in-the-Loop (HITL) Governance](#9-human-in-the-loop-hitl-governance)
10. [Gen AI Concepts Matrix](#10-gen-ai-concepts-matrix)
11. [Local Setup & Installation](#11-local-setup--installation)
12. [API Key Configuration](#12-api-key-configuration)
13. [Running the Application](#13-running-the-application)
14. [Testing & Verification](#14-testing--verification)
15. [Deployment to Streamlit Community Cloud](#15-deployment-to-streamlit-community-cloud)
16. [Example Test Decisions](#16-example-test-decisions)

---

## 1. Executive Summary

Traditional AI decision tools often suffer from single-persona bias, superficial averaging, or rigid, hardcoded role classifications. **DecisionLens AI** solves this by implementing a **dynamic multi-agent decision intelligence architecture** that accepts **any arbitrary decision dilemma** (spanning education, healthcare, cloud architecture, organizational strategy, and finance).

Instead of forcing users to pick viewpoints, DecisionLens AI:
1. **Dynamically deduces 3 to 8 relevant stakeholder and analytical perspectives** tailored specifically to the problem.
2. **Retrieves localized decision frameworks** from a Markdown-backed RAG pipeline using LangChain and ChromaDB.
3. **Conducts empirical external research** using a ReAct-style research agent.
4. **Executes parallel deep analyses** across perspectives using a single reusable OpenAI specialist agent.
5. **Synthesizes trade-offs, consensus, and friction** into an authoritative recommendation via a Judge Agent.
6. **Audits the recommendation** through an independent Self-Reflection Reviewer Agent that can trigger self-correction loops.
7. **Presents a Human-in-the-Loop approval gate** for executive sign-off.

---

## 2. End-to-End System Architecture

```
                                USER DECISION QUERY
                                         │
                                         ▼
                            PERSPECTIVE PLANNER AGENT
                      (Dynamically selects 3–8 perspectives)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
        RAG KNOWLEDGE AGENT                            ReAct RESEARCH AGENT
   (ChromaDB + Local Markdown)                      (WebSearch / Benchmarks)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                           PERSPECTIVE ANALYSTS (Parallel)
                     (1 Reusable Dynamic OpenAI Specialist Agent)
                                         │
                                         ▼
                                    JUDGE AGENT
                     (Consensus, Friction, Trade-offs, Verdict)
                                         │
                                         ▼
                             SELF-REFLECTION REVIEWER
                    (Evidence, Contradictions, Calibration Audit)
                                         │
                      ┌──────────────────┴──────────────────┐
             [NEEDS_REVISION & count < 2]               [APPROVED]
                      │                                     │
                      ▼                                     ▼
                (Revise Judge)                      HUMAN APPROVAL GATE
                                                  (Approve / Re-analyze)
                                                            │
                                                            ▼
                                                   FINAL DECISION REPORT
```

---

## 3. Logical Agent Responsibilities

| Agent | Technology | Primary Mandate |
| :--- | :--- | :--- |
| **1. Perspective Planner** | OpenAI Agents SDK / LLM | Analyzes the decision query and dynamically selects 3–8 distinct perspectives from a 44+ domain catalog or generates specialized roles. |
| **2. ReAct Researcher** | OpenAI Agents SDK + Web Tools | Gathers empirical benchmarks, regulatory precedents, and industry statistics using a ReAct search loop. |
| **3. RAG Knowledge Agent** | LangChain + ChromaDB | Indexes and retrieves passages from local decision framework Markdown documents. |
| **4. Perspective Analyst** | OpenAI Agents SDK (`Agent` + `Runner`) | ONE reusable specialist agent that dynamically assumes any stakeholder persona (Finance, Security, Student, Engineering, Ethics, etc.) and runs concurrently. |
| **5. Decision Judge** | OpenAI Agents SDK | Synthesizes multi-perspective analyses, RAG frameworks, and research into consensus, friction, major risks, trade-offs, and a calibrated recommendation. |
| **6. Self-Reflection Reviewer**| OpenAI Agents SDK | Performs independent metacognitive checks on evidence grounding, contradictions, missing angles, and calibration, triggering revisions if needed (max 2 iterations). |
| **7. Human Approval Gate** | Streamlit HITL Interface | Provides active executive oversight to approve the recommendation or trigger re-analysis. |

---

## 4. Dynamic Perspective Selection Mechanism

DecisionLens AI avoids hardcoded question routing. When given a query like *"Should our college adopt AI-assisted assessment?"* or *"Should our fintech migrate to microservices?"*, the **Perspective Planner Agent** evaluates the problem semantics and selects between 3 and 8 perspectives.

The system maintains a broad reference catalog of over 40 domains:
- **Technical & Architecture**: Software Engineering, Cloud, Infrastructure, Database, Performance, Scalability, AI/ML, Data.
- **Security & Governance**: Security, Privacy, Legal, Compliance, Ethics, Safety, Accessibility, Policy, Regulatory.
- **Organizational & Human**: Student, Faculty, Customer, Employee, Manager, Executive, HR, Education, Healthcare.
- **Business & Financial**: Finance, Cost-Benefit, Product, Marketing, Sales, Operations, Risk, Market Analysis.

---

## 5. Local RAG Knowledge Pipeline

The local RAG knowledge base provides structured theoretical grounding for all analyses:

```
knowledge_base/
├── decision_frameworks.md     # MCDA, Cynefin, Six Thinking Hats, Type 1 vs 2
├── cost_benefit.md            # TCO, NPV, IRR, Opportunity Cost, OpEx vs CapEx
├── risk_management.md         # 5x5 Matrix, FMEA, 4 T's of Risk, Blast Radius
├── technology_evaluation.md   # Build vs Buy, Tech Debt, Vendor Lock-in, SLOs
├── privacy_ethics.md          # GDPR, CCPA, Algorithmic Bias, PII Scrubbing
├── ai_adoption.md             # Gen AI Maturity, HITL Patterns, Kotter's Change
└── software_architecture.md   # Modular Monoliths, Microservices, Observability
```

**Ingestion & Retrieval Workflow:**
1. Files in `knowledge_base/` are loaded via LangChain document loaders.
2. `RecursiveCharacterTextSplitter` segments content into semantic chunks (700 chars, 100 overlap).
3. `OpenAIEmbeddings` (`text-embedding-3-small`) vectors are stored in a local ChromaDB collection (`.chroma_db/`).
4. Queries execute semantic similarity retrieval with top-$k$ ranking and a graceful in-memory keyword fallback.

---

## 6. LangGraph Cyclic Orchestration Workflow

The orchestration layer is implemented with a **LangGraph StateGraph** that manages shared state across nodes:

```python
class DecisionState(TypedDict):
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
    human_approved: Optional[bool]
```

**Conditional Routing:**
- After `reviewer` executes, the conditional edge `should_revise` checks:
  - If `status == "NEEDS_REVISION"` and `revision_count < 2`: routes back to `judge` with revision instructions.
  - If `status == "APPROVED"` or `revision_count >= 2`: routes to `END`.

---

## 7. ReAct External Research Workflow

The Research Agent follows the **ReAct (Reason + Act)** pattern:
1. **Thought**: Deconstruct the decision dilemma into empirical information requirements.
2. **Action**: Query external sources for industry statistics, adoption trends, and benchmarks.
3. **Observation**: Inspect returned evidence and evaluate relevance.
4. **Synthesis**: Produce grounded research findings with source citations and relevance mappings.

*Note: Safe high-level activity is surfaced to the user ("Searching for evidence", "Evaluating sources") without exposing raw hidden chain-of-thought.*

---

## 8. Self-Reflection and Metacognitive Reviewer

The Reviewer Agent acts as an independent quality auditor:
- **Evidence Verification**: Ensures claims reference explicit RAG frameworks or empirical research.
- **Contradiction Detection**: Reconciles unaddressed trade-offs between stakeholders.
- **Missing Angle Detection**: Confirms critical viewpoints were not overlooked.
- **Confidence Calibration**: Verifies that confidence is proportionate to available evidence.

---

## 9. Human-in-the-Loop (HITL) Governance

After the multi-agent workflow completes, DecisionLens AI presents an active Human Approval Gate:
- **✅ Approve Recommendation**: Finalizes the decision report for executive distribution.
- **🔄 Re-analyze Decision**: Allows the user to trigger another analysis iteration with adjusted parameters.
- **📥 Export Report**: Exports the complete decision intelligence artifact in JSON format.

---

## 10. Gen AI Concepts Matrix

| Gen AI Training Concept | Concrete Implementation in DecisionLens AI |
| :--- | :--- |
| **Prompt Engineering** | Structured system prompts with role constraints and Pydantic validation schemas. |
| **ReAct Pattern** | `agents/research.py` iterative query-action-grounding workflow. |
| **RAG (Retrieval-Augmented Generation)** | LangChain Markdown loaders + `RecursiveCharacterTextSplitter` + ChromaDB vector store. |
| **Vector Embeddings** | `OpenAIEmbeddings` (`text-embedding-3-small`) semantic similarity retrieval. |
| **LangGraph Multi-Agent Orchestration** | StateGraph with shared state, node functions, and conditional loop edges. |
| **OpenAI Agents SDK** | `Agent` and `Runner.run_sync` with structured Pydantic `output_type`. |
| **Dynamic Specialization** | Single reusable `Perspective Analyst` assuming dynamic stakeholder personas. |
| **Self-Reflection & Self-Correction**| `Reviewer Agent` auditing output and routing back to `Judge` for revision. |
| **Human-in-the-Loop (HITL)** | Interactive Streamlit approval gates and re-analysis triggers. |

---

## 11. Local Setup & Installation

### Prerequisites
- Python 3.10 to 3.13
- Git

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DecisionLens-AI.git
cd DecisionLens-AI

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 12. API Key Configuration

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and insert your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-proj-your-openai-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   DEMO_MODE=false
   ```

> [!NOTE]
> You can also enter or override your OpenAI API key dynamically in the Streamlit sidebar UI at runtime. If no API key is provided, the application runs seamlessly in **Interactive Demo Mode**.

---

## 13. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 14. Testing & Verification

Run the automated smoke test suite (validates imports, schemas, RAG indexing, LangGraph compilation, and demo workflow):

```bash
python tests/test_smoke.py
```

---

## 15. Deployment to Streamlit Community Cloud

DecisionLens AI is designed for single-command zero-configuration deployment to [Streamlit Community Cloud](https://share.streamlit.io):

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of DecisionLens AI"
   git remote add origin https://github.com/<your-username>/DecisionLens-AI.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
   - Select your repository, branch (`main`), and set **Main file path** to `app.py`.
3. **Configure Secrets**:
   - In Streamlit Cloud App Settings, navigate to **Secrets** and add:
     ```toml
     OPENAI_API_KEY = "sk-proj-..."
     OPENAI_MODEL = "gpt-4o-mini"
     ```
   - Click **Save & Deploy**.

---

## 16. Example Test Decisions

Test DecisionLens AI with diverse real-world decisions across domains:

1. **Higher Education**:
   > *"Should our university introduce AI-assisted assessment in undergraduate engineering courses?"*
2. **Software Architecture**:
   > *"Should our fintech startup decompose our monolithic billing system into event-driven microservices?"*
3. **Healthcare Operations**:
   > *"Should our regional hospital network deploy autonomous AI triage kiosks in urgent care emergency rooms?"*
4. **Organizational Strategy**:
   > *"Should our 200-person technology company mandate a 3-day weekly in-office return policy?"*
5. **Product & Privacy**:
   > *"Should our consumer mobile app integrate facial recognition for one-touch biometric checkout?"*

---

*DecisionLens AI — Transforming complex organizational dilemmas into structured, multi-perspective decision intelligence.*
