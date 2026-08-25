"""
DecisionLens AI - Smoke Test and Verification Suite
Validates imports, Pydantic data schemas, RAG ingestion/retrieval,
LangGraph assembly, and end-to-end execution in Demo Mode without requiring API calls.
"""

import sys
import unittest
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.schemas import (
    PerspectiveItem,
    PerspectivePlannerOutput,
    PerspectiveAnalysisOutput,
    ResearchOutput,
    RAGOutput,
    JudgeOutput,
    ReviewerOutput,
    FullDecisionReport,
)
from utils.config import (
    KNOWLEDGE_BASE_DIR,
    OPENAI_MODEL,
    is_live_ai_available,
)
from rag.ingest import load_raw_knowledge_documents, get_split_chunks
from rag.retriever import retrieve_relevant_frameworks
from decision_agents.planner import plan_perspectives, PERSPECTIVE_CATALOG
from decision_agents.perspective import analyze_all_perspectives_parallel
from decision_agents.research import conduct_research
from decision_agents.judge import synthesize_decision
from decision_agents.reviewer import review_recommendation
from workflow.graph import build_decision_lens_graph, run_workflow_stream


class TestDecisionLensSmoke(unittest.TestCase):
    """Core verification tests for DecisionLens AI."""

    def test_01_knowledge_base_files_exist(self):
        """Verify all 7 core knowledge base markdown files exist and load properly."""
        docs = load_raw_knowledge_documents()
        self.assertGreaterEqual(len(docs), 7, "Expected at least 7 knowledge base markdown documents.")
        chunks = get_split_chunks()
        self.assertGreater(len(chunks), 10, "Expected document splitting to produce semantic chunks.")
        print(f"[OK] Knowledge Base: Loaded {len(docs)} documents ({len(chunks)} chunks).")

    def test_02_rag_retrieval(self):
        """Verify RAG retrieval returns structured RAGOutput with grounded passages."""
        query = "Should we adopt microservices for our core banking platform?"
        rag_res = retrieve_relevant_frameworks(query, perspectives=["Software Engineering", "Security"])
        self.assertIsInstance(rag_res, RAGOutput)
        self.assertGreaterEqual(len(rag_res.retrieved_docs), 1)
        self.assertTrue(bool(rag_res.summary))
        print(f"[OK] RAG Retriever: {rag_res.status} returned {len(rag_res.retrieved_docs)} docs.")

    def test_03_perspective_planner_dynamic(self):
        """Verify perspective planner dynamically produces 3-8 perspectives with schema validation."""
        test_questions = [
            "Should our university introduce AI-assisted assessment in engineering courses?",
            "Should our healthcare clinic migrate electronic health records to a multi-tenant cloud?",
            "Should our 50-person startup enforce a mandatory 5-day return-to-office policy?"
        ]
        for q in test_questions:
            res = plan_perspectives(q, is_demo=True)
            self.assertIsInstance(res, PerspectivePlannerOutput)
            self.assertGreaterEqual(len(res.perspectives), 3)
            self.assertLessEqual(len(res.perspectives), 8)
            self.assertTrue(bool(res.domain))
            self.assertTrue(bool(res.decision_summary))
        print(f"[OK] Perspective Planner: Successfully generated dynamic perspectives for {len(test_questions)} diverse domains.")

    def test_04_perspective_catalog(self):
        """Verify the catalog contains at least 30 diverse perspective categories."""
        self.assertGreaterEqual(len(PERSPECTIVE_CATALOG), 30)
        print(f"[OK] Perspective Catalog: Verified {len(PERSPECTIVE_CATALOG)} perspectives.")

    def test_05_langgraph_compilation(self):
        """Verify LangGraph StateGraph builds and compiles without errors."""
        graph = build_decision_lens_graph()
        self.assertIsNotNone(graph)
        print("[OK] LangGraph: StateGraph compiled successfully.")

    def test_06_end_to_end_workflow_demo_run(self):
        """Verify complete end-to-end multi-agent execution pipeline in Demo Mode."""
        decision_prompt = "Should our healthcare system deploy autonomous triage kiosks in urgent care centers?"
        
        final_state = None
        step_names = []
        for update in run_workflow_stream(decision_prompt, is_demo=True):
            step_names.append(update["step"])
            final_state = update["state"]

        self.assertIsNotNone(final_state)
        self.assertEqual(final_state.get("status"), "completed")
        self.assertGreaterEqual(len(final_state.get("selected_perspectives", [])), 3)
        self.assertGreaterEqual(len(final_state.get("perspective_results", [])), 3)
        self.assertTrue(bool(final_state.get("judge_result", {}).get("final_recommendation")))
        self.assertEqual(final_state.get("review_result", {}).get("status"), "APPROVED")
        
        print(f"[OK] End-to-End Workflow: Completed steps {step_names}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
