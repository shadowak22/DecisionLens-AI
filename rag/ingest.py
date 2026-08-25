"""
DecisionLens AI - RAG Knowledge Base Ingestion Pipeline
Loads markdown files from knowledge_base/, splits into semantic chunks, and builds ChromaDB vector store.
"""

import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    is_live_ai_available,
)

# Global in-memory cache of raw documents for offline fallback
_LOCAL_DOC_CHUNKS: List[Document] = []
_CHROMA_VECTOR_STORE = None


def load_raw_knowledge_documents() -> List[Document]:
    """Reads all .md files in knowledge_base/ directory."""
    documents: List[Document] = []
    if not KNOWLEDGE_BASE_DIR.exists():
        return documents

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path.name, "file_path": str(file_path)},
                )
                documents.append(doc)
        except Exception as e:
            print(f"[RAG Ingest] Warning: could not read {file_path.name}: {e}")

    return documents


def get_split_chunks(chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
    """Splits knowledge base documents into chunks."""
    global _LOCAL_DOC_CHUNKS
    if _LOCAL_DOC_CHUNKS:
        return _LOCAL_DOC_CHUNKS

    raw_docs = load_raw_knowledge_documents()
    if not raw_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    _LOCAL_DOC_CHUNKS = splitter.split_documents(raw_docs)
    return _LOCAL_DOC_CHUNKS


def initialize_chroma_vectorstore(api_key: str = None):
    """
    Initializes or loads the persistent Chroma vector store with OpenAI Embeddings.
    Falls back gracefully if embeddings or vector store cannot be built.
    """
    global _CHROMA_VECTOR_STORE
    if _CHROMA_VECTOR_STORE is not None:
        return _CHROMA_VECTOR_STORE

    chunks = get_split_chunks()
    if not chunks:
        return None

    active_key = api_key or OPENAI_API_KEY
    if not is_live_ai_available(active_key):
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=active_key,
        )

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name="decisionlens_knowledge",
        )
        _CHROMA_VECTOR_STORE = vector_store
        return _CHROMA_VECTOR_STORE
    except Exception as e:
        print(f"[RAG Ingest] Note: Chroma vector store initialization skipped/failed ({e}). Using local document index fallback.")
        return None
