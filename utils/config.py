"""
DecisionLens AI - Configuration and Environment Settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env file if present
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_PERSIST_DIR = BASE_DIR / ".chroma_db"

# Core Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()

# DEMO_MODE setting: If true or if OPENAI_API_KEY is empty, falls back gracefully to demo simulation
DEMO_MODE_ENV = os.getenv("DEMO_MODE", "false").strip().lower()
DEFAULT_DEMO_MODE = DEMO_MODE_ENV in ("true", "1", "yes") or not OPENAI_API_KEY

def is_live_ai_available(api_key: str = None) -> bool:
    """Returns True if a valid-looking OpenAI API key is supplied."""
    key = api_key if api_key is not None else OPENAI_API_KEY
    return bool(key and len(key) > 10 and not key.startswith("your-") and not key.startswith("sk-..."))

def get_active_model(override_model: str = None) -> str:
    """Returns the active model name."""
    if override_model and override_model.strip():
        return override_model.strip()
    return OPENAI_MODEL or "gpt-4o-mini"
