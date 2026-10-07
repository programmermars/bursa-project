"""Central settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # Data and storage
    raw_dir: Path = field(default_factory=lambda: Path(_env("RAG_RAW_DIR", str(ROOT / "data" / "raw"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("RAG_INDEX_DIR", str(ROOT / "index"))))

    # Chunking
    chunk_size: int = int(_env("RAG_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "150"))
    # Prefix each chunk with "company, year, section" before indexing (cheap contextual retrieval)
    context_prefix: bool = _env("RAG_CONTEXT_PREFIX", "1") not in ("0", "false", "False")

    # Embeddings: ollama | hf | tfidf
    embed_backend: str = _env("EMBED_BACKEND", "ollama")
    ollama_embed_model: str = _env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    hf_embed_model: str = _env("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval: vector | bm25 | hybrid
    retrieval_mode: str = _env("RETRIEVAL_MODE", "hybrid")
    top_k: int = int(_env("TOP_K", "5"))

    # Generation: ollama | groq | none
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    ollama_llm_model: str = _env("OLLAMA_LLM_MODEL", "qwen2.5:7b")
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature: float = float(_env("LLM_TEMPERATURE", "0"))


def get_settings() -> Settings:
    return Settings()
