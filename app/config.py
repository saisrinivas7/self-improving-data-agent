"""
Central configuration, loaded from .env and validated at import time.

Design note on having two LLM providers:

The spec's Rule 6 says not to use multiple LLM providers unless necessary.
It became necessary. Gemini's free tier allows 20 requests per day on
gemini-3.8-flash, while one Learning Lab cycle costs about 10 calls and the
three-way benchmark needs roughly 3,000. So the default provider is a local
Ollama model, which has no quota at all.

This is deliberately a thin switch, not a plugin framework. One enum, two
small adapter methods in app/llm.py, and the same prompt text either way.
The point is that the benchmark can be re-run against a hosted model to
check whether the findings hold, without touching agent code.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(StrEnum):
    OLLAMA = "ollama"
    GEMINI = "gemini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- provider selection ----------------
    llm_provider: LLMProvider = LLMProvider.OLLAMA

    # ---------------- Ollama (default, no quota) ----------------
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "granite4.2:8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"

    # ---------------- Gemini (optional fallback) ----------------
    gemini_api_key: str = ""
    gemini_model: str = ""
    gemini_embedding_model: str = ""

    # ---------------- embeddings ----------------
    # Baked into feedback_memory.embedding as vector(N). Changing it means
    # dropping the column and re-embedding every stored lesson, so it is
    # validated against the live model by `make verify-llm`.
    embedding_dim: int = 1024

    # ---------------- database ----------------
    database_url: str
    database_url_ro: str

    # ---------------- agent engine ----------------
    # "graph"      the LangGraph state machine in app/agent/graph.py
    # "sequential" the plain function pipeline in app/agent/baseline.py
    #
    # Both share every tool and prompt, so they should produce the same
    # answers. The switch exists so the graph can be validated against the
    # pipeline it replaces, rather than being trusted because it compiles.
    agent_engine: str = "sequential"

    # ---------------- agent limits ----------------
    max_sql_attempts: int = 3
    sql_row_limit: int = 5000
    sql_timeout_ms: int = 15000
    feedback_top_k: int = 5

    # ---------------- determinism ----------------
    llm_temperature: float = 0.0
    data_seed: int = 42

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chat_model(self) -> str:
        """The chat model actually in use, whichever provider is selected."""
        return (
            self.ollama_model
            if self.llm_provider is LLMProvider.OLLAMA
            else self.gemini_model
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embedding_model(self) -> str:
        """The embedding model actually in use.

        Recorded on every stored memory row, because comparing vectors from
        two different embedding models is meaningless even when the
        dimensions match.
        """
        return (
            self.ollama_embedding_model
            if self.llm_provider is LLMProvider.OLLAMA
            else self.gemini_embedding_model
        )

    @model_validator(mode="after")
    def _check_provider_configured(self) -> "Settings":
        if self.llm_provider is LLMProvider.GEMINI:
            missing = [
                name
                for name, val in (
                    ("GEMINI_API_KEY", self.gemini_api_key),
                    ("GEMINI_MODEL", self.gemini_model),
                    ("GEMINI_EMBEDDING_MODEL", self.gemini_embedding_model),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    f"LLM_PROVIDER=gemini but these are unset in .env: {', '.join(missing)}"
                )
        if self.embedding_dim > 2000:
            raise ValueError(
                f"EMBEDDING_DIM={self.embedding_dim} exceeds pgvector's 2000-dimension "
                "index limit; similarity search could never use an index"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Import this rather than instantiating Settings()."""
    return Settings()  # type: ignore[call-arg]
