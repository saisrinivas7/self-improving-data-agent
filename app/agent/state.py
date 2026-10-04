"""
The agent's state object (spec section 9).

One object carries everything through the run: the question in, the answer
out, and every intermediate step. Two reasons it is this explicit:

  - LangGraph (Phase 3) passes exactly this between nodes, so defining it
    now means the baseline and the graph share one shape.
  - The trace, the benchmark metrics and the Learning Lab all read from it.
    When an analyst gives feedback, what they are correcting is the content
    of this object, so it has to record enough to explain itself: which SQL
    was tried, which failed and why, which lessons were retrieved and which
    were actually used.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SystemVariant = Literal["baseline", "feedback_rag", "verified_feedback"]


@dataclass
class SQLAttempt:
    """One try at writing and running SQL, successful or not."""

    attempt: int
    sql: str
    ok: bool
    error: str = ""
    row_count: int = 0
    latency_s: float = 0.0


@dataclass
class RetrievedLesson:
    """A feedback memory that retrieval returned, and what we did with it."""

    feedback_id: str
    lesson: str
    feedback_type: str
    status: str
    confidence: float
    similarity: float
    # Set by the re-ranking step in the verified system. The baseline and
    # plain RAG leave it empty.
    used: bool = False
    rejected_reason: str = ""


@dataclass
class AgentState:
    question: str
    system_variant: SystemVariant = "baseline"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    snapshot: str = "live"

    # ---- what the tools produced ----
    schema_text: str = ""
    tables_used: list[str] = field(default_factory=list)
    generated_sql: str = ""
    sql_result_rows: list[dict] = field(default_factory=list)
    sql_result_table: str = ""
    analysis_text: str = ""
    analysis_findings: list[dict] = field(default_factory=list)
    final_answer: str = ""

    # ---- feedback (empty for the baseline, by construction) ----
    retrieved_lessons: list[RetrievedLesson] = field(default_factory=list)

    # ---- bookkeeping ----
    sql_attempts: list[SQLAttempt] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "running"

    chat_model: str = ""
    embedding_model: str = ""

    # ---------------------------------------------------------- helpers

    @property
    def sql_attempt_count(self) -> int:
        return len(self.sql_attempts)

    @property
    def lessons_used(self) -> list[RetrievedLesson]:
        return [l for l in self.retrieved_lessons if l.used]

    @property
    def lessons_rejected(self) -> list[RetrievedLesson]:
        return [l for l in self.retrieved_lessons if not l.used]

    def record_llm(self, resp: Any) -> None:
        """Accumulate token and call counts from one LLM response."""
        self.llm_calls += 1
        self.prompt_tokens += getattr(resp, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(resp, "output_tokens", 0) or 0
        self.thinking_tokens += getattr(resp, "thinking_tokens", 0) or 0

    def summary(self) -> dict:
        """Flat dict for the trace table and benchmark rows."""
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "system_variant": self.system_variant,
            "snapshot": self.snapshot,
            "status": self.status,
            "final_answer": self.final_answer,
            "generated_sql": self.generated_sql,
            "sql_attempts": self.sql_attempt_count,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "latency_s": round(self.latency_s, 3),
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "tables_used": self.tables_used,
            "errors": self.errors,
            "feedback_retrieved": [l.feedback_id for l in self.retrieved_lessons],
            "feedback_used": [l.feedback_id for l in self.lessons_used],
            "feedback_rejected": [l.feedback_id for l in self.lessons_rejected],
        }
