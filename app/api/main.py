"""
HTTP API (spec section 25).

    make run        ->  http://localhost:8000/docs

    POST /analyze           ask a question
    POST /feedback          submit a correction
    GET  /feedback/{id}     one stored lesson, with its verification trail
    GET  /feedback          browse memory
    GET  /trace/{id}        the full step-by-step trace of a run
    GET  /health            readiness of each dependency

Deliberately a thin layer. Every endpoint delegates to the same functions
the Learning Lab and the benchmark use, so there is no behaviour that only
exists over HTTP - a third code path would be a third thing that can
disagree with the other two.

Two notes on the design:

  Requests are SLOW. Local inference means /analyze takes 30-90 seconds.
  That is stated in the response model rather than hidden, and /health
  reports whether the model is even loaded, because "it hung" and "the
  model is not running" look identical to a client otherwise.

  /feedback returns the full verification trail, not just a status. A caller
  that is told its feedback was REJECTED should be able to see which claim
  failed and against which measured number, or the system is just saying no.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings

app = FastAPI(
    title="Self-Improving Data Analyst Agent",
    description=(
        "Answers business questions with SQL and improves from verified "
        "human corrections held in a persistent pgvector memory. The model "
        "is never retrained; what changes is what reaches its prompt."
    ),
    version="0.1.0",
)


# ------------------------------------------------------------------ models

class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Why did revenue decrease in March 2026?"])
    variant: Literal["baseline", "feedback_rag", "verified_feedback"] = "verified_feedback"
    snapshot: str = Field("live", description="Which memory to retrieve from.")


class LessonOut(BaseModel):
    feedback_id: str
    lesson: str
    status: str
    confidence: float
    similarity: float
    feedback_type: str


class AnalyzeResponse(BaseModel):
    answer: str
    sql: str
    trace_id: str
    status: str
    tables_used: list[str]
    feedback_used: list[LessonOut]
    analysis: str
    row_count: int
    sql_attempts: int
    latency_s: float
    tokens: int
    note: str = Field(
        "Local inference: expect 30-90 seconds per question.",
        description="Timing expectation, so a slow response is not mistaken for a hang.",
    )


class FeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=3,
                          examples=["You only looked at March. Compare it to February and check refunds."])
    question: str = Field(..., min_length=3)
    trace_id: str | None = None
    sql: str = ""
    answer: str = ""
    snapshot: str = "live"
    verify: bool = Field(True, description="False stores it unchecked, as PENDING.")


class ClaimOut(BaseModel):
    claim: str
    kind: str
    verdict: str
    method: str
    detail: str
    measured: dict[str, Any] = {}


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    verified: bool
    confidence: float
    lesson_kind: str
    lesson: str
    feedback_type: str
    reason: str
    claims: list[ClaimOut]
    conflicts_with: list[str]


# --------------------------------------------------------------- endpoints

@app.get("/health")
def health() -> dict:
    """Per-dependency readiness.

    Reports each component separately rather than one boolean, because the
    useful question when something breaks is WHICH part is down - the
    database, the model, or the memory schema.
    """
    s = get_settings()
    out: dict[str, Any] = {
        "status": "ok",
        "provider": str(s.llm_provider),
        "chat_model": s.chat_model,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
        "agent_engine": s.agent_engine,
        "checks": {},
    }

    try:
        from app.database.connection import ping

        out["checks"]["database"] = {"ok": True, **ping()}
    except Exception as e:  # noqa: BLE001
        out["checks"]["database"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["status"] = "degraded"

    try:
        import httpx

        r = httpx.get(f"{s.ollama_host}/api/version", timeout=3.0)
        out["checks"]["llm"] = {"ok": r.status_code == 200, **r.json()}
    except Exception as e:  # noqa: BLE001
        out["checks"]["llm"] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "hint": "start it with: make llm-up",
        }
        out["status"] = "degraded"

    try:
        from app.feedback import memory as fm

        out["checks"]["memory"] = {"ok": True, **fm.memory_stats("live")}
    except Exception as e:  # noqa: BLE001
        out["checks"]["memory"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["status"] = "degraded"

    return out


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    from app.agent.runner import answer_question

    try:
        state = answer_question(
            req.question, variant=req.variant, snapshot=req.snapshot, save=True
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e

    return AnalyzeResponse(
        answer=state.final_answer,
        sql=state.generated_sql,
        trace_id=state.trace_id,
        status=state.status,
        tables_used=state.tables_used,
        feedback_used=[
            LessonOut(
                feedback_id=l.feedback_id, lesson=l.lesson, status=l.status,
                confidence=l.confidence, similarity=round(l.similarity, 4),
                feedback_type=l.feedback_type,
            )
            for l in state.retrieved_lessons
        ],
        analysis=state.analysis_text,
        row_count=len(state.sql_result_rows),
        sql_attempts=state.sql_attempt_count,
        latency_s=round(state.latency_s, 2),
        tokens=state.prompt_tokens + state.output_tokens + state.thinking_tokens,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit(req: FeedbackRequest) -> FeedbackResponse:
    from app.feedback.pipeline import submit_feedback

    try:
        res = submit_feedback(
            feedback=req.feedback,
            question=req.question,
            sql=req.sql,
            answer=req.answer,
            trace_id=req.trace_id,
            snapshot=req.snapshot,
            verify=req.verify,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e

    return FeedbackResponse(
        feedback_id=res.feedback_id,
        status=str(res.outcome.status),
        verified=res.outcome.verified,
        confidence=res.outcome.confidence,
        lesson_kind=str(res.outcome.lesson_kind),
        lesson=res.processed.lesson,
        feedback_type=res.processed.feedback_type,
        reason=res.outcome.reason,
        claims=[
            ClaimOut(
                claim=c.claim, kind=c.kind, verdict=str(c.verdict),
                method=c.method, detail=c.detail, measured=c.measured,
            )
            for c in res.outcome.claim_results
        ],
        conflicts_with=[c.existing_id for c in res.conflicts],
    )


@app.get("/feedback")
def list_feedback(snapshot: str = "live", limit: int = 50) -> dict:
    from app.feedback import memory as fm

    rows = fm.list_lessons(snapshot, limit=limit)
    return {
        "snapshot": snapshot,
        "stats": fm.memory_stats(snapshot),
        "lessons": [
            {
                "feedback_id": str(r["feedback_id"]),
                "status": r["status"],
                "feedback_type": r["feedback_type"],
                "confidence": float(r["confidence"]),
                "lesson": r["lesson"],
                "original_question": r["original_question"],
                "schema_context": list(r["schema_context"] or []),
                "times_retrieved": r["times_retrieved"],
                "times_applied": r["times_applied"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
    }


@app.get("/feedback/{feedback_id}")
def get_feedback(feedback_id: str, snapshot: str = "live") -> dict:
    from app.feedback import memory as fm

    for r in fm.list_lessons(snapshot, limit=500):
        if str(r["feedback_id"]) == feedback_id:
            return {
                "feedback_id": feedback_id,
                "status": r["status"],
                "verified": r["verified"],
                "confidence": float(r["confidence"]),
                "feedback_type": r["feedback_type"],
                "lesson": r["lesson"],
                "mistake": r["mistake"],
                "correction": r["correction"],
                "raw_feedback": r["raw_feedback"],
                "original_question": r["original_question"],
                "schema_context": list(r["schema_context"] or []),
                "conflicts_with": [str(c) for c in (r["conflicts_with"] or [])],
                # The audit trail: which claims were checked, how, and
                # against which measured numbers.
                "verification_notes": r["verification_notes"],
                "times_retrieved": r["times_retrieved"],
                "times_applied": r["times_applied"],
                "created_at": r["created_at"].isoformat(),
            }
    raise HTTPException(404, f"no lesson {feedback_id} in snapshot '{snapshot}'")


@app.get("/trace/{trace_id}")
def trace(trace_id: str) -> dict:
    from app.observability.tracing import get_trace

    tr = get_trace(trace_id)
    if tr is None:
        raise HTTPException(404, f"no trace {trace_id}")
    for k in ("created_at",):
        if tr.get(k):
            tr[k] = tr[k].isoformat()
    for ev in tr.get("events", []):
        if ev.get("started_at"):
            ev["started_at"] = ev["started_at"].isoformat()
    tr["feedback_retrieved"] = [str(x) for x in (tr.get("feedback_retrieved") or [])]
    tr["feedback_used"] = [str(x) for x in (tr.get("feedback_used") or [])]
    tr["feedback_rejected"] = [str(x) for x in (tr.get("feedback_rejected") or [])]
    tr["trace_id"] = str(tr["trace_id"])
    return tr


@app.get("/traces")
def traces(limit: int = 25) -> dict:
    from app.observability.tracing import recent_traces

    rows = recent_traces(limit)
    return {
        "traces": [
            {
                **{k: v for k, v in r.items() if k not in ("created_at", "trace_id")},
                "trace_id": str(r["trace_id"]),
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]
    }
