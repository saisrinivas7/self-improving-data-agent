-- ============================================================
-- Feedback memory and agent traces  (schema: memory)
--
-- WHY A SEPARATE SCHEMA
-- agent_ro has no USAGE on this schema, so SQL the agent writes cannot read
-- it. That matters for one specific reason: a REJECTED poisoned lesson is
-- still a row in this table. If agent-generated SQL could SELECT from
-- feedback_memory, a question like "what lessons do you have?" would pull
-- rejected content straight into the agent's context, and the
-- wrong-feedback-adoption metric would be measuring a side channel rather
-- than the retrieval policy. The grant boundary makes that impossible
-- rather than merely unlikely.
--
-- Retrieval reads this schema through the `analyst` role, in code we wrote,
-- with explicit filtering - never through model-authored SQL.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS memory;

-- Explicitly ensure the agent role cannot reach it, even if a future
-- ALTER DEFAULT PRIVILEGES is added carelessly.
REVOKE ALL ON SCHEMA memory FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA memory FROM agent_ro;

DROP TABLE IF EXISTS memory.trace_events    CASCADE;
DROP TABLE IF EXISTS memory.traces          CASCADE;
DROP TABLE IF EXISTS memory.feedback_memory CASCADE;

-- ============================================================
-- feedback_memory
--
-- One row per lesson extracted from analyst feedback. The spec's section 7
-- field list is all here; the extra columns exist to satisfy the memory
-- policy in section 13, which a bare lesson+embedding table cannot:
--
--   Rule 3 (provenance)        -> source, raw_feedback, trace_id, created_at
--   Rule 4 (can be rejected)   -> status, verification_notes
--   Rule 5 (conflicts)         -> conflicts_with, superseded_by
--   Rule 6 (prioritise)        -> confidence, status, times_applied
--   Rule 7 (explain influence) -> lesson, correction kept separate
-- ============================================================
CREATE TABLE memory.feedback_memory (
    feedback_id       uuid PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- ---- classification (spec section 7) ----
    feedback_type     text NOT NULL CHECK (feedback_type IN (
                          'SQL_ERROR','MISSING_ANALYSIS','WRONG_ASSUMPTION',
                          'MISSING_DATA_SOURCE','BUSINESS_LOGIC',
                          'INTERPRETATION','OTHER')),

    -- ---- the content ----
    original_question text NOT NULL,
    mistake           text NOT NULL,
    correction        text NOT NULL,
    -- The generalised, reusable instruction. This is what gets injected into
    -- future prompts. Storing the raw comment instead is the classic failure:
    -- it is too specific to one question to ever help again.
    lesson            text NOT NULL,

    -- ---- provenance (Rule 3) ----
    -- Verbatim human text, kept so we can always show what was actually said
    -- rather than only our interpretation of it.
    raw_feedback      text,
    source            text NOT NULL DEFAULT 'human_analyst',
    -- The agent run this feedback was given against.
    trace_id          uuid,

    -- ---- verification (spec section 12) ----
    status            text NOT NULL DEFAULT 'PENDING' CHECK (status IN (
                          'PENDING','VERIFIED','REJECTED','CONFLICTING')),
    verified          boolean NOT NULL DEFAULT false,
    confidence        numeric(4,3) NOT NULL DEFAULT 0.500
                          CHECK (confidence >= 0 AND confidence <= 1),
    -- What the verifier actually checked, the SQL it ran, and the numbers it
    -- got back. This is what makes a stored lesson auditable instead of a
    -- claim we have to trust.
    verification_notes jsonb NOT NULL DEFAULT '{}'::jsonb,

    -- ---- conflict handling (Rule 5) ----
    conflicts_with    uuid[] NOT NULL DEFAULT '{}',
    superseded_by     uuid REFERENCES memory.feedback_memory (feedback_id),

    -- ---- retrieval context ----
    -- Which tables/columns the lesson is about. Used to re-rank by schema
    -- relevance, so a lesson about refunds does not surface on a question
    -- that never touches refunds (spec section 14).
    schema_context    text[] NOT NULL DEFAULT '{}',
    context           jsonb  NOT NULL DEFAULT '{}'::jsonb,

    -- ---- embedding ----
    embedding         vector({EMBEDDING_DIM}),
    -- Comparing vectors from two different embedding models is meaningless
    -- even when the dimensions match, so the model is recorded per row and
    -- retrieval filters on it.
    embedding_model   text NOT NULL,

    -- ---- experiment isolation ----
    -- 'live' is the Learning Lab's own memory, which starts EMPTY per the
    -- spec's human-in-the-loop requirement. Benchmarks run against named
    -- snapshots ('clean', 'poisoned', 'mixed') so each condition has a known,
    -- reproducible memory state without touching live.
    snapshot          text NOT NULL DEFAULT 'live',

    -- ---- usage counters, for the feedback-utilisation metric (section 20.4) ----
    times_retrieved   integer NOT NULL DEFAULT 0,
    times_applied     integer NOT NULL DEFAULT 0,
    times_rejected    integer NOT NULL DEFAULT 0,

    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

-- Vector similarity index. Cosine is the right operator here because the
-- embedding model returns unit-length vectors (verified in verify_llm.py).
-- HNSW rather than IVFFlat: no training step, and it behaves well on the
-- small, steadily-growing table this will be.
CREATE INDEX idx_fm_embedding ON memory.feedback_memory
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_fm_snapshot ON memory.feedback_memory (snapshot);
CREATE INDEX idx_fm_status   ON memory.feedback_memory (status);
CREATE INDEX idx_fm_type     ON memory.feedback_memory (feedback_type);
CREATE INDEX idx_fm_schema   ON memory.feedback_memory USING gin (schema_context);
-- Lexical fallback alongside vector search, so an exact term like "refund"
-- still matches when the embedding ranking is ambiguous.
CREATE INDEX idx_fm_lesson_trgm ON memory.feedback_memory
    USING gin (lesson gin_trgm_ops);

-- ============================================================
-- traces  (spec section 23)
--
-- One row per agent run. Deliberately simple: a parent row plus ordered
-- child events. The spec warns against over-engineering observability, and
-- two tables in the same Postgres we already run is the cheapest thing that
-- answers "what did the agent do, and how long did it take".
-- ============================================================
CREATE TABLE memory.traces (
    trace_id        uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    question        text NOT NULL,
    -- baseline | feedback_rag | verified_feedback
    system_variant  text NOT NULL,
    -- Which memory snapshot this run retrieved from.
    snapshot        text,

    final_answer    text,
    generated_sql   text,
    status          text NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','completed','failed')),
    error           text,

    -- Metrics the benchmark reports (section 20).
    sql_attempts    integer NOT NULL DEFAULT 0,
    tool_calls      integer NOT NULL DEFAULT 0,
    llm_calls       integer NOT NULL DEFAULT 0,
    prompt_tokens   integer NOT NULL DEFAULT 0,
    output_tokens   integer NOT NULL DEFAULT 0,
    thinking_tokens integer NOT NULL DEFAULT 0,
    latency_s       numeric(10,3),

    -- Recorded per run so results stay interpretable after a model change.
    chat_model      text,
    embedding_model text,

    -- Which feedback was retrieved, which was used, which was rejected.
    -- Denormalised here so a single row answers the utilisation metric.
    feedback_retrieved uuid[] NOT NULL DEFAULT '{}',
    feedback_used      uuid[] NOT NULL DEFAULT '{}',
    feedback_rejected  uuid[] NOT NULL DEFAULT '{}',

    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_traces_created ON memory.traces (created_at DESC);
CREATE INDEX idx_traces_variant ON memory.traces (system_variant);

CREATE TABLE memory.trace_events (
    event_id    bigserial PRIMARY KEY,
    trace_id    uuid NOT NULL REFERENCES memory.traces (trace_id) ON DELETE CASCADE,
    -- Ordering within the run. The LangGraph node sequence.
    seq         integer NOT NULL,
    node        text NOT NULL,

    input       jsonb,
    output      jsonb,
    error       text,

    latency_s   numeric(10,3),
    prompt_tokens   integer NOT NULL DEFAULT 0,
    output_tokens   integer NOT NULL DEFAULT 0,
    thinking_tokens integer NOT NULL DEFAULT 0,

    started_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (trace_id, seq)
);

CREATE INDEX idx_trace_events_trace ON memory.trace_events (trace_id, seq);

-- ============================================================
-- Final guard: make absolutely sure the agent role cannot read any of this.
-- verify_db.py asserts this by trying and expecting failure.
-- ============================================================
REVOKE ALL ON ALL TABLES    IN SCHEMA memory FROM agent_ro;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA memory FROM agent_ro;
REVOKE USAGE ON SCHEMA memory FROM agent_ro;
