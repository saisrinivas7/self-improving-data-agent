"""
Phase 0 LLM verification.

Run:  make verify-llm

What this does and why it isn't just a "hello world":

1. Lists the models your API key can actually see. Model IDs change, and
   hardcoding one from a blog post is how projects break six months later.
2. Picks a Flash chat model and an embedding model from that live list.
3. Makes one real generate call and one real embed call.
4. Reports the embedding dimension, then writes it to .env.

Point 4 is the important one. The embedding dimension is baked into the
feedback_memory table as `embedding vector(N)`. Changing it later means
dropping the column and re-embedding every stored lesson. So we settle it
before writing any DDL.

It also caps the dimension at 1536. pgvector's HNSW and IVFFlat indexes
support at most 2000 dimensions, and some Gemini embedding models default to
3072 - which would store fine but could never be indexed, making similarity
search a full table scan. Gemini's embedding models are trained with
Matryoshka representation learning, so truncating is a supported operation,
not a hack - but truncated vectors must be re-normalised to unit length.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# pgvector index ceiling. Staying at or below this keeps HNSW available.
MAX_INDEXABLE_DIM = 2000
PREFERRED_DIM = 1536


def fail(msg: str) -> "None":
    print(f"\n  FAILED: {msg}\n", file=sys.stderr)
    sys.exit(1)


def load_env() -> dict[str, str]:
    from dotenv import dotenv_values

    if not ENV_PATH.exists():
        fail(f"{ENV_PATH} not found. Run: cp .env.example .env")
    return dotenv_values(ENV_PATH)  # type: ignore[return-value]


def write_env_values(updates: dict[str, str]) -> None:
    """Rewrite specific KEY=value lines in .env, leaving everything else alone."""
    text = ENV_PATH.read_text()
    for key, value in updates.items():
        pattern = rf"^{re.escape(key)}=.*$"
        if re.search(pattern, text, flags=re.MULTILINE):
            text = re.sub(pattern, f"{key}={value}", text, flags=re.MULTILINE)
        else:
            text = text.rstrip("\n") + f"\n{key}={value}\n"
    ENV_PATH.write_text(text)


# Models that advertise generateContent but are not general-purpose text
# models. A TTS model will happily accept our prompt and return audio, which
# fails in a confusing way three layers deep in the agent. Excluded up front.
NON_TEXT_MARKERS = (
    "tts",        # text-to-speech
    "image",      # image generation
    "imagen",
    "veo",        # video
    "audio",      # native-audio dialog models
    "live",       # realtime streaming models
    "robotics",
    "embedding",  # handled separately
    "aqa",        # attributed question answering, different contract
    "learnlm",
)


def is_text_chat_model(name: str) -> bool:
    n = name.lower()
    return "gemini" in n and not any(t in n for t in NON_TEXT_MARKERS)


def score_chat_model(name: str) -> tuple[int, ...]:
    """Rank candidate chat models. Higher is better.

    Preference order: Flash (cheap + fast, and what the spec asks for),
    stable over preview/experimental, newer version over older, full over lite.
    """
    n = name.lower()
    is_flash = "flash" in n
    is_lite = "lite" in n
    unstable = any(t in n for t in ("exp", "preview", "thinking"))
    version = 0.0
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    if m:
        version = float(m.group(1))
    # flash first, then stable, then higher version, then non-lite
    return (int(is_flash), int(not unstable), int(version * 10), int(not is_lite))


def score_embed_model(name: str) -> tuple[int, ...]:
    n = name.lower()
    unstable = any(t in n for t in ("exp", "preview"))
    # prefer the newer unified "gemini-embedding" family over legacy text-embedding
    is_gemini_family = "gemini-embedding" in n
    version = 0.0
    m = re.search(r"(\d+)$", n)
    if m:
        version = float(m.group(1))
    return (int(not unstable), int(is_gemini_family), int(version))


def verify_gemini() -> None:
    env = load_env()
    api_key = (env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()

    if not api_key:
        fail(
            "GEMINI_API_KEY is empty in .env\n"
            "  Get a free key at https://aistudio.google.com/apikey\n"
            "  then paste it after GEMINI_API_KEY= in .env"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # ---------- 1. discover models ----------
    print("Querying models available to this API key...")
    try:
        all_models = list(client.models.list())
    except Exception as e:  # noqa: BLE001
        fail(f"Could not list models - is the key valid?\n  {type(e).__name__}: {e}")

    chat_models, embed_models = [], []
    for m in all_models:
        actions = set(getattr(m, "supported_actions", None) or [])
        short = (m.name or "").replace("models/", "")
        if "generateContent" in actions and is_text_chat_model(short):
            chat_models.append(short)
        if "embedContent" in actions:
            embed_models.append(short)

    if not chat_models:
        fail("No models supporting generateContent were returned.")
    if not embed_models:
        fail("No models supporting embedContent were returned.")

    print(f"  {len(all_models)} models visible")
    print(f"  {len(chat_models)} support generateContent")
    print(f"  {len(embed_models)} support embedContent")

    chat_models.sort(key=score_chat_model, reverse=True)
    embed_models.sort(key=score_embed_model, reverse=True)

    print("\n  Top chat candidates:")
    for n in chat_models[:6]:
        print(f"    - {n}")
    print("  Top embedding candidates:")
    for n in embed_models[:6]:
        print(f"    - {n}")

    # ---------- pinning ----------
    # Once a model is chosen it is PINNED in .env, and re-running this script
    # must not silently change it. Two ways that would bite:
    #   - A transient 503 on the pinned chat model would fall back to a
    #     different model mid-benchmark, invalidating the comparison.
    #   - A different embedding model would produce vectors of the same 1536
    #     dimensions in a DIFFERENT vector space. Every similarity score
    #     against already-stored lessons becomes meaningless, with no error.
    # So: discover only when unset. When set, verify and fail loudly.
    pinned_chat = (env.get("GEMINI_MODEL") or "").strip()
    pinned_embed = (env.get("GEMINI_EMBEDDING_MODEL") or "").strip()
    pinned_dim = (env.get("EMBEDDING_DIM") or "").strip()
    is_pinned = bool(pinned_chat and pinned_embed and pinned_dim)

    if is_pinned:
        print("\n  Models are PINNED in .env - verifying them, not re-selecting:")
        print(f"    chat      : {pinned_chat}")
        print(f"    embedding : {pinned_embed}")
        print(f"    dimension : {pinned_dim}")
        if pinned_chat not in chat_models:
            fail(
                f"Pinned GEMINI_MODEL '{pinned_chat}' is not available to this key.\n"
                f"  Available: {', '.join(chat_models[:5])}\n"
                f"  To re-select, blank GEMINI_MODEL in .env and re-run."
            )
        if pinned_embed not in embed_models:
            fail(
                f"Pinned GEMINI_EMBEDDING_MODEL '{pinned_embed}' is not available.\n"
                f"  Re-selecting would invalidate every stored embedding.\n"
                f"  Available: {', '.join(embed_models)}"
            )
        embed_model = pinned_embed
    else:
        embed_model = embed_models[0]
        print(f"\n  No pin found - selecting embedding model: {embed_model}")

    # ---------- 2. real generate call, with retry and model fallback ----------
    # The newest Flash model is the most popular and so the most likely to
    # return 503 "high demand". We retry with backoff, then fall back to the
    # next-best candidate. Whichever model actually answers is the one we
    # write to .env, so the project is configured against reality.
    print("\nTesting generate_content (temperature=0 for determinism)...")

    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=2048,
        # We drive tool use explicitly through LangGraph nodes, so the SDK's
        # automatic function calling is off. Also silences an SDK warning.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # Thinking off by default. Measured on an SQL-generation prompt:
        # default = 34.7s / 1003 tokens (863 thinking); budget 0 = 13.1s /
        # 138 tokens, same SQL quality. Leaving it on would distort the
        # latency and token metrics and cut benchmark throughput ~7x.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "high demand")

    def try_generate(model: str, attempts: int = 3) -> "str | None":
        for i in range(attempts):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents="Reply with exactly the single word: OK",
                    config=gen_config,
                )
                usage = getattr(resp, "usage_metadata", None)
                if usage:
                    thoughts = usage.thoughts_token_count or 0
                    print(
                        f"    tokens -> prompt: {usage.prompt_token_count}, "
                        f"output: {usage.candidates_token_count}, "
                        f"thinking: {thoughts}, "
                        f"total: {usage.total_token_count}"
                    )
                    if thoughts > 0:
                        print(
                            f"    \033[33mwarning\033[0m thinking_budget=0 was requested "
                            f"but {thoughts} thinking tokens were billed"
                        )
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                transient = any(t in msg for t in TRANSIENT)
                if transient and i < attempts - 1:
                    wait = 2 ** (i + 1)
                    print(f"    transient error, retrying in {wait}s ({i + 1}/{attempts})")
                    time.sleep(wait)
                    continue
                short = msg.split("\n")[0][:120]
                print(f"    {model}: {type(e).__name__}: {short}")
                return None
        return None

    chat_model = None
    if is_pinned:
        # Pinned: verify this exact model. No fallback - falling back would
        # change the model under a running benchmark.
        text = try_generate(pinned_chat, attempts=4)
        if text is None:
            fail(
                f"Pinned model '{pinned_chat}' did not respond after 4 attempts.\n"
                "  This is usually transient API load - wait and re-run.\n"
                "  Deliberately NOT falling back: a silent model swap would\n"
                "  invalidate benchmark comparisons."
            )
        chat_model = pinned_chat
        print(f"  \033[32mresponse\033[0m: {text!r}")
    else:
        for candidate in chat_models[:4]:
            print(f"  trying {candidate}")
            text = try_generate(candidate)
            if text is not None:
                chat_model = candidate
                print(f"  \033[32mresponse\033[0m: {text!r}")
                break
            print(f"  {candidate} unavailable, falling back")

        if chat_model is None:
            fail(
                "No chat model responded. Tried: "
                + ", ".join(chat_models[:4])
                + "\n  This usually means transient API load - try again in a few minutes."
            )
    print(f"\n  Chat model: {chat_model}")

    # ---------- 3. real embed call, and settle the dimension ----------
    print("\nTesting embed_content...")
    native_dim = None
    try:
        r = client.models.embed_content(model=embed_model, contents="revenue declined in March")
        native_dim = len(r.embeddings[0].values)
        print(f"  native dimension: {native_dim}")
    except Exception as e:  # noqa: BLE001
        fail(f"embed_content failed\n  {type(e).__name__}: {e}")

    # If a dimension is already pinned, that is the only acceptable answer:
    # feedback_memory.embedding is vector(pinned), and anything else cannot
    # be inserted into it.
    target_dim = int(pinned_dim) if is_pinned else (
        PREFERRED_DIM if native_dim > MAX_INDEXABLE_DIM else native_dim
    )

    chosen_dim = native_dim
    if target_dim != native_dim:
        why = (
            f"pinned at {target_dim} in .env"
            if is_pinned
            else f"native {native_dim} exceeds pgvector's {MAX_INDEXABLE_DIM}-dim index limit"
        )
        print(f"  {why}; requesting output_dimensionality={target_dim}")
        try:
            r = client.models.embed_content(
                model=embed_model,
                contents="revenue declined in March",
                config=types.EmbedContentConfig(output_dimensionality=target_dim),
            )
            chosen_dim = len(r.embeddings[0].values)
            print(f"  truncated dimension: {chosen_dim}")
        except Exception as e:  # noqa: BLE001
            fail(
                f"Model does not support output_dimensionality={target_dim}, and "
                f"{native_dim} dims cannot be indexed by pgvector.\n"
                f"  {type(e).__name__}: {e}"
            )

    if chosen_dim != target_dim:
        fail(f"expected {target_dim} dimensions, got {chosen_dim}")
    if chosen_dim > MAX_INDEXABLE_DIM:
        fail(
            f"{chosen_dim} dims exceeds pgvector's {MAX_INDEXABLE_DIM}-dim index "
            "limit - similarity search could never use an index"
        )

    # ---------- 4. sanity-check that embeddings are semantically useful ----------
    # If this fails, retrieval will never work, and it's far better to find out
    # now than to debug it through three layers of agent code later.
    print("\nSanity-checking semantic similarity...")
    cfg = (
        types.EmbedContentConfig(output_dimensionality=chosen_dim)
        if chosen_dim != native_dim
        else None
    )
    probes = [
        "Why did revenue decrease in March?",
        "What caused the March revenue decline?",
        "How many support tickets mention shipping delays?",
    ]

    # IMPORTANT, and a genuine footgun:
    #   embed_content(contents=["a", "b", "c"])  -> 1 embedding
    #   embed_content(contents=[Content(a), Content(b), Content(c)]) -> 3
    # A list of plain strings is treated as the PARTS OF ONE document, so you
    # get a single combined vector and the inputs are silently merged. Only a
    # list of Content objects batches per item. Verified: batched vectors are
    # identical to individual calls (cosine 1.000000).
    # We assert the count so this can never regress into silent data loss.
    batch = [types.Content(parts=[types.Part(text=p)]) for p in probes]
    r = client.models.embed_content(model=embed_model, contents=batch, config=cfg)
    if len(r.embeddings) != len(probes):
        fail(
            f"batched embed returned {len(r.embeddings)} vectors for "
            f"{len(probes)} inputs - inputs were merged, not batched"
        )
    print(f"  batched {len(probes)} inputs -> {len(r.embeddings)} vectors")
    raw_vecs: list[list[float]] = [list(e.values) for e in r.embeddings]

    import math

    def l2(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    # Measured: Gemini returns unit-length vectors at both 3072 and truncated
    # 1536, so no normalisation is needed. We assert it rather than assume it,
    # because if it ever stopped holding, our hand-rolled dot product below
    # (and any inner-product pgvector operator) would be quietly wrong.
    norms = [l2(v) for v in raw_vecs]
    if not all(abs(n - 1.0) < 1e-3 for n in norms):
        fail(
            "embeddings are not unit length "
            f"(norms: {[round(n, 4) for n in norms]}) - normalise before storing"
        )
    print(f"  all vectors unit length (L2 = {norms[0]:.6f})")
    vecs = raw_vecs

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    sim_related = cos(vecs[0], vecs[1])
    sim_unrelated = cos(vecs[0], vecs[2])
    print(f"  similar questions   : {sim_related:.4f}")
    print(f"  unrelated questions : {sim_unrelated:.4f}")
    if sim_related <= sim_unrelated:
        fail(
            "Paraphrased questions scored no higher than unrelated ones. "
            "Embeddings are not usable for retrieval."
        )
    print(f"  margin: +{sim_related - sim_unrelated:.4f}  (paraphrase ranks higher)")

    # ---------- 5. persist, but only on first run ----------
    if is_pinned:
        print("\n  .env unchanged (models already pinned).")
    else:
        write_env_values(
            {
                "GEMINI_MODEL": chat_model,
                "GEMINI_EMBEDDING_MODEL": embed_model,
                "EMBEDDING_DIM": str(chosen_dim),
            }
        )
        print("\n  Written to .env and now PINNED. Re-running this script will")
        print("  verify these models rather than re-select them.")

    print("\n" + "=" * 62)
    print("LLM LAYER OK")
    print("=" * 62)
    print(f"  GEMINI_MODEL           = {chat_model}")
    print(f"  GEMINI_EMBEDDING_MODEL = {embed_model}")
    print(f"  EMBEDDING_DIM          = {chosen_dim}")
    print(f"\n  feedback_memory.embedding will be vector({chosen_dim}).")
    print("  Changing the embedding model later means re-embedding every")
    print("  stored lesson - which is why these values are pinned.")


def verify_ollama() -> None:
    """Verify the local Ollama stack through the real LLMClient.

    This deliberately exercises app/llm.py rather than calling the HTTP API
    directly, so what we verify here is the exact code path the agent uses.
    """
    import httpx

    env = load_env()
    host = (env.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    chat_model = (env.get("OLLAMA_MODEL") or "").strip()
    embed_model = (env.get("OLLAMA_EMBEDDING_MODEL") or "").strip()

    # ---------- 1. server reachable ----------
    print(f"Checking Ollama at {host} ...")
    try:
        v = httpx.get(f"{host}/api/version", timeout=5.0).json()
        print(f"  server up, version {v.get('version')}")
    except Exception as e:  # noqa: BLE001
        fail(
            f"Cannot reach Ollama at {host} ({type(e).__name__}).\n"
            "  Start it with:  make llm-up"
        )

    # ---------- 2. required models present ----------
    tags = httpx.get(f"{host}/api/tags", timeout=10.0).json()
    installed = {m["name"] for m in tags.get("models", [])}
    # ollama reports "name:tag"; a bare name means ":latest"
    def present(name: str) -> bool:
        return name in installed or f"{name}:latest" in installed

    print(f"  {len(installed)} model(s) installed")
    for label, name in (("chat", chat_model), ("embedding", embed_model)):
        if present(name):
            size = next(
                (m.get("size", 0) for m in tags.get("models", []) if m["name"].startswith(name)),
                0,
            )
            print(f"  {label:<10} {name}  ({size / 1e9:.1f} GB)")
        else:
            fail(
                f"{label} model '{name}' is not installed.\n"
                f"  Install it with:  ollama pull {name}\n"
                f"  Installed: {', '.join(sorted(installed)) or '(none)'}"
            )

    # ---------- 3. chat through the real client ----------
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.llm import LLMClient  # noqa: PLC0415

    print("\nTesting chat (plain text)...")
    with LLMClient() as llm:
        r = llm.complete("Reply with exactly the single word: OK", max_tokens=64)
        print(f"  response: {r.text.strip()[:60]!r}")
        print(
            f"  tokens -> prompt: {r.prompt_tokens}, output: {r.output_tokens}  "
            f"latency: {r.latency_s:.2f}s"
        )
        if r.output_tokens == 0:
            fail("model returned no output tokens")

        # ---------- 4. JSON SCHEMA ENFORCEMENT ----------
        # This is the capability the whole feedback system rests on. The
        # classifier must turn free-text analyst feedback into a typed record.
        # If the model cannot be constrained to a schema, every downstream
        # stage has to defensively repair malformed output.
        print("\nTesting constrained JSON output (feedback classifier contract)...")
        schema = {
            "type": "object",
            "properties": {
                "feedback_type": {
                    "type": "string",
                    "enum": [
                        "SQL_ERROR",
                        "MISSING_ANALYSIS",
                        "WRONG_ASSUMPTION",
                        "MISSING_DATA_SOURCE",
                        "BUSINESS_LOGIC",
                        "INTERPRETATION",
                        "OTHER",
                    ],
                },
                "mistake": {"type": "string"},
                "correction": {"type": "string"},
                "lesson": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["feedback_type", "mistake", "correction", "lesson", "confidence"],
        }
        prompt = (
            "An analyst reviewed a data agent's answer and said:\n\n"
            '"You focused only on product sales. You should also check refunds, '
            'because refund volume increased substantially in March."\n\n'
            "The agent had answered: 'Revenue decreased 12% in March primarily "
            "because Product A sales declined.'\n\n"
            "Convert the analyst's feedback into the structured schema. The "
            "'lesson' must be phrased generally enough to apply to future "
            "questions, not just this one."
        )
        r2 = llm.complete(prompt, json_schema=schema, max_tokens=512)
        try:
            parsed = r2.json()
        except ValueError as e:
            fail(f"schema-constrained call did not produce valid JSON\n  {e}")

        missing = [k for k in schema["required"] if k not in parsed]
        if missing:
            fail(f"JSON missing required fields: {missing}\n  got: {parsed}")
        if parsed["feedback_type"] not in schema["properties"]["feedback_type"]["enum"]:
            fail(f"feedback_type not in enum: {parsed['feedback_type']!r}")

        print(f"  feedback_type : {parsed['feedback_type']}")
        print(f"  lesson        : {str(parsed['lesson'])[:90]}")
        print(f"  confidence    : {parsed['confidence']}")
        print(f"  latency       : {r2.latency_s:.2f}s")
        if parsed["feedback_type"] != "MISSING_ANALYSIS":
            print(
                f"  \033[33mnote\033[0m expected MISSING_ANALYSIS for this example, "
                f"got {parsed['feedback_type']} - classification prompt may need tuning"
            )

        # ---------- 5. embeddings ----------
        print("\nTesting embeddings...")
        probes = [
            "Why did revenue decrease in March?",
            "What caused the March revenue decline?",
            "How many support tickets mention shipping delays?",
        ]
        # Bypass the client's dimension assertion on first run, since
        # EMBEDDING_DIM in .env may still hold the Gemini value.
        raw = llm._ollama_embed(probes)  # noqa: SLF001
        if len(raw) != len(probes):
            fail(f"batched embed returned {len(raw)} vectors for {len(probes)} inputs")
        dim = len(raw[0])
        print(f"  batched {len(probes)} inputs -> {len(raw)} vectors, dim={dim}")

        if dim > MAX_INDEXABLE_DIM:
            fail(
                f"{dim} dimensions exceeds pgvector's {MAX_INDEXABLE_DIM}-dim index "
                "limit; pick a smaller embedding model"
            )

        import math

        def l2(v: list[float]) -> float:
            return math.sqrt(sum(x * x for x in v))

        norms = [l2(v) for v in raw]
        normalised = all(abs(n - 1.0) < 1e-3 for n in norms)
        print(
            f"  L2 norms: {[round(n, 4) for n in norms]} "
            f"-> {'already unit length' if normalised else 'NOT unit length, will normalise on write'}"
        )

        def unit(v: list[float]) -> list[float]:
            m = l2(v) or 1.0
            return [x / m for x in v]

        u = [unit(v) for v in raw]
        sim_rel = sum(a * b for a, b in zip(u[0], u[1]))
        sim_unrel = sum(a * b for a, b in zip(u[0], u[2]))
        print(f"  similar questions   : {sim_rel:.4f}")
        print(f"  unrelated questions : {sim_unrel:.4f}")
        if sim_rel <= sim_unrel:
            fail(
                "Paraphrased questions scored no higher than unrelated ones. "
                "This embedding model cannot support feedback retrieval."
            )
        margin = sim_rel - sim_unrel
        print(f"  margin: +{margin:.4f}  (paraphrase ranks higher)")
        if margin < 0.05:
            print(
                "  \033[33mnote\033[0m margin is small; retrieval ranking will be "
                "noisy. Gemini's embedding model measured +0.3515 on the same probe."
            )

    # ---------- 6. persist the dimension ----------
    current = (env.get("EMBEDDING_DIM") or "").strip()
    if current != str(dim):
        write_env_values({"EMBEDDING_DIM": str(dim)})
        print(f"\n  EMBEDDING_DIM updated: {current or '(unset)'} -> {dim}")
        print("  feedback_memory.embedding will be vector(" + str(dim) + ").")
    else:
        print(f"\n  EMBEDDING_DIM already {dim}, .env unchanged.")

    print("\n" + "=" * 62)
    print("LLM LAYER OK  (provider: ollama)")
    print("=" * 62)
    print(f"  chat model      = {chat_model}")
    print(f"  embedding model = {embed_model}")
    print(f"  EMBEDDING_DIM   = {dim}")


def main() -> None:
    env = load_env()
    provider = (env.get("LLM_PROVIDER") or "ollama").strip().lower()
    print(f"LLM_PROVIDER = {provider}\n")
    if provider == "ollama":
        verify_ollama()
    elif provider == "gemini":
        verify_gemini()
    else:
        fail(f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama' or 'gemini'.")


if __name__ == "__main__":
    main()
