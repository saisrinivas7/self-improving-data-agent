"""
The single place the project talks to a language model.

Everything the agent does goes through `LLMClient.complete()` or
`LLMClient.embed()`. Two reasons it is centralised:

1. Observability. The spec wants token counts, latency and tool-call counts
   per run (§20, §23). Those are only trustworthy if every call is measured
   in one place, so `LLMResponse` carries them and nothing bypasses it.

2. Provider swapping. Local Ollama by default (no quota), Gemini optionally.
   The agent code never knows which is in use.

A note on determinism, which matters for the benchmark: Ollama accepts a
`seed`, so repeated runs of the same prompt are genuinely reproducible.
Hosted Gemini does not offer that, and we measured its thinking-token count
varying between identical calls. That makes the local model the better choice
for the controlled comparison, not merely the cheaper one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import LLMProvider, Settings, get_settings


@dataclass(slots=True)
class LLMResponse:
    """One model call, with everything a trace needs to record."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens

    def json(self) -> Any:
        """Parse the response as JSON, tolerating markdown fences.

        Small models wrap JSON in ```json blocks even when told not to, so
        we strip those rather than failing. If it still isn't valid JSON we
        raise, because callers need to handle that explicitly - silently
        returning None would let a failed feedback classification look like
        an empty one.
        """
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("```")[1] if "```" in t[3:] else t[3:]
            if t.lstrip().lower().startswith("json"):
                t = t.lstrip()[4:]
            t = t.strip().rstrip("`").strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"model did not return valid JSON: {e}\n--- raw ---\n{self.text[:600]}"
            ) from e


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0))
        if self.s.llm_provider is LLMProvider.GEMINI:
            from google import genai

            self._genai = genai.Client(api_key=self.s.gemini_api_key)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ chat

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> LLMResponse:
        """One turn, no history.

        json_schema: when given, the model is constrained to emit JSON
        matching it. Both backends enforce this server-side, which is far
        more reliable than asking politely in the prompt and then repairing
        the output. The feedback classifier depends on this.
        """
        temp = self.s.llm_temperature if temperature is None else temperature
        seed = self.s.data_seed if seed is None else seed
        t0 = time.perf_counter()

        if self.s.llm_provider is LLMProvider.OLLAMA:
            resp = self._ollama_chat(prompt, system, json_schema, temp, max_tokens, seed)
        else:
            resp = self._gemini_chat(prompt, system, json_schema, temp, max_tokens)

        resp.latency_s = time.perf_counter() - t0
        return resp

    def _ollama_chat(
        self,
        prompt: str,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.s.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
            # Disable the model's own chain-of-thought. Granite and Qwen both
            # support a thinking mode; leaving it on inflates latency and
            # token counts for no gain on these structured tasks.
            "think": False,
        }
        if json_schema is not None:
            body["format"] = json_schema

        try:
            r = self._http.post(f"{self.s.ollama_host}/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Ollama returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.RequestError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.s.ollama_host}. Is it running?\n"
                f"  Start it with: make llm-up\n  ({type(e).__name__}: {e})"
            ) from e

        d = r.json()
        return LLMResponse(
            text=(d.get("message") or {}).get("content", ""),
            model=self.s.ollama_model,
            provider="ollama",
            prompt_tokens=d.get("prompt_eval_count", 0) or 0,
            output_tokens=d.get("eval_count", 0) or 0,
            raw=d,
        )

    def _gemini_chat(
        self,
        prompt: str,
        system: str | None,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        from google.genai import types

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            # Measured: default thinking cost 863 tokens and 34.7s on an SQL
            # prompt vs 0 tokens and 13.1s at budget 0, for equivalent output.
            # include_thoughts=False is also needed; budget alone was not
            # reliably honoured.
            "thinking_config": types.ThinkingConfig(
                thinking_budget=0, include_thoughts=False
            ),
        }
        if system:
            cfg_kwargs["system_instruction"] = system
        if json_schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = json_schema

        try:
            r = self._genai.models.generate_content(
                model=self.s.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Gemini call failed: {type(e).__name__}: {e}") from e

        u = getattr(r, "usage_metadata", None)
        return LLMResponse(
            text=(r.text or ""),
            model=self.s.gemini_model,
            provider="gemini",
            prompt_tokens=(getattr(u, "prompt_token_count", 0) or 0) if u else 0,
            output_tokens=(getattr(u, "candidates_token_count", 0) or 0) if u else 0,
            thinking_tokens=(getattr(u, "thoughts_token_count", 0) or 0) if u else 0,
        )

    # ------------------------------------------------------------- embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings, returning one vector per input.

        The count is asserted on return. Both providers have a failure mode
        where a list of inputs collapses into a single combined vector, which
        would silently mis-associate every stored lesson with the wrong text.
        """
        if not texts:
            return []

        if self.s.llm_provider is LLMProvider.OLLAMA:
            vecs = self._ollama_embed(texts)
        else:
            vecs = self._gemini_embed(texts)

        if len(vecs) != len(texts):
            raise LLMError(
                f"embedding batch returned {len(vecs)} vectors for {len(texts)} "
                "inputs - inputs were merged rather than batched"
            )
        for v in vecs:
            if len(v) != self.s.embedding_dim:
                raise LLMError(
                    f"embedding has {len(v)} dimensions, expected "
                    f"{self.s.embedding_dim} (feedback_memory.embedding is "
                    f"vector({self.s.embedding_dim}))"
                )
        return vecs

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = self._http.post(
                f"{self.s.ollama_host}/api/embed",
                json={"model": self.s.ollama_embedding_model, "input": texts},
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Ollama embed returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except httpx.RequestError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.s.ollama_host}. Is it running?\n"
                f"  Start it with: make llm-up\n  ({type(e).__name__}: {e})"
            ) from e
        return r.json().get("embeddings", [])

    def _gemini_embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        # A list of plain strings is treated as the parts of ONE document and
        # returns a single combined vector. Only Content objects batch per
        # item. Verified against individual calls (cosine 1.000000).
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        try:
            r = self._genai.models.embed_content(
                model=self.s.gemini_embedding_model,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.s.embedding_dim
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Gemini embed failed: {type(e).__name__}: {e}") from e
        return [list(e.values) for e in (r.embeddings or [])]
