"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 1 - Multi-query expansion
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 The pipeline's single query-understanding point. The LLM rewrites the input
 query into 2-3 paraphrased variants, shifting toward the corpus's formal
 legal register — bridging the gap between how a worker phrases a question
 colloquially and how the Labour Act itself is worded.

 ONE stage, ONE structured output. Everything downstream needs from query
 understanding comes out of one JSON envelope:

   {
     "original":      "<the query as received>",
     "variants":      ["<formal-register paraphrase 1>", "...2", "...3"],
     "multi_aspect":  true | false,   # enables Stage 5b (MMR)
     "historical":    true | false    # relaxes Stage 2's filter
   }

   - multi_aspect: set when generating variants surfaces that the question has
     genuinely distinct aspects (e.g. "is this fee legal AND what's the real
     wage"). This flag conditionally enables Stage 5b — MMR must NOT run by
     default on single-aspect queries, where it hurts precision.
   - historical: set when the question is explicitly about law over time
     ("what did the law say before 2018?", "what changed in the 2026
     amendment?"). Stage 2 consumes it; nothing else in the pipeline guesses
     at history.
   - A reserved `asr` field will eventually carry transcript-cleanup signals —
     DELIBERATELY ABSENT NOW (RETRIEVAL_PIPELINE.md Flagged follow-ups).

 Model choice is a tunable (config.EXPAND_MODEL), not a given. The demo LLM is
 ollama/llama3.2 — weak at Bangla, and Bangla->formal-legal-English register
 shifting is the riskiest LLM-dependent step in this pipeline. Rewrite quality
 on the Bangla test set is a MEASURED eval item, not an assumption.

 The fallback path (defined, loud): if expansion fails (LLM unavailable or
 malformed JSON), run with the original query as the sole variant; all flags
 false. The filter stays default-STRICT — never widen legality filters on a
 degraded run. Logged in the trace, never silent.

 Deferred for later: once real ASR is in the loop, this stage may also need to
 absorb transcript-cleanup duties (garbled numbers, dialect normalization)
 before expansion. Not needed now — typed/clean text input only.

 Used by: orchestrator.py (Stage 1 of the retrieve() flow).

 Objective of this file: take a query string and return the Stage 1 envelope
 (variants + flags), with the fallback, with trace entries for the variants,
 flags, and any fallback triggered. This is the only place query intent is
 interpreted — Stages 2-7 consume the envelope, never re-guess.
============================================================================
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from retrieval import config as cfg

logger = logging.getLogger("retrieval.stage1")

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class _Reply:
    """Minimal stand-in for a LangChain chat response: only .content is used."""

    def __init__(self, content: str) -> None:
        self.content = content


class GeminiChat:
    """Hosted Gemini chat model with the same .invoke(messages) shape as ChatOllama.

    Uses the REST API via the standard library (no extra dependency). The API key
    is read from the GEMINI_API_KEY environment variable (a .env file is loaded if
    python-dotenv is present). A missing key or any HTTP/network error is raised
    as an exception; expand_query() catches it and records a Stage 1 fallback, so
    retrieval degrades loudly instead of crashing.
    """

    def __init__(self, model: str, timeout_s: int = 60) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def invoke(self, messages: List[Dict[str, str]]) -> _Reply:
        key = _gemini_api_key()
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = [
            {"role": "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") == "user"
        ]
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                # Ask for JSON directly; _parse_envelope still tolerates fences.
                "responseMimeType": "application/json",
            },
        }
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        req = urllib.request.Request(
            _GEMINI_ENDPOINT.format(model=self.model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError("Gemini HTTP %d: %s" % (exc.code, detail)) from exc

        try:
            parts = payload["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned no text: %s" % str(payload)[:300]) from exc
        return _Reply(text)


def _gemini_api_key() -> str:
    """Return GEMINI_API_KEY, loading .env first if python-dotenv is installed."""
    if not os.environ.get("GEMINI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (env var or .env file)")
    return key

# How many variants the LLM is asked to produce (capped at config.MAX_VARIANTS).
_N_VARIANTS = 3

# The system prompt that asks the model to shift toward the corpus's formal
# legal register. This is where the register-bridging instruction lives.
_SYSTEM_PROMPT = (
    "You are helping build a legal retrieval query for Bangladesh's Labour Act, "
    "Labour Rules, and wage gazettes.\n\n"
    "A worker has asked a question in colloquial Bangla (or English). Rewrite the "
    "question into 2-3 formal, legal-register paraphrases that a retriever could "
    "match against the official legal text. Use the formal terms the law actually "
    "uses — for example, use 'recruitment fee' or 'placement charge' rather than "
    "'dalal wants money', use 'notice of discharge' rather than 'they threw me out', "
    "use 'overtime' rather than 'extra hours'.\n\n"
    "Rules:\n"
    "- Return a JSON object with keys: original, variants (list of 2-3 strings), "
    "multi_aspect (true/false), historical (true/false).\n"
    "- Do NOT answer the question. Only produce the paraphrases.\n"
    "- multi_aspect is true when the question clearly has two or more genuinely "
    "different things being asked (e.g. a legal question AND a job/hiring question).\n"
    "- historical is true when the question is explicitly about how the law was in "
    "the past or about a specific amendment/change over time.\n"
    "- Keep each variant short (one or two sentences) and in the SAME language as the "
    "original if it is Bangla; if the original is English, keep variants in English.\n"
    "- Preserve any numbers, dates, or named entities exactly (factory names, sections, "
    "amounts).\n"
    "- If you cannot produce meaningful variants, return the original as the sole "
    "variant and set both flags to false."
)


def expand_query(
    original: str,
    trace: Any,
    model: Any = None,
    n_variants: int = _N_VARIANTS,
) -> Dict[str, Any]:
    """Run Stage 1 multi-query expansion and return the envelope.

    Parameters
    ----------
    original : str
        The query as received (Bangla or English text).
    trace : Trace (retrieval.trace.Trace)
        The in-progress trace. This function records the variants, flags, model,
        and any fallback into it via trace.add_variant() / trace.set_flags() /
        trace.add_fallback().
    model : langchain ChatOllama, optional
        The LLM to use for expansion. If None, a ChatOllama is created from
        config.EXPAND_MODEL (the dev stand-in, ollama/llama3.2). The model is a
        tunable — swap it without touching this function's logic.
    n_variants : int
        How many variants to ask the LLM for (default 3, capped at
        config.MAX_VARIANTS).

    Returns
    -------
    dict with keys:
        "original"     : the query as received (passed through)
        "variants"     : list of variant strings (2-3, or 1 if fallback)
        "multi_aspect" : bool  (flag; False in fallback)
        "historical"   : bool  (flag; False in fallback)
    """
    capped = min(n_variants, cfg.MAX_VARIANTS)
    model = model or _default_model()

    # Ask the model for the structured envelope.
    user_message = _user_prompt(original, capped)
    try:
        response = model.invoke([_system_to_msg(), _human_to_msg(user_message)])
    except Exception as exc:
        _fallback(original, trace, "llm_unavailable", str(exc))
        return _fallback_envelope(original)

    text = _first_content_text(response)
    if text is None:
        _fallback(original, trace, "no_model_output", "LLM returned no content")
        return _fallback_envelope(original)

    envelope = _parse_envelope(text, original, capped)
    if envelope is None:
        _fallback(original, trace, "malformed_json", "could not parse model JSON")
        return _fallback_envelope(original)

    # Record the successful expansion into the trace.
    for variant in envelope["variants"]:
        trace.add_variant(variant)
    trace.set_flags(envelope["multi_aspect"], envelope["historical"])
    logger.info(
        "Stage 1 expansion OK: %d variants, multi_aspect=%s, historical=%s",
        len(envelope["variants"]),
        envelope["multi_aspect"],
        envelope["historical"],
    )
    return envelope


# ------------------------------------------------------------------
# Small helpers — model creation, prompting, parsing, fallback
# ------------------------------------------------------------------

def _default_model() -> Any:
    """Create the default expansion LLM from config.EXPAND_PROVIDER.

    "gemini" (default): hosted Gemini via GeminiChat; needs GEMINI_API_KEY.
    "ollama": local Ollama server via LangChain (imported lazily, so the Gemini
    path does not need langchain installed).

    Bangla->formal-legal-English register shifting is the riskiest LLM-dependent
    step in the pipeline — its rewrite quality on the Bangla test set is a measured
    eval item, not an assumption.
    """
    if cfg.EXPAND_PROVIDER == "gemini":
        return GeminiChat(cfg.EXPAND_GEMINI_MODEL, timeout_s=cfg.EXPAND_GEMINI_TIMEOUT_S)
    if cfg.EXPAND_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg.EXPAND_OLLAMA_MODEL, temperature=0.1)
    raise ValueError("unknown EXPAND_PROVIDER %r (use 'gemini' or 'ollama')" % cfg.EXPAND_PROVIDER)


def _system_to_msg() -> Dict[str, str]:
    """System message carrying the register-bridging instructions."""
    return {"role": "system", "content": _SYSTEM_PROMPT}


def _human_to_msg(content: str) -> Dict[str, str]:
    """User message carrying the query + the number of variants to produce."""
    return {"role": "user", "content": content}


def _user_prompt(original: str, n: int) -> str:
    """The user-side prompt asking for n formal-register variants."""
    lang = "Bangla" if _looks_bangla(original) else "English"
    return (
        f"The worker asked this question in {lang}:\n\n"
        f"'{original}'\n\n"
        f"Rewrite it into {n} formal legal-register paraphrases as JSON. "
        f"See the system instructions for the exact JSON shape and the rules."
    )


def _looks_bangla(text: str) -> bool:
    """Rough Bangla-detection: presence of any Bengali block character."""
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return True
    return False


def _first_content_text(response: Any) -> Optional[str]:
    """Pull the first piece of text content out of a LangChain chat response."""
    if response is None:
        return None
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    # Some response objects carry message.content instead.
    msg = getattr(response, "message", None)
    if msg is not None:
        c = getattr(msg, "content", None)
        if c is not None:
            return str(c)
    return None


def _parse_envelope(
    text: str, original: str, capped: int
) -> Optional[Dict[str, Any]]:
    """Parse the model's JSON response into the envelope shape.

    Returns the envelope dict, or None if the JSON is missing / malformed /
    does not match the expected shape. The expected shape is:
        { "original": ..., "variants": [...], "multi_aspect": bool,
          "historical": bool }
    We tolerate a few common failures gracefully (missing keys filled with
    defaults, variants capped to `capped`, non-list variants coerced).
    """
    # Hosted models often wrap JSON in a ```json fence; strip it before parsing.
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        variants = [str(variants)] if variants else []
    # Keep the first `capped` variants, each as a plain string.
    clean_variants: List[str] = []
    for v in variants[:capped]:
        if v is None:
            continue
        clean_variants.append(str(v).strip())
    if not clean_variants:
        # Model returned a JSON object but no usable variants — treat as
        # fallback (original as sole variant).
        return None

    return {
        "original": original,
        "variants": clean_variants,
        "multi_aspect": bool(payload.get("multi_aspect", False)),
        "historical": bool(payload.get("historical", False)),
    }


def _fallback(original: str, trace: Any, reason: str, detail: str) -> None:
    """Record a Stage 1 fallback into the trace (loud, never silent).

    The fallback envelope is NOT returned here — the caller returns it. This
    function only records the event so the trace + eval know expansion degraded.
    """
    trace.add_fallback(1, reason, "original query as sole variant")
    logger.warning("Stage 1 expansion fallback: %s — %s", reason, detail)


def _fallback_envelope(original: str) -> Dict[str, Any]:
    """The fallback envelope: original as the sole variant, both flags false.

    The filter stays default-STRICT on a degraded run — we never widen the
    legality filters (in_force / legal_status) just because expansion failed.
    Convenience features (rewriting) may degrade; correctness filters may not.
    """
    return {
        "original": original,
        "variants": [original],
        "multi_aspect": False,
        "historical": False,
    }
