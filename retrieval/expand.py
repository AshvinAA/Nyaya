"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 1 - Multi-query expansion
============================================================================
 The pipeline's single query-understanding point (per RETRIEVAL_PIPELINE.md).
 One LLM call rewrites the input into 2-3 formal-register variants that
 bridge the gap between colloquial spoken Bangla and the formal legal
 English register of the corpus (the Act says "recruitment fee", not
 "dalal wants money"), AND emits the two intent flags everything downstream
 consumes - multi_aspect (enables Stage 5b) and historical (swaps Stage 2's
 filter). Nothing downstream re-guesses query intent.

 Structured envelope (the contract):
     {
       "original":     "<the query as received>",
       "variants":     ["<formal-register paraphrase 1>", "...2", "...3"],
       "multi_aspect": true | false,
       "historical":   true | false
     }

 Failure mode is defined and loud: if the LLM is unavailable or the JSON is
 malformed, run with the ORIGINAL query as the sole variant, all flags
 false (the filter stays default-strict - never widen legality filters on
 a degraded run), and record `fallback: true` in the trace.

 A reserved `asr` field will eventually carry transcript-cleanup signals -
 deliberately absent now; voice is deferred.
============================================================================
"""

import json
import re
import time

from retrieval.common import eprint

# ---------------------------------------------------------------------------
# Prompt - raw string, no f-string: the JSON template chars must survive
# ---------------------------------------------------------------------------

_EXPAND_PROMPT = """You rewrite a worker's question so a legal-document search \
engine can answer it.

The corpus is Bangladesh labour law: the Labour Act 2006 (with 2013/2018 \
amendments), the Labour Rules 2015, wage gazettes, and policy reports. \
Documents are written in FORMAL LEGAL English. Workers often ask in \
colloquial Bangla or everyday English.

Rewrite the question into {n} paraphrased variants that:
1. Use the formal legal vocabulary the corpus itself uses \
(e.g. "recruitment fee", "minimum wage", "termination", "notice period").
2. Preserve the question's MEANING exactly - add no new legal claims.
3. Are each self-contained (no pronouns referring to the original).

Also judge:
- "multi_aspect": true only if the question has GENUINELY distinct aspects \
(e.g. "is this fee legal AND what is the wage") that a single relevant chunk \
could not answer together.
- "historical": true only if the question is explicitly about law OVER TIME \
("what did the law say before 2018?", "what changed in the 2025 Ordinance?").

Answer with ONLY a JSON object, no markdown, no commentary:
{{"variants": ["...", "..."], "multi_aspect": false, "historical": false}}

Question: {query}
JSON:"""


def _extract_json(text):
    """Pull the first JSON object out of an LLM response. Tolerates markdown
    fences and trailing chatter - the demo model is small and chatty."""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)          # first {...} block, greedy-inside
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def expand_query(query, client=None, model=None, host=None, n_variants=None,
                 timeout_s=None, verbose=False):
    """Stage 1: one LLM call -> the structured envelope. Never raises; on
    any failure returns the degraded-but-strict envelope described in the
    module docstring. Returns (envelope, diagnostics) for the trace."""

    from retrieval import config

    model = model or config.EXPAND_MODEL
    host = host or config.OLLAMA_HOST
    n = max(1, int(n_variants or config.MAX_VARIANTS))
    timeout = timeout_s or config.EXPAND_TIMEOUT_S

    envelope = {"original": query, "variants": [], "multi_aspect": False,
                "historical": False}
    diag = {"stage": "1_expand", "model": model, "host": host,
            "fallback": False, "reason": None, "elapsed_s": None}

    prompt = _EXPAND_PROMPT.format(n=n, query=query)

    started = time.time()
    raw = None
    try:
        if client is None:
            from ollama import Client            # local import: Ollama is optional
            client = Client(host=host)
        resp = client.generate(model=model, prompt=prompt,
                               options={"temperature": 0.0},
                               keep_alive="5m")  # keep the model warm for batches
        raw = resp.get("response", "")
    except Exception as exc:                     # connection refused / model absent
        diag["reason"] = "llm_error: %s" % exc
        diag["fallback"] = True
    diag["elapsed_s"] = round(time.time() - started, 2)

    if raw is not None:
        parsed = _extract_json(raw)
        if parsed and isinstance(parsed.get("variants"), list):
            variants = []
            for v in parsed["variants"]:
                if isinstance(v, str):
                    v = v.strip()
                    if v and v.lower() != query.strip().lower() and v not in variants:
                        variants.append(v)
            envelope["variants"] = variants[:n]
            envelope["multi_aspect"] = bool(parsed.get("multi_aspect"))
            envelope["historical"] = bool(parsed.get("historical"))
        else:
            diag["fallback"] = True
            diag["reason"] = (diag["reason"] or "malformed_json")
            diag["raw_preview"] = (raw or "")[:200]

    if not envelope["variants"]:                 # hard fallback rule
        envelope["variants"] = []
        envelope["multi_aspect"] = False
        envelope["historical"] = False
        # degraded run: original query is the sole search key, flags false,
        # and the Stage 2 filter stays default-strict. Logged, never silent.
        if not diag["fallback"]:
            diag["fallback"] = True
            diag["reason"] = diag["reason"] or "no_usable_variants"
        eprint("[stage1] fallback: %s" % diag["reason"])

    if verbose:
        eprint("[stage1] variants=%s multi_aspect=%s historical=%s (%.2fs, fallback=%s)"
               % (envelope["variants"], envelope["multi_aspect"],
                  envelope["historical"], diag["elapsed_s"], diag["fallback"]))

    return envelope, diag
