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
     ("what did the law say before 2018?", "what changed in the 2025
     Ordinance?"). Stage 2 consumes it; nothing else in the pipeline guesses
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

# ---------------------------------------------------------------------------
# Stage 1 contract
# ---------------------------------------------------------------------------
# expand_query(original: str, trace: dict) -> envelope dict
#   original  : the query as received (Bangla or English text)
#   trace     : the in-progress trace object (so this stage records its
#               variants, flags, model used, and any fallback into it)
#   returns   : the envelope dict with keys original, variants, multi_aspect,
#               historical (and, later, asr — not built now)
#
# The function is the ONLY query-understanding point in the pipeline. It does
# NOT decide what to retrieve, what to filter, or what to rank — it only
# interprets the query enough to produce variants + two flags. Everything else
# is a downstream consumer of this envelope.

# ---------------------------------------------------------------------------
# Variant generation (the register-bridging step)
# ---------------------------------------------------------------------------
# The LLM is asked to produce 2-3 formal-register paraphrases of the original
# query — shifting colloquial Bangla ("দালাল কী টাকা চাইলে বলবো?") toward
# the corpus's legal English register ("is the recruitment fee demanded by a
# middleman lawful under the Labour Act?").
#
# This is the step that exists because the corpus demanded it: the Act says
# "recruitment fee" / "placement charge", not "dalal wants money". Without
# expansion, semantic search has to generalize across the register gap by luck;
# with expansion, the variants explicitly carry the formal vocabulary into the
# search. The eval harness measures this as the "variants off -> recall delta"
# ablation row (the register-bridging value).
#
# Variant quality on Bangla is a measured item, not an assumption: the demo
# llama3.2 is weak at Bangla, so the Bangla register-gap queries in the test
# set are the spot-check set that decides whether the expansion model is good
# enough or whether a stronger local model / two-step translate-then-expand
# path is needed. Decided by measurement, not preference.

# ---------------------------------------------------------------------------
# Flag derivation (multi_aspect, historical)
# ---------------------------------------------------------------------------
# These flags are derived by the SAME expansion step that produces the
# variants — one structured output, one query-understanding point. They are
# NOT inferred downstream by re-examining the query.
#
# multi_aspect (bool):
#   true when generating variants surfaces that the question has genuinely
#   distinct aspects. E.g. a query that is both a legal-question AND a
#   job-search ("is this fee legal and who's hiring lockstitch operators in
#   Gazipur?"). This flag is what conditionally enables Stage 5b (MMR).
#   The flag's correctness is an eval item: the harness measures how often it
#   fires and whether it fires correctly against the multi-aspect vs
#   single-aspect control pairs in the test set.
#
# historical (bool):
#   true when the question is explicitly about law over time ("what did the
#   law say before 2018?", "what changed in the 2025 Ordinance?"). Stage 2
#   consumes this flag to SWAP the filter (not loosen it). Nothing else in the
#   pipeline guesses at history — if Stage 1 didn't flag it, the default
#   applicability filter holds.

# ---------------------------------------------------------------------------
# Fallback (defined, loud — never silent)
# ---------------------------------------------------------------------------
# If the LLM is unavailable or returns malformed JSON:
#   - variants = [original]   (the original query as the sole variant)
#   - multi_aspect = false
#   - historical = false
#   - trace records: expansion_fallback_triggered = true, reason, model attempted
#
# CRITICAL asymmetry: on a degraded run, the filter stays default-STRICT. We
# never widen legality filters (in_force / legal_status) on a degraded run just
# because expansion failed. Convenience features may degrade; legal-applicability
# filters may not. This is the same fails-loud ethos as the ingestion gate.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search (Stage 2).
#   - It does not interpret the variants for retrieval — it only produces them.
#   - It does not decide final context — that's Stages 2-7.
#   - It does not handle ASR noise — that's deferred (reserved asr field exists
#     for later; do not build against it now).
#   - It does not generate an answer — that's the generation stage (built much
#     later, after retrieval is validated).
