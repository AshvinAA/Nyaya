"""
============================================================================
 Nyaya - Retrieval Pipeline :: orchestrator
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Wires Stages 1-7 into one entry point:

   retrieve(query, ...) -> (final_context_set, trace)

 Flow (in order):

   1. Stage 1 — expand_query(original) -> envelope {variants, multi_aspect,
      historical}
   2. Stage 2 — hybrid_search(envelope, registry) -> per_variant_lists,
      filter_applied
   3. Stage 3 — rrf_fuse(per_variant_lists) -> fused_pool (capped ~30)
   4. Stage 4 — retrieval_dedup(fused_pool, registry) -> deduped_pool
   5. Stage 5 — rerank(deduped_pool, original_query) -> final_topk (~5)
   6. Stage 5b — optional_mmr(final_topk, dense_vectors, envelope)
      -> mmr_topk  (only if envelope["multi_aspect"] is True; else the Stage 5
      final_topk passes through)
   7. Stage 6 — resolve_parents(mmr_or_stage5_topk, registry) ->
      resolved_context
   8. Stage 7 — resolve_overlay(resolved_context) -> final_context_set

 Every stage records its inputs/outputs into the trace as it runs. Every fallback
 is logged, never silent. The function returns BOTH the final context set (what
 generation receives) and the trace (what evaluation consumes).

 Used by: the generation stage (built much later) as its retrieval call; and by
          cli.py for demo / trace inspection.

 Objective of this file: be the single "query in, labeled context out" entry point
 that the rest of the system calls, with a complete trace as the side effect. No
 stage is bypassed; no fallback is silent.
============================================================================
"""

# ---------------------------------------------------------------------------
# retrieve() — the single entry point
# ---------------------------------------------------------------------------
# retrieve(query: str, query_id: str, registry, trace_writer) -> (context_set, trace)
#   query       : the query as received (Bangla or English text)
#   query_id    : stable id for this query (matches eval/test_queries.json id so
#                 traces and gold labels can be joined)
#   registry    : the index read registry (openers, dense_search, resolve_parent,
#                 chunk_metadata, ...)
#   trace_writer: the trace writer (trace.write_trace)
#
#   returns     : (final_context_set, trace)
#     final_context_set : the labeled, parent-resolved, deduplicated, reranked
#                         (and MMR-diversified if applicable) context set ready for
#                         generation
#     trace             : the complete per-query trace, written to
#                         retrieval_traces/<query_id>.json
#
# The orchestrator is the ONLY place that sequences the stages. No stage calls
# another stage directly — each stage is a pure function of (its inputs, the
# registry, the trace). This keeps each stage testable in isolation and keeps the
# ablation config able to swap / skip any stage by re-routing the orchestrator's
# sequence.

# ---------------------------------------------------------------------------
# Trace lifecycle
# ---------------------------------------------------------------------------
# One trace object is created at the start of retrieve() and passed to every
# stage. Each stage appends its own record into trace["stages"]["<n>"] and any
# stage-global fields (flags, final_context_set, timings, overall_fallbacks).
#
# The trace is written ONCE at the end of retrieve() (trace_writer.write_trace),
# after all stages have run. If retrieve() is interrupted partway (exception,
# early return), the partial trace should still be written (with the stages that
# ran recorded and the ones that didn't marked as not_run) — a partial trace is
# more useful than a missing one, because it shows WHERE the failure happened.
#
# Timings: each stage records its wall-clock time into trace["timings"]["<n>"].
# The reranker (Stage 5) is the expected bottleneck on CPU; the timings make that
# visible rather than assumed.

# ---------------------------------------------------------------------------
# Fallback propagation
# ---------------------------------------------------------------------------
# Each stage's fallback (defined in its own file) records into the trace and
# returns a degraded-but-usable result. The orchestrator does NOT re-decide
# fallbacks — it just passes the degraded result to the next stage. This is the
# degradation policy: every stage degrades on its own terms, loudly, and the
# orchestrator sequences whatever each stage returned.
#
# The one asymmetry the orchestrator ENFORCES (not the stages): if Stage 1
# triggered its fallback (expansion failed), the filter stays default-STRICT for
# Stage 2 — the orchestrator should not let a degraded expansion widen the
# legality filters. The stages record their own fallbacks; the orchestrator
# enforces the one cross-stage constraint (don't widen filters on a degraded
# expansion run).

# ---------------------------------------------------------------------------
# Ablation hook (eval harness concern — structured here so the harness can use it)
# ---------------------------------------------------------------------------
# The orchestrator's stage sequence is what the eval harness's ablation config
# rewires. To support ablation, the orchestrator should be written so that each
# stage is a separable step that can be:
#   - skipped (e.g. "reranker off" -> Stage 5 returns the deduped pool as the
#     final top-k, in RRF order)
#   - replaced (e.g. "RRF off" -> Stage 3 returns one dense list instead of the
#     fused pool)
#   - forced on/off (e.g. "MMR forced on" -> Stage 5b runs even if multi_aspect
#     is False; "MMR forced off" -> Stage 5b always skips)
#
# The ablation config (in eval/) rewires these without touching the stage files
# themselves. The stage files don't know about ablation — the orchestrator's
# sequence does.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not implement any stage's logic — that's each stage file's job.
#   - It does not generate an answer — that's the generation stage (built much
#     later), which receives this stage's final_context_set.
#   - It does not write the trace mid-flow (it writes once at the end, after all
#     stages ran or degraded).
#   - It does not decide the generation prompt — that's the generation stage (and
#     the Stage 7 open design question about dual-layer chunks).
#   - It does not handle ASR — no asr field is built yet; if one is added later,
#     it enters at Stage 1's envelope, not here.
