"""
============================================================================
 Nyaya - Retrieval Pipeline :: per-query trace (the evaluation substrate)
============================================================================
 Every query produces ONE JSON trace written to retrieval_traces/<query_id>.json.

 This is not optional instrumentation. It is the substrate the eval harness
 consumes:
   - RAGAS context metrics are computed from the final context set
   - stage-level ablations are computed from the per-stage ranked lists
   - a wrong answer is debugged down to the stage that caused it by reading
     the trace

 The trace schema below is the single source of truth for what a trace contains.
 The orchestrator builds one trace object per query and hands it to the writer.

 Objective of this file: define the trace contract so every stage records the
 same shape, and so the eval harness can depend on that shape without guessing.

 TODO: implement the trace object + writer once Stages 1-7 exist. Right now
 this is the schema shell with each field documented.
============================================================================
"""

# ---------------------------------------------------------------------------
# Trace schema (one per query)
# ---------------------------------------------------------------------------
# A trace is a JSON object with these top-level fields:

#   query_id            str   — stable id for this query (e.g. q0001); matches
#                                the id in eval/test_queries.json so traces and
#                                gold labels can be joined by id
#   original            str   — the query as received (Bangla or English text)
#   variants            list  — the Stage 1 envelope's variants (the formal-
#                                register paraphrases), in order
#   flags               dict  — {multi_aspect: bool, historical: bool}
#                                (the Stage 1 envelope's flags; the generation
#                                stage may also carry an asr field later — not
#                                built now)
#   stages              dict  — per-stage record, keyed by stage number:
#       "1"  expansion  : {model, variants, flags, fallback_triggered, notes}
#       "2"  search     : {filter_applied: "default"|"historical",
#                          Variants: [{chunk_id, score, rank, ...}, ...],
#                          dense_vectors_retained: bool,  # for Stage 5b MMR
#                          fallback_triggered, notes}
#       "3"  fuse       : {fused_pool: [{chunk_id, rrf_score, rank}, ...],
#                          constituent_lists: [list-of-list refs],
#                          fallback_triggered, notes}
#       "4"  dedup      : {deduped_pool: [{chunk_id, ...}, ...],
#                          collapsed_pairs: [{winner_id, loser_id, reason}, ...],
#                          guard_hits: [{pair, why_not_collapsed}, ...],
#                          fallback_triggered, notes}
#       "5"  rerank     : {final_topk: [{chunk_id, rerank_score, rank}, ...],
#                          reranker: "bge-reranker-v2-m3"|"fallback_rrf",
#                          fallback_triggered, notes}
#       "5b" mmr        : {mmr_applied: bool,
#                          mmr_topk: [{chunk_id, ...}, ...] | null,
#                          reason_if_skipped, notes}
#       "6"  resolve    : {resolved_context: [{child_id, parent_id,
#                          parent_kind: "prose"|"table"|"self"|"missing",
#                          parent_text_or_json_ref, truncation_label|null}, ...],
#                          missing_parent_events: [{child_id, reason}, ...],
#                          fallback_triggered, notes}
#       "7"  overlay    : {dual_layer_pairs: [{base_chunk_id, overlay_chunk_id,
#                          base_label, overlay_label}, ...],
#                          labels_applied: [chunk_id -> label], notes}
#
#   final_context_set   list  — the context set handed to generation:
#                                [{chunk_id, label, parent_id, parent_kind,
#                                text_or_json_preview}, ...]
#                                (the full parent text / table JSON is fetched
#                                by the generation stage or passed alongside;
#                                the trace carries the labels + ids + preview)
#
#   timings             dict  — per-stage wall-clock seconds (for latency
#                                profiling; the reranker is the expected
#                                bottleneck on CPU)
#
#   overall_fallbacks   list  — every fallback triggered across all stages,
#                                with stage + reason + what was used instead.
#                                Convenience features may degrade; correctness
#                                filters may not — this list is the audit of
#                                which happened.

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
# write_trace(trace) -> path
#   pretty-prints the trace as UTF-8 JSON to TRACE_DIR/<query_id>.json
#   (TRACE_DIR from config; already in .gitignore)
#
# The writer is dumb: it serializes what it's given. Validation of the schema
# (every stage recorded, no missing required fields) belongs in the eval
# harness's trace-invariance check, not in the writer.

# ---------------------------------------------------------------------------
# Trace-invariance check (eval harness concern, documented here)
# ---------------------------------------------------------------------------
# The eval harness should assert, for every trace:
#   - all 7 stages present in trace["stages"]
#   - trace["final_context_set"] non-empty (a query that returns nothing is a
#     retrieval failure worth diagnosing, not a silent empty answer)
#   - if flags.multi_aspect is true, Stage 5b recorded whether MMR ran or was
#     skipped, with a reason
#   - if any fallback_triggered is true in a stage, overall_fallbacks records it
#   - if dual_layer pairs exist, labels_applied covers both chunks
#
# These are eval-time assertions on the trace, not runtime checks in the
# pipeline — the pipeline's job is to record honestly; the harness's job is to
# verify the record is complete enough to measure against.
