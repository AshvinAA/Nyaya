"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 2 - Hybrid search, with metadata
 pre-filtering
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Each query variant runs against the BGE-M3 index using both dense and sparse
 representations (sparse per the upgrade path; dense-only until then).

 METADATA FILTERING HAPPENS HERE, NOT AS AN AFTERNOOTH. The index already
 contains only chunks that passed the ingestion gate (retrievable=true, no
 front matter, no demoted duplicates), so the filter below is the ENTIRE
 applicability control:

   Default filter:  in_force = true  AND  legal_status != superseded
   Historical filter (when Stage 1 flagged historical): SWAP the filter
                  entirely — drop the in_force / legal_status constraints.
                  Not loosen at the edges. A superseded provision is by
                  definition not in force, so partial relaxation is incoherent.
                  Stage 7's labeling takes over the job of saying which text
                  governed when.

 The 2025 Ordinance survives the default filter BY DESIGN: it is
 in_force=true with legal_status=pending_ratification, so it competes normally
 in default mode — and Stage 7 labels it. The filter's job is applicability,
 not ratification status; conflating them would silently hide the current
 operative text. This is the retrieval-side half of ingestion Stage 0's overlay
 rule.

 This is the retrieval-side half of the ingestion doc's Stage 0 overlay rule.

 Dense-only until the sparse upgrade lands (BGE-M3 learned lexical weights in
 Qdrant named vectors — NOT BM25). When sparse exists, Stage 2 produces a
 second list per variant and Stage 3 fuses across both — no downstream change.
 The upgrade is an ingestion-side change to Stage 7 only.

 Used by: orchestrator.py (Stage 2 of the retrieve() flow).

 Objective of this file: take the Stage 1 envelope + the index registry, run
 each variant through hybrid search with the right filter, and return per-variant
 ranked lists (chunk_id, score, rank) plus the filter that was applied (so the
 trace records filter: default or filter: historical). Dense-only for now;
 written for the general two-list case from the start.
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 2 contract
# ---------------------------------------------------------------------------
# hybrid_search(envelope, registry, trace) -> per_variant_lists, filter_applied
#   envelope      : the Stage 1 envelope (variants + flags)
#   registry      : the index read registry (dense_search, chunk_metadata, ...)
#   trace        : the in-progress trace (so this stage records per-variant
#                   ranked lists, the filter applied, dense_vectors_retained,
#                   and any fallback)
#   returns       : (per_variant_lists, filter_applied)
#     per_variant_lists : list of ranked lists, one per variant
#       [{chunk_id, score, rank, metadata}, ...]  (rank-positioned, not raw
#       score — because Stage 3 fuses by rank, and dense/sparse scores aren't
#       comparable)
#     filter_applied : "default" | "historical"   (recorded in the trace)

# ---------------------------------------------------------------------------
# Per-variant dense search (current path)
# ---------------------------------------------------------------------------
# For each variant in envelope["variants"]:
#   - run dense_search against nyaya_children with the appropriate filter
#     (DEFAULT_FILTER or HISTORICAL_FILTER from config, chosen by
#     envelope["historical"])
#   - collect the ranked list (chunk_id, score, rank, metadata)
#   - RETAIN the dense vector per candidate if MMR may run later (Stage 5b
#     uses the BGE-M3 dense vectors already computed here — do not recompute;
#     so Stage 2 must carry dense vectors alongside the ranked list when
#     config.SPARSE_AVAILABLE is False and MMR might apply)
#
# Dense scores are cosine similarities; they're comparable across dense lists
# from the same model, but the pipeline fuses by rank anyway (Stage 3), so
# return rank-positioned lists, not raw-score-sorted-for-fusion lists.

# ---------------------------------------------------------------------------
# Metadata pre-filter (the applicability control)
# ---------------------------------------------------------------------------
# The filter is applied INSIDE the search call (Chroma where-filter on the
# stored metadata fields), not as a post-hoc check on the results. This is the
# design principle: filter at the source.
#
# Default filter (envelope["historical"] is False):
#   in_force = true  AND  legal_status != superseded
#   - 2025 Ordinance (in_force=true, legal_status=pending_ratification) SURVIVES
#     and competes normally — Stage 7 labels it, the filter does not hide it.
#
# Historical filter (envelope["historical"] is True):
#   SWAP the filter entirely — drop in_force / legal_status constraints.
#   - This is a SWAP, not a loosening. Partial relaxation ("allow superseded but
#     still require in_force") is incoherent because a superseded provision is by
#     definition not in force.
#   - Stage 7 labeling takes over saying which text governed when.
#
# The filter's job is APPLICABILITY, not ratification status. Confusing the two
# would silently hide the current operative text (the 2025 Ordinance), which is
# exactly what the overlay rule exists to prevent.

# ---------------------------------------------------------------------------
# Sparse path (future — only when the upgrade lands)
# ---------------------------------------------------------------------------
# When config.SPARSE_AVAILABLE is True (Qdrant named sparse vectors exist),
# hybrid_search also calls sparse_search for each variant and produces a second
# ranked list per variant. Stage 3 then fuses across 2 modes x N variants =
# 2N lists instead of N lists.
#
# NO downstream stage changes for this. Stage 3 fuses ranked lists generically;
# Stages 4-7 consume ranked lists regardless of how many modes produced them.
# The sparse upgrade is an ingestion Stage 7 change + this file's second list.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not fuse lists — that's Stage 3 (RRF).
#   - It does not dedup — that's Stage 4 (retrieval-time dedup).
#   - It does not rerank — that's Stage 5 (cross-encoder vs original query).
#   - It does not resolve parents — that's Stage 6.
#   - It does not label overlays — that's Stage 7.
#   - It does not interpret query intent — that's Stage 1's envelope; Stage 2
#     only reads envelope["variants"] and envelope["historical"].
#
#   - It does NOT run BM25. The plan's sparse is BGE-M3 learned lexical weights
#     in Qdrant named vectors, NOT BM25. A true BM25 side-channel is optional,
#     not a requirement of the design. Do not build a BM25 index "to be safe."
