"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 4 - Retrieval-time dedup
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Ingestion-time dedup (ingestion doc, Stage 6) is a STATIC, ONE-TIME corpus
 cleanup. This is a DYNAMIC, PER-QUERY problem, and the two must never be
 conflated.

 What this stage actually does:

   1. Distinct-ID near-dup collapse. Multi-query fan-out can surface two
      DIFFERENT chunk IDs carrying the same fact from different angles — and
      the final context budget is small (top-k ~5), so a duplicate burns a slot
      that should carry new evidence. Ingestion dedup removed same-layer,
      same-act duplicates before indexing; what can still co-occur here are
      pairs ingestion REFUSED to merge (different source_act or text_layer) plus
      threshold stragglers. Resolve by the SAME authority ranking as ingestion
      (authority_rank: official gazette > official translation > reputable
      secondary > advocacy), applied PER-QUERY instead of once over the corpus.

   2. Hard guard, inherited from ingestion: NEVER collapse a pair across a
      different text_layer or source_act. A base-text provision and its
      2025-Ordinance overlay are near-identical by construction — and Stage 7
      exists precisely to surface both with labels. If this stage collapsed
      them, it would destroy the very thing the overlay rule protects. When in
      doubt, leave both in and let the reranker and the small final-k do the
      trimming.

   3. Runs BEFORE reranking on purpose: the cross-encoder is the most expensive
      call in the pipeline — never spend it re-scoring a duplicate.

   4. Exact-ID duplicates need no handling at this stage at all: RRF already
      merged them into one entry (same ID in multiple lists = one RRF entry).

 Used by: orchestrator.py (Stage 4 of the retrieve() flow).

 Objective of this file: take the fused pool from Stage 3, collapse per-query
 near-duplicate distinct IDs by authority rank (with the hard cross-layer guard),
 and return a smaller deduped pool to Stage 5 — plus the collapsed pairs + guard
 hits for the trace and the eval ablation row "dedup off -> wasted slots /
 double-counted evidence."
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 4 contract
# ---------------------------------------------------------------------------
# retrieval_dedup(fused_pool, registry, trace) -> deduped_pool
#   fused_pool  : the Stage 3 fused pool [{chunk_id, rrf_score, rank, ...}, ...]
#   registry    : the index read registry (chunk_metadata for authority_rank /
#                 text_layer / source_act lookup per chunk_id)
#   trace       : the in-progress trace (so this stage records collapsed pairs,
#                 guard hits, and the deduped pool)
#   returns     : deduped_pool = [{chunk_id, rrf_score, rank, ...}, ...]
#                 (smaller than the fused pool; what Stage 5 receives)
#
# This stage does NOT touch ingestion-time dedup. Ingestion dedup already ran
# (or will run) on the corpus and demoted exact/near duplicates to
# retrievable=false BEFORE indexing. This stage only handles what can still
# co-occur at query time: distinct IDs that ingestion refused to merge plus
# threshold stragglers.

# ---------------------------------------------------------------------------
# Near-dup collapse (per-query, authority-resolved)
# ---------------------------------------------------------------------------
# For each PAIR of DISTINCT IDs in the fused pool:
#   - compute similarity (the fused pool is ~30 chunks, so pairwise is cheap —
#     reuse the cached MiniLM scorer from ingestion Stage 6, or embed the small
#     pool with BGE-M3; do NOT use the cross-encoder here — that's Stage 5)
#   - if similarity >= config.RETRIEVAL_NEAR_DUP_THRESHOLD (a SEPARATE tunable
#     from ingestion's 0.92 — different population, different purpose):
#       - if the pair shares the same text_layer AND same source_act:
#           -> collapse to the higher-authority chunk (authority_rank from
#              ingestion's AUTHORITY_ORDER; reuse it, don't re-declare)
#           -> record the collapse in the trace (winner_id, loser_id, reason,
#              similarity)
#       - if the pair DIFFERS on text_layer OR source_act:
#           -> DO NOT COLLAPSE (hard guard inherited from ingestion)
#           -> record the guard hit in the trace (pair, why not collapsed)
#   - if similarity < threshold: not a pair, leave both
#
# The RETRIEVAL_NEAR_DUP_THRESHOLD is a tunable to be set and swept against the
# test set's duplicate-rate metric. Start conservative; the population (fused
# pool candidates, ~30 chunks) is different from ingestion's (whole corpus), so
# the threshold semantics may differ. Don't assume ingestion's 0.92 transfers.

# ---------------------------------------------------------------------------
# Hard guard (inherited from ingestion — non-negotiable)
# ---------------------------------------------------------------------------
# NEVER collapse a pair across a different text_layer or source_act.
#
# Rationale: a base-text provision (text_layer=consolidated_base) and its
# 2025-Ordinance overlay (text_layer=amendment_overlay) are near-identical by
# construction — the overlay is a delta applied to the same provision. Stage 7
# exists precisely to surface BOTH with explicit labels. If this stage collapsed
# them, it would destroy the overlay rule and the system would silently pick one
# version — exactly the failure the whole layering design prevents.
#
# When in doubt (similarity high, guard hit), leave BOTH in and let the reranker
# (Stage 5) and the small final-k do the trimming. The guard is stricter than
# "collapse unless sure" — it's "don't collapse across layers, period."

# ---------------------------------------------------------------------------
# Authority ranking (reuse ingestion's, applied per-query)
# ---------------------------------------------------------------------------
# Same ranking as ingestion Stage 6 near-dup resolution:
#   official_gazette > official_translation > unofficial_translation >
#   reputable_secondary > advocacy_report
#   (AUTHORITY_ORDER in ingestion.deduplicator or ingestion.doc_metadata)
#
# Import it from ingestion — do NOT re-declare it here. The ranking is a shared
# project constant; retrieval applies it per-query instead of once over the
# corpus. Re-declaring it would let the two drift apart, which is exactly the
# kind of inconsistency the dedup-design doc warns against.

# ---------------------------------------------------------------------------
# Why before reranking
# ---------------------------------------------------------------------------
# The cross-encoder (Stage 5) is the most expensive call in the pipeline. Never
# spend it re-scoring a duplicate. Stage 4 collapses dupes first; Stage 5 scores
# the deduped pool. The pool entering Stage 5 is what the FINAL_K ~5 final
# context comes from — every duplicate slot in that pool is a slot that should
# have carried new evidence.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search or fuse — that's Stages 2/3 (it receives the fused
#     pool).
#   - It does not rerank — that's Stage 5 (it produces the deduped pool for
#     Stage 5).
#   - It does not resolve parents — that's Stage 6.
#   - It does not label overlays — that's Stage 7.
#   - It does not touch ingestion-time dedup (exact hash + near-dup demotion
#     before indexing) — that's a separate, static pass; this stage is dynamic
#     and per-query. Don't conflate the two.
#   - It does NOT collapse same-ID entries — RRF already merged those into one
#     entry; Stage 4 only handles DISTINCT IDs.
