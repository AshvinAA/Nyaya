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
      amendment overlay are near-identical by construction — and Stage 7
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
# amendment overlay (text_layer=amendment_overlay) are near-identical by
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

# ===========================================================================
# IMPLEMENTATION (below the design contract; mirrors the contract exactly)
# ===========================================================================

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ingestion import config as icfg
from ingestion.deduplicator import AUTHORITY_ORDER
from retrieval import config as cfg

logger = logging.getLogger("retrieval.stage4")

# Lazily-built MiniLM scorer, shared per process (same model ingestion Stage 6
# uses — DEDUP_MODEL in ingestion/config.py). Fused pools are ~30 chunks, so
# encoding the pool once per query is cheap.
_scorer_model = None


def _authority_rank(entry: Dict[str, Any]) -> int:
    """Map a fused-pool entry to ingestion's authority rank (lower = better).

    Uses the AUTHORITY_ORDER imported from ingestion.deduplicator — the SAME
    ranking ingestion applied, applied here PER-QUERY. Unknown/missing ranks
    sort last (9), matching ingestion's `_rank` fallback.
    """
    meta = entry.get("metadata") or {}
    ar = meta.get("authority_rank")
    return AUTHORITY_ORDER.get(ar, 9)


def _embed_pool_texts(texts: List[str]) -> np.ndarray:
    """Encode the fused pool's texts with the cached MiniLM dedup scorer.

    Uses the same model as ingestion Stage 6 (ingestion.config.DEDUP_MODEL).
    Returns an (n, d) float32 matrix of L2-normalised vectors for cosine
    similarity via dot products.
    """
    global _scorer_model
    if _scorer_model is None:
        from sentence_transformers import SentenceTransformer
        _scorer_model = SentenceTransformer(icfg.DEDUP_MODEL)
    vecs = _scorer_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def retrieval_dedup(
    fused_pool: List[Dict[str, Any]],
    registry: Any,
    trace: Any,
) -> Dict[str, Any]:
    """Collapse per-query near-duplicate distinct IDs by authority rank.

    Parameters
    ----------
    fused_pool : list of dict
        The Stage 3 fused pool — entries carry at least
        {chunk_id, rrf_score, rank, appears_in}.
    registry : retrieval.registry (the module)
        Used to read each chunk's metadata (authority_rank / text_layer /
        source_act, via registry.chunk_metadata) and bare generation text (via
        registry.chunk_document) for similarity scoring.
    trace : Trace
        The in-progress trace. Records the collapsed pairs, guard hits, and the
        deduped pool into stages["4"].

    Returns
    -------
    dict (also recorded into trace["stages"]["4"]):
        {
          "deduped_pool": [  # surviving entries in RRF-rank order
              {chunk_id, rrf_score, rank, appears_in, metadata, text}, ...],
          "collapsed_pairs": [ {winner_id, loser_id, reason, similarity}, ...],
          "guard_hits":      [ {pair: [id_a, id_b], similarity, why}, ...],
          "fallback_triggered": bool,
          "notes": str,
        }
    """
    t0 = time.time()

    if not fused_pool:
        record = {
            "deduped_pool": [],
            "collapsed_pairs": [],
            "guard_hits": [],
            "fallback_triggered": True,
            "notes": "no fused pool provided to dedup",
        }
        trace.record_stage(4, record)
        trace.add_fallback(4, "empty_fused_pool", "nothing to dedup")
        logger.warning("Stage 4: empty fused pool — nothing to dedup")
        return record

    # Read metadata + generation text for every distinct-ID candidate in one
    # batch pass, then re-attach them to the pooled entries in place.
    ids = [e.get("chunk_id") for e in fused_pool if e.get("chunk_id")]
    metas_by_id = registry.chunk_metadata_batch(ids) if hasattr(registry, "chunk_metadata_batch") else {}
    for e in fused_pool:
        cid = e.get("chunk_id")
        meta = metas_by_id.get(cid) if cid else None
        if meta is None:
            meta = registry.chunk_metadata(cid) if cid else None
        e["metadata"] = meta or {}
        e["text"] = registry.chunk_document(cid) if cid else ""

    collapsed_pairs: List[Dict[str, Any]] = []
    guard_hits: List[Dict[str, Any]] = []
    dropped: set = set()

    threshold = float(cfg.RETRIEVAL_NEAR_DUP_THRESHOLD)

    if len(fused_pool) > 1:
        texts = [e.get("text") or "" for e in fused_pool]
        vecs = _embed_pool_texts(texts)
        sims = vecs @ vecs.T   # cosine (vectors are L2-normalised)
        n = len(fused_pool)
        for i in range(n):
            if i in dropped:
                continue
            for j in range(i + 1, n):
                if j in dropped:
                    continue
                a, b = fused_pool[i], fused_pool[j]
                sim = float(sims[i, j])
                if sim < threshold:
                    continue
                layer_a = (a.get("metadata") or {}).get("text_layer")
                layer_b = (b.get("metadata") or {}).get("text_layer")
                act_a = (a.get("metadata") or {}).get("source_act")
                act_b = (b.get("metadata") or {}).get("source_act")

                # HARD GUARD: never collapse across text_layer OR source_act.
                # A base provision and its amendment overlay are near-
                # identical by construction; Stage 7 exists to surface both.
                if layer_a != layer_b or act_a != act_b:
                    guard_hits.append(
                        {
                            "pair": [a["chunk_id"], b["chunk_id"]],
                            "similarity": round(sim, 6),
                            "why": (
                                "text_layer differs (%s vs %s); source_act differs (%s vs %s)"
                                % (
                                    layer_a,
                                    layer_b,
                                    act_a,
                                    act_b,
                                )
                            ),
                        }
                    )
                    continue

                # Same layer + same act -> collapse to the higher authority;
                # tie-break on the higher RRF score (earlier fused rank).
                rank_a, rank_b = _authority_rank(a), _authority_rank(b)
                if rank_a < rank_b:
                    winner, loser = a, b
                elif rank_b < rank_a:
                    winner, loser = b, a
                else:
                    # Same authority tier: the higher RRF score wins
                    # (earlier fused rank = more list consensus upstream).
                    if a.get("rank", 10**9) <= b.get("rank", 10**9):
                        winner, loser = a, b
                    else:
                        winner, loser = b, a
                dropped.add(fused_pool.index(loser))
                collapsed_pairs.append(
                    {
                        "winner_id": winner["chunk_id"],
                        "loser_id": loser["chunk_id"],
                        "reason": "near-dup; authority_rank %d beats %d"
                        % (min(rank_a, rank_b), max(rank_a, rank_b)),
                        "similarity": round(sim, 6),
                    }
                )

    deduped_pool: List[Dict[str, Any]] = [
        e for k, e in enumerate(fused_pool) if k not in dropped
    ]

    elapsed = time.time() - t0
    record = {
        "deduped_pool": deduped_pool,
        "collapsed_pairs": collapsed_pairs,
        "guard_hits": guard_hits,
        "fallback_triggered": False,
        "notes": _notes(len(fused_pool), len(deduped_pool), collapsed_pairs, guard_hits, elapsed),
    }
    trace.record_stage(4, record)
    logger.info(
        "Stage 4: pool_in=%d collapsed=%d guard_hits=%d pool_out=%d elapsed=%.3fs",
        len(fused_pool),
        len(collapsed_pairs),
        len(guard_hits),
        len(deduped_pool),
        elapsed,
    )
    return record


def _notes(
    pool_in: int,
    pool_out: int,
    collapsed: List[Dict[str, Any]],
    guards: List[Dict[str, Any]],
    elapsed: float,
) -> str:
    """A short human-readable note about what Stage 4 did, for the trace."""
    return (
        "pool_in=%d pool_out=%d collapsed=%d guard_hits=%d threshold=%s elapsed=%.3fs"
        % (
            pool_in,
            pool_out,
            len(collapsed),
            len(guards),
            cfg.RETRIEVAL_NEAR_DUP_THRESHOLD,
            elapsed,
        )
    )
