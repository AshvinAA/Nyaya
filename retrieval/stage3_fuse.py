"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 3 - RRF fusion
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 With 2-3 variants x 2 retrieval modes (dense/sparse), expect 4-6 ranked lists
 per query. Merge by RANK POSITION (not raw score, since dense and sparse scores
 aren't on the same scale):

   score(chunk) = sum( 1 / (k + rank_in_list) )   — summed across all lists the
                                                    chunk appears in

 with the standard constant k = 60 and a per-list depth of 10 as starting points
 (config.SEARCH_PER_LIST_K, config.RRF_K).

 Two properties worth making explicit:

   1. Same-ID fan-out merges for free. If the same chunk surfaces from four of
      six lists, RRF sums four contributions — that is a RELEVANCE SIGNAL, not
      a bug. The double-counting problem Stage 4 addresses is a DIFFERENT one
      (distinct chunks, same fact).
   2. Consensus beats luck. Chunks that rank well across multiple variants and
      modes surface to the top — genuine relevance rather than a single lucky
      embedding match.

 The fused pool is capped (top ~30 by RRF score) before Stage 4 — enough to
 survive dedup and reranking with a real final-k, small enough to keep the
 cross-encoder cheap.

 Dense-only right now means 2-3 lists instead of 4-6 — RRF's fusion signal is
 thinner than the design assumes, which is fine as long as we measure the
 pre-sparse recall@k baseline and watch the table/numeric lookup category
 specifically (the predicted dense-only weak spot). When sparse lands, Stage 3
 automatically fuses across more lists — no change to this file.

 Used by: orchestrator.py (Stage 3 of the retrieve() flow).

 Objective of this file: take the per-variant ranked lists from Stage 2, fuse
 them by RRF rank position into one capped fused pool, and return it (plus the
 constituent-list references and per-chunk fusion scores, for the trace and for
 the eval ablation row "RRF off -> use one dense list").
============================================================================
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from retrieval import config as cfg

logger = logging.getLogger("retrieval.stage3")


def rrf_fuse(
    per_variant_lists: List[List[Dict[str, Any]]],
    trace: Any,
) -> Dict[str, Any]:
    """Fuse per-variant ranked lists into one capped RRF-ranked pool.

    Parameters
    ----------
    per_variant_lists : list of list of dict
        One ranked list per variant (later one per variant x mode). Each inner
        list is a list of dicts with at least chunk_id and rank (1-based within
        that list). Example entry: {"chunk_id": "chk0003", "score": 0.81,
        "rank": 1, "metadata": {...}}.
    trace : Trace
        The in-progress trace. This function records the fused pool, per-chunk
        RRF scores, the constituent list counts, and any fallback.

    Returns
    -------
    dict (recorded into trace["stages"]["3"] by the orchestrator):
        {
          "fused_pool": [  # top config.FUSED_POOL_CAP by RRF score, rank order
              {"chunk_id": ..., "rrf_score": ..., "rank": ..., "appears_in": int},
              ...
          ],
          "constituent_list_counts": [int, ...],   # number of candidates per input list
          "fallback_triggered": bool,
          "notes": str,
        }
    """
    t0 = time.time()

    if not per_variant_lists:
        record = _empty_record()
        trace.record_stage(3, record)
        logger.warning("Stage 3: no variant lists to fuse")
        return record

    list_counts = [len(lst) for lst in per_variant_lists]

    # Build RRF score per chunk across all lists it appears in.
    # rrf_score[chunk_id] = sum( 1 / (k + rank_in_list) ) over lists containing it.
    scores: Dict[str, float] = {}
    appearances: Dict[str, int] = {}
    for lst in per_variant_lists:
        for entry in lst:
            cid = entry.get("chunk_id")
            if not cid:
                continue
            rank = entry.get("rank", 1)
            if rank is None:
                rank = 1
            # RRF contribution from this list for this chunk.
            contrib = 1.0 / (cfg.RRF_K + rank)
            scores[cid] = scores.get(cid, 0.0) + contrib
            appearances[cid] = appearances.get(cid, 0) + 1

    # Sort by RRF score descending and cap to FUSED_POOL_CAP.
    capped = int(cfg.FUSED_POOL_CAP)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:capped]

    fused_pool: List[Dict[str, Any]] = []
    for rank, (cid, rrf) in enumerate(ranked, start=1):
        fused_pool.append(
            {
                "chunk_id": cid,
                "rrf_score": round(rrf, 6),
                "rank": rank,
                "appears_in": appearances.get(cid, 0),
            }
        )

    elapsed = time.time() - t0
    record = {
        "fused_pool": fused_pool,
        "constituent_list_counts": list_counts,
        "fallback_triggered": False,
        "notes": _notes(len(per_variant_lists), len(fused_pool), elapsed),
    }
    logger.info(
        "Stage 3: lists=%d unique_chunks=%d fused_pool=%d elapsed=%.3fs",
        len(per_variant_lists),
        len(scores),
        len(fused_pool),
        elapsed,
    )
    return record


def _empty_record() -> Dict[str, Any]:
    """A valid Stage 3 record for the degenerate case (no lists to fuse)."""
    return {
        "fused_pool": [],
        "constituent_list_counts": [],
        "fallback_triggered": True,
        "notes": "no variant lists provided to fuse",
    }


def _notes(n_lists: int, pool_size: int, elapsed: float) -> str:
    """A short human-readable note about what Stage 3 did, for the trace."""
    return (
        "lists="
        + str(n_lists)
        + " fused_pool="
        + str(pool_size)
        + " k="
        + str(cfg.RRF_K)
        + " elapsed=%.3fs"
        + str(round(elapsed, 3))
    )


# ------------------------------------------------------------------
# RRF formula, exposed as a pure function for the eval harness
# ------------------------------------------------------------------

def rrf_score_across(
    chunk_id: str,
    per_variant_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> float:
    """Compute the RRF score for a single chunk across all input lists.

    Pure function (no side effects, no trace). Useful for the eval harness when
    it wants to recompute or ablate RRF per chunk rather than read the fused pool.

    Parameters
    ----------
    chunk_id : str
        The chunk to score.
    per_variant_lists : list of list of dict
        The same input Stage 3 receives.
    k : int
        The RRF constant (default 60 = config.RRF_K).

    Returns
    -------
    float — sum of 1/(k+rank) across all lists containing the chunk.
    """
    total = 0.0
    for lst in per_variant_lists:
        for entry in lst:
            if entry.get("chunk_id") == chunk_id:
                rank = entry.get("rank", 1) or 1
                total += 1.0 / (k + rank)
    return total
