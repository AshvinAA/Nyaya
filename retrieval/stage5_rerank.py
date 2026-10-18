"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 5 - Reranking (BGE-Reranker-v2-M3)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 BGE-Reranker-v2-M3 re-scores the deduplicated, fused candidate set directly
 against the ORIGINAL input query (NOT the Stage 1 expanded variants) using a
 cross-encoder — more expensive, more accurate than embedding similarity alone.

 Produces the final top-k (~5).

 WHY AGAINST THE ORIGINAL: the expanded variants exist to WIDEN THE NET in
 search; the reranker's job is to judge relevance to what the user actually
 asked. Re-ranking against a variant would optimize for the paraphrase, not the
 question.

 Practical notes:
   - The reranker is multilingual (Bangla queries score natively — no
     translation step).
   - It is the pipeline's latency bottleneck on CPU (hence the Stage 4 pool cap
     of ~30 entering it).
   - If the model cannot be loaded, the RRF order stands and the failure is
     recorded in the trace (reranker="fallback_rrf") — retrieval never hard-fails
     because an optional re-scorer is down.

 Used by: orchestrator.py (Stage 5 of the retrieve() flow).

 Objective of this file: take the deduped pool from Stage 4, re-score each
 candidate against the ORIGINAL query with the cross-encoder, return the final
 top-k (~5) by rerank score — plus rerank scores and fallback status for the
 trace and the eval ablation row "reranker off -> RRF order straight to final-k."
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 5 contract
# ---------------------------------------------------------------------------
# rerank(deduped_pool, original_query, trace) -> final_topk
#   deduped_pool   : the Stage 4 deduped pool [{chunk_id, ..., metadata}, ...]
#   original_query : the ORIGINAL query string from the Stage 1 envelope
#                    (NOT the variants — variants widened the net; this judges
#                    the actual question)
#   trace          : the in-progress trace (so this stage records rerank scores,
#                    the reranker used or fallback, and any fallback triggered)
#   returns        : final_topk = [{chunk_id, rerank_score, rank}, ...]
#                    capped to config.FINAL_K (~5) by rerank score
#
# The final top-k is what Stage 5b (MMR, if applied) and Stage 6 (parent
# resolution) receive.

# ---------------------------------------------------------------------------
# Cross-encoder scoring vs the original query
# ---------------------------------------------------------------------------
# For each candidate in the deduped pool:
#   - score = cross_encoder.score(original_query, candidate_text)
#   - candidate_text = the bare chunk text (the generation text Chroma stored,
#     NOT the embedding prefix — the prefix was for the embedding vector, never
#     for the LLM; the reranker should score the text the LLM would actually read)
#
# The cross-encoder scores (query, passage) pairs directly — more accurate than
# embedding cosine for relevance, at higher cost. This is why the pool entering
# it is capped at ~30 (Stage 4 / fused-pool cap).
#
# Score against the ORIGINAL query, not the variants. The variants exist to
# widen the search net; the reranker's job is to judge relevance to the question
# the user actually asked. Re-ranking against a variant would optimize for the
# paraphrase, not the question.

# ---------------------------------------------------------------------------
# Multilingual note
# ---------------------------------------------------------------------------
# BGE-Reranker-v2-M3 is multilingual — Bangla queries score natively, no
# translation step. This is a property worth recording in the trace (the doc
# says to make it explicit), not a hidden assumption. Don't translate the query
# before reranking "to be safe" — that would add a translation hop the reranker
# doesn't need and could introduce number/entity errors.

# ---------------------------------------------------------------------------
# Final top-k
# ---------------------------------------------------------------------------
# Sort by rerank_score descending, take the top config.FINAL_K (~5). This is the
# final context set entering Stage 5b (MMR, if applied) and then Stage 6 (parent
# resolution). The FINAL_K is a tunable — small enough for a small local LLM's
# context, large enough to survive MMR diversification if it runs.

# ---------------------------------------------------------------------------
# Fallback (defined, loud — never silent)
# ---------------------------------------------------------------------------
# If the reranker model cannot be loaded (no network, no disk, model missing):
#   - the RRF order (from Stage 3 / Stage 4 order) STANDS as the final order
#   - trace records: reranker = "fallback_rrf", fallback_triggered = true,
#     reason
#   - retrieval does NOT hard-fail because an optional re-scorer is down
#
# This is the degradation policy asymmetry again: convenience features (reranking)
# may degrade; correctness filters (the Stage 2 applicability filter) may not.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search, fuse, or dedup — that's Stages 2/3/4 (it receives
#     the deduped pool).
#   - It does not apply MMR — that's Stage 5b (which receives this stage's
#     final top-k if multi_aspect is flagged).
#   - It does not resolve parents — that's Stage 6.
#   - It does not label overlays — that's Stage 7.
#   - It does not interpret query intent — that's Stage 1's envelope.
#   - It does NOT score against the variants. If it did, it would optimize for
#     the paraphrase, not the question — that's the explicit reason the doc says
#     to score against the original.

# ===========================================================================
# IMPLEMENTATION (below the design contract; mirrors the contract exactly)
# ===========================================================================

import logging
import time
from typing import Any, Dict, List, Optional

from retrieval import config as cfg

logger = logging.getLogger("retrieval.stage5")

# Lazily-built cross-encoder, shared per process. Loaded from
# config.RERANKER_MODEL (BAAI/bge-reranker-v2-m3, multilingual:
# Bangla queries score natively — no translation step). Load failures are
# caught in rerank() and degrade to the RRF-order fallback (reranker=
# "fallback_rrf" in the trace), never a hard pipeline failure.
_reranker = None
_reranker_load_failed = False


def _load_reranker() -> Any:
    """Build and cache the BGE-Reranker-v2-M3 cross-encoder.

    Raises on failure — the caller (rerank) catches and falls back loudly.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(cfg.RERANKER_MODEL)
    return _reranker


def _candidate_text(entry: Dict[str, Any], registry: Any) -> str:
    """The bare generation text to score for a deduped pool entry.

    Prefers entry["text"] (attached by Stage 4); falls back to reading the
    stored document via registry.chunk_document when absent. This is the bare
    chunk text Chroma stored as the document — NOT the hierarchical embedding
    prefix, which exists only inside the embedding vector and was never
    generation-facing. Never score the prefix by accident.
    """
    text = entry.get("text")
    if text:
        return str(text)
    if registry is not None:
        return str(registry.chunk_document(entry.get("chunk_id")) or "")
    return ""


def rerank(
    deduped_pool: List[Dict[str, Any]],
    original_query: str,
    trace: Any,
    registry: Any = None,
) -> Dict[str, Any]:
    """Cross-encoder re-score the deduped pool against the ORIGINAL query.

    Parameters
    ----------
    deduped_pool : list of dict
        The Stage 4 deduped pool — entries carry at least {chunk_id, text}.
    original_query : str
        The ORIGINAL query string from the Stage 1 envelope (NOT a variant).
        The variants exist to widen the search net; the reranker judges the
        question the user actually asked. Never score against a variant.
    trace : Trace
        The in-progress trace. Records the final top-k with rerank scores, the
        reranker used (or the fallback), and any fallback.
    registry : retrieval.registry (the module), optional
        Used only to fetch a candidate's generation text when the entry does
        not already carry one. The default orchestrator always passes it.

    Returns
    -------
    dict (also recorded into trace["stages"]["5"]):
        {
          "final_topk": [ {chunk_id, rerank_score, rank}, ... ],
                        # top config.FINAL_K (~5) by rerank score
          "reranker": str,   # config.RERANKER_MODEL or "fallback_rrf"
          "fallback_triggered": bool,
          "reason": str | None,   # set when fallback triggered
          "notes": str,
        }
    """
    t0 = time.time()

    if not deduped_pool:
        record = {
            "final_topk": [],
            "reranker": "no_candidates",
            "fallback_triggered": True,
            "reason": "empty deduped pool from Stage 4",
            "notes": "nothing to rerank",
        }
        trace.record_stage(5, record)
        trace.add_fallback(5, "empty_deduped_pool", "final_topk empty")
        logger.warning("Stage 5: empty deduped pool — nothing to rerank")
        return record

    # Master knob (config.RERANKER_ENABLED): when off, the RRF order STANDS
    # immediately — the ablation row "reranker off -> RRF order straight to
    # final-k". Recorded loudly (fallback_triggered=True), never silent.
    if not getattr(cfg, "RERANKER_ENABLED", True):
        reason = "reranker disabled by config (RERANKER_ENABLED=False)"
        logger.warning("Stage 5: %s — RRF order stands", reason)
        trace.add_fallback(5, "reranker_disabled", "RRF order stands")
        scores = [0.0] * len(deduped_pool)
        reranker_label = "fallback_rrf"
        fallback_triggered = True
        reason_record = reason
        scored = []
        for entry, score in zip(deduped_pool, scores):
            scored.append(
                {
                    "chunk_id": entry.get("chunk_id"),
                    "rerank_score": round(score, 6),
                    "rrf_rank": entry.get("rank"),
                }
            )
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        final_topk = []
        for rank, item in enumerate(scored[: int(cfg.FINAL_K)], start=1):
            final_topk.append(
                {"chunk_id": item["chunk_id"], "rerank_score": item["rerank_score"], "rank": rank}
            )
        elapsed = time.time() - t0
        record = {
            "final_topk": final_topk,
            "reranker": reranker_label,
            "fallback_triggered": fallback_triggered,
            "reason": reason_record,
            "notes": _notes(len(deduped_pool), len(final_topk), reranker_label, elapsed),
        }
        trace.record_stage(5, record)
        return record

    texts = [_candidate_text(entry, registry) for entry in deduped_pool]

    try:
        cross_encoder = _load_reranker()
        scores = cross_encoder.predict(
            [(original_query, text) for text in texts],
            show_progress_bar=False,
        )
        reranker_label = cfg.RERANKER_MODEL
        fallback_triggered = False
        reason = None
    except Exception as exc:
        # Fallback (loud): the RRF order from Stage 3/4 stands. Retrieval never
        # hard-fails because an optional re-scorer is down. Convenience features
        # may degrade; correctness filters may not.
        logger.warning(
            "Stage 5: reranker load/inference failed (%s: %s) — falling back to RRF order",
            type(exc).__name__,
            exc,
        )
        scores = [0.0] * len(deduped_pool)
        reranker_label = "fallback_rrf"
        fallback_triggered = True
        reason = "%s: %s" % (type(exc).__name__, exc)
        trace.add_fallback(5, "reranker_load_or_inference_failed", "RRF order stands")

    # Sort by rerank score descending; ties keep the RRF upstream order
    # (Stage 4 preserved rrf rank order, and Python's sort is stable).
    scored = []
    for entry, score in zip(deduped_pool, [float(s) for s in scores]):
        scored.append(
            {
                "chunk_id": entry.get("chunk_id"),
                "rerank_score": round(score, 6),
                "rrf_rank": entry.get("rank"),   # upstream RRF rank, for tie visibility
            }
        )
    scored.sort(key=lambda d: d["rerank_score"], reverse=True)
    final_topk = []
    for rank, item in enumerate(scored[: int(cfg.FINAL_K)], start=1):
        final_topk.append({"chunk_id": item["chunk_id"], "rerank_score": item["rerank_score"], "rank": rank})

    elapsed = time.time() - t0
    record = {
        "final_topk": final_topk,
        "reranker": reranker_label,
        "fallback_triggered": fallback_triggered,
        "reason": reason,
        "notes": _notes(len(deduped_pool), len(final_topk), reranker_label, elapsed),
    }
    trace.record_stage(5, record)
    logger.info(
        "Stage 5: pool_in=%d final_topk=%d reranker=%s elapsed=%.3fs",
        len(deduped_pool),
        len(final_topk),
        reranker_label,
        elapsed,
    )
    return record


def _notes(pool_in: int, final_k: int, reranker: str, elapsed: float) -> str:
    """A short human-readable note about what Stage 5 did, for the trace."""
    return (
        "pool_in=%d final_topk=%d reranker=%s scored_against=original_query elapsed=%.3fs"
        % (pool_in, final_k, reranker, elapsed)
    )
