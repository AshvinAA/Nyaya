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

import logging
import time
from typing import Any, Dict, List, Optional

from retrieval import config as cfg
from retrieval.registry import dense_search

logger = logging.getLogger("retrieval.stage2")

# Per-list depth: how many candidates each variant x mode returns before fusion.
# Starting point from the doc (config.SEARCH_PER_LIST_K = 10).


def hybrid_search(
    envelope: Dict[str, Any],
    registry: Any,
    trace: Any,
    model: Any = None,
) -> Dict[str, Any]:
    """Run Stage 2 hybrid search per variant and return the search record.

    Parameters
    ----------
    envelope : dict
        The Stage 1 envelope: {original, variants, multi_aspect, historical}.
    registry : retrieval.registry (the module, or any object exposing dense_search)
        The index read registry. This function calls registry.dense_search() for
        each variant. When the sparse upgrade lands, registry also exposes
        sparse_search() and this function adds a second list per variant.
    trace : Trace
        The in-progress trace. This function records the per-variant ranked lists,
        the filter applied, dense_vectors_retained, and any fallback.
    model : optional
        Unused here (search is model-free), accepted for symmetry with the
        orchestrator's staged calling convention.

    Returns
    -------
    dict (recorded into trace["stages"]["2"] by the orchestrator):
        {
          "filter_applied": "default" | "historical",
          "variants": [  # one ranked list per variant
              [  # ranked list: [{chunk_id, score, rank, metadata, text}, ...]
              ],
              ...
          ],
          "dense_vectors_retained": bool,   # True when vectors were fetched for MMR
          "fallback_triggered": bool,
          "notes": str,
        }
    """
    t0 = time.time()
    historical = bool(envelope.get("historical", False))

    # Choose the applicability filter. This is the retrieval-side half of the
    # ingestion Stage 0 overlay rule.
    if historical:
        # SWAP the filter entirely — drop in_force / legal_status constraints.
        # A superseded provision is by definition not in force, so partial
        # relaxation is incoherent. Stage 7 labeling takes over for saying which
        # text governed when.
        filter_params: Optional[Dict[str, Any]] = None
        filter_label = "historical"
    else:
        # Default filter: in_force=true AND legal_status != superseded.
        # The 2025 Ordinance survives by design (in_force=true,
        # legal_status=pending_ratification) and competes normally; Stage 7
        # labels it. The filter's job is applicability, not ratification status.
        filter_params = {
            "in_force": True,
            "legal_status_not": "superseded",
        }
        filter_label = "default"

    # Decide whether to retain the dense vector per candidate. Stage 5b MMR
    # reuses the vectors already computed here — it does not re-embed. We retain
    # them when MMR might run (multi_aspect flagged) AND sparse is not available
    # (dense vectors are BGE-M3's; if we later run sparse-only, we may not have
    # dense vectors to retain).
    multi_aspect = bool(envelope.get("multi_aspect", False))
    retain_vectors = multi_aspect and not cfg.SPARSE_AVAILABLE

    per_variant_lists: List[List[Dict[str, Any]]] = []
    for variant in envelope.get("variants", []):
        if not variant:
            continue
        try:
            ranked = dense_search(
                query_text=variant,
                filter_params=filter_params,
                k=cfg.SEARCH_PER_LIST_K,
                retain_vectors=retain_vectors,
            )
        except Exception as exc:
            logger.warning(
                "Stage 2 dense_search failed for variant %d: %s", len(per_variant_lists), exc
            )
            # A single variant failing is a partial degradation — drop that variant's
            # list and continue with the others (don't hard-fail the whole search).
            continue
        per_variant_lists.append(ranked)

    elapsed = time.time() - t0
    record = {
        "filter_applied": filter_label,
        "variants": per_variant_lists,
        "dense_vectors_retained": retain_vectors,
        "fallback_triggered": False,
        "notes": _notes(filter_label, len(per_variant_lists), retain_vectors),
    }
    logger.info(
        "Stage 2: filter=%s variants=%d lists retained_vectors=%s elapsed=%.3fs",
        filter_label,
        len(per_variant_lists),
        retain_vectors,
        elapsed,
    )
    return record


def _notes(
    filter_label: str,
    n_lists: int,
    retain_vectors: bool,
) -> str:
    """A short human-readable note about what Stage 2 did, for the trace."""
    parts = [f"filter={filter_label}"]
    if n_lists == 0:
        parts.append("no variants produced a result list")
    else:
        parts.append("variant_lists=" + str(n_lists))
    parts.append("dense_vectors_retained=" + str(retain_vectors))
    if cfg.SPARSE_AVAILABLE:
        parts.append("sparse_available=True (not used in this record; sparse_search would add a second list per variant)")
    else:
        parts.append("sparse_available=False (dense-only; one list per variant)")
    return "; ".join(parts)
