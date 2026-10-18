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
# Trace lifecycle
# ---------------------------------------------------------------------------
# One trace object is created at the start of retrieve() and passed to every
# stage. Each stage appends its own record into trace["stages"]["<n>"]. The
# trace is written ONCE at the end of retrieve() (trace.write_trace), after all
# stages have run. If retrieve() fails partway, the partial trace is STILL
# written (with the stages that ran recorded and the ones that didn't marked
# not_run) — a partial trace is more useful than a missing one, because it
# shows WHERE the failure happened.
#
# Timings: each stage records its wall-clock time into trace["timings"]["<n>"].
# The reranker (Stage 5) is the expected bottleneck on CPU; the timings make
# that visible rather than assumed.

# ---------------------------------------------------------------------------
# Fallback propagation
# ---------------------------------------------------------------------------
# Each stage's fallback (defined in its own file) records into the trace and
# returns a degraded-but-usable result. The orchestrator does NOT re-decide
# fallbacks — it just passes the degraded result to the next stage. The one
# asymmetry the orchestrator ENFORCES: if Stage 1 triggered its fallback
# (expansion failed), the historical flag is treated as False — the Stage 2
# filter stays default-STRICT. The stages record their own fallbacks; the
# orchestrator enforces only this one cross-stage constraint (don't widen
# legality filters on a degraded expansion run).

# ===========================================================================
# IMPLEMENTATION
# ===========================================================================

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from retrieval import config as cfg
from retrieval import registry as default_registry
from retrieval import trace as trace_mod
from retrieval.stage1_expand import expand_query
from retrieval.stage2_search import hybrid_search
from retrieval.stage3_fuse import rrf_fuse
from retrieval.stage4_dedup import retrieval_dedup
from retrieval.stage5_rerank import rerank
from retrieval.stage5b_mmr import optional_mmr
from retrieval.stage6_resolve import resolve_parents
from retrieval.stage7_overlay import resolve_overlay

logger = logging.getLogger("retrieval.orchestrator")

_REQUIRED_STAGES = ("1", "2", "3", "4", "5", "5b", "6", "7")


def retrieve(
    query: str,
    query_id: str,
    registry: Any = default_registry,
    write: bool = True,
    use_llm: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run Stages 1-7 for one query and return (final_context_set, trace).

    Parameters
    ----------
    query : str
        The query as received (Bangla or English text).
    query_id : str
        Stable id for this query (matches eval/test_queries.json ids so traces
        and gold labels can be joined). Also the trace's filename.
    registry : retrieval.registry (the module)
        The index read registry. Default is retrieval.registry itself.
    write : bool
        When True (default), the finalised trace is written to
        retrieval_traces/<query_id>.json. The final trace dict is returned
        either way.
    use_llm : bool
        Stage 1 needs the expansion LLM (Ollama). When use_llm=False, Stage 1
        skips the LLM call and uses fallback envelope semantics (original
        query as sole variant, both flags false) — recorded loudly as a
        Stage 1 fallback. This keeps search-only smoke tests runnable without
        a live Ollama.

    Returns
    -------
    (final_context_set, trace_dict)
        final_context_set : the labeled, parent-resolved, deduplicated,
                            reranked (and MMR-diversified if applicable)
                            context set ready for generation.
        trace_dict        : the complete per-query trace (also written to
                            retrieval_traces/<query_id>.json when write=True).
    """
    t_start = time.time()
    timings: Dict[str, float] = {}
    t = trace_mod.Trace(query_id=query_id, original=query)

    envelope: Dict[str, Any] = {}
    stage2_record: Dict[str, Any] = {}
    stage3_record: Dict[str, Any] = {}
    stage4_record: Dict[str, Any] = {}
    stage5_record: Dict[str, Any] = {}
    stage5b_record: Dict[str, Any] = {}
    stage6_record: Dict[str, Any] = {}
    stage7_record: Dict[str, Any] = {}

    ran: List[str] = []

    try:
        # ------------------------------------------------------------------
        # Stage 1 — multi-query expansion
        # ------------------------------------------------------------------
        t0 = time.time()
        if use_llm:
            envelope = expand_query(query, t)
            ran.append("1")
        else:
            # LLM deliberately skipped (smoke tests / no Ollama). Same fallback
            # envelope shape as expansion failure: original as sole variant,
            # both flags false, recorded loudly — never silent.
            envelope = {
                "original": query,
                "variants": [query],
                "multi_aspect": False,
                "historical": False,
            }
            t.add_variant(query)
            t.add_fallback(1, "llm_skipped_by_caller", "original query as sole variant")
            t.record_stage(
                1,
                {
                    "variants": [query],
                    "multi_aspect": False,
                    "historical": False,
                    "fallback_triggered": True,
                    "reason": "use_llm=False (LLM not invoked)",
                },
            )
            ran.append("1")
        timings["1"] = round(time.time() - t0, 3)

        # Orchestrator-enforced cross-stage constraint: a degraded expansion
        # (fallback triggered) must NOT widen the filters. Force historical to
        # False when Stage 1 degraded so Stage 2's filter stays default-STRICT.
        historical = bool(envelope.get("historical", False)) and not _stage1_degraded(
            t, use_llm
        )
        envelope["historical"] = historical

        # ------------------------------------------------------------------
        # Stage 2 — hybrid search per variant, metadata pre-filter
        # ------------------------------------------------------------------
        t0 = time.time()
        stage2_record = hybrid_search(envelope, registry, t)
        # stage2_search only records its own trace entry in its error branch;
        # the orchestrator guarantees the record lands either way.
        if "2" not in t._stages:
            t.record_stage(2, stage2_record)
        ran.append("2")
        timings["2"] = round(time.time() - t0, 3)

        # ------------------------------------------------------------------
        # Stage 3 — RRF fusion
        # ------------------------------------------------------------------
        t0 = time.time()
        stage3_record = rrf_fuse(stage2_record.get("variants", []), t)
        # stage3_fuse only records its own trace entry in its error branch;
        # the orchestrator guarantees the record lands either way.
        if "3" not in t._stages:
            t.record_stage(3, stage3_record)
        ran.append("3")
        timings["3"] = round(time.time() - t0, 3)

        # ------------------------------------------------------------------
        # Stage 4 — retrieval-time dedup
        # ------------------------------------------------------------------
        t0 = time.time()
        stage4_record = retrieval_dedup(stage3_record.get("fused_pool", []), registry, t)
        ran.append("4")
        timings["4"] = round(time.time() - t0, 3)

        # ------------------------------------------------------------------
        # Stage 5 — rerank against the ORIGINAL query
        # ------------------------------------------------------------------
        t0 = time.time()
        stage5_record = rerank(
            stage4_record.get("deduped_pool", []),
            envelope.get("original", query),
            t,
            registry=registry,
        )
        ran.append("5")
        timings["5"] = round(time.time() - t0, 3)

        # ------------------------------------------------------------------
        # Stage 5b — optional MMR (conditional on multi_aspect)
        # ------------------------------------------------------------------
        t0 = time.time()
        stage5b_record = optional_mmr(
            stage5_record.get("final_topk", []),
            _collect_dense_vectors(stage2_record),
            envelope,
            t,
        )
        ran.append("5b")
        timings["5b"] = round(time.time() - t0, 3)
        # The Stage 6 input is the MMR-ordered top-k when MMR ran, else the
        # Stage 5 final top-k (MMR is not the default path).
        if stage5b_record.get("mmr_applied"):
            stage6_children = [
                {
                    "chunk_id": item["chunk_id"],
                    "rerank_score": item.get("rerank_score"),
                    "mmr_rank": item.get("mmr_rank"),
                }
                for item in stage5b_record["mmr_topk"]
            ]
        else:
            stage6_children = [
                {
                    "chunk_id": item["chunk_id"],
                    "rerank_score": item.get("rerank_score"),
                    "rank": item.get("rank"),
                }
                for item in stage5_record.get("final_topk", [])
            ]
        # Stage 6 resolves parents via registry.resolve_parent(child), which
        # reads child["parent_id"]; Stage 7 reads text_layer / section_number
        # from metadata. Both live in Stage 4's pooled metadata — attach them
        # here (top-k entries carry only ids + scores).
        metadata_by_id = _collect_metadata_by_id(stage4_record)
        for child in stage6_children:
            meta = metadata_by_id.get(child.get("chunk_id"), {})
            child["parent_id"] = meta.get("parent_id")

        # ------------------------------------------------------------------
        # Stage 6 — parent/table resolution, token budgeting
        # ------------------------------------------------------------------
        t0 = time.time()
        stage6_record = resolve_parents(
            stage6_children,
            registry,
            t,
            metadata_by_id=metadata_by_id,
        )
        ran.append("6")
        timings["6"] = round(time.time() - t0, 3)

        # ------------------------------------------------------------------
        # Stage 7 — overlay vs base, dual labeling (last stage)
        # ------------------------------------------------------------------
        t0 = time.time()
        stage7_record = resolve_overlay(stage6_record.get("resolved_context", []), t)
        ran.append("7")
        timings["7"] = round(time.time() - t0, 3)

        final_context_set = stage7_record.get("final_context_set", [])

    except Exception as exc:
        # Partial-trace guarantee: write whatever ran, mark the rest not_run.
        logger.exception("retrieve() failed mid-flow: %s", exc)
        for stage_nb in _REQUIRED_STAGES:
            if stage_nb not in ran and stage_nb not in t._stages:
                t.record_stage(stage_nb, {"not_run": True, "reason": str(exc)})
        final_context_set = []

    finalised = t.finalize(final_context_set, timings)
    if write:
        path = trace_mod.write_trace(finalised, query_id)
        logger.info("trace written: %s", path)
    return final_context_set, finalised


def _stage1_degraded(t: trace_mod.Trace, use_llm: bool) -> bool:
    """Whether Stage 1's fallback path fired (LLM unavailable, malformed, or skipped)."""
    return any(
        f.get("stage") == "1" for f in getattr(t, "_fallbacks", [])
    ) or not use_llm


def _collect_metadata_by_id(stage4_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Flatten Stage 4's deduped pool into {chunk_id: metadata}.

    Stage 4 attaches each pooled candidate's stored metadata (authority_rank,
    text_layer, source_act, section_number, section_path, parent_id, ...). The
    surviving entries are the source of truth for everything Stages 6/7 need
    about a chunk without re-querying Chroma.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for entry in stage4_record.get("deduped_pool", []) or []:
        cid = entry.get("chunk_id")
        if cid:
            out[cid] = entry.get("metadata") or {}
    return out


def _collect_dense_vectors(stage2_record: Dict[str, Any]) -> Dict[str, List[float]]:
    """Flatten Stage 2's per-variant lists into {chunk_id: dense_vector}.

    Stage 2 retained vectors per candidate ONLY when it decided MMR might run
    (multi_aspect flagged, dense-only). Duplicates across variants are the same
    stored vector — take the first occurrence.
    """
    vectors: Dict[str, List[float]] = {}
    for lst in stage2_record.get("variants", []) or []:
        for entry in lst:
            cid = entry.get("chunk_id")
            vec = entry.get("dense_vector")
            if cid and vec is not None and cid not in vectors:
                vectors[cid] = vec
    return vectors
