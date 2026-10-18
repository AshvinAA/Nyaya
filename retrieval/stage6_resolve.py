"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 6 - Parent/table resolution
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 For each surviving top-k CHILD chunk, fetch from the document store via the
 child's parent_id:

   - Its PARENT SECTION (prose) — e.g. matched on Section 23(3) but handing the
     LLM all of Section 23, including the misconduct exceptions in 23(1) that
     qualify the compensation right in 23(3). Parents are stored in
     parent_store.json keyed by the same IDs the index points at; self-parented
     chunks (sentence-fallback pieces with no section context) resolve to
     themselves, honestly.

   - Its FULL STRUCTURED TABLE JSON (tables) — the LLM reasons over clean
     row/column data, never a flattened table-as-prose summary.

 The LLM NEVER sees bare, context-stripped child chunks.

 Token budgeting is the open issue here, not resolution mechanics: parents were
 built with a soft cap and a part-splitting rule at ingestion, but assembling 3-5
 full sections plus a table JSON can still overflow a small local LLM's context.
 The starting policy: order parents by rerank score, include WHOLE parents until
 the budget is reached, and mark any overflow parent as truncated IN ITS LABEL so
 generation knows the text is partial — never silently cut. The exact budget is a
 tunable (config.PARENT_TOKEN_BUDGET) the eval harness sweeps.

 Used by: orchestrator.py (Stage 6 of the retrieve() flow).

 Objective of this file: take the surviving top-k child chunks from Stage 5/5b,
 fetch each one's parent (prose section text or full table JSON) from the parent
 store by parent_id, apply token budgeting with truncation labeling, and return
 the resolved context set (which children -> which parent text/JSON, with labels)
 — plus trace entries for the parent ids fetched and any missing-parent /
 truncation events.
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 6 contract
# ---------------------------------------------------------------------------
# resolve_parents(topk_children, registry, trace) -> resolved_context
#   topk_children : the surviving top-k child chunks from Stage 5/5b
#                   [{chunk_id, parent_id, kind, metadata, ...}, ...]
#   registry      : the index read registry (resolve_parent for fetching a parent
#                   by parent_id; parent_text / parent_json conveniences)
#   trace         : the in-progress trace (records the resolved context, parent
#                   ids fetched, missing-parent events, truncation events)
#   returns       : resolved_context = [{child_id, parent_id, parent_kind,
#                   parent_text_or_json_ref, truncation_label|null, label}, ...]
#                   — the context set that enters Stage 7 (overlay labeling) and
#                   then generation
#
# The LLM never sees the bare child chunks — only the resolved parent text / table
# JSON. This is a MANDATORY step, not optional: returning bare children to
# generation would violate the embed-small-generate-large principle and would hand
# the LLM anaphoric, context-stripped clauses ("said worker", "the amount referred
# to in sub-section (1)") without the surrounding section that gives them meaning.

# ---------------------------------------------------------------------------
# Parent resolution (prose)
# ---------------------------------------------------------------------------
# For each child in topk_children:
#   - fetch the parent from the parent store by child["parent_id"] via
#     registry.resolve_parent()
#   - prose parent:
#       -> parent_kind = "prose"
#       -> parent_text = the full section text (the haystack around the needle)
#       -> the child matched on a clause (e.g. 23(3)); the parent is the whole
#          section (all of 23, including 23(1)'s exceptions that qualify 23(3)'s
#          right) — this is the whole point of parent-child indexing
#   - self-parented chunk (sentence-fallback pieces with no section context):
#       -> parent_kind = "self"
#       -> parent_text = the child's own text, honestly labeled as self-parented
#       -> these chunks have no real section context; pretending they do would be
#          worse than honestly labeling them as bare-ish

# ---------------------------------------------------------------------------
# Parent resolution (table)
# ---------------------------------------------------------------------------
# For each table child:
#   - the parent is the full structured table_json (the LLM reasons over clean
#     row/column data, never the flattened summary that got embedded)
#   - parent_kind = "table"
#   - parent_text_or_json_ref = the table_json dict (table_id, caption, col_labels,
#     row_labels, data with coerced numerics, metadata block)
#   - the child's text is the summary (what got embedded); the parent's table_json
#     is what the LLM receives — same parent-child principle applied to tables

# ---------------------------------------------------------------------------
# Token budgeting (the open issue — starting policy, tunable)
# ---------------------------------------------------------------------------
# Assemble the resolved context set by ordering parents by rerank score (the
# Stage 5/5b order), and include WHOLE parents until config.PARENT_TOKEN_BUDGET
# is reached.
#
#   - Order parents by rerank score (most relevant first).
#   - Include whole parents (don't split a parent mid-section to fit — a split
#     parent would hand the LLM a partial section without telling it, which is
#     the exact failure parent-child indexing exists to prevent).
#   - Mark any overflow parent (a parent that didn't fit in the budget) as
#     truncated IN ITS LABEL, so generation knows the text is partial — never
#     silently cut a parent.
#
# The exact budget (config.PARENT_TOKEN_BUDGET) is a tunable the eval harness
# sweeps against the target LLM's context window. Start with a reasonable value;
# the doc leaves the exact number open as a flagged follow-up.

# ---------------------------------------------------------------------------
# Missing parent (bug alarm — should be impossible by construction)
# ---------------------------------------------------------------------------
# If a child's parent_id is NOT found in the parent store:
#   - return the child with parent_kind = "missing"
#   - label it "[bare chunk — parent missing]"
#   - record a missing-parent event in the trace (child_id, reason)
#
# This should be IMPOSSIBLE by construction: ingestion Stage 4 guarantees every
# retrievable child has a parent_id pointing into the parent store. Its occurrence
# is a BUG ALARM, not a handled case — but handle it loudly anyway (per the
# degradation policy: never silently drop a chunk). A missing parent means either
# the parent store and the index are out of sync (re-ingestion needed) or a parent
# write was lost — both are pipeline bugs worth diagnosing, not normal outcomes.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search, fuse, dedup, rerank, or MMR — that's Stages 2-5b
#     (it receives the surviving top-k).
#   - It does not label overlays — that's Stage 7 (it passes the resolved context
#     set, with parent ids and truncation labels, to Stage 7).
#   - It does not interpret query intent — that's Stage 1's envelope.
#   - It does NOT split parents to fit the budget — it includes whole parents and
#     labels overflow as truncated. Splitting a parent mid-section would hand
#     the LLM a partial section without telling it, which is the failure this
#     stage exists to prevent.
#   - It does NOT silently drop a child whose parent is missing — it labels it and
#     traces it; dropping would silently lose evidence.

# ===========================================================================
# IMPLEMENTATION (below the design contract; mirrors the contract exactly)
# ===========================================================================

import logging
import time
from typing import Any, Dict, List, Optional

from ingestion.common import estimate_tokens
from retrieval import config as cfg

logger = logging.getLogger("retrieval.stage6")

# Traceable labels attached to every resolved entry (honest, never silent).
LABEL_PARENT_MISSING = "[bare chunk — parent missing]"
LABEL_TRUNCATED = "[truncated: parent exceeds token budget]"
LABEL_SELF_PARENTED = "[self-parented — no broader section context]"


def _parent_content(parent: Dict[str, Any]) -> Any:
    """Pull the parent's LLM-facing content out of a parent-store record.

    table parent -> the 'table_json' dict (structured, what the LLM reasons over)
    prose / self parent -> the 'text' str (the full section)
    anything else -> parent.get('text') or ''
    """
    if parent.get("kind") == "table":
        return parent.get("table_json") or {}
    return parent.get("text") or ""


def _content_tokens(parent: Dict[str, Any]) -> int:
    """Rough token estimate for a parent record (text or structured JSON).

    - str content  -> ingestion.common.estimate_tokens (the same estimator the
      ingestion cap decisions used, so budget semantics stay consistent)
    - dict content -> len(str(content))/4 as a deterministic, dependency-free
      proxy (the eval harness sweeps the budget against the target LLM window).
    """
    content = _parent_content(parent)
    if isinstance(content, str):
        return estimate_tokens(content)
    return max(1, round(len(str(content)) / 4))


def _label_for_kind(kind: str, base_label: Optional[str]) -> str:
    """Attach an honest label to a resolved entry based on parent kind."""
    if kind == "missing":
        return LABEL_PARENT_MISSING
    if kind == "self":
        return (base_label + " " if base_label else "") + LABEL_SELF_PARENTED
    return base_label or ""


def resolve_parents(
    topk_children: List[Dict[str, Any]],
    registry: Any,
    trace: Any,
    metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve each surviving top-k child into its parent section / table.

    Parameters
    ----------
    topk_children : list of dict
        The surviving top-k child chunks from Stage 5/5b — entries carry at
        least {chunk_id, ...}. The parent_id is read from the parent store via
        registry.resolve_parent(child) (which accepts a child dict carrying a
        'parent_id' key).
    registry : retrieval.registry (the module)
        Used to fetch each parent (registry.resolve_parent). Parent records
        are keyed by the SAME IDs the index points at (ingestion Stage 4
        guarantee), so a missing parent is a bug alarm, not a normal case.
    trace : Trace
        The in-progress trace. Records the resolved context, parent ids, and
        every missing-parent / truncation event.
    metadata_by_id : dict chunk_id -> metadata, optional
        Metadata read from the index (Stage 4 attaches it to the pool; the
        orchestrator passes it through). Carried into every resolved entry so
        Stage 7 can read text_layer / section_number without re-querying.

    Returns
    -------
    dict (also recorded into trace["stages"]["6"]):
        {
          "resolved_context": [
              { child_id, parent_id, parent_kind, parent_text_or_json,
                est_tokens, truncation_label, label, metadata,
                rerank_score, rank },
              ...
          ],
          "parent_ids_fetched": [str, ...],
          "missing_parent_events": [ {child_id, reason}, ... ],
          "truncation_events":    [ {child_id, reason}, ... ],
          "fallback_triggered": bool,
          "notes": str,
        }
    """
    t0 = time.time()
    metadata_by_id = metadata_by_id or {}

    resolved: List[Dict[str, Any]] = []
    parent_ids: List[str] = []
    missing_events: List[Dict[str, Any]] = []
    truncation_events: List[Dict[str, Any]] = []
    fallback_triggered = False

    # -- Pass 1: fetch every parent (whole parents, no splitting) ------------
    # Order parents by the incoming rank (Stage 5/5b order = relevance order).
    for child in topk_children:
        cid = child.get("chunk_id")
        try:
            parent = registry.resolve_parent(child)   # child dict carries parent_id
        except FileNotFoundError as exc:
            # The parent STORE itself is absent (ingestion not re-run in this
            # checkout). Same policy as a per-child miss: label loudly, keep
            # the entry, never silently drop — the trace shows the environment
            # cause so the fix (re-run ingestion) is obvious.
            missing_events.append({"child_id": cid, "reason": "parent store unavailable: %s" % exc})
            trace.add_fallback(
                6, "parent_store_unavailable", "child kept, labeled '%s'" % LABEL_PARENT_MISSING
            )
            fallback_triggered = True
            resolved.append(
                {
                    "child_id": cid,
                    "parent_id": child.get("parent_id"),
                    "parent_kind": "missing",
                    "parent_text_or_json": None,
                    "est_tokens": 0,
                    "truncation_label": None,
                    "label": LABEL_PARENT_MISSING,
                    "metadata": metadata_by_id.get(cid, {}),
                    "rerank_score": child.get("rerank_score"),
                    "rank": child.get("rank") if child.get("rank") is not None else child.get("mmr_rank"),
                }
            )
            continue
        if parent is None:
            # BUG ALARM, not a handled case — never silently drop. Label loudly
            # and trace, so the eval/debug path sees the event.
            missing_events.append(
                {"child_id": cid, "reason": "parent_id not found in parent store"}
            )
            trace.add_fallback(
                6, "missing_parent", "child kept, labeled '%s'" % LABEL_PARENT_MISSING
            )
            fallback_triggered = True
            resolved.append(
                {
                    "child_id": cid,
                    "parent_id": child.get("parent_id"),
                    "parent_kind": "missing",
                    "parent_text_or_json": None,
                    "est_tokens": 0,
                    "truncation_label": None,
                    "label": LABEL_PARENT_MISSING,
                    "metadata": metadata_by_id.get(cid, {}),
                    "rerank_score": child.get("rerank_score"),
                    "rank": child.get("rank") if child.get("rank") is not None else child.get("mmr_rank"),
                }
            )
            continue

        pid = parent.get("parent_id") or child.get("parent_id")
        kind = parent.get("kind", "prose")
        content = _parent_content(parent)
        parent_ids.append(pid)
        resolved.append(
            {
                "child_id": cid,
                "parent_id": pid,
                "parent_kind": kind,
                "parent_text_or_json": content,
                "est_tokens": _content_tokens(parent),
                "truncation_label": None,
                "label": _label_for_kind(kind, None),
                "metadata": metadata_by_id.get(cid, {}),
                "rerank_score": child.get("rerank_score"),
                "rank": child.get("rank") if child.get("rank") is not None else child.get("mmr_rank"),
            }
        )

    # -- Pass 2: token budgeting (whole parents, overflow labeled loud) --------
    # Include whole parents until the budget is reached. Never split a parent
    # mid-section — that would hand the LLM a partial section without telling
    # it, which is the exact failure parent-child indexing prevents. Overflow
    # parents are STILL included (never silently dropped) and labeled truncated.
    budget = int(cfg.PARENT_TOKEN_BUDGET)
    used = 0
    for entry in resolved:
        if entry["parent_kind"] == "missing":
            continue
        tokens = entry["est_tokens"]
        if used + tokens > budget and used > 0:
            entry["truncation_label"] = LABEL_TRUNCATED
            entry["label"] = (entry["label"] + " " if entry["label"] else "") + LABEL_TRUNCATED
            truncation_events.append(
                {
                    "child_id": entry["child_id"],
                    "reason": "parent est_tokens=%d exceeds remaining budget (budget=%d, used=%d)"
                    % (tokens, budget, used),
                }
            )
            # Truncation is a convenience degradation, not a filter failure.
            trace.add_fallback(6, "parent_truncated", "entry kept with truncation label")
            fallback_triggered = True
            continue
        used += tokens

    elapsed = time.time() - t0
    record = {
        "resolved_context": resolved,
        "parent_ids_fetched": parent_ids,
        "missing_parent_events": missing_events,
        "truncation_events": truncation_events,
        "fallback_triggered": fallback_triggered,
        "notes": _notes(len(topk_children), len(resolved), missing_events, truncation_events, elapsed),
    }
    trace.record_stage(6, record)
    logger.info(
        "Stage 6: children=%d resolved=%d missing=%d truncated=%d tokens_used=%d elapsed=%.3fs",
        len(topk_children),
        len(resolved),
        len(missing_events),
        len(truncation_events),
        used,
        elapsed,
    )
    return record


def _notes(
    children_in: int,
    resolved: int,
    missing: List[Dict[str, Any]],
    truncations: List[Dict[str, Any]],
    elapsed: float,
) -> str:
    """A short human-readable note about what Stage 6 did, for the trace."""
    return (
        "children_in=%d resolved=%d missing=%d truncated=%d budget=%s elapsed=%.3fs"
        % (
            children_in,
            resolved,
            len(missing),
            len(truncations),
            cfg.PARENT_TOKEN_BUDGET,
            elapsed,
        )
    )
