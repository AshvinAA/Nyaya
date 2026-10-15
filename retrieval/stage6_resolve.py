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
#     parent would hand the LLM a partial section without telling it, which is the
#     exact failure parent-child indexing exists to prevent).
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
#     labels overflow as truncated. Splitting a parent mid-section would hand the
#     LLM a partial section without telling it, which is the failure this stage
#     exists to prevent.
#   - It does NOT silently drop a child whose parent is missing — it labels it and
#     traces it; dropping would silently lose evidence.
