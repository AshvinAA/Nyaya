"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 7 - Overlay vs. base resolution
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Implements the ingestion doc's Stage 0 layering rule at the point where the
 context set is finalized:

   "overlay supersedes base, but both are surfaced with labels — never silently
    pick one"

 If the resolved context set (from Stage 6) includes both:
   - a base-text provision (text_layer = consolidated_base), AND
   - an overlay provision (text_layer = amendment_overlay, e.g. the 2025
     Ordinance)
 covering the SAME topic / same section_number:

   - DO NOT discard either. The reranker may order them; this stage may NOT drop
     one.
   - Tag both explicitly for the generation prompt:
       - "[current text — ratification pending]"   for the overlay
       - "[settled law since 2018]"                for the base
   - This lets generation produce something like:
       "under a 2025 amendment currently in effect but not yet ratified, X —
        this replaces the previous rule that said Y"
     rather than silently picking one version and stating it as uncontested fact.

   - Record the pair in the trace (dual_layer: true, with both chunk ids) so
     evaluation can measure conflict-handling DIRECTLY instead of inferring it
     from answers.

 Used by: orchestrator.py (Stage 7 of the retrieve() flow, the LAST stage before
          the final context set is handed to generation).

 Objective of this file: take the resolved context set from Stage 6, detect
 base+overlay pairs covering the same provision, label both explicitly, record
 dual_layer pairs in the trace, and NEVER drop one. The final labeled context set
 is what generation receives.
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 7 contract
# ---------------------------------------------------------------------------
# resolve_overlay(resolved_context, trace) -> final_context_set
#   resolved_context : the Stage 6 resolved context set
#                      [{child_id, parent_id, parent_kind, parent_text_or_json_ref,
#                        truncation_label|null, label}, ...]
#   trace            : the in-progress trace (records dual_layer pairs and
#                      labels_applied)
#   returns          : final_context_set = [{chunk_id, label, parent_id,
#                      parent_kind, text_or_json_preview, ...}, ...]
#                      — the FINAL labeled context set handed to generation
#
# This is the LAST retrieval stage. After this, the context set is complete,
# labeled, parent-resolved, deduplicated, reranked, and (if applicable) MMR-
# diversified — ready for the generation stage (built much later) to reason over
# and produce an answer in Bangla or English.

# ---------------------------------------------------------------------------
# Base+overlay pair detection
# ---------------------------------------------------------------------------
# For each pair of entries in the resolved context set:
#   - if one has text_layer = consolidated_base and the other has
#     text_layer = amendment_overlay, AND they cover the same provision
#     (same section_number, or same topic in the overlapping-coverage sense the
#     ingestion metadata already encodes), then:
#       -> this is a dual_layer pair
#       -> record it in the trace: dual_layer_pairs.append({
#            base_chunk_id, overlay_chunk_id, base_label, overlay_label })
#       -> label both for the generation prompt:
#            base    -> config.LABEL_BASE_SETTLED
#            overlay -> config.LABEL_OVERLAY_PENDING
#       -> do NOT drop either from the final context set
#
# "Cover the same provision" is detected from the metadata the ingestion already
# attached (section_number, section_path, topic_tags). The ingestion stage
# already tagged each chunk with its text_layer and section_number; Stage 7
# doesn't re-derive them — it reads them from the resolved context entries.

# ---------------------------------------------------------------------------
# Labeling (both surfaced, never silently picked)
# ---------------------------------------------------------------------------
# The labels are the human-readable statements of the Stage 0 layering rule,
# attached to each chunk for the generation prompt:
#
#   config.LABEL_OVERLAY_PENDING = "[current text — ratification pending]"
#   config.LABEL_BASE_SETTLED     = "[settled law since 2018]"
#
# These labels are what lets generation produce a correctly-caveated answer
# (overlay supersedes base where they overlap, but both are shown, and the
# overlay's pending-ratification status is explicit) rather than silently picking
# one version and stating it as uncontested fact.
#
# Do NOT drop the base when the overlay exists, and do NOT drop the overlay when
# the base exists. Both stay in the final context set, both carry their labels.

# ---------------------------------------------------------------------------
# Trace record (so evaluation can measure conflict-handling directly)
# ---------------------------------------------------------------------------
# For each dual_layer pair, the trace records:
#   dual_layer_pairs: [{base_chunk_id, overlay_chunk_id, base_label,
#                       overlay_label}, ...]
#
# This lets the eval harness measure conflict-handling correctness on the
# base-vs-overlay test items DIRECTLY (does the final context set contain both
# labeled chunks for a dual-layer query?) instead of inferring it from the
# generated answer. The eval ablation row is "labels off -> does the answer state
# pending law as settled?"

# ---------------------------------------------------------------------------
# Open design question (carried from ingestion Stage 0 / RETRIEVAL_PIPELINE.md)
# ---------------------------------------------------------------------------
# The EXACT prompt template that turns these dual-labeled chunks into a coherent,
# correctly-caveated answer belongs to the GENERATION stage, not retrieval.
#
# Retrieval's obligation is only to keep both layers present and clearly labeled.
# It does NOT need to know how generation will phrase the conflict. That question
# is carried open here so it isn't lost between "retrieval is done" and "generation
# prompt is designed."
#
# Do NOT try to resolve the conflict here (e.g. "pick the overlay and say the base
# is superseded"). That would violate the rule this stage exists to enforce.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search, fuse, dedup, rerank, MMR, or parent resolution —
#     that's Stages 2-6 (it receives the resolved context set).
#   - It does not interpret query intent — that's Stage 1's envelope.
#   - It does NOT drop one side of a dual_layer pair — that's the rule it exists
#     to enforce. Dropping would silently resolve a legal conflict, which is the
#     failure the whole layering design prevents.
#   - It does NOT generate an answer — that's the generation stage (built much
#     later), which receives this stage's final labeled context set.
#   - It does NOT decide the generation prompt template for dual-layer chunks —
#     that's the open design question carried to the generation stage.
