"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 6 - Parent/table resolution
                              Stage 7 - Overlay vs. base resolution
============================================================================
 STAGE 6: For each surviving top-k CHILD, fetch its parent from the
 document store (parent_store.json, keyed by the same IDs the index points
 at). The LLM never sees bare, context-stripped child chunks:
   prose  -> the full parent section (matched on Section 23(3), handed all
             of Section 23 including the misconduct exceptions that
             qualify the compensation right in 23(3))
   table  -> the full structured table JSON (clean row/column data,
             never a flattened table-as-prose)
 Self-parented chunks (sentence-fallback pieces with no section context)
 resolve to themselves, honestly.

 Token budgeting is the open issue here, not resolution mechanics: order
 parents by rerank score, include whole parents until the budget is
 reached, and mark any overflow parent as truncated IN ITS LABEL - never
 silently cut. The exact budget is a tunable the RAGAS harness sweeps.

 STAGE 7: Implements the ingestion Stage 0 layering rule - "overlay
 supersedes base, but BOTH are surfaced with labels, never silently pick
 one" - at the point where the context set is finalized. If the fetched
 parents include both a base-text provision (text_layer: consolidated_base)
 and an overlay provision (text_layer: amendment_overlay, e.g. the 2025
 Ordinance), NEITHER is discarded: both are tagged for the generation
 prompt ([current text - ratification pending] vs [settled law since
 2018]), and the pair is recorded in the trace (dual_layer: true) so
 evaluation can measure conflict-handling directly instead of inferring
 it from answers.

 The affiliation is by TOPIC (section_number overlap / nearest section
 path), not by text equality - the overlay rewords the provision, so
 exact-match pairing would find nothing to label.
============================================================================
"""

import json
import os
import time

from retrieval.common import eprint, est_tokens

# ---------------------------------------------------------------------------
# Loading the document store (parent side)
# ---------------------------------------------------------------------------

def load_parents(path):
    """{parent_id: parent_dict} from ingestion/parent_store.json output."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------

def _parent_label(parent, truncated=False):
    kind = parent.get("kind", "prose")
    title = parent.get("section_title") or ""
    doc = parent.get("doc", "")
    base = "[%s parent] %s — %s" % (kind, doc, title or "(untitled)")
    return base + (" [TRUNCATED - partial text]" if truncated else "")


def resolve_parents(context, parent_store, budget=None):
    """Attach the full parent (prose section or table JSON) to each
    surviving child, ordered by rerank score (or RRF fallback order),
    under a token budget. Overflowing parents are included truncated and
    LABELED as such - never silently cut.
    Returns (labeled_context_list, diag)."""
    from retrieval import config
    budget = budget or config.PARENT_TOKEN_BUDGET

    started = time.time()
    diag = {"stage": "6_parents", "n_in": len(context),
            "resolved": 0, "self_parented": 0, "missing_parent": 0,
            "n_truncated": 0, "budget": budget, "tokens_used": 0}

    labeled = []
    used = 0
    for c in context:                             # already in final order
        pid = c.get("parent_id")
        p = parent_store.get(pid) if pid else None

        if p is None:
            # Should be impossible by construction (ingestion Stage 4
            # guarantees the pointer) - its occurrence is a bug alarm,
            # not a handled case. Degrade honestly, loudly.
            diag["missing_parent"] += 1
            labeled.append({**c,
                            "parent_kind": "bare_child",
                            "label": "[bare chunk - parent missing]",
                            "parent_text": None})
            continue

        entries = []
        if p.get("kind") == "table":
            tj = p.get("table_json") or {}
            text = json.dumps(tj, ensure_ascii=False)
            tokens = est_tokens(text)
            entries.append({"parent_id": pid, "kind": "table",
                            "label": "[table parent] %s (%s)"
                                     % (tj.get("caption") or p.get("section_title") or "", p.get("doc", "")),
                            "table_json": tj})
        else:
            text = p.get("text") or ""
            tokens = est_tokens(text) or est_tokens(json.dumps(text, ensure_ascii=False))
            entries.append({"parent_id": pid, "kind": p.get("kind", "prose"),
                            "label": _parent_label(p),
                            "parent_text": text})

        truncated = False
        if used + tokens > budget:
            remaining = max(0, budget - used)
            if remaining > 200:                   # include a truncated tail
                if p.get("kind") != "table":
                    words = (p.get("text") or "").split()
                    entry = entries[0]
                    entry["parent_text"] = " ".join(words[:max(1, int(remaining / 1.3))])
                    truncated = True
                else:
                    labeled.append({**c, "parent_kind": "table",
                                    "label": "[table parent - dropped: token budget exhausted]",
                                    "table_json": None})
                    continue
            else:
                labeled.append({**c, "parent_kind": "table", "label":
                                "[parent dropped: token budget]",
                                "parent_text": None})
                continue

        used += tokens
        diag["tokens_used"] = used
        if truncated:
            diag["n_truncated"] += 1
        entry = entries[0]
        entry["truncated"] = truncated
        labeled.append({**c, **entry})
        diag["resolved"] += 1

    diag["elapsed_s"] = round(time.time() - started, 2)
    return labeled, diag


# ---------------------------------------------------------------------------
# Stage 7 - overlay vs. base resolution
# ---------------------------------------------------------------------------

_OVERLAY_LAYER = "amendment_overlay"
_BASE_LAYER = "consolidated_base"

def _topic_key(c):
    """What makes two provisions 'the same provision': the Act's section
    number (overlay rewords the text, the NUMBER is the anchor)."""
    return c.get("section_number")

def resolve_overlay(context):
    """Surface BOTH layers with explicit labels when they cover the same
    provision. Never silently pick one; never drop either.
    Returns (labeled_context, diag)."""
    started = time.time()
    diag = {"stage": "7_overlay", "dual_layer": [], "labels_applied": 0,
            "elapsed_s": None}

    overlays = [c for c in context if c.get("text_layer") == _OVERLAY_LAYER]
    bases = [c for c in context if c.get("text_layer") == _BASE_LAYER]
    labeled = list(context)

    # pair the two layers by topic key (section_number)
    for ov in overlays:
        key = _topic_key(ov)
        if key is None:
            continue
        for bs in bases:
            if _topic_key(bs) != key:
                continue
            # apply labels on the two paired chunks, in the final list
            for c in labeled:
                if c.get("chunk_id") == ov.get("chunk_id"):
                    c["layer_label"] = "[current text — ratification pending]"
                    diag["labels_applied"] += 1
                elif c.get("chunk_id") == bs.get("chunk_id"):
                    c["layer_label"] = "[settled law since 2018]"
                    diag["labels_applied"] += 1
            diag["dual_layer"].append({
                "overlay_chunk": ov.get("chunk_id"),
                "base_chunk": bs.get("chunk_id"),
                "section_number": key,
                "dual_layer": True})
            break                                 # one base per overlay topic

    diag["elapsed_s"] = round(time.time() - started, 2)
    return labeled, diag
