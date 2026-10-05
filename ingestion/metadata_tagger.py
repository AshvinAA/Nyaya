"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 5 - Metadata tagging (per chunk)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Every chunk - prose or table - carries metadata that is NOT used for
 semantic matching but for FILTERING and PRIORITIZATION that similarity
 alone can't express:

   source_act         which law (Labour Act vs EPZ Act vs wage gazette) -
                      prevents citing the wrong act for the wrong worker
   text_layer         consolidated_base vs amendment_overlay - drives the
                      overlay-supersedes-base rule from Stage 0
   section_path       full hierarchical path (Chapter -> Section ->
                      Sub-section -> Clause), filterable
   section_number     precise citation; overlap-coverage detection in QA
   topic_tags         THIN, manually-curated topic vocabulary (termination,
                      maternity, ...) - workers ask by colloquial topic, not
                      by the Act's structural path. Keyword-matched from a
                      deliberately small vocabulary; register bridging stays
                      the job of multi-query expansion at retrieval.
   effective_date     which version this text belongs to
   in_force           APPLICABILITY - is this the text that governs today?
   legal_status       AUTHORITY - ratified / pending_ratification / ...
                      (two different fields, both required: see Stage 0)
   translation_status official vs unofficial - prefer authoritative sources
   source_type        primary_law vs policy_advocacy - CPD's *proposed*
                      figures can never surface as the operative wage
   authority_rank     authority tier - retrieval prefers the authoritative
                      version when several sources carry the same fact
   layer_note         one human-readable sentence stating the layer status,
                      so generation can label the overlay instead of
                      silently picking a version

 This stage also mints each chunk's stable `chunk_id` (the id Stage 6's
 dedup cross-references and Stage 7's vector index key on) and provides
 build_embed_prefix() - the embed-time context injection used by Stage 7:
 the hierarchical path prefix anchors every child to its section's
 vocabulary, so a query sharing none of the clause's own words can still
 surface it. The prefix goes into the EMBEDDING VECTOR, never into the
 LLM's context window as if it were law text.

 Used by: run_pipeline.py (tags children after Stage 4), indexer.py.
============================================================================
"""

import re

from ingestion.common import next_id
from ingestion.doc_metadata import doc_metadata, layer_note

# ---------------------------------------------------------------------------
# Thin topic vocabulary (deliberately small; extend only with real queries)
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "termination":    ("termination", "dismiss", "discharge", "retrench"),
    "notice_period":  ("notice of", "notice period", "one hundred and twenty day"),
    "wages":          ("wage", "wages", "remuneration", "payable"),
    "overtime":       ("overtime", "over-time"),
    "maternity":      ("maternity", "pregnan", "expectant"),
    "leave":          ("leave with wages", "casual leave", "annual leave",
                       "sick leave", "earned leave"),
    "working_hours":  ("working hour", "hours of work", "spread-over",
                       "interval for rest"),
    "safety":         ("safety", "dangerous", "fencing", "explosive"),
    "welfare":        ("welfare", "canteen", "washing", "first-aid"),
    "child_labour":   ("child", "adolescent", "young person"),
    "trade_union":    ("trade union", "collective bargaining", "participation committee"),
    "apprentice":     ("apprentice", "apprenticeship"),
}


def topic_tags(text, cap=3):
    """Match the chunk text against the thin vocabulary; return at most
    `cap` tags. Pure keyword containment - cheap, auditable, zero deps."""
    low = (text or "").lower()
    tags = [tag for tag, pats in TOPIC_KEYWORDS.items()
            if any(p in low for p in pats)]
    return tags[:cap]


# ---------------------------------------------------------------------------
# section_path assembly
# ---------------------------------------------------------------------------
def build_section_path(c):
    """Full hierarchical path as one filterable string, e.g.
    'Chapter II · Section 23 · Sub-section (3) · Clause (b)'.
    Fallback/table/front-matter chunks get honest coarse paths."""
    parts = []
    ch = (c.get("chapter") or "").strip()
    if ch and ch != c.get("doc"):                  # doc_label used as chapter
        parts.append(ch)                           # by fallback - not a real one
    node = c.get("node") or ""
    marker = c.get("section_marker") or ""
    kind = c.get("kind")
    if kind == "table":
        parts.append("page %s table %s" % (c.get("page"), node))
    elif kind == "front_matter":
        parts.append("front matter")
    elif re.fullmatch(r"part\d+", node):           # sentence-fallback piece
        parts.append("prose part %s" % node.replace("part", ""))
    else:
        if re.fullmatch(r"\d{1,3}", marker):
            parts.append("Section %s" % marker)
        if node.startswith(marker) and "(" in node:
            rest = node[len(marker):]              # e.g. "(3)(b)" / "(proviso)"
            for m in re.findall(r"\(([^)]+)\)", rest):
                if m.isdigit():
                    parts.append("Sub-section (%s)" % m)
                elif m.isalpha():
                    parts.append("Clause (%s)" % m)
                else:
                    parts.append(m)                # "proviso"
        elif not parts:
            parts.append(node or "unnumbered")
    return " · ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Embed-time context injection (used by Stage 7)
# ---------------------------------------------------------------------------
def build_embed_prefix(c):
    """The hierarchical path prefix that gets PREPENDED to a child's text
    before embedding - 'Labour Act 2006 · Chapter II · Conditions of Service
    · Section 23 · Sub-section (3)'. The prefix anchors every child to its
    section's vocabulary; it is reconstructable from section_path metadata,
    which stays the single source of truth."""
    parts = [c.get("source_act") or c.get("doc", "")]
    ch = (c.get("chapter") or "").strip()
    if ch and ch != c.get("doc"):
        parts.append(ch)
    path = c.get("section_path") or ""
    for seg in path.split(" · "):                  # reuse the assembled path
        if seg and seg not in parts:
            parts.append(seg)
    if c.get("section_title") and c.get("kind") == "leaf":
        parts.append(c["section_title"])           # heading anchors the clause
    return " · ".join(dict.fromkeys(p for p in parts if p))   # dedupe, ordered


# ---------------------------------------------------------------------------
# The tagging pass
# ---------------------------------------------------------------------------
def tag_children(children, doc_meta_all):
    """Attach Stage 5 metadata to every child chunk, in place, and mint the
    stable chunk_id. `doc_meta_all` is slug -> document-level metadata
    (registry defaults merged with front-matter fields at intake).
    Returns nothing - chunks are mutated (they flow on to Stages 6/7)."""
    for c in children:
        slug = c["doc"]
        meta = dict(doc_meta_all.get(slug) or doc_metadata(slug))
        c["chunk_id"] = next_id("chk")             # stable id for dedup + index
        c["source_type"] = meta.get("source_type")
        c["source_act"] = meta.get("source_act")
        c["effective_date"] = meta.get("effective_date")
        c["legal_status"] = meta.get("legal_status")
        c["in_force"] = bool(meta.get("in_force"))
        c["authority_rank"] = meta.get("authority_rank")
        c["translation_status"] = meta.get("translation_status")
        c["text_layer"] = meta.get("text_layer")
        m = re.match(r"(\d{1,3})", c.get("section_marker") or "")
        c["section_number"] = m.group(1) if m else None
        c["section_path"] = build_section_path(c)
        c["topic_tags"] = topic_tags(c.get("text", ""))
        c["layer_note"] = layer_note(meta) if c.get("text_layer") else None
        if c.get("kind") == "table":
            # the table JSON is what the LLM receives - keep its metadata
            # block consistent with the chunk's flat fields (schema contract)
            tj = c.get("table_json") or {}
            tj["metadata"] = {**tj.get("metadata", {}),
                              "section_path": c["section_path"],
                              "section_number": c["section_number"],
                              "topic_tags": c["topic_tags"],
                              "layer_note": c["layer_note"],
                              "chunk_id": c["chunk_id"]}
