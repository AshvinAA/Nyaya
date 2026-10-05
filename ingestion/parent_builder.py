"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 4 - Parent-child indexing
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 What gets embedded and searched is deliberately SMALLER and more precise
 than what gets handed to the LLM for generation:

     |        | Child (embedded, searched)       | Parent (returned to LLM)   |
     | prose  | a single clause - e.g. 23(3),    | the full surrounding       |
     |        | small and specific               | section - all of 23,       |
     |        |                                  | including 23(1)'s          |
     |        |                                  | exceptions that qualify    |
     |        |                                  | the right granted in 23(3) |
     | tables | the table's natural-language     | the full structured        |
     |        | summary                          | table JSON                 |

 Why this exists: legal text is full of exactly the pattern where a right
 is granted in one sub-section and qualified in the next. A retriever
 matching only on the small chunk still finds the right needle; the LLM
 generating the answer then sees the full haystack around that needle -
 so it doesn't miss the exception that changes the answer.

 Mechanics:
   - prose leaves are grouped by (document, section marker); each group
     becomes one parent whose text is the full section. Children get a
     `parent_id` pointer. Oversized sections are split into parent PARTS
     (never one unbounded blob), and each part carries the section header.
   - chunks with no section context (sentence-fallback pieces) self-parent:
     their parent is themselves, honestly labeled, rather than pretending
     they have structure they don't.
   - table chunks already carry their parent inline (table_json); the
     parent store formalizes it under a stable id.
   - front-matter blocks get no parent and never reach the index.

 The parent store (parent_id -> text/JSON + metadata) is the DOCUMENT
 STORE of the architecture: parents are keyed by the same ids the child
 index points at, written to parent_store.json alongside the outputs.

 Used by: run_pipeline.py (after the Stage 3 gate).
============================================================================
"""

from ingestion.config import MAX_PARENT_CHILDREN, PARENT_TOKEN_SOFT_CAP
from ingestion.common import next_id, estimate_tokens


def _parent_header(children):
    """First non-empty section title + marker across a group's children -
    the heading line every parent part starts with."""
    for c in children:
        if c.get("section_title"):
            return c["section_title"]
    return children[0].get("section_marker") or "section"


def _new_parent(doc, children, part_no, n_parts, chapter):
    """Assemble one parent record from a list of prose children."""
    header = _parent_header(children)
    text = "\n\n".join(c["text"] for c in children)   # the LLM-facing haystack
    pages = sorted({c.get("page") for c in children if c.get("page") is not None})
    suffix = "" if n_parts <= 1 else " (part %d/%d)" % (part_no, n_parts)
    return {"parent_id": next_id("par"), "kind": "prose", "doc": doc,
            "section_marker": children[0].get("section_marker", ""),
            "section_title": header + suffix, "chapter": chapter or "",
            "text": text, "pages": pages,
            "n_children": len(children),
            "est_tokens": estimate_tokens(text)}


def _split_parent_parts(children):
    """Safety valve: never assemble one unbounded parent. A group bigger
    than MAX_PARENT_CHILDREN leaves or PARENT_TOKEN_SOFT_CAP tokens is cut
    into ordered parts; each part repeats the section header so it still
    stands alone in the LLM's context."""
    parts, cur, cur_tokens = [], [], 0
    for c in children:
        t = c.get("est_tokens") or estimate_tokens(c["text"])
        if cur and (len(cur) >= MAX_PARENT_CHILDREN
                    or cur_tokens + t > PARENT_TOKEN_SOFT_CAP):
            parts.append(cur)                      # flush the current part
            cur, cur_tokens = [], 0
        cur.append(c)                              # child joins the part
        cur_tokens += t
    if cur:
        parts.append(cur)
    return parts


def build_parents(prose_children, table_children):
    """Annotate every child with a parent_id and build the parent store.
    Only retrievable, non-front-matter chunks should reach this function
    (run_pipeline filters). Returns (children_annotated, parents, stats).
    """
    parents = {}

    # ---- prose: group leaves by (doc, section marker) ----------------------
    groups, order = {}, []                         # keep first-appearance order
    for c in prose_children:
        key = (c["doc"], c.get("section_marker") or "")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)

    for (doc, marker), members in ((k, groups[k]) for k in order):
        if not marker:                             # no section context:
            for c in members:                      # self-parent, honestly
                pid = next_id("par")
                parents[pid] = {"parent_id": pid, "kind": "self", "doc": doc,
                                "text": c["text"], "section_marker": "",
                                "section_title": c.get("section_title", ""),
                                "n_children": 1,
                                "est_tokens": c.get("est_tokens", 0)}
                c["parent_id"] = pid
            continue
        chapter = next((c.get("chapter") for c in members if c.get("chapter")), "")
        for i, part in enumerate(_split_parent_parts(members), start=1):
            p = _new_parent(doc, part, i, 1, chapter)   # n_parts patched below
            parents[p["parent_id"]] = p
            for c in part:
                c["parent_id"] = p["parent_id"]
        # patch part counters now that the split is known
        pids = [k for k, v in parents.items()
                if v["doc"] == doc and v["section_marker"] == marker
                and v["kind"] == "prose"]
        n_parts = len(pids)
        for i, pid in enumerate(pids, start=1):
            if n_parts > 1:
                parents[pid]["section_title"] = \
                    "%s (part %d/%d)" % (parents[pid]["section_title"], i, n_parts)

    # ---- tables: parent = the full structured JSON -------------------------
    for c in table_children:
        pid = "tbl::%s::%s" % (c["doc"], c["node"])
        parents[pid] = {"parent_id": pid, "kind": "table", "doc": c["doc"],
                        "table_json": c.get("table_json", {}),
                        "section_marker": c.get("section_marker", ""),
                        "section_title": c.get("section_title", ""),
                        "n_children": 1,
                        "est_tokens": c.get("est_tokens", 0)}
        c["parent_id"] = pid

    # front-matter blocks never get a parent and never reach the index
    for c in prose_children:
        c.setdefault("parent_id", None)

    stats = {"children": len(prose_children) + len(table_children),
             "parents": len(parents),
             "prose_parents": sum(1 for p in parents.values()
                                  if p["kind"] == "prose"),
             "self_parents": sum(1 for p in parents.values()
                                 if p["kind"] == "self"),
             "table_parents": sum(1 for p in parents.values()
                                  if p["kind"] == "table")}
    return prose_children + table_children, parents, stats
