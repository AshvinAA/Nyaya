"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 2a - The prose path
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Hierarchical, structure-aware chunking. Prose content is split along the
 document's OWN logical hierarchy rather than by a fixed token count:

     Chapter -> Section -> Sub-section -> Clause

 The chunker walks the document's actual numbering and headings, producing
 one chunk per leaf node - typically a clause or sub-section, the smallest
 unit that still stands on its own. A chunk boundary never cuts across two
 different clauses.

 Failure mode this prevents: a worker's question about Section 23 retrieving
 a chunk that starts mid-sentence in Section 22.

 Two symmetric safety rules (both from INGESTION_PIPELINE.md):
   TOO LONG  -> a single leaf over ~TOKEN_CAP tokens is split within itself
                into overlapping sentence groups (split_long_text below).
   TOO SHORT -> a bare structural heading ("PRELIMINARY", ~1 token) carries
                no retrieval signal, so it never stands alone: it merges
                FORWARD, becoming a prefix on the next real node's chunk -
                exactly how the heading functions in the source document.
                A bare heading at the very END of a document merges
                backward as a last resort.

 Documents without legal numbering (study reports, guides) fall back to
 sentence grouping with the same token cap and overlap.

 Used by: document_intake.py (which feeds it the routed prose text).
============================================================================
"""

import re

from ingestion.config import MIN_LEAF_TOKENS, FRONT_MATTER_MAX_LINES, \
    TOKEN_CAP, OVERLAP_RATIO
from ingestion.common import normalize, estimate_tokens
from ingestion.page_router import (is_front_matter_line,
                                      extract_front_matter_fields,
                                      STRONG_FM_RE)

# ---------------------------------------------------------------------------
# Legal-numbering patterns, e.g.:  "23. (1) ..."  "23(1)"  "(2) ..."  "3. ..."
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(r"^[0-9]{1,3}+\.")        # "23."  chapter-sections
SUBSEC_RE  = re.compile(r"^\(([0-9]{1,2})\)")     # "(3)"  sub-sections
CLAUSE_RE  = re.compile(r"^\(([a-z]{1,2})\)")     # "(b)"  lettered clauses


def split_long_text(text, max_tokens=TOKEN_CAP, overlap_ratio=OVERLAP_RATIO):
    """Too-long fallback: split oversized text into overlapping, token-capped
    sentence groups. Note the Bangla daari sign is treated as a sentence
    ender too, so Bangla prose splits on real sentence boundaries."""
    sentences = re.split(r"(?<=[.!?।॥])\s+", text.strip())
    pieces, cur, cur_tokens = [], [], 0
    for sentence in sentences:                       # group sentences greedily
        cost = estimate_tokens(sentence)             # token cost of this sentence
        if cur and cur_tokens + cost > max_tokens:   # would overflow the cap?
            pieces.append(" ".join(cur))             # -> flush the current piece
            keep = max(1, int(len(cur) * overlap_ratio))  # overlap = last ~15% sentences
            cur = cur[-keep:]                        # carry them into the next piece
            cur_tokens = estimate_tokens(" ".join(cur))
        cur.append(sentence)                         # add sentence to the piece
        cur_tokens += cost                           # running token total
    if cur:                                          # flush whatever is left
        pieces.append(" ".join(cur))
    return pieces or [text]                          # never return an empty list


# ---------------------------------------------------------------------------
# Block assembly
# ---------------------------------------------------------------------------
def flush_pending(state, pending, blocks):
    """Close the current block: bundle accumulated lines into one structured
    leaf unit stamped with the hierarchy position tracked in `state`."""
    if pending:
        blocks.append({"kind": "leaf",                  # a real content block
                       "node": state["node"], "text": normalize(" ".join(pending)),
                       "chapter": state["chapter"],
                       "section_marker": state["section_marker"],
                       "section_title": state["section_title"]})
        pending.clear()        # careful: caller passes the live list


def merge_short_leaves_forward(blocks, min_tokens=MIN_LEAF_TOKENS):
    """The too-short rule: a structural leaf or bare heading whose own text
    is too short to carry retrieval signal ("PRELIMINARY", ~1 token) must
    never be emitted standalone - it merges FORWARD, becoming a prefix on
    the next surviving node ("PRELIMINARY" ends up opening the Rule 1
    chunk). Symmetric to the too-long token-split fallback.
    Returns (blocks, n_merged)."""
    merged, carry, n_merged = [], "", 0
    for b in blocks:
        if b["kind"] in ("leaf", "section_header") \
                and estimate_tokens(b["text"]) < min_tokens:
            carry = (carry + " " + b["text"]).strip()  # hold as prefix for
            n_merged += 1                              # the next real node
            continue
        if carry:                                  # first real node after the
            b["text"] = normalize(carry + " " + b["text"])   # short run gets
            carry = ""                             # the heading as prefix
        merged.append(b)
    if carry and merged:                           # trailing bare heading with
        merged[-1]["text"] = normalize(            # no next node: merge
            merged[-1]["text"] + " " + carry)      # backward as last resort
        n_merged += 1
    elif carry:                                    # the whole page was bare
        merged.append({"kind": "leaf", "node": "", "text": carry,
                       "chapter": "", "section_marker": "",
                       "section_title": ""})       # headings - keep them
    return merged, n_merged


# ---------------------------------------------------------------------------
# The hierarchy walker
# ---------------------------------------------------------------------------
def hierarchy_chunk(text, doc_label, state=None):
    """Stage 2a main entry. Walks a document's lines, tracking Chapter ->
    Section -> Sub-section -> Clause numbers. Produces one structured block
    per leaf (sub-section/clause). Falls back to sentence grouping for
    documents without legal numbering (fewer than 3 structural hits).

    `state` carries the hierarchy tracker ACROSS PAGES: legal sections span
    page boundaries, so a page that starts mid-section ('(4) ...') inherits
    the governing section from the previous page instead of emitting a
    parentless clause leaf (which Stage 3 would flag as a broken parse).
    Pass the state returned by the previous page's call; None starts fresh.

    Returns (blocks, stats, state) where blocks carry:
        kind          leaf | section_header | front_matter | prose_fallback
        node          full hierarchical marker, e.g.  23 / 23(3) / 23(3)(b)
        chapter       the last PART/CHAPTER heading seen above this block
        section_marker  the governing section number, e.g. "23"
        section_title the section's title text, e.g. 'Compensation for injury'
    """
    if state is None:                              # fresh document (or a
        state = {"node": "", "chapter": "",       # caller that wants no
                 "section_marker": "",            # cross-page context)
                 "section_title": ""}
    lines = [ln for ln in text.splitlines() if ln.strip()]  # non-empty, in order

    # --- front-matter strip: a gazette masthead is document identity, not --
    # law. Cut the leading run off the body; keep it as one non-retrievable
    # kind=front_matter block whose fields feed document-level metadata.
    cut, grace = 0, 0
    for i, raw in enumerate(lines[:FRONT_MATTER_MAX_LINES]):
        if is_front_matter_line(raw.strip(), cut > 0):
            cut, grace = i + 1, 0                  # extend the masthead run
        elif cut == 0 and grace < 2:               # allow a short non-matching
            grace += 1                             # lead-in (logo / mojibake
                                                   # title line) before the
        else:                                      # masthead proper begins
            break                                  # real content begins here
    fm_blocks = []
    head_text = normalize(" ".join(ln.strip() for ln in lines[:cut]))
    if cut and STRONG_FM_RE.search(head_text):     # require a real masthead
        fm_blocks = [{"kind": "front_matter",      # signal - never strip a
                      "node": "front_matter",      # body paragraph that just
                      "retrievable": False,        # mentions the gazette
                      "text": head_text,
                      "doc_fields": extract_front_matter_fields(head_text),
                      "chapter": doc_label, "section_marker": "",
                      "section_title": ""}]
    else:
        cut, head_text = 0, ""                     # not a masthead: the page
    remainder_text = normalize(" ".join(           # keeps ALL of its lines
        ln.strip() for ln in lines[cut:]))

    blocks, pending = [], []                     # output blocks + current buffer
    # NOTE: `state` (the hierarchy tracker) arrives from the caller - it may
    # already carry section context from the previous page of this document.
    struct_hits = 0                              # lines that actually shaped
                                                 # the hierarchy (>=3 = structured)
    for raw in lines[cut:]:                      # walk once, top to bottom
        line = raw.strip()                       # trimmed copy for matching
        # --- pattern A: numbered section heading like "23. Title here" ------
        if SECTION_RE.match(line):
            flush_pending(state, pending, blocks)  # close what came before
            m = SECTION_RE.match(line)             # capture the number
            heading_text = normalize(line[m.end():])  # title after the number
            if heading_text:                       # "23. Wages" -> header block
                blocks.append({"node": m.group(0).rstrip("."), "text": heading_text,
                               "kind": "section_header", "chapter": state["chapter"],
                               "section_marker": m.group(0).rstrip("."),
                               "section_title": heading_text})
            # update tracker - BEFORE matching body lines that follow
            state["section_marker"] = m.group(0).rstrip(".")
            state["section_title"] = heading_text
            state["node"] = state["section_marker"]
            struct_hits += 1                       # this line shaped the hierarchy
            continue
        # --- pattern B: sub-section start "(3) ..." --------------------------
        if SUBSEC_RE.match(line):
            flush_pending(state, pending, blocks)
            m = SUBSEC_RE.match(line)
            state["node"] = "%s(%s)" % (state["section_marker"], m.group(1))
            struct_hits += 1
            pending.append(line)                   # THIS line is clause content
            continue
        # --- pattern C: lettered clause "(b) ..." ----------------------------
        if CLAUSE_RE.match(line):
            flush_pending(state, pending, blocks)
            m = CLAUSE_RE.match(line)
            state["node"] = "%s(%s)" % (state["node"].split("(")[0] if "(" in state["node"]
                                        else state["node"], m.group(1))
            struct_hits += 1
            pending.append(line)
            continue
        # --- pattern D: PART / CHAPTER style headers -------------------------
        if re.match(r"^(PART|CHAPTER)\s+[IVXLC]+", line, re.I):
            flush_pending(state, pending, blocks)
            state["chapter"] = line                # remember as running context
            state["node"] = ""
            struct_hits += 1
            continue
        # --- pattern E: "PROVIDED THAT" provisos open a sub-block ------------
        if re.match(r"^Provided", line):
            flush_pending(state, pending, blocks)
            state["node"] = (state["node"].split("(")[0] if "(" in state["node"]
                             else state["node"]) + "(proviso)"
            struct_hits += 1
            pending.append(line)
            continue
        # --- anything else: continuation of the current block ----------------
        pending.append(line)
        # observed rare failure: a numbered "23." inside body text is NOT
        # caught as a heading; acceptable for the demo - the Stage 3 gate
        # flags continuity breaks downstream.

    flush_pending(state, pending, blocks)          # close the final open block

    # --- min-token merge rule: bare headings never stand alone -------------
    blocks, tiny_merged = merge_short_leaves_forward(blocks)

    # --- decide: did structure actually work for this document? ------------
    structured = struct_hits >= 3
    if structured:
        return fm_blocks + blocks, {
            "strategy": "hierarchical",
            "leaves": len(blocks) + len(fm_blocks),
            "struct_hits": struct_hits, "tiny_merged": tiny_merged}, state

    # --- fallback path: documents without legal numbering -------------------
    # (state still flows through unchanged: the next page may resume a
    # numbered section even if THIS page had none)
    fb_blocks = sentence_fallback(remainder_text, doc_label)
    return fm_blocks + fb_blocks, {"strategy": "sentence_fallback",
                                   "struct_hits": struct_hits,
                                   "tiny_merged": 0}, state


def sentence_fallback(text, doc_label):
    """>= 90%-prose documents without CHAPTER/section/clause numbering get
    sentence-grouped chunks (~TOKEN_CAP tokens, ~15% overlap forwards)."""
    norm = normalize(text)                         # flatten line breaks first
    if not norm:                                   # page was all front matter
        return []                                  # -> nothing left to chunk
    pieces = split_long_text(norm)                 # overlapping token-capped pieces
    return [{"kind": "prose_fallback", "node": "part%02d" % (i + 1),
             "text": p, "chapter": doc_label, "section_marker": "",
             "section_title": ""} for i, p in enumerate(pieces)]
