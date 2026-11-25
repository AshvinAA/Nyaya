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
from ingestion.common import normalize, estimate_tokens, ascii_digits
from ingestion.page_router import (is_front_matter_line,
                                      extract_front_matter_fields,
                                      STRONG_FM_RE)

# ---------------------------------------------------------------------------
# Legal-numbering patterns, e.g.:  "23. (1) ..."  "23(1)"  "(2) ..."  "3. ..."
# ---------------------------------------------------------------------------
# "23."  chapter-sections; "196A." inserted sections (the base Act's heading form,
# e.g. "196A. Anti-trade union discrimination.-", p.83 of the 2006 PDF) too.
SECTION_RE = re.compile(r"^[0-9]{1,3}[A-Z]?\.")
SUBSEC_RE  = re.compile(r"^\(([0-9]{1,2})\)")     # "(3)"  sub-sections
CLAUSE_RE  = re.compile(r"^\(([a-z]{1,2})\)")     # "(b)"  lettered clauses

# Bangla-original gazettes (hand transcripts, see transcripts.py) number the
# same hierarchy with Bangla digits and letters:  "৩। ..."  "(২ক)"  "(ক)"
SECTION_BN_RE = re.compile(r"^([০-৯]{1,3})।\s*")              # "৩।" section
SUBSEC_BN_RE  = re.compile(r"^\(([০-৯]{1,2}[ক-হ]?)\)")       # "(২)" / "(২ক)"
CLAUSE_BN_RE  = re.compile(r"^\(([অ-৥]{1,4})\)")   # "(ক)" / "(অ)"
# "১। সংক্ষিপ্ত শিরোনাম ও প্রবর্তন।—(১) এই আইন ..." -> title | body
HEADING_BN_RE = re.compile(r"^(.{1,250}?)।\s*[—–-]+\s*(.*)$")


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
def group_lines(lines, max_tokens=TOKEN_CAP, overlap_ratio=OVERLAP_RATIO):
    """Too-long rule for a structured leaf: cut it at its own LINE boundaries
    (paragraphs, table rows) into token-capped groups with a small overlap,
    falling back to sentence splitting for a single oversized line."""
    pieces, cur, cur_tokens = [], [], 0
    for line in lines:
        cost = estimate_tokens(line)
        if cost > max_tokens:                        # one huge paragraph
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_tokens = [], 0
            pieces.extend(split_long_text(line, max_tokens, overlap_ratio))
            continue
        if cur and cur_tokens + cost > max_tokens:   # would overflow the cap
            pieces.append(" ".join(cur))
            keep = int(len(cur) * overlap_ratio)     # carry ~15% of the lines
            cur = cur[-keep:] if keep else []
            cur_tokens = estimate_tokens(" ".join(cur)) if cur else 0
        cur.append(line)
        cur_tokens += cost
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def flush_pending(state, pending, blocks):
    """Close the current block: bundle accumulated lines into one structured
    leaf unit stamped with the hierarchy position tracked in `state`. A leaf
    over TOKEN_CAP is split at line boundaries (the too-long rule); every
    piece keeps the same hierarchy stamp."""
    if pending:
        text = normalize(" ".join(pending))
        pieces = ([text] if estimate_tokens(text) <= TOKEN_CAP
                  else [normalize(p) for p in group_lines(pending)])
        for i, piece in enumerate(pieces, start=1):
            block = {"kind": "leaf",                    # a real content block
                     "node": state["node"], "text": piece,
                     "chapter": state["chapter"],
                     "section_marker": state["section_marker"],
                     "section_title": state["section_title"]}
            if len(pieces) > 1:
                block["split_part"] = "%d/%d" % (i, len(pieces))
            blocks.append(block)
        pending.clear()        # careful: caller passes the live list


def merge_short_leaves_forward(blocks, min_tokens=MIN_LEAF_TOKENS):
    """The too-short rule: a structural leaf or bare heading whose own text
    is too short to carry retrieval signal ("PRELIMINARY", ~1 token) must
    never be emitted standalone - it merges FORWARD, becoming a prefix on
    the next surviving node ("PRELIMINARY" ends up opening the Rule 1
    chunk). Symmetric to the too-long token-split fallback.
    A carried run that belongs to a numbered section (its heading, or a
    numbered unit of it) never crosses into a DIFFERENT section: a whole
    section that is short ("39. Section 184 is omitted.") is emitted as its
    own leaf instead of being glued onto section 40.
    Returns (blocks, n_merged)."""
    merged, carry, n_merged = [], [], 0

    def anchored(blk):
        # a section's own heading line was seen on this page. (A leaf's
        # section_marker alone proves nothing: running headers and chapter
        # titles at the top of a page inherit the PREVIOUS page's marker.)
        return blk["kind"] == "section_header"

    def carry_text():
        return normalize(" ".join(c["text"] for c in carry))

    def standalone_carry(next_marker):
        anchor = next((c for c in carry if anchored(c)), None)
        return (anchor is not None
                and anchor.get("section_marker") != next_marker
                and estimate_tokens(carry_text()) >= min_tokens)

    def emit_carry_standalone():
        anchor = next(c for c in carry if anchored(c))
        merged.append({**anchor, "kind": "leaf", "text": carry_text()})
        carry.clear()

    for b in blocks:
        short = estimate_tokens(b["text"]) < min_tokens
        # a short NUMBERED unit ("(2) It shall come into force at once.")
        # is content of its own section, not a heading for the next one:
        # it merges BACKWARD into the previous leaf of the same section
        if (short and not carry and b["kind"] == "leaf"
                and "(" in (b.get("node") or "") and merged
                and merged[-1]["kind"] == "leaf"
                and merged[-1].get("section_marker") == b.get("section_marker")):
            merged[-1]["text"] = normalize(merged[-1]["text"] + " " + b["text"])
            n_merged += 1
            continue
        # a carried section must not leak into the next section - when it is
        # a provision in its own right. (A numbered schedule list - "1. Lead
        # poisoning;" - also parses as headings; those stay a forward prefix.)
        if standalone_carry(b.get("section_marker")):
            emit_carry_standalone()
        # a section heading is never content on its own, whatever its length
        # (Bangla headings like "২০০৬ সনের ৪২ নং আইনের ধারা ১৭ এর সংশোধন"
        # run past MIN_LEAF_TOKENS) - it always prefixes the next node
        if b["kind"] == "section_header" or (b["kind"] == "leaf" and short):
            carry.append(b)                            # hold as prefix for
            n_merged += 1                              # the next real node
            continue
        if carry:                                  # first real node after the
            b["text"] = normalize(carry_text() + " " + b["text"])  # short run
            carry.clear()                          # gets the heading as prefix
        merged.append(b)
    if carry:
        if standalone_carry(merged[-1].get("section_marker") if merged else None):
            emit_carry_standalone()                # a short section at the end
        elif merged:                               # trailing bare heading with
            merged[-1]["text"] = normalize(        # no next node: merge
                merged[-1]["text"] + " " + carry_text())   # backward as last
            n_merged += 1                          # resort
        else:                                      # the whole page was bare
            merged.append({"kind": "leaf", "node": "", "text": carry_text(),
                           "chapter": "", "section_marker": "",
                           "section_title": ""})   # headings - keep them
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
    quote_lines = 0                              # '> ' lines (quoted text)
    table_rows = 0                               # '|' data rows emitted
    table_header = None                          # column-header row of the
    header_used = False                          # current '|' table run

    def end_table_run():
        # a header row with no data rows under it is still content
        nonlocal table_header, header_used
        if table_header is not None and not header_used:
            pending.append(table_header)
        table_header, header_used = None, False

    for raw in lines[cut:]:                      # walk once, top to bottom
        line = raw.strip()                       # trimmed copy for matching
        # --- transcript convention: '> ' = text QUOTED by the document ------
        # (e.g. the wording an amending Act inserts). Its own numbering -
        # "(ক)", "৬১ক।" - belongs to the quoted law, not to this document,
        # so it is body text of the provision that quotes it.
        if line.startswith(">"):
            end_table_run()
            pending.append(line[1:].strip())
            quote_lines += 1
            continue
        # --- table row "| a | b |": one leaf per row, header prefixed ------
        if line.startswith("|"):
            flush_pending(state, pending, blocks)
            state["node"] = state["section_marker"]   # a row is not inside the
                                                      # clause that preceded it
            if table_header is None:             # first row of a run is the
                table_header = line              # column header
            else:
                pending.extend([table_header, line])
                header_used = True
                flush_pending(state, pending, blocks)
                table_rows += 1
            struct_hits += 1
            continue
        end_table_run()
        # --- pattern A-bn: Bangla section "৩। শিরোনাম।—(১) ..." -------------
        m = SECTION_BN_RE.match(line)
        if m:
            flush_pending(state, pending, blocks)
            marker = str(int(ascii_digits(m.group(1))))      # "০২" -> "2"
            rest = line[m.end():]
            h = HEADING_BN_RE.match(rest)
            title, body = (h.group(1).strip(), h.group(2).strip()) if h else ("", rest)
            state["section_marker"] = marker
            state["section_title"] = normalize(title)
            state["node"] = marker
            if title:
                blocks.append({"node": marker, "text": normalize(title),
                               "kind": "section_header", "chapter": state["chapter"],
                               "section_marker": marker,
                               "section_title": normalize(title)})
            sm = SUBSEC_BN_RE.match(body)        # body opens with "(১)"?
            if sm:
                state["node"] = "%s(%s)" % (marker, ascii_digits(sm.group(1)))
            if body:
                pending.append(body)
            struct_hits += 1
            continue
        # --- pattern B-bn / C-bn: "(২)" sub-section, "(ক)" clause ----------
        m = SUBSEC_BN_RE.match(line)
        if m:
            flush_pending(state, pending, blocks)
            state["node"] = "%s(%s)" % (state["section_marker"], ascii_digits(m.group(1)))
            struct_hits += 1
            pending.append(line)
            continue
        m = CLAUSE_BN_RE.match(line)
        if m:
            flush_pending(state, pending, blocks)
            base = state["node"].split("(")[0] if "(" in state["node"] else state["node"]
            state["node"] = "%s(%s)" % (base, m.group(1))
            struct_hits += 1
            pending.append(line)
            continue
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

    end_table_run()
    flush_pending(state, pending, blocks)          # close the final open block

    # --- min-token merge rule: bare headings never stand alone -------------
    blocks, tiny_merged = merge_short_leaves_forward(blocks)

    # --- decide: did structure actually work for this document? ------------
    # A page that is all quoted text (e.g. a schedule an amending Act
    # substitutes) is still structured: it continues the open section. So is
    # a page whose table rows were already cut one leaf per row.
    structured = (struct_hits >= 3 or table_rows > 0
                  or bool(quote_lines and state["section_marker"]))
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
