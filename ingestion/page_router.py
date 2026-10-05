"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 1 - Intake & content-type routing
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Every raw document enters through a router that classifies each PAGE (or
 whole plain-text file) as either prose or table BEFORE any chunking
 happens. Prose and tables have fundamentally different failure modes if
 mishandled:
   - prose loses meaning when cut at the wrong boundary
   - tables lose meaning when flattened into text AT ALL
 ...so the two paths must never mix (Stage 2a vs Stage 2b).

 Labels emitted per page:
   'table'                    tables dominate the page area
   'prose'                    flowing paragraphs, no meaningful tables
   'mixed'                    mostly prose with embedded tables
   'empty'                    cover pages / blank pages
   'visual_routing_required'  extraction came out as garbage -> the page must
                              be rendered and classified by LAYOUT, not text
                              (the Bangla-gazette case from the plan). These
                              pages are flagged for manual review, never
                              silently chunked.

 This file also owns FRONT-MATTER detection: a gazette's masthead run
 ("The Bangladesh Gazette (Extraordinary)...", registration number,
 publication date, ministry line) is document identity, not law. It must
 populate document-level metadata and NEVER reach the index as an answer
 candidate - a letterhead chunk semantically matches any query containing
 "Bangladesh" or "Ministry of Labour" and would be handed to the LLM as if
 it were a provision.

 Used by: document_intake.py (the driver that applies this router).
============================================================================
"""

import re

import pdfplumber

from ingestion.config import (TABLE_PAGE_RATIO, EMPTY_PAGE_WORDS,
                              FRONT_MATTER_MAX_LINES)
from ingestion.common import normalize

# ---------------------------------------------------------------------------
# Front-matter (gazette masthead) patterns
# ---------------------------------------------------------------------------
# A masthead line is any of these patterns; a RUN of them at the top of a
# page is the front matter block.
FRONT_MATTER_RE = re.compile(
    r"translated\s+from|registered\s+no|gazette|published\s+by\s+the\s+authority"
    r"|ministry\s+of|dated\s*:|bengali\s+year|people['’]?s\s+republic"
    r"|additional\s+issue|extraordinary|\bs\.?\s?r\.?\s?o\.?\b"
    r"|(mon|tues|wednes|thurs|fri|satur|sun)day\b", re.I)
# A run only counts as front matter when at least one of these STRONG masthead
# signals is present - stops a body paragraph that merely mentions the gazette
# from being stripped off the top of a page.
STRONG_FM_RE = re.compile(
    r"registered\s+no|published\s+by\s+the\s+authority|translated\s+from"
    r"|gazette\s+notification|people['’]?s\s+republic", re.I)
# ALL-CAPS lines inside a masthead run only count as front matter when they
# name the instrument - this stops a bare heading like "PRELIMINARY" (also
# upper-case) from being swallowed into the preamble.
GAZETTE_TITLE_KW = re.compile(
    r"gazette|rules|act\b|ordinance|order\b|notification|s\.?r\.?o\.?", re.I)


def is_front_matter_line(line, in_run):
    # a masthead pattern match, or an instrument-title line continuing a run
    if FRONT_MATTER_RE.search(line):
        return True
    return bool(in_run and line.isupper() and len(line) <= 90
                and GAZETTE_TITLE_KW.search(line))


def extract_front_matter_fields(preamble):
    """Pull document-identity fields straight out of the masthead text - raw
    strings, merged into the document-level metadata registry (Stage 5 can
    normalize them later)."""
    fields = {"registration_no": None, "gazette_name": None,
              "title_line": None, "publication_date": None,
              "issuing_authority": None, "dated_reference": None}
    m = re.search(r"Registered\s+No\.?\s*([A-Za-z]+\s*-\s*\S+|\S+)", preamble, re.I)
    if m:
        fields["registration_no"] = re.sub(r"-\s+", "-", m.group(1)).rstrip(".")
    m = re.search(r"(The\s+Bangladesh\s+Gazette[^.\n]*?)(?=\s+Published\b|\.|\s*$)",
                  preamble, re.I)
    if m:
        fields["gazette_name"] = normalize(m.group(1))
    m = re.search(r"((?:The\s+)?Bangladesh\s+(?:Labour\s+)?(?:Rules|Act|Ordinance)"
                  r"[^.\n]{0,60})", preamble, re.I)
    if m:
        fields["title_line"] = normalize(m.group(1))
    m = re.search(r"((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
                  r"[^,\n]*,\s*[A-Za-z]+\s+\d{1,2},\s*\d{4})", preamble)
    if m:
        fields["publication_date"] = normalize(m.group(1))
    m = re.search(r"(Ministry\s+of[^\n.]*)", preamble, re.I)
    if m:
        fields["issuing_authority"] = normalize(m.group(1))
    m = re.search(r"Dated\s*:\s*(.+)", preamble, re.I)
    if m:
        fields["dated_reference"] = normalize(m.group(1))
    return fields


# ---------------------------------------------------------------------------
# PDF page router
# ---------------------------------------------------------------------------
def classify_page(page):
    """Stage 1 router for one pdfplumber page.

    Returns (label, evidence_dict). The evidence is kept so the routing
    decision is auditable - the routing_report.json shows WHY a page was
    routed the way it was.
    """
    text = page.extract_text() or ""           # body text of this page
    words = page.extract_words()               # word boxes, for garbage heuristics
    tbls = page.find_tables()                  # detected table bboxes + cells

    # ---- evidence collection ---------------------------------------------
    n_words = len(words)                       # extracted word count
    total_chars = len(normalize(text))         # normalized character count
    table_area = 0.0                           # fraction of page covered by tables
    table_word_share = 0.0                     # fraction of words inside table boxes
    boxes = ""

    if tbls:
        pw, ph = float(page.width), float(page.height)   # page dimensions
        boxes = " ; ".join(                       # printable bbox list for the report
            "(%.0f,%.0f %.0fx%.0f)" % (t.bbox[0], t.bbox[1],
                                       t.bbox[2] - t.bbox[0],
                                       t.bbox[3] - t.bbox[1])
            for t in tbls)
        covered = sum((t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
                      for t in tbls)             # sum of table rectangle areas
        table_area = covered / float(pw * ph)    # ... as a fraction of the page
        in_box = sum(1 for w in words            # words whose center falls in a
                     for t in tbls               # detected table rectangle
                     if t.bbox[0] <= w["x0"] <= t.bbox[2]
                     and t.bbox[1] <= (w["top"] + w["bottom"]) / 2 <= t.bbox[3])
        table_word_share = in_box / n_words if n_words else 0.0

    # ---- garbage heuristics (the "visual_routing_required" signal) --------
    # (a) transliteration spam: pdfminer often maps Bangla glyphs to romanized
    #     junk sequences; one letter per "word".
    ghost_one_letter = 0.0
    if n_words:
        ghost_one_letter = sum(1 for w in words if len(w["text"]) <= 1) / n_words
    # (b) replaced-glyph spam: extraction emitted the SAME symbol over and over
    #     (e.g. the circled-number glyph repeated thousands of times).
    top_share = 0.0
    if n_words:
        counts = {}
        for w in words:                          # tally every distinct token
            counts[w["text"]] = counts.get(w["text"], 0) + 1
        top_share = max(counts.values()) / n_words
    # (c) sparsely-worded but long: thousands of chars, almost no real words
    #     (Bangla conjuncts often collapse to few huge "words") - covered by
    #     the total_chars / n_words ratio below.
    # (d) legacy-Bangla-font mojibake: gazettes exported from Bijoy/SutonnyMJ
    #     fonts carry dagger marks where Bangla vowel signs belong -
    #     statistically impossible in any real language's running text.
    legacy_share = 0.0
    if n_words:
        legacy_share = sum(1 for w in words
                           if "\u2020" in w["text"] or "\u2021" in w["text"]) / n_words
    garbage = n_words > 60 and (
        ghost_one_letter > 0.45 or top_share > 0.30 or total_chars / n_words > 40
        or legacy_share > 0.10
    )

    # ---- decision ladder (first match wins) -------------------------------
    if garbage:
        label = "visual_routing_required"        # trust nothing extracted here
    elif table_area >= TABLE_PAGE_RATIO:
        label = "table"                          # tables dominate the page
    elif not tbls and (n_words < EMPTY_PAGE_WORDS):
        label = "empty"                          # cover page / near-blank
    elif tbls:
        label = "mixed"                          # prose with embedded tables
    else:
        label = "prose"                          # plain paragraphs

    evidence = {"n_words": n_words, "n_chars": total_chars,
                "n_tables": len(tbls), "table_area": round(table_area, 3),
                "table_word_share": round(table_word_share, 3),
                "one_letter_word_share": round(ghost_one_letter, 3),
                "top_token_share": round(top_share, 3),
                "legacy_font_share": round(legacy_share, 3),
                "table_bboxes": boxes}
    return label, evidence


# ---------------------------------------------------------------------------
# Plain-text router (same contract, no PDF layer)
# ---------------------------------------------------------------------------
def classify_text_document(text):
    """Fallback router for .txt/.md files (used when a study document has no
    PDF). Same contract as classify_page: (label, evidence)."""
    norm = normalize(text)
    words = norm.split()
    garbage = bool(re.search(r"\u09c1|\u09c2", norm)) if norm else False
    # ^ diacritic marks with no base consonant -> broken Bangla extraction
    lines = [ln for ln in text.splitlines() if ln.strip()]
    table_like = sum(1 for ln in lines                       # 2+ column blocks
                     for m in [re.findall(r"\S+", ln)]
                     if len(m) >= 4 and len(m) * 6 < len(ln) - 4)
    if garbage:
        label = "visual_routing_required"
    elif not norm:
        label = "empty"
    elif table_like >= 3 and table_like > len(lines) * 0.3:
        label = "table"
    elif lines and (table_like >= 2 or any(re.match(r"\s*\S+(\s{2,})\S+", ln)
                                           for ln in lines)):
        label = "mixed"
    else:
        label = "prose"
    return label, {"n_words": len(words), "n_chars": len(norm),
                   "n_tables": max(0, table_like if label != "prose" else 0),
                   "table_area": None, "table_word_share": None,
                   "one_letter_word_share": None, "top_token_share": None,
                   "table_bboxes": ""}
