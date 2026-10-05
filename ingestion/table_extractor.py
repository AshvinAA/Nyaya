"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 2b - The table path
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Structured extraction, never flattening. Tables never become flowing text:
 each detected grid is extracted into structured JSON - explicit row
 labels, column labels, and coerced numeric values - plus a short
 natural-language summary.

 Summary-embeds, JSON-generates (the parent-child principle applied to
 tables): the SUMMARY is what gets embedded for semantic search; the FULL
 structured JSON is what gets handed to the LLM once that summary is
 retrieved. A retrieved table chunk hands the LLM a dict - and the metadata
 block inside it is why CPD's *proposed* wage figures can never surface as
 the operative gazette wage.

 Validation gate (implemented INLINE at extraction, per the plan): every
 table passes two concrete checks before it can become a chunk -
   1. no empty column headers (a blank header makes the data uninterpretable)
   2. no flattened-line-break cells (a cell matching digit-space-digit -
      "23 9", "229 132" - is the signature of two column values merged into
      one string: exactly the corruption Stage 2b exists to prevent)
 A failing table is retried ONCE with text-strategy settings (camelot's
 lattice -> stream switch); if it still fails it is QUARANTINED to the
 manual-review queue and never indexed - the gate fails loud, per table.

 Used by: document_intake.py (which routes 'table'/'mixed' pages here).
============================================================================
"""

import re

from ingestion.common import normalize, clean_cell
from ingestion.doc_metadata import doc_metadata, summarize_table

# ---------------------------------------------------------------------------
# The inline validation gate
# ---------------------------------------------------------------------------
DIGIT_SPACE_DIGIT_RE = re.compile(r"\d\s+\d")      # "23 9", "229 132": a
                                                   # line break flattened
                                                   # into a space


def validate_table(rows_c):
    """Inline Stage 3 table gate (its two concrete checks run here at
    extraction time; rejected tables go straight to the manual queue and
    are NEVER indexed). Returns (ok, reasons)."""
    reasons = []
    header = rows_c[0]                             # first row = column labels
    empty_cols = [i for i, c in enumerate(header) if not c]
    if empty_cols:
        reasons.append("empty column header(s) at index %s" % empty_cols)
    for ri, row in enumerate(rows_c):
        for ci, cell in enumerate(row):
            if DIGIT_SPACE_DIGIT_RE.search(cell):
                reasons.append("row %d col %d: digit-space-digit in '%s' "
                               "(flattened line break)" % (ri, ci, cell))
    return (not reasons), reasons


def coerce_cell(value):
    # '17568' -> 17568, '19,310' -> 19310, '3.5' -> 3.5, else the string -
    # numeric cells should land in the JSON as numbers, per the schema, so
    # the LLM can compute with them instead of string-matching
    s = str(value).strip()
    if re.fullmatch(r"-?\d[\d,]*", s):
        return int(s.replace(",", ""))
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------
def table_chunk(rows, page, doc_label, tbl_index, evidence, caption=""):
    """Build one structured table chunk in the agreed schema -
    table_id / caption / col_labels / row_labels / data / metadata.
    Returns (chunk, None) when the grid passes validation, or (None, reason)
    when it must be quarantined to the manual queue instead."""
    rows_c = [[clean_cell(c) for c in row] for row in rows]     # clean every cell
    width = max(len(r) for r in rows_c) if rows_c else 0        # widest row
    rows_c = [r + [""] * (width - len(r)) for r in rows_c]      # pad short rows
    rows_c = [r for r in rows_c if any(c for c in r)]           # drop blank rows
    if not rows_c:                                              # empty grid ->
        return None, None                                       # nothing to see
    ok, reasons = validate_table(rows_c)           # inline gate
    if not ok:
        return None, "; ".join(reasons)            # -> manual queue, no index
    table_id = "p%03d_t%02d" % (page, tbl_index)
    col_labels = rows_c[0]                         # header row = column labels
    data = {}                                      # row_label -> {col: value}
    for row in rows_c[1:]:
        label = row[0] or "row_%d" % (len(data) + 1)   # 1st column = row label
        key, n = label, 2
        while key in data:                         # disambiguate duplicate
            key = "%s (%d)" % (label, n); n += 1   # row labels
        data[key] = {col: coerce_cell(cell)
                     for col, cell in zip(col_labels[1:], row[1:])}
    summary = summarize_table(rows_c)              # embed this
    return {"kind": "table", "node": table_id,
            "retrievable": True,                   # passed validation
            "text": summary,                       # what gets embedded
            "table_json": {                        # what the LLM would get
                "table_id": table_id,
                "caption": caption,                # filled at extraction
                "col_labels": col_labels,
                "row_labels": list(data),
                "data": data,
                "source_doc": doc_label,
                "page": page,
                "metadata": doc_metadata(doc_label)},   # Stage 5 fields
            "summary": summary,
            "chapter": doc_label,                  # hierarchy metadata
            "section_marker": "page %d" % page,
            "section_title": caption,
            "extraction_evidence": evidence}, None  # transparency for QA


def find_caption(page, bbox, window=90):
    """Best-effort caption: the closest line of text just ABOVE the table's
    top edge (within `window` points). Returns '' when nothing is there - a
    missing caption is metadata debt, not a validation failure."""
    x0, top, x1, bottom = bbox
    above = [w for w in page.extract_words()
             if w["bottom"] <= top and w["bottom"] >= top - window]
    if not above:
        return ""
    closest = max(w["top"] for w in above)         # the line nearest the table
    line = sorted((w for w in above if abs(w["top"] - closest) < 3),
                  key=lambda w: w["x0"])           # reading order left -> right
    return normalize(" ".join(w["text"] for w in line))[:200]


def table_chunk_with_retry(page, tbl, page_number, doc_label, tbl_index, evidence):
    """Extract + validate one detected table. On validation failure, retry
    ONCE on the same region with text-strategy settings (the lattice ->
    stream switch); if the retry also fails, return the combined reason for
    the manual queue. Returns (chunk_or_None, reason_or_None)."""
    caption = find_caption(page, tbl.bbox)         # best-effort label
    chunk, reason = table_chunk(tbl.extract() or [], page_number, doc_label,
                                tbl_index, evidence, caption)
    if chunk or reason is None:                    # passed, or empty grid
        return chunk, reason
    try:                                           # retry with text strategy
        rows2 = page.crop(tbl.bbox).extract_table(
            {"vertical_strategy": "text", "horizontal_strategy": "text",
             "snap_tolerance": 3, "intersection_tolerance": 3,
             "text_x_tolerance": 2}) or []
    except Exception as exc:                       # a failed retry still
        return None, "%s | retry raised: %s" % (reason, exc)   # quarantines
    chunk2, reason2 = table_chunk(rows2, page_number, doc_label,
                                  tbl_index, evidence, caption)
    if chunk2:
        return chunk2, None
    return None, "%s | retry(text-strategy): %s" % (reason, reason2)


# ---------------------------------------------------------------------------
# Quarantine flags (the manual-review queue)
# ---------------------------------------------------------------------------
def flag_table_rejected(node_id, page, doc_label, reason):
    """A table that failed validation is quarantined - it goes to the manual
    queue and is never indexed (a garbled wage figure that 'looks
    structured' is worse than a missing one)."""
    return {"kind": "table_validation_failed", "node": node_id,
            "text": "[TABLE REJECTED BY VALIDATION - quarantined for manual "
                    "review; never indexed]",
            "doc": doc_label, "page": page, "reason": reason}


def flag_unextractable(page, doc_label, reason):
    """Route a page to manual transcription when the PDF layer cannot yield
    a real table (or any legible text) but the router said it should."""
    return {"kind": "needs_manual_transcription", "node": "p%03d" % page,
            "text": "[TABLE PAGE - PDF-layer returned no table structure; "
                    "marked for one-time manual transcription]",
            "doc": doc_label, "page": page, "reason": reason}
