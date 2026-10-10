"""
============================================================================
 Nyaya - Ingestion Pipeline :: Intake driver (Stages 1 + 2 wiring)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 This is the per-document engine: it opens one PDF (or plain-text file),
 runs the Stage 1 router over every page, and dispatches each page to the
 right Stage 2 path BEFORE any chunking happens:

     prose pages   -> prose_chunker.hierarchy_chunk   (structure-aware)
     table pages   -> table_extractor (structured extraction, never flattened)
     mixed pages   -> tables pulled OUT first, then prose chunked on the
                      remaining text (so a chunk never contains table cells)
     empty pages   -> skipped entirely
     visual_routing_required pages -> flagged for manual review, never
                      chunked on garbage extraction

 It assembles the per-document record consumed by every later stage:
     rec = {doc, path, routing[], prose[], tables[], flags[], stats[],
            doc_metadata, route_counts, node_ids_filled}

 Used by: run_pipeline.py (the only caller).
============================================================================
"""

import os

import pdfplumber

from ingestion.common import doc_slug, normalize, estimate_tokens
from ingestion.doc_metadata import doc_metadata
from ingestion.page_router import classify_page, classify_text_document
from ingestion.prose_chunker import hierarchy_chunk
from ingestion.table_extractor import (table_chunk_with_retry,
                                      flag_table_rejected,
                                      flag_unextractable)
from ingestion.validation_gate import ensure_node_ids


def text_outside_tables(page, tbls):
    """pdfplumber helper: re-extract the page's text with every detected
    table region filtered OUT, so prose chunking never sees table cells
    (the 'mixed' page case)."""
    boxes = [t.bbox for t in tbls]                 # rectangle of each table
    def keep(obj):                                 # pdfplumber object filter
        cx = (obj["x0"] + obj["x1"]) / 2.0        # object center x
        cy = (obj["top"] + obj["bottom"]) / 2.0   # object center y
        for x0, top, x1, bottom in boxes:          # center inside a table?
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False                       # -> drop it (table cell)
        return True                                # -> keep (real prose)
    return page.filter(keep).extract_text() or ""  # text of prose-only regions


# ---------------------------------------------------------------------------
# PDF documents
# ---------------------------------------------------------------------------
def process_pdf(path, max_pages):
    """Run Stage 1 (routing) + Stage 2a/2b (chunking) over one PDF.
    `max_pages` caps pages per document (0 = the whole document)."""
    slug = doc_slug(path)                          # stable id used in all records
    rec = {"doc": slug, "path": path, "routing": [], "prose": [],
           "tables": [], "flags": [], "stats": [], "doc_metadata": {}}
    counts = {}                                    # label -> how many pages
    fm_seen = False                                # front-matter block emitted
                                                   # once per document
    hier_state = None                              # hierarchy tracker carried
                                                   # ACROSS pages (sections span
                                                   # page boundaries; see prose_chunker)

    with pdfplumber.open(path) as pdf:             # open lazily, page by page
        pages = pdf.pages[:max_pages] if max_pages > 0 else pdf.pages
        for page in pages:                         # ---- Stage 1: per page ----
            label, evidence = classify_page(page)  # route BEFORE any chunking
            counts[label] = counts.get(label, 0) + 1
            rec["routing"].append({"page": page.page_number,
                                   "label": label, "evidence": evidence})

            if label == "empty":                   # cover page / blank page
                continue                           # nothing to chunk at all

            if label == "visual_routing_required": # Bangla-gazette case:
                # extraction is garbage -> per the plan, this page needs
                # RENDER-then-visual-routing; here we flag it for the manual
                # transcription queue instead of pretending we understood it
                rec["flags"].append(flag_unextractable(
                    page.page_number, slug,
                    "text-layer garbage (top-token share %.2f, one-letter %.2f)"
                    % (evidence["top_token_share"], evidence["one_letter_word_share"])))
                continue

            if label == "table":                   # ---- Stage 2b: tables ----
                found = page.find_tables()         # detected grids (bboxes let
                made = rejected = 0                # us retry + caption them)
                for i, t in enumerate(found or []):
                    chunk, reason = table_chunk_with_retry(
                        page, t, page.page_number, slug, i, evidence)
                    if chunk:                               # structured JSON chunk
                        rec["tables"].append({**chunk, "doc": slug,
                                              "path": os.path.basename(path),
                                              "page": page.page_number,
                                              "est_tokens": estimate_tokens(
                                                  chunk.get("summary") or "")})
                        made += 1
                    elif reason:                   # failed validation -> manual
                        # queue, never indexed (inline Stage 3 table gate)
                        rec["flags"].append(flag_table_rejected(
                            "p%03d_t%02d" % (page.page_number, i),
                            page.page_number, slug, reason))
                        rejected += 1
                if made == 0 and rejected == 0:    # routed as table but the
                    # grid yielded no readable cells -> manual queue, never
                    # flatten the page text just to have "a chunk"
                    rec["flags"].append(flag_unextractable(
                        page.page_number, slug,
                        "table page: %d grid(s) detected but no cells readable"
                        % len(found or [])))
                continue                           # table pages never reach Stage 2a

            # ---- prose / mixed -> Stage 2a -------------------------------
            page_tables = []                       # tables found on this page
            if label == "mixed":                   # pull tables out FIRST so
                tbls = page.find_tables()          # prose never happens to
                kept = rejected = 0                # contain their cells
                for i, t in enumerate(tbls):
                    rows = t.extract()
                    if rows and len(rows) >= 2:    # skip 1-row false positives
                        ch, reason = table_chunk_with_retry(
                            page, t, page.page_number, slug, i, evidence)
                        if ch:
                            page_tables.append(ch)
                            kept += 1
                        elif reason:               # rejected grid: quarantine
                            rec["flags"].append(flag_table_rejected(
                                "p%03d_t%02d" % (page.page_number, i),
                                page.page_number, slug, reason))
                            rejected += 1
                rec["tables"].extend({**t, "doc": slug,
                                      "path": os.path.basename(path),
                                      "page": t["table_json"].get("page")
                                              if isinstance(t.get("table_json"), dict) else None,
                                      "est_tokens": estimate_tokens(t["summary"])
                                                     if t.get("summary") else t.get("est_tokens", 0)}
                                     for t in page_tables)
                # strip table regions from the prose text whenever a grid was
                # real OR rejected (rejected cells are garbage, not prose)
                text = (text_outside_tables(page, tbls) if (kept or rejected)
                        else page.extract_text()) or ""
            else:
                text = page.extract_text() or ""  # pure prose page
                # KNOWN LIMITATION: two-column layouts (e.g. Legal 500's Q&A
                # grid) interleave in raw extraction order. pdfplumber's
                # layout=True preserves geometry but pads lines with spaces
                # and still needs real column detection (x0 clustering) to
                # fix reading order - that belongs in Stage 3 QA, not here.

            blocks, stats, hier_state = hierarchy_chunk(   # Stage 2a chunking
                text, slug, hier_state)                    # ...state carried on
            for b in blocks:                       # attach doc/page metadata
                if b["kind"] == "front_matter":    # masthead repeats on later
                    if fm_seen:                    # pages are page furniture;
                        continue                   # keep only the first and lift
                    fm_seen = True                 # its fields into the
                    rec["doc_metadata"] = {        # document-level registry
                        **doc_metadata(slug),
                        **b.get("doc_fields", {})}
                rec["prose"].append({**b, "doc": slug,
                                     "path": os.path.basename(path),
                                     "page": page.page_number,
                                     "retrievable": b.get("retrievable", True),
                                     "est_tokens": estimate_tokens(b["text"])})
            rec["stats"].append({"page": page.page_number, "route": label,
                                 "strategy": stats.get("strategy"),
                                 "leaves": len(blocks),
                                 "tables": len(page_tables)})

    # ---- Stage 3-style inline assert: every surviving chunk needs an ID ----
    # (parent pointers in Stage 4 and dedup hashes in Stage 6 key off it)
    rec["node_ids_filled"] = (ensure_node_ids(rec["prose"])
                              + ensure_node_ids(rec["tables"]))
    rec["route_counts"] = counts                   # routing histogram
    return rec


# ---------------------------------------------------------------------------
# Plain-text documents
# ---------------------------------------------------------------------------
def process_text_file(path):
    """Plain-text study documents (no PDF layer) go through the same
    Stage 1 contract via the text router, then straight to Stage 2a."""
    slug = doc_slug(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()                           # raw file contents
    label, evidence = classify_text_document(text)  # Stage 1 on plain text
    rec = {"doc": slug, "path": path, "routing": [{"page": None,
           "label": label, "evidence": evidence}], "prose": [],
           "tables": [], "flags": [], "stats": [], "doc_metadata": {},
           "route_counts": {label: 1}}
    if label in ("prose", "mixed"):                # only text types here
        blocks, stats, _ = hierarchy_chunk(text, slug)  # Stage 2a (fallback)
        rec["stats"].append({"route": label, "strategy": stats.get("strategy"),
                             "leaves": len(blocks), "tables": 0})
        for b in blocks:                           # attach metadata
            if b["kind"] == "front_matter":        # plain-text docs can have
                rec["doc_metadata"] = {**doc_metadata(slug),   # mastheads too
                                       **b.get("doc_fields", {})}
            rec["prose"].append({**b, "doc": slug,
                                 "path": os.path.basename(path),
                                 "page": None,
                                 "retrievable": b.get("retrievable", True),
                                 "est_tokens": estimate_tokens(b["text"])})
    rec["node_ids_filled"] = ensure_node_ids(rec["prose"])   # ID assert pass
    return rec
