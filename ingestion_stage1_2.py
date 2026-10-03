"""
============================================================================
 Nyaya - Ingestion Pipeline: Stage 1 (routing) + Stage 2 (chunking)
============================================================================
 Runnable demo of the architecture in INGESTION_PIPELINE.md:

   Stage 1  Intake & content-type routing
            Every PDF page is classified (prose / table / mixed / empty /
            visual_routing_required) BEFORE any chunking happens.

   Stage 2a Prose path - hierarchical, structure-aware chunking
            Walks Chapter -> Section -> Sub-section -> Clause numbering.
            Documents without legal numbering fall back to sentence
            grouping with a token cap and overlap.

   Stage 2b Table path - structured extraction, never flattening
            Tables become JSON (headers + rows + a summary). The summary
            is what gets embedded; the JSON is what the LLM gets.
            Un-extractable tables are flagged for manual transcription
            instead of being flattened into prose.

 Run it:
     venv/Scripts/python ingestion_stage1_2.py                 # 30 pages/doc
     venv/Scripts/python ingestion_stage1_2.py --max-pages 0   # whole corpus

 Output -> ./ingestion_output/
     routing_report.json    one record per page (the Stage 1 manifest)
     prose_chunks.json      Stage 2a chunks with hierarchy metadata
     tables_extracted.json  Stage 2b structured tables with summaries
============================================================================
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse            # parses --max-pages / --docs-dir / --out CLI flags
import json                # serializes the three output JSON files
import os                  # lists the docs folder, joins paths, builds slugs
import re                  # regexes for the legal-numbering structure walker
import sys                 # reconfigures stdout to UTF-8 (Bangla text on Windows)

import pdfplumber          # Stage 1 text extraction + Stage 2b table detection

# ---------------------------------------------------------------------------
# Configuration - every knob in one place
# ---------------------------------------------------------------------------
DOCS_DIR = "docs"              # folder holding the raw corpus PDFs
OUT_DIR = "ingestion_output"   # folder where the JSON outputs are written
DEFAULT_MAX_PAGES = 30         # per-document page cap so a demo run is quick

TOKEN_CAP = 350                # Stage 2a: max ~tokens per chunk (plan: 256-384)
OVERLAP_RATIO = 0.15           # Stage 2a: 15% overlap between split pieces

TABLE_PAGE_RATIO = 0.5         # page is "table" when tables cover >= 50% of area
MIXED_PAGE_RATIO = 0.08        # page is "mixed" when tables cover >= 8% of area
EMPTY_PAGE_WORDS = 15          # fewer words + no tables = cover/blank page

# ---------------------------------------------------------------------------
# Small shared helpers used by every stage
# ---------------------------------------------------------------------------
def normalize(text):
    # collapse every whitespace run (newlines, tabs) into single spaces
    return re.sub(r"\s+", " ", text or "").strip()

def estimate_tokens(text):
    # rough token estimate (~1.3 tokens per word) - no tiktoken dependency
    return max(1, round(len(text.split()) * 1.3))

def clean_cell(value):
    # pdfplumber table cells carry embedded newlines; flatten and trim them
    return normalize(str(value)) if value is not None else ""

def doc_slug(path):
    # "docs/BangladeshGagetteSep2015.pdf" -> "bangladeshgagettesep2015"
    base = os.path.splitext(os.path.basename(path).lower())[0]  # name minus .pdf
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")          # non-alnum -> "_"

def preview(text, limit=500):
    # shortens text for console display so samples stay readable
    return text if len(text) <= limit else text[:limit] + " ..."

def split_long_text(text, max_tokens=TOKEN_CAP, overlap_ratio=OVERLAP_RATIO):
    # splits oversized text into overlapping token-capped pieces
    # note: the Bangla daari sign (৷/।) is treated as a sentence ender too
    sentences = re.split(r"(?<=[.!?\u0964\u0965])\s+", text.strip())
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
# Stage 1 - Content-type routing: every page is classified before chunking
# ---------------------------------------------------------------------------
def classify_page(page):
    """Stage 1 router. Classifies one pdfplumber page as:

      'table'                  tables dominate the page area
      'prose'                  flowing paragraphs, no meaningful tables
      'mixed'                  mostly prose with embedded tables
      'empty'                  cover pages / blank pages
      'visual_routing_required' extraction came out as garbage, so this page
                               must be rendered and classified by LAYOUT, not
                               text (the Bangla-gazette case from the plan)

    Returns (label, evidence_dict). The evidence is kept so the routing
    decision is auditable - and so the debug JSON can show why a page was
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
        covered = 0.0                            # union area (partial overlaps ok)
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
    #     (Bangla conjuncts often collapse to few huge "words").
    # (d) legacy-Bangla-font mojibake: gazettes exported from Bijoy/SutonnyMJ
    #     fonts carry dagger marks (^, ^^) where Bangla vowel signs belong -
    #     statistically impossible in any real language's running text
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


# ---------------------------------------------------------------------------
# Stage 2a - The prose path: hierarchical, structure-aware chunking
# ---------------------------------------------------------------------------
# Legal numbering, e.g.:  "23. (1) ..."  "23(1)"  "(2) ..."  "3. ..."
SECTION_RE    = re.compile(r"^[0-9]{1,3}+\.")          # "23."  Chapter-sections
SUBSEC_RE     = re.compile(r"^\(([0-9]{1,2})\)")       # "(3)"
CLAUSE_RE     = re.compile(r"^\(([a-z]{1,2})\)")       # "(b)"


def flush_pending(state, pending, blocks):
    # close the current block: bundle accumulated lines into a structured unit
    if pending:
        blocks.append({"kind": "leaf",                  # a real content block
                       "node": state["node"], "text": normalize(" ".join(pending)),
                       "chapter": state["chapter"],
                       "section_marker": state["section_marker"],
                       "section_title": state["section_title"]})
        pending.clear()        # careful: caller passes the live list


def hierarchy_chunk(text, doc_label):
    """Stage 2a. Walks a document's lines, tracking Chapter -> Section ->
    Sub-section -> Clause numbers. Produces one structured block per leaf
    (sub-section/clause). Falls back to sentence grouping for documents
    without legal numbering.

    Returns (blocks, stats) where blocks carry:
        kind          leaf | body | section_header | prose_fallback
        node          full hierarchical marker, e.g.  23 / (3) / (b)
        chapter       the last SECTION-heading seen above this block
        section       the highest-level heading above this block
        section_title the heading text, e.g. 'Compensation for injury'
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]  # non-empty, in order
    blocks, pending = [], []                     # output blocks + current buffer
    state = {"node": "", "chapter": "", "section_marker": "",
             "section_title": ""}                # hierarchy tracker
    struct_hits = 0                              # leaves actually cut by structure

    for raw in lines:                            # walk once, top to bottom
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
            struct_hits +=1
            pending.append(line)
            continue
        # --- pattern D: PART / CHAPTER style headers -------------------------
        if re.match(r"^(PART|CHAPTER)\s+[IVXLC]+", line, re.I):
            flush_pending(state, pending, blocks)
            state["chapter"] = line                # remember as running context
            state["node"] = ""
            struct_hits += 1
            continue
        # --- pattern E: "PROVIDED THAT", provisos open a sub-block ----------
        if re.match(r"^Provided", line):
            flush_pending(state, pending, blocks)
            state["node"] = (state["node"].split("(")[0] if "(" in state["node"]
                             else state["node"]) + "(proviso)"
            struct_hits += 1
            pending.append(line)
            continue
        # --- anything else: continuation of the current block ---------------
        pending.append(line)
        # observed rare failure: numbered "23." inside body text is NOT caught;
        # acceptable for a demo - the validation gate (Stage 3 post-step) would
        # flag continuity breaks later.

    flush_pending(state, pending, blocks)          # close the final open block

    # --- decide: did structure actually work for this document? ------------
    structured = struct_hits >= 3
    if structured:
        return blocks, {"strategy": "hierarchical", "leaves": len(blocks),
                        "struct_hits": struct_hits}

    # --- fallback path: documents without legal numbering ------------------
    return sentence_fallback(text, doc_label), {"strategy": "sentence_fallback"}


def sentence_fallback(text, doc_label):
    """>= 90%-prose documents without CHAPTER/section/callout numbering get
    sentence-grouped chunks (~cap tokens, ~15% overlap forwards)."""
    norm = normalize(text)                         # flatten line breaks first
    pieces = split_long_text(norm)                 # overlapping token-capped pieces
    return [{"kind": "prose_fallback", "node": "part%02d" % (i + 1),
             "text": p, "chapter": doc_label, "section_marker": "",
             "section_title": ""} for i, p in enumerate(pieces)]


# ---------------------------------------------------------------------------
# Stage 2b - The table path: structured extraction, never flattening
# ---------------------------------------------------------------------------
def summarize_table(rows):
    """Generate the natural-language summary that GETS EMBEDDED for semantic
    search (the plan: summary embeds, JSON generates). Deliberately rule-based
    so the demo has zero API dependency."""
    if not rows:
        return "Empty table."
    header = rows[0]                               # first row = column labels
    cells = [c for row in rows for c in row if c]  # every non-empty cell
    width = len(header)                            # number of columns
    return ("Structured data table with %d columns: %s. "
            "It contains %d data rows and %d populated cells; sample values: %s.") % (
        width,
        ", ".join(header[:6]) or "unlabeled",
        max(0, len(rows) - 1),
        len(cells),
        ", ".join(cells[:5]) or "none")


def table_chunk(rows, page, doc_label, tbl_index, evidence):
    """Stage 2b: build one structured table chunk (JSON payload + summary).
    Returns None when the grid carries no recoverable text at all - callers
    then route the page to the manual-transcription queue instead."""
    rows_c = [[clean_cell(c) for c in row] for row in rows]     # clean every cell
    width = max(len(r) for r in rows_c) if rows_c else 0        # widest row
    rows_c = [r + [""] * (width - len(r)) for r in rows_c]      # pad short rows
    rows_c = [r for r in rows_c if any(c for c in r)]           # drop blank rows
    if not rows_c:                                              # empty grid ->
        return None                                             # reject here
    summary = summarize_table(rows_c)                           # embed this
    return {"kind": "table",
            "node": "p%03d_t%02d" % (page, tbl_index),
            "text": summary,                       # what gets embedded
            "table_json": {"columns": rows_c[0] if rows_c else [],
                           "rows": rows_c[1:]},    # what the LLM would get
            "summary": summary,
            "chapter": doc_label,                  # hierarchy metadata placeholders
            "section_marker": "page %d" % page,
            "section_title": "",
            "extraction_evidence": evidence}       # transparency for QA


def flag_unextractable(page, doc_label, reason):
    """Stage 2b: route a page to manual transcription when pdfplumber cannot
    see a real table but the router said this page should be one."""
    return {"kind": "needs_manual_transcription", "node": "p%03d" % page,
            "text": "[TABLE PAGE - PDF-layer returned no table structure; "
                    "marked for one-time manual transcription]",
            "doc": doc_label, "page": page, "reason": reason}


# ---------------------------------------------------------------------------
# Driver: route every page of a PDF, then chunk accordingly (Stages 1 + 2)
# ---------------------------------------------------------------------------
def text_outside_tables(page, tbls):
    """ pdfplumber helper: re-extract the page's text with every detected
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


def process_pdf(path, max_pages):
    """Run Stage 1 (routing) + Stage 2a/2b (chunking) over one PDF."""
    slug = doc_slug(path)                          # stable id used in all records
    rec = {"doc": slug, "path": path, "routing": [], "prose": [],
           "tables": [], "flags": [], "stats": []}
    counts = {}                                    # label -> how many pages

    with pdfplumber.open(path) as pdf:             # open lazily, page by page
        pages = pdf.pages[: max_pages] if max_pages > 0 else pdf.pages
        for page in pages:                         # ---- Stage 1: per page ----
            label, evidence = classify_page(page)  # route BEFORE any chunking
            counts[label] = counts.get(label, 0) + 1
            rec["routing"].append({"page": page.page_number,
                                    "label": label, "evidence": evidence})

            if label == "empty":                   # cover page / blank page
                continue                           # nothing to chunk at all

            if label == "visual_routing_required":  # Bangla-gazette case:
                # extraction is garbage -> per the plan, this page needs
                # RENDER-then-visual-routing; here we flag it for the manual
                # transcription queue instead of pretending we understood it
                rec["flags"].append(flag_unextractable(
                    page.page_number, slug,
                    "text-layer garbage (top-token share %.2f, one-letter %.2f)"
                    % (evidence["top_token_share"], evidence["one_letter_word_share"])))
                continue

            if label == "table":                   # ---- Stage 2b: tables ----
                found = page.extract_tables()      # detected grids -> cells
                made = 0                                    # tables kept
                for i, rows in enumerate(found or []):
                    chunk = table_chunk(rows, page.page_number, slug, i, evidence)
                    if chunk:                               # structured JSON chunk
                        rec["tables"].append(chunk)
                        made += 1
                if made == 0:                      # routed as table but the
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
                kept = 0                           # tables kept from this page
                for i, t in enumerate(tbls):       # contain their cells
                    rows = t.extract()
                    if rows and len(rows) >= 2:    # skip 1-row false positives
                        ch = table_chunk(rows, page.page_number, slug, i, evidence)
                        if ch:
                            page_tables.append(ch)
                            kept += 1
                rec["tables"].extend(page_tables)
                text = (text_outside_tables(page, tbls) if kept
                        else page.extract_text()) or ""  # prose-only text
            else:
                text = page.extract_text() or ""  # pure prose page
                # KNOWN LIMITATION: two-column layouts (e.g. Legal 500's Q&A
                # grid) interleave in raw extraction order. pdfplumber's
                # layout=True preserves geometry but pads lines with spaces
                # and still needs real column detection (x0 clustering) to
                # fix reading order - that belongs in Stage 3 QA, not here.

            blocks, stats = hierarchy_chunk(text, slug)  # Stage 2a chunking
            for b in blocks:                       # attach doc/page metadata
                rec["prose"].append({**b, "doc": slug,
                                     "path": os.path.basename(path),
                                     "page": page.page_number,
                                     "est_tokens": estimate_tokens(b["text"])})
            rec["stats"].append({"page": page.page_number, "route": label,
                                 "strategy": stats.get("strategy"),
                                 "leaves": len(blocks),
                                 "tables": len(page_tables)})
    rec["route_counts"] = counts                   # routing histogram
    return rec


def process_text_file(path):
    """Plain-text study documents (no PDF layer) go through the same
    Stage 1 contract via the text router, then straight to Stage 2a."""
    slug = doc_slug(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()                           # raw file contents
    label, evidence = classify_text_document(text)  # Stage 1 on plain text
    rec = {"doc": slug, "path": path, "routing": [{"page": None,
            "label": label, "evidence": evidence}], "prose": [],
            "tables": [], "flags": [], "stats": [], "route_counts": {label: 1}}
    if label in ("prose", "mixed"):                # only text types here
        blocks, stats = hierarchy_chunk(text, slug) # Stage 2a (fallback)
        rec["stats"].append({"route": label, "strategy": stats.get("strategy"),
                             "leaves": len(blocks), "tables": 0})
        rec["prose"] = [{**b, "doc": slug, "path": os.path.basename(path),
                         "page": None, "est_tokens": estimate_tokens(b["text"])}
                        for b in blocks]           # attach metadata
    return rec


# ---------------------------------------------------------------------------
# main: run over documents/, print chunk samples to the console, dump JSONs
# ---------------------------------------------------------------------------
def main():
    # Windows terminals default to cp1252; our corpus contains Bangla glyphs,
    # so force stdout to UTF-8 (with replacement) before printing anything
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Nyaya ingestion demo: Stage 1 routing + Stage 2 chunking")
    ap.add_argument("--docs-dir", default=DOCS_DIR)      # where the corpus lives
    ap.add_argument("--out-dir", default=OUT_DIR)        # where JSONs get written
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                    help="pages per PDF (0 = the whole document)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)             # mkdir ./ingestion_output
    docs = sorted(os.path.join(args.docs_dir, f)         # every pdf/txt/md in order
                  for f in os.listdir(args.docs_dir)
                  if f.lower().endswith((".pdf", ".txt", ".md")))

    routing_all, prose_all, tables_all, flags_all = [], [], [], []
    totals = {"table": 0, "prose": 0, "mixed": 0, "empty": 0,
              "visual_routing_required": 0}              # corpus-wide histogram

    for path in docs:                                    # ---- per document ----
        rec = (process_pdf(path, args.max_pages) if path.lower().endswith(".pdf")
               else process_text_file(path))

        # ---- console preview: how the chunks ACTUALLY look ----------------
        print("\n" + "=" * 76)
        print("DOCUMENT: %s" % os.path.basename(path))
        print("route:    %s" % rec["route_counts"])
        print("=" * 76)
        for c in rec["prose"][:3]:               # sample: first 3 prose chunks
            print("\n[PROSE] node=%s  page=%s  ~%d tok  strategy-relevant: kind=%s"
                  % (c["node"], c["page"], c["est_tokens"], c["kind"]))
            print("  %s" % preview(c["text"], 280))
        if rec["tables"]:
            t = rec["tables"][0]                 # sample: first table chunk
            cols = t["table_json"]["columns"]
            print("\n[TABLE] node=%s  page=%s" % (t["node"], t["section_marker"]))
            print("  summary : %s" % preview(t["summary"], 220))
            print("  columns : %s" % (", ".join(cols[:8]) or "(none)"))
            if t["table_json"]["rows"]:
                print("  row[0]  : %s" % ", ".join(t["table_json"]["rows"][0][:8]))
        for f in rec["flags"][:2]:               # sample: flagged pages
            print("\n[FLAGGED] %s  (%s)" % (f["node"], f["reason"]))

        # ---- accumulate the full record set ----
        for r in rec["routing"]:                 # one record per routed page
            routing_all.append({"doc": rec["doc"], **r})
        prose_all.extend(rec["prose"])
        tables_all.extend(rec["tables"])
        flags_all.extend(rec["flags"])
        for k, v in rec["route_counts"].items(): # corpus-wide routing tally
            totals[k] = totals.get(k, 0) + v

    # ---- write the three deliverables -------------------------------------
    def dump(name, payload):                     # tiny helper: pretty UTF-8 json
        out = os.path.join(args.out_dir, name)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        return out

    dump("routing_report.json", routing_all)     # Stage 1 manifest
    dump("prose_chunks.json", prose_all)         # Stage 2a output
    dump("tables_extracted.json", tables_all)    # Stage 2b output
    dump("flags_manual_review.json", flags_all)  # visual-routing/manual queue

    # ---- totals ------------------------------------------------------------
    print("\n" + "=" * 76)
    print("CORPUS TOTALS  (docs=%d)" % len(docs))
    print("  pages routed: %s" % totals)
    print("  prose chunks      : %d" % len(prose_all))
    print("  table chunks      : %d" % len(tables_all))
    print("  flagged for manual: %d" % len(flags_all))
    print("  full details -> %s/{routing_report,prose_chunks,tables_extracted,flags_manual_review}.json"
          % args.out_dir)


if __name__ == "__main__":                       # run: venv/Scripts/python ingestion_stage1_2.py
    main()
