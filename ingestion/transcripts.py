"""
============================================================================
 Nyaya - Ingestion Pipeline :: page transcripts (Stage 1 text-layer override)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Some official gazettes ship without a usable text layer: the 2026 Labour
 (Amendment) Act and the Dec-2023 garments wage gazette are set in fonts
 with no Unicode map, so pdfplumber returns "(cid:123)" garbage and Stage 1
 routes every page to visual_routing_required. Those pages were transcribed
 by hand from the rendered page images into

     docs/transcriptions/<doc_slug>/p001.txt, p002.txt, ...

 and this module hands that text to the intake driver in place of the PDF's
 own text layer. The conventions the transcripts follow (and that
 prose_chunker relies on) are documented in docs/transcriptions/README.md:

   - one file per PDF page, UTF-8, one paragraph per line
   - lines starting with '#' are page furniture (masthead, running header,
     page number, printer's imprint) or a transcriber's note; they are
     dropped here and never reach a chunk
   - lines starting with '> ' are text QUOTED by the document - e.g. the new
     wording an amending Act inserts. The chunker treats them as body text of
     the provision that quotes them, never as the document's own sections
   - lines starting with '|' are table rows

 Used by: document_intake.py
============================================================================
"""

import os

TRANSCRIPTS_SUBDIR = "transcriptions"   # lives inside the docs folder


def transcript_dir(pdf_path, slug):
    """docs/transcriptions/<slug>/ for a PDF at docs/<file>.pdf"""
    return os.path.join(os.path.dirname(pdf_path), TRANSCRIPTS_SUBDIR, slug)


def load_page_transcript(pdf_path, slug, page_number):
    """Return the transcribed text of one PDF page (1-based), with furniture
    lines removed, or None when that page has no transcript."""
    path = os.path.join(transcript_dir(pdf_path, slug), "p%03d.txt" % page_number)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    kept = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    return "\n".join(kept)
