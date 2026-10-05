"""
============================================================================
 Nyaya - Ingestion Pipeline :: shared helpers (used by every stage)
============================================================================
 This file holds the tiny, dependency-light utilities that every stage
 module needs: text normalization, token estimation, slug building, the
 chunk-id factory, and the standard JSON dump used by the run report.

 Nothing stage-specific belongs here. If a function is only used by one
 stage, it lives in that stage's file instead - that keeps each stage
 module readable as a self-contained unit.

 Used by: every pipeline module and run_pipeline.py
============================================================================
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def normalize(text):
    """Collapse every whitespace run (newlines, tabs, double spaces) into
    single spaces and trim the ends. Every chunk's text goes through this
    exactly once, so downstream hashes and token counts are stable."""
    return re.sub(r"\s+", " ", text or "").strip()


def estimate_tokens(text):
    """Rough token estimate (~1.3 tokens per word) - deliberately avoids a
    tiktoken dependency; chunk-size thresholds are tuned against THIS
    estimate, so the estimator itself must stay deterministic."""
    return max(1, round(len((text or "").split()) * 1.3))


def preview(text, limit=280):
    """Shortens text for console display so run-report samples stay readable."""
    return text if len(text) <= limit else text[:limit] + " ..."


def doc_slug(path):
    # "docs/BangladeshGagetteSep2015.pdf" -> "bangladeshgagettesep2015"
    base = os.path.splitext(os.path.basename(path).lower())[0]  # name minus .pdf
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")          # non-alnum -> "_"


def clean_cell(value):
    # pdfplumber table cells carry embedded newlines; flatten and trim them
    return normalize(str(value)) if value is not None else ""


# ---------------------------------------------------------------------------
# Chunk id factory
# ---------------------------------------------------------------------------
_id_counters = {}   # prefix -> how many ids issued; resets only on process restart


def next_id(prefix):
    """Deterministic sequential id like 'par0007' / 'blk0042'. Guarantees
    uniqueness within one process run, which is all Stage 6 dedup and the
    Chroma write need (ids never have to be stable across re-runs - each
    full rebuild rewrites the index from scratch)."""
    _id_counters[prefix] = _id_counters.get(prefix, 0) + 1
    return "%s%04d" % (prefix, _id_counters[prefix])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def dump_json(out_dir, name, payload):
    """Pretty-print `payload` to out_dir/name as UTF-8 JSON (Bangla text in
    mastheads stays readable). Returns the written path for the run report."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    return out
