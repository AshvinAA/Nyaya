"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 3 - Structural validation gate
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 A structure-aware chunker that silently mis-parses is WORSE than a naive
 fixed-size splitter: the naive splitter at least announces its crudeness,
 while a broken hierarchy parser produces chunks that LOOK authoritative
 and are trusted all the way into the index. So every structured parse
 must pass this validation gate before its chunks proceed:

   1. Section-number continuity  - the section numbers extracted from the
      Act must form a plausible sequence. A missing Section 24 is a parse
      failure, not missing law. (Expected counts/ranges are meant to be
      recorded once from each gazette's own table of contents during
      Stage 0; until that registry is filled, the gate checks continuity
      of the OBSERVED range - gaps and backward jumps are review items.)
   2. Hierarchy integrity        - every clause's parent section must
      exist. A leaf like "23(3)" with no governing section marker is
      parentless (typically a page that starts mid-section). A few of
      these are page-boundary artifacts (review); a document that is
      mostly parentless has a broken parse (hard failure).
   3. Header/footer pollution    - page furniture (running headers, page
      numbers, gazette stamps) must not survive into chunk text. Short
      texts repeated across many pages of one document are furniture:
      they are stripped from the index (retrievable=False), not deleted.
   4. Non-empty node IDs         - every surviving chunk must carry a
      non-empty `node` ID. Stage 4's parent pointers and Stage 6's dedup
      key hang off it, so a blank ID is a validation failure, not a
      silent gap. (Blanks found at the end of extraction are filled
      deterministically and counted in the run report.)
   5. Front-matter invariants    - a document carries at most ONE
      front-matter block, and it is never retrievable.
   (+ Table validation is implemented INLINE at extraction time in
      table_extractor.py, per the plan; this gate aggregates its counts.)

 Output: a per-document validation report. Documents that FAIL are
 blocked from the index and queued for human review - the gate fails
 loud, never silently. The gate re-runs on EVERY ingestion (it is part
 of the pipeline run, not a one-time step), and per-document pass-rates
 are reported alongside retrieval scores as a corpus-quality signal.

 Used by: document_intake.py (ensure_node_ids), run_pipeline.py (the gate).
============================================================================
"""

import re
from collections import Counter

from ingestion.config import (VALIDATION_STRICT, PARENTLESS_FAIL_RATE,
                              POLLUTION_MAX_TOKENS, POLLUTION_MIN_PAGES)
from ingestion.common import normalize

# page-number furniture: a bare number, optionally "page 12" / "p. 12"
PAGE_NO_RE = re.compile(r"^(page|p\.)?\s*\d{1,4}$", re.I)


# ---------------------------------------------------------------------------
# Node-ID assert (also used at the end of intake)
# ---------------------------------------------------------------------------
def ensure_node_ids(chunks):
    """Every surviving chunk needs a non-empty node ID - Stage 4 parent
    pointers and Stage 6 dedup hashes key off it. Fills any blank with a
    deterministic page-local counter ID and returns how many were filled
    (the driver reports the total; it must be counted, never assumed)."""
    filled = 0
    for i, c in enumerate(chunks):
        if not c.get("node"):
            c["node"] = "blk%03d" % (i + 1)
            filled += 1
    return filled


# ---------------------------------------------------------------------------
# Check 1: section-number continuity
# ---------------------------------------------------------------------------
def check_continuity(prose):
    """Collect numeric section markers in encounter order; flag gaps inside
    the observed range and backward jumps. These are REVIEW items (a
    --max-pages cap makes truncated tails routine); they become failures
    only under --strict. Returns (review_items, n_sections_seen)."""
    seen = []                                      # encounter order
    for c in prose:
        m = c.get("section_marker") or ""
        if re.fullmatch(r"\d{1,3}", m):
            n = int(m)
            if not seen or seen[-1] != n:
                seen.append(n)
    review = []
    if len(seen) >= 2:
        highest = seen[0]
        for n in seen[1:]:
            if n < highest:                        # numbering went backward
                review.append("backward section jump: %d after %d" % (n, highest))
            else:
                highest = n
        uniq = sorted(set(seen))
        for a, b in zip(uniq, uniq[1:]):           # gaps inside observed range
            if b - a > 1:
                review.append("section gap: %d -> %d (missing %s)"
                              % (a, b, ", ".join(str(x) for x in range(a + 1, b))))
    return review, len(seen)


# ---------------------------------------------------------------------------
# Check 2: hierarchy integrity (parentless leaves)
# ---------------------------------------------------------------------------
def check_hierarchy(prose):
    """A clause leaf like '23(3)' MUST have a governing section marker - its
    Stage 4 parent pointer and its citation both depend on it. Returns
    (review_items, n_leaves, n_parentless)."""
    leaves = [c for c in prose if c.get("kind") == "leaf"]
    parentless = [c for c in leaves
                  if "(" in (c.get("node") or "") and not c.get("section_marker")]
    review = []
    if parentless:
        sample = ", ".join("%s p%s" % (c["node"], c.get("page")) for c in parentless[:3])
        review.append("%d parentless clause leaf(s) (e.g. %s)"
                      % (len(parentless), sample))
    return review, len(leaves), len(parentless)


# ---------------------------------------------------------------------------
# Check 3: header/footer pollution
# ---------------------------------------------------------------------------
def check_pollution(prose, n_pages):
    """Short texts repeated across many pages are page furniture - running
    headers, page numbers, gazette stamps. They are STRIPPED from the index
    (retrievable=False), never deleted, and reported. Returns
    (stripped_chunks, review_items)."""
    candidates = [c for c in prose
                  if c.get("retrievable", True)
                  and c.get("est_tokens", 99) <= POLLUTION_MAX_TOKENS]
    pages_per_text = Counter()                     # normalized text -> page set
    text_pages = {}
    for c in candidates:
        t = normalize(c.get("text", "")).lower()
        if not t:
            continue
        text_pages.setdefault(t, set()).add(c.get("page"))
    threshold = max(POLLUTION_MIN_PAGES, int(0.4 * n_pages)) if n_pages else \
        POLLUTION_MIN_PAGES
    furniture = {t for t, pgs in text_pages.items() if len(pgs) >= threshold}
    stripped = []
    for c in candidates:
        t = normalize(c.get("text", "")).lower()
        if t in furniture or PAGE_NO_RE.match(t):
            c["retrievable"] = False               # out of the index...
            c["pollution"] = "page_furniture"      # ...reason kept on the chunk
            stripped.append(c)
    review = []
    if stripped:
        review.append("%d furniture chunk(s) stripped (e.g. %s)"
                      % (len(stripped),
                         normalize(stripped[0]["text"])[:40]))
    return stripped, review


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------
def validate_document(rec, strict=VALIDATION_STRICT):
    """Run every check over one document's record. MUTATES rec (strips
    furniture, appends flags) and returns the per-document report:
      failures  - hard parse failures; the document is blocked from the index
      review    - suspicious-but-tolerable items; blocking only under --strict
      pass_rate - 1 - (failed checks / total checks), an evaluation artifact
    """
    doc = rec["doc"]
    prose, tables = rec["prose"], rec["tables"]
    n_pages = len([r for r in rec["routing"] if r["label"] in ("prose", "mixed")])
    failures, review = [], []

    # 4. non-empty node IDs (blanks were already filled at intake; anything
    #    left is a hard failure)
    blank = sum(1 for c in prose + tables if not c.get("node"))
    if blank:
        failures.append("%d chunk(s) with blank node ID after fill" % blank)
    rec["node_ids_filled"] = rec.get("node_ids_filled", 0)

    # 5. front-matter invariants
    fms = [c for c in prose if c.get("kind") == "front_matter"]
    if any(c.get("retrievable", True) for c in fms):
        failures.append("front-matter chunk marked retrievable")
    if len(fms) > 1:
        failures.append("%d front-matter blocks (max 1 per document)" % len(fms))

    # empty text is never a valid chunk
    empty = [c for c in prose + tables if not (c.get("text") or "").strip()]
    if empty:
        failures.append("%d empty-text chunk(s)" % len(empty))

    # 1. continuity + 2. hierarchy + 3. pollution
    cont_items, n_sections = check_continuity(prose)
    review += ["continuity: " + it for it in cont_items]
    hier_items, n_leaves, n_parentless = check_hierarchy(prose)
    if n_leaves and n_parentless / n_leaves > PARENTLESS_FAIL_RATE:
        failures.append("hierarchy broken: %d/%d leaves parentless (> %.0f%%)"
                        % (n_parentless, n_leaves, PARENTLESS_FAIL_RATE * 100))
    else:
        review += ["hierarchy: " + it for it in hier_items]
    stripped, poll_items = check_pollution(prose, n_pages)
    review += ["pollution: " + it for it in poll_items]
    for c in stripped:                             # record the strip loudly
        rec["flags"].append({"kind": "furniture_stripped", "node": c["node"],
                             "doc": doc, "page": c.get("page"),
                             "text": c.get("text", "")[:80],
                             "reason": "page furniture (Stage 3 pollution check)"})

    # table rejections were quarantined inline at extraction; aggregate them
    table_rejections = [f for f in rec["flags"]
                        if f.get("kind") == "table_validation_failed"]

    checks_total = 5 + len(cont_items) + len(hier_items) + len(poll_items)
    checks_failed = len(failures) + len(review)
    blocked = bool(failures) or (strict and bool(review))
    if blocked:
        rec["blocked"] = True                      # downstream stages skip it
        rec["flags"].append({"kind": "document_blocked", "node": "DOC",
                             "doc": doc, "page": None,
                             "reason": "; ".join(failures or review)})

    return {"doc": doc, "ok": not blocked, "blocked": blocked,
            "failures": failures, "review": review,
            "sections_seen": n_sections, "leaves": n_leaves,
            "parentless_leaves": n_parentless,
            "furniture_stripped": len(stripped),
            "table_rejections": len(table_rejections),
            "checks_total": checks_total, "checks_failed": checks_failed,
            "pass_rate": round(1 - checks_failed / max(1, checks_total), 3)}


def validate_corpus(records, strict=VALIDATION_STRICT):
    """Gate over the whole corpus. Returns (reports, n_blocked).
    `reports` is the per-document list that becomes validation_report.json;
    blocked documents are queued for review in the report itself."""
    reports = [validate_document(rec, strict=strict) for rec in records]
    n_blocked = sum(1 for r in reports if r["blocked"])
    return reports, n_blocked
