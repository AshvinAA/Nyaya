"""
============================================================================
 Nyaya - Ingestion Pipeline :: run_pipeline.py (CLI driver)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 The end-to-end entry point. It runs the seven stages IN ORDER over the
 corpus in docs/ and writes every artifact to ingestion_output/ (plus the
 Chroma vector index under db/):

   Stage 1+2  document_intake   route every page, chunk prose/tables
   Stage 3    validation_gate  structural validation gate (blocks bad docs)
   Stage 4    parent_builder   parents + child parent-pointers
   Stage 5    metadata_tagger  per-chunk metadata tagging + chunk ids
   Stage 6    deduplicator     exact-hash + near-dup demotion by authority
   Stage 7    indexer          embed with context injection, write Chroma

 Output files (ingestion_output/):
   routing_report.json      Stage 1 manifest - one record per routed page
   prose_chunks.json        Stage 2a chunks, hierarchy metadata, fully tagged
   tables_extracted.json    Stage 2b structured tables with summaries
   flags_manual_review.json quarantined tables / visual-routing pages /
                            blocked documents / stripped furniture
   doc_metadata.json        document-level metadata registry
   validation_report.json   Stage 3 per-document gate report (pass rates)
   parent_store.json        Stage 4 document store (parents keyed by id)
   dedup_report.json        Stage 6 demotion audit trail
   index_manifest.json      Stage 7 index build summary

 Run it (from the project root):
     venv/Scripts/python ingestion/run_pipeline.py            # 30 pages/doc
     venv/Scripts/python ingestion/run_pipeline.py --full     # whole corpus
     venv/Scripts/python ingestion/run_pipeline.py --stages 123   # skip 4-7
     venv/Scripts/python ingestion/run_pipeline.py --no-embed # build chunks,
                                                              # skip vector DB
============================================================================
"""

import argparse
import os
import sys

# allow "python ingestion/run_pipeline.py" from the project root: the script's
# own directory is sys.path[0], so add the PROJECT ROOT for the package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion import config
from ingestion.common import dump_json, preview
from ingestion.document_intake import process_pdf, process_text_file
from ingestion.validation_gate import validate_corpus
from ingestion.parent_builder import build_parents
from ingestion.metadata_tagger import tag_children
from ingestion.deduplicator import dedup
from ingestion.indexer import embed_and_index


def banner(title):
    print("\n" + "=" * 76 + "\n %s\n" % title + "=" * 76)


# ---------------------------------------------------------------------------
# Stages 1 + 2: intake over every document
# ---------------------------------------------------------------------------
def run_intake(docs, max_pages):
    """Run the intake driver over each document; returns (records, flags)."""
    records = []
    for path in docs:
        is_pdf = path.lower().endswith(".pdf")
        rec = process_pdf(path, max_pages) if is_pdf else process_text_file(path)
        records.append(rec)
        # ---- console preview: how the chunks ACTUALLY look -----------------
        print("\n" + "-" * 76)
        print("DOCUMENT: %s" % os.path.basename(path))
        print("route:   %s" % rec["route_counts"])
        for c in rec["prose"][:2]:                 # sample: first 2 prose chunks
            print("[PROSE] node=%-12s page=%-4s ~%dtok kind=%s"
                  % (c["node"], c["page"], c["est_tokens"], c["kind"]))
            print("   %s" % preview(c["text"]))
        if rec["tables"]:
            t = rec["tables"][0]
            tj = t["table_json"]
            print("[TABLE] node=%s page=%s caption=%s"
                  % (t["node"], tj["page"], tj["caption"] or "(none)"))
            print("   %s" % preview(t["summary"]))
        for f in rec["flags"][:2]:
            print("[FLAG ] %s (%s)" % (f["node"], f["reason"]))
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # Windows terminals default to cp1252; our corpus contains Bangla glyphs,
    # so force stdout to UTF-8 (with replacement) before printing anything
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Nyaya ingestion pipeline: Stages 1-7 end to end")
    ap.add_argument("--docs-dir", default=config.DOCS_DIR,
                    help="folder holding the raw corpus (default: docs)")
    ap.add_argument("--out-dir", default=config.OUT_DIR,
                    help="folder for JSON outputs (default: ingestion_output)")
    ap.add_argument("--full", action="store_true",
                    help="ingest whole documents (overrides --max-pages)")
    ap.add_argument("--max-pages", type=int, default=config.DEFAULT_MAX_PAGES,
                    help="pages per PDF, 0 = whole document (default: %d)"
                         % config.DEFAULT_MAX_PAGES)
    ap.add_argument("--stages", default="1234567",
                    help="which stages to run, e.g. '123' or '1234567'")
    ap.add_argument("--strict", action="store_true",
                    help="Stage 3: review items block documents too")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip Stage 7 model load + index write")
    ap.add_argument("--embed-model", default=config.EMBED_MODEL,
                    help="embedding model for Stage 7 (default: %s)"
                         % config.EMBED_MODEL)
    ap.add_argument("--db-dir", default=config.DB_DIR,
                    help="Chroma persistence dir (default: %s)" % config.DB_DIR)
    args = ap.parse_args()
    max_pages = 0 if args.full else args.max_pages
    stages = set(args.stages)                      # e.g. {'1','2','3'}

    docs = sorted(os.path.join(args.docs_dir, f)   # every pdf/txt/md in order
                  for f in os.listdir(args.docs_dir)
                  if f.lower().endswith((".pdf", ".txt", ".md")))
    if not docs:
        print("No documents found in %s" % args.docs_dir)
        return

    # ---- Stages 1+2: intake ------------------------------------------------
    banner("STAGES 1+2 - intake & chunking (%d docs, %s)"
           % (len(docs), "full" if max_pages == 0 else "max %d pages" % max_pages))
    records = run_intake(docs, max_pages)

    # ---- Stage 3: validation gate ------------------------------------------
    banner("STAGE 3 - structural validation gate%s" % (" (strict)" if args.strict else ""))
    reports, n_blocked = validate_corpus(records, strict=args.strict)
    for r in reports:
        status = "BLOCKED" if r["blocked"] else "pass"
        print("  %-55s %s  (pass_rate %.2f, review %d)"
              % (r["doc"][:55], status, r["pass_rate"], len(r["review"])))
    print("  blocked documents: %d / %d" % (n_blocked, len(records)))

    # ---- collect the chunk streams -----------------------------------------
    prose_all = [c for rec in records for c in rec["prose"]]
    tables_all = [c for rec in records for c in rec["tables"]]
    flags_all = [f for rec in records for f in rec["flags"]]
    doc_meta_all = {rec["doc"]: rec.get("doc_metadata") or {}
                    for rec in records if rec.get("doc_metadata")}

    # chunks eligible for the index = passed the gate, retrievable
    ok_docs = {r["doc"] for r in reports if not r["blocked"]}
    index_children = [c for c in prose_all + tables_all
                      if c["doc"] in ok_docs and c.get("retrievable", True)]

    # ---- Stage 4: parents ---------------------------------------------------
    parents = {}
    if "4" in stages:
        banner("STAGE 4 - parent-child indexing")
        prose_kids = [c for c in index_children if c.get("kind") != "table"]
        table_kids = [c for c in index_children if c.get("kind") == "table"]
        index_children, parents, pstats = build_parents(prose_kids, table_kids)
        print("  children=%(children)d parents=%(parents)d "
              "(prose %(prose_parents)d, self %(self_parents)d, "
              "table %(table_parents)d)" % pstats)

    # ---- Stage 5: metadata tagging (ALL chunks, so outputs stay complete) ---
    if "5" in stages:
        banner("STAGE 5 - metadata tagging")
        tag_children(prose_all + tables_all, doc_meta_all)
        n_tagged = sum(1 for c in prose_all + tables_all
                       if c.get("section_path"))
        print("  tagged %d chunks with section_path / topic_tags / authority"
              % n_tagged)

    # ---- Stage 6: dedup (index-bound children only) --------------------------
    dedup_report = None
    if "6" in stages:
        banner("STAGE 6 - deduplication")
        dedup_report, index_children = dedup(index_children)
        print("  scored %d children (%s): %d exact + %d near duplicates demoted"
              % (dedup_report["n_scored"], dedup_report["backend"],
                 dedup_report["n_exact_dups"], dedup_report["n_near_dups"]))

    # ---- Stage 7: embedding + index write -----------------------------------
    index_manifest = None
    if "7" in stages:
        banner("STAGE 7 - embedding & index write")
        if args.no_embed:
            index_manifest = {"skipped": True, "n_indexable": len(index_children),
                              "requested_model": args.embed_model}
            print("  skipped (--no-embed); %d children remain index-ready"
                  % len(index_children))
        else:
            index_manifest = embed_and_index(
                index_children, db_dir=args.db_dir,
                model_name=args.embed_model)
            print("  indexed %(n_indexed)d children -> %(db_dir)s "
                  "(collection %(collection)s, count %(collection_count)d)"
                  % index_manifest)

    # ---- write all artifacts -------------------------------------------------
    dump_json(args.out_dir, "prose_chunks.json", prose_all)
    dump_json(args.out_dir, "tables_extracted.json", tables_all)
    dump_json(args.out_dir, "flags_manual_review.json", flags_all)
    dump_json(args.out_dir, "doc_metadata.json", doc_meta_all)
    dump_json(args.out_dir, "validation_report.json",
              {"documents": reports, "n_docs": len(records),
               "n_blocked": n_blocked, "strict": args.strict})
    dump_json(args.out_dir, "parent_store.json", parents)
    dump_json(args.out_dir, "dedup_report.json", dedup_report or {})
    dump_json(args.out_dir, "index_manifest.json", index_manifest or {})
    routing_all = [{"doc": rec["doc"], **r}                   # Stage 1 manifest
                   for rec in records for r in rec["routing"]]
    dump_json(args.out_dir, "routing_report.json", routing_all)

    # ---- totals ---------------------------------------------------------------
    banner("RUN TOTALS  (docs=%d)" % len(docs))
    print("  prose chunks        : %d  (front matter: %d)"
          % (len(prose_all),
             sum(1 for c in prose_all if c.get("kind") == "front_matter")))
    print("  table chunks        : %d" % len(tables_all))
    print("  flagged for manual  : %d" % len(flags_all))
    print("  validation blocked  : %d / %d docs" % (n_blocked, len(records)))
    print("  index-ready children: %d" % len(index_children))
    print("  outputs             -> %s/ + index in %s"
          % (args.out_dir, args.db_dir if index_manifest and
             not index_manifest.get("skipped") else "(not written)"))


if __name__ == "__main__":
    main()
