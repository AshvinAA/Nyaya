"""
============================================================================
 Nyaya - Retrieval Pipeline :: run_pipeline.py (CLI driver)
============================================================================
 End-to-end entry point for retrieval (per RETRIEVAL_PIPELINE.md):

   Stage 1   expand          multi-query expansion -> structured envelope
   Stage 2   search          filtered dense search, per variant
   Stage 3   rrf_fuse        reciprocal rank fusion across variants
   Stage 4   retrieve_dedup  per-query near-dup collapse by authority
   Stage 5   rerank          cross-encoder re-score vs the ORIGINAL query
   Stage 5b  mmr             only when Stage 1 flagged multi_aspect
   Stage 6   resolve_parents parent/table resolution under a token budget
   Stage 7   resolve_overlay surface overlay + base with explicit labels

 Everything is written to a per-query JSON trace in retrieval_traces/ -
 not optional instrumentation but the substrate the evaluation harness
 consumes (RAGAS context metrics from the final context set; stage-level
 ablations from the per-stage lists) and the only way to debug a wrong
 answer down to the stage that caused it.

 Fallback policy: every stage has a defined fallback and every fallback
 is logged, never silent. Convenience features may degrade; legal-
 applicability filters may not.

 Run it (from the project root):
     python retrieval/run_pipeline.py --query "who is hiring in Gazipur"
     python retrieval/run_pipeline.py --interactive
     python retrieval/run_pipeline.py --query "..." --json   # context only
============================================================================
"""

import argparse
import os
import sys
import time

# project-root imports whether run as a script or as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import config
from retrieval.common import dump_json, eprint, est_tokens, load_json, cosine_matrix
from retrieval.expand import expand_query
from retrieval.search import (search_variant_text, load_docstore, rrf_fuse)


def banner(title):
    print("\n" + "=" * 76 + "\n %s\n" % title + "=" * 76)


class Tracker:
    """Times every stage and accumulates fallbacks for the trace."""
    def __init__(self):
        self.events = []
        self._t = None

    def start(self):
        self._t = time.time()

    def stop(self, stage, **extra):
        el = round(time.time() - self._t, 3) if self._t else None
        self._t = None
        ev = {"stage": stage, "elapsed_s": el, **extra}
        self.events.append(ev)
        return ev


def _flat_index_docs(chroma_docs, ids):
    """The documents array from Chroma's col.get() -> {chunk_id: full text}.
    These are the bare generation texts ingestion Stage 7 stored (the embed
    prefix lives only in the vectors, never here)."""
    return dict(zip(ids, chroma_docs))


def run_query(query, verbose=True, json_out=False, gate=None):
    """One text query -> the final labeled context set + full trace dict."""
    trace = {"original": query, "stages": {}, "fallbacks": [],
             "timings_s": {}, "query_id": "q_%s" % format(time.time(), ".6f")}

    # ---- load stores (the ingestion Stage 7 output is the contract) -------
    docstore, chroma_docs = load_docstore(config.DB_DIR, config.COLLECTION)
    ids = list(docstore)
    index_docs = _flat_index_docs(chroma_docs, ids)    # chunk_id -> text
    parent_store = load_parents(config.PARENT_STORE)
    if verbose:
        banner("RETRIEVAL PIPELINE  (index: %d children, %d parents)"
               % (len(docstore), len(parent_store)))

    # ---- embeddings --------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")

    # ---- Stage 1: multi-query expansion ------------------------------------
    tracker = Tracker()
    tracker.start()
    envelope, s1diag = expand_query(query, verbose=verbose)
    trace["stages"]["1_expand"] = s1diag
    if s1diag.get("fallback"):
        trace["fallbacks"].append({"stage": "1_expand",
                                   "reason": s1diag.get("reason")})
    variants = [query] + envelope["variants"]        # original ALWAYS searches
    trace["variants"] = variants
    trace["flags"] = {"multi_aspect": envelope["multi_aspect"],
                      "historical": envelope["historical"]}

    # ---- Stage 2: filtered dense search, per variant -----------------------
    s2lists, s2diag = [], []
    tracker.start()
    for v in variants:
        res = search_variant_text(v, model, index_docs, docstore, config,
                                  historical=envelope["historical"])
        s2lists.append(res)
        s2diag.append({"query": v, "n_results": len(res["ranked"]),
                       "filter_mode": res["filter_mode"],
                       "error": res["error"]})
    trace["stages"]["2_search"] = {"lists": s2diag}
    if any(r["error"] for r in s2lists):
        trace["fallbacks"].append({"stage": "2_search",
                                   "reason": "one or more variants failed to embed"})

    # ---- Stage 3: RRF fusion ----------------------------------------------
    fused, s3diag = rrf_fuse(s2lists)
    trace["stages"]["3_rrf"] = s3diag

    if not fused:
        trace["final_context"] = []
        trace["error"] = "no results after fusion"
        if json_out:
            import json as _json
            print(_json.dumps(trace, ensure_ascii=False, indent=1))
        return trace

    # ---- Stage 4: retrieval-time dedup -------------------------------------
    from retrieval.dedup_rerank import retrieve_dedup
    fused_ext = [{**f,
                  "preview": index_docs[f["chunk_id"]][:512]}
                 for f in fused]
    fused_ext = [{**f,
                  **docstore[f["chunk_id"]]} for f in fused_ext]
    survivors, s4diag = retrieve_dedup(fused_ext, docstore)
    trace["stages"]["4_dedup"] = s4diag

    # ---- Stage 5: rerank against the ORIGINAL query ------------------------
    from retrieval.dedup_rerank import rerank
    reranked, s5diag = rerank(query, survivors, index_docs)
    trace["stages"]["5_rerank"] = s5diag
    if s5diag.get("fallback"):
        trace["fallbacks"].append({"stage": "5_rerank",
                                   "reason": "reranker unavailable - RRF order stands"})
    ordered = reranked if reranked is not None else \
        sorted(survivors, key=lambda c: (-c["rrf_score"], c["chunk_id"]))

    # ---- Stage 5b: MMR, conditional (multi_aspect only) ---------------------
    if envelope["multi_aspect"] and ordered:
        from retrieval.dedup_rerank import mmr
        try:
            qv = model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
            cvs = model.encode([index_docs[c["chunk_id"]] for c in ordered],
                               normalize_embeddings=True,
                               convert_to_numpy=True,
                               show_progress_bar=False)
            order = mmr(qv, cvs, k=len(ordered))
            ordered = [ordered[i] for i in order]
            trace["stages"]["5b_mmr"] = {"applied": True, "lambda": config.MMR_LAMBDA}
        except Exception as exc:
            trace["stages"]["5b_mmr"] = {"applied": False, "error": str(exc)}
    else:
        trace["stages"]["5b_mmr"] = {"applied": False}

    # ---- final-k + Stage 6: parent/table resolution -------------------------
    topk = ordered[:config.FINAL_K]
    from retrieval.resolve import resolve_parents, resolve_overlay
    resolved, s6diag = resolve_parents(topk, parent_store)
    trace["stages"]["6_parents"] = s6diag

    # ---- Stage 7: overlay vs. base labels -----------------------------------
    final, s7diag = resolve_overlay(resolved)
    trace["stages"]["7_overlay"] = s7diag

    # ---- trace summary of the final set -------------------------------------
    trace["final_context"] = [{
        "chunk_id": c.get("chunk_id"),
        "parent_id": c.get("parent_id"),
        "rrf_score": c.get("rrf_score"),
        "rerank_score": c.get("rerank_score"),
        "layer_label": c.get("layer_label"),
        "label": c.get("label"),
        "truncated": c.get("truncated", False),
        "text_preview": (c.get("parent_text") or (c.get("table_json") is not None and "(structured table JSON)" or "")).strip()[:280]
    } for c in final]

    # ---- per-stage ranked lists for the ablation harness --------------------
    trace["ablation_lists"] = {
        "2_search": {"variant_%d" % i: [r["chunk_id"] for r in l["ranked"]]
                     for i, l in enumerate(s2lists)},
        "3_rrf": [f["chunk_id"] for f in fused],
        "4_dedup": [c["chunk_id"] for c in survivors],
        "5_rerank": [c["chunk_id"] for c in ordered],
        "final_k": [c["chunk_id"] for c in final],
    }

    # ---- write the trace -----------------------------------------------------
    os.makedirs(config.TRACE_DIR, exist_ok=True)
    path = os.path.join(config.TRACE_DIR, "%s.json" % trace["query_id"])
    dump_json(trace, path)
    if verbose:
        banner("TRACE -> %s" % path)
        print("  variants:  %s" % variants)
        print("  flags:     %s" % trace["flags"])
        print("  fused:     %d  ->  deduped: %d  ->  reranked: %s  ->  final: %d"
              % (s3diag["fused_pool"], s4diag["out"],
                 s5diag.get("n_scored") or "(rrf fallback)", len(final)))
        for c in trace["final_context"]:
            tags = []
            if c.get("layer_label"):
                tags.append(c["layer_label"])
            if c.get("truncated"):
                tags.append("[truncated]")
            print("  [%-8s] %s %s" % (c["chunk_id"],
                                      " ".join(tags),
                                      (c.get("label") or "")[:110]))

    if json_out:
        import json as _json
        print("\nFINAL CONTEXT (for generation):\n")
        print(_json.dumps(final, ensure_ascii=False, indent=1))

    return trace


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description="Nyaya retrieval pipeline: Stages 1-7, text in / labeled context out")
    ap.add_argument("--query", help="a single query to run")
    ap.add_argument("--interactive", action="store_true",
                    help="loop reading queries from stdin, one per line")
    ap.add_argument("--json", action="store_true",
                    help="print the final context set as JSON after the run")
    args = ap.parse_args()

    if not (args.query or args.interactive):
        ap.error("give --query or --interactive")

    if args.query:
        run_query(args.query, json_out=args.json)

    if args.interactive:
        print("Interactive mode. Type a query, blank line to quit.\n")
        while True:
            try:
                q = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q)


if __name__ == "__main__":
    main()
