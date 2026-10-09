"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 2 - Filtered dense search (+ Stage 3)
============================================================================
 Each query variant runs against the ingestion Stage 7 index. The index is
 dense-only for now (Chroma, cosine, collection `nyaya_children`) - the
 plan's dense+sparse hybrid (BGE-M3's learned lexical weights in Qdrant
 named vectors, NOT BM25) is the retrieval-stage upgrade path; when it
 lands, Stage 3 fuses 4-6 ranked lists instead of 2-3 and nothing here
 changes (every consumer takes ranked lists, not search modes).

 Metadata pre-filtering happens HERE, not as a post-hoc check:
   default filter  : in_force = true AND legal_status != superseded
   historical swap : when Stage 1 flagged the query, the constraints are
                     dropped ENTIRELY (a superseded provision is by
                     definition not in force - partial relaxation would be
                     incoherent) and Stage 7's labeling takes over.
 The 2025 Ordinance survives the default filter by design: it is
 in_force=true with legal_status=pending_ratification - the filter's job
 is applicability, not ratification status.

 Embedding: queries must be embedded by the SAME model that built the
 index (BGE-M3 via sentence-transformers), with embed-time context
 injection semantics in mind: index texts carry the hierarchical path
 prefix, so a bare query is embedded bare - matching already worked at
 index build time.

 Stage 3 (RRF) lives in this module too: with 2-3 variants x modes we get
 2-3 ranked lists (4-6 once sparse lands). Merge by rank position, not raw
 score (dense/sparse scores aren't on one scale):
     score(chunk) = sum over lists of  1 / (k + rank_in_list)
 with k = 60. Consensus beats luck: chunks ranking well across several
 variants surface on top, and same-ID fan-out merges for free (that sum is
 a relevance signal, not double-counting - double-counting is Stage 4's
 problem, and only for DISTINCT chunks carrying the same fact).
============================================================================
"""

import re
import time

from retrieval.common import eprint, cosine_matrix

# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def _variant_query(queries_by_variant, verbose=False):
    return queries_by_variant


def search_variant_text(query, model, index_texts_by_id, docstore, cfg,
                        historical=False):
    """One variant -> one ranked list of {chunk_id, score}.

    Brute-force dense cosine against the persisted index texts: the corpus
    is small (hundreds of children), this avoids depending on whichever
    Chroma version is installed for filtered search, and it keeps the
    pre-filter semantics fully explicit and auditable.

      docstore: {chunk_id: flat_metadata_dict} as written by ingestion
      Stage 7 (in_force bool, legal_status str, ...) - loaded via
      load_docstore() from the Chroma collection itself, so this module
      has no ingestion import.

    Fallback behavior: a variant whose embedding fails is recorded and
    skipped - the other variants still carry the query.
    """
    started = time.time()
    out = {"query": query, "ranked": [], "error": None,
           "elapsed_s": None, "n_candidates": 0, "filter_mode":
           "historical" if historical else "default"}

    try:
        qv = model.encode([query], normalize_embeddings=True,
                          convert_to_numpy=True)[0]
    except Exception as exc:
        out["error"] = "embed_failed: %s" % exc
        eprint("[stage2] %s: %s" % (out["error"], query[:60]))
        out["elapsed_s"] = round(time.time() - started, 2)
        return out

    # ---- apply the applicability filter BEFORE scoring ---------------------
    cands = {cid: md for cid, md in docstore.items()
             if _filter_allows(md, historical)}
    if not cands:
        out["error"] = "no_candidates_after_filter"
        out["elapsed_s"] = round(time.time() - started, 2)
        return out

    ids = list(cands)
    texts = [index_texts_by_id[cid] for cid in ids]
    D = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False, batch_size=64)
    sims = cosine_matrix(qv.reshape(1, -1).astype("float32"),
                         D.astype("float32"))[0]
    order = sorted(range(len(ids)), key=lambda i: -float(sims[i]))
    depth = cfg.N_RESULTS_VARIANT
    out["ranked"] = [{"chunk_id": ids[i], "score": float(sims[i]),
                      "rank": r + 1}
                     for r, i in enumerate(order[:depth])]
    out["n_candidates"] = len(cands)
    out["elapsed_s"] = round(time.time() - started, 2)
    return out


def _filter_allows(md, historical):
    """The entire applicability control. Default-strict; historical mode
    swaps the constraints out entirely (never partially relaxed)."""
    if historical:
        return True
    if not bool(md.get("in_force")):
        return False
    if (md.get("legal_status") or "") == "superseded":
        return False
    return True


def load_docstore(db_dir, collection):
    """Read the persisted Chroma collection into a plain dict
    {chunk_id: flat_metadata}. Reading the SAME store ingestion Stage 7
    wrote - no ingestion import needed, the store is the contract."""
    import chromadb
    client = chromadb.PersistentClient(path=db_dir)
    try:
        col = client.get_collection(collection)
    except Exception as exc:
        raise RuntimeError(
            "Chroma collection '%s' not found under %s (%s).\n"
            "Run the ingestion pipeline first:  "
            "venv/Scripts/python ingestion/run_pipeline.py --full"
            % (collection, db_dir, exc))
    got = col.get(include=["metadatas", "documents"])
    docstore = {}
    for cid, md, doc in zip(got["ids"], got["metadatas"], got["documents"]):
        docstore[cid] = md
    return docstore, got["documents"]


# ---------------------------------------------------------------------------
# Stage 3 - Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists, k=None, pool_cap=None):
    """Merge ranked lists by rank position (never raw score - dense and
    sparse scales aren't comparable). Same-ID fan-out merges for free;
    the fused pool is capped before Stage 4 keeps the cross-encoder cheap.
    Returns (fused_list_of_dicts, diagnostics)."""
    from retrieval import config
    k = k or config.RRF_K
    pool_cap = pool_cap or config.RRF_POOL_CAP

    scores, meta = {}, {}
    lists_used = 0
    for lst in ranked_lists:
        if not lst or not lst.get("ranked"):
            continue
        lists_used += 1
        for item in lst["ranked"]:               # rank 1..N, already 1-based
            cid = item["chunk_id"]
            contrib = 1.0 / (k + item["rank"])
            scores[cid] = scores.get(cid, 0.0) + contrib
            if cid not in meta:
                meta[cid] = {"lists": 0, "best_rank": item["rank"]}
            meta[cid]["lists"] += 1
            meta[cid]["best_rank"] = min(meta[cid]["best_rank"], item["rank"])

    fused = [{"chunk_id": cid, "rrf_score": round(s, 6),
              "n_lists": meta[cid]["lists"],
              "best_rank": meta[cid]["best_rank"]}
             for cid, s in sorted(scores.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    merged_lists = [f for f in fused if f["n_lists"] > 1]
    fused = fused[:pool_cap]
    diag = {"stage": "3_rrf", "lists_in": len(ranked_lists),
            "lists_used": lists_used, "fused_pool": len(fused),
            "pool_cap": pool_cap, "multi_list_chunks": len(merged_lists)}
    return fused, diag
