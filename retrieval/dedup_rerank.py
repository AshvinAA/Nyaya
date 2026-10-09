"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 4 dedup · Stage 5 rerank · Stage 5b MMR
============================================================================
 STAGE 4 - Retrieval-time dedup. NOT the same as ingestion-time dedup:
 ingestion's is a static one-time corpus cleanup; this is a dynamic,
 per-query problem. Multi-query fan-out can surface two DIFFERENT chunk
 IDs carrying the same fact from different angles, and with a final-k of
 ~5 a duplicate burns a slot that should carry new evidence. Resolved by
 the same authority ranking as ingestion, applied per-query. Two hard
 rules inherited from ingestion:
   1. NEVER collapse a pair across a different text_layer or source_act -
      a base provision and its 2025-Ordinance overlay are near-identical
      by construction, and Stage 7 exists precisely to surface both.
   2. Runs BEFORE reranking on purpose - the cross-encoder is the most
      expensive call in the pipeline; never spend it on a duplicate.

 STAGE 5 - Rerank the deduplicated, fused pool with a multilingual
 cross-encoder against the ORIGINAL query (the variants exist to widen
 the net in search; the reranker judges relevance to what was actually
 asked). If the reranker cannot load, the RRF order stands and the
 failure is recorded (`reranker: fallback_rrf`) - retrieval never
 hard-fails because an optional re-scorer is down.

 STAGE 5b - MMR, conditional, NOT default. Only runs when Stage 1 flagged
 the query multi_aspect: on a single-aspect legal question MMR can reduce
 precision by demoting the most relevant chunk for diversity nobody
 needed. Its conditional trigger is itself an eval item for the harness.
============================================================================
"""

import time

import numpy as np

from retrieval.common import eprint

# ---------------------------------------------------------------------------
# Stage 4 - retrieval-time near-dup collapse
# ---------------------------------------------------------------------------

def _rank_of(c):
    from retrieval import config
    return config.AUTHORITY_ORDER.get(c.get("authority_rank"), 9)


def _pair_allowed(a, b):
    """Version-history protection, inherited from ingestion. Near-dup must
    never merge different law layers or different instruments."""
    return (a.get("text_layer") or "") == (b.get("text_layer") or "") \
        and (a.get("source_act") or "") == (b.get("source_act") or "")


def retrieve_dedup(fused, docstore, model=None, threshold=None):
    """Collapse distinct-ID near-duplicates, resolve by authority_rank.
    Named retrieve_dedup (not `dedup`) to be unmistakable next to the
    ingestion-time pass of the same name. Returns (survivors, diag).

    `model`: an already-loaded sentence-transformers encoder; if None or
    unloadable, the stage falls back to plain authority-based collapse of
    EXACT text matches only, and says so in the trace."""
    from retrieval import config
    threshold = threshold or config.DEDUP_SIM_THRESHOLD

    diag = {"stage": "4_dedup", "in": len(fused), "collapsed": [],
            "backend": None, "skipped": 0}

    survivors = list(fused)
    if len(survivors) <= 1:
        diag["backend"] = "noop"
        return survivors, diag

    # pair candidates: distinct IDs, both resolvable in the docstore
    pairs = []
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = docstore.get(survivors[i]["chunk_id"]), docstore.get(survivors[j]["chunk_id"])
            if a is None or b is None:
                continue
            if not _pair_allowed(a, b):
                continue
            pairs.append((i, j, a, b))
    diag["candidate_pairs"] = len(pairs)

    backend = None
    sims = None
    if pairs and model is not None:
        try:
            texts_a = [a.get("preview") or "" for (_, _, a, _) in pairs]
            texts_b = [b.get("preview") or "" for (_, _, _, b) in pairs]
            E_a = model.encode(texts_a, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False,
                               batch_size=64)
            E_b = model.encode(texts_b, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False,
                               batch_size=64)
            sims = [float((E_a[i] * E_b[i]).sum()) for i in range(len(pairs))]
            backend = "embedding"
        except Exception as exc:
            eprint("[stage4] embedding scorer failed (%s) - exact-text collapse "
                    "only" % exc)
            backend = "fallback_exact_only"
    if not pairs:
        backend = "noop"

    survivors = list(fused)
    alive = [True] * len(survivors)
    for pair_idx, (i, j, a, b) in enumerate(pairs):
        sim = sims[pair_idx] if sims is not None else None
        if not alive[i] or not alive[j]:
            continue
        if sim is not None and sim < threshold:
            continue
        winner_id, loser_id = survivors[i], survivors[j]
        ra, rb = _rank_of(a), _rank_of(b)
        if rb < ra:                                  # b is more authoritative
            winner_id, loser_id, a, b, ra, rb = (loser_id, winner_id, b, a,
                                                 rb, ra)
        elif ra == rb:
            # same authority: keep the better RRF score (list regression = worse)
            if (loser_id["n_lists"], -loser_id["best_rank"]) > \
               (winner_id["n_lists"], -winner_id["best_rank"]):
                winner_id, loser_id = loser_id, winner_id
        alive[survivors.index(loser_id)] = False
        diag["collapsed"].append({
            "kept": winner_id["chunk_id"], "dropped": loser_id["chunk_id"],
            "similarity": None if sim is None else round(sim, 3),
            "text_layers": sorted({a.get("text_layer"), b.get("text_layer")}),
            "source_acts": sorted({a.get("source_act"), b.get("source_act")})})

    out = [s for s, ok in zip(survivors, alive) if ok]
    diag["out"] = len(out)
    diag["backend"] = backend or "noop"
    return out, diag


# ---------------------------------------------------------------------------
# Stage 5 - cross-encoder reranking
# ---------------------------------------------------------------------------

def _load_reranker():
    from retrieval import config
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANKER_MODEL, max_length=512)
    except Exception as exc:
        eprint("[stage5] reranker unavailable (%s) - RRF order stands" % exc)
        return None


def rerank(original_query, candidates, index_docs_by_id, reranker=None):
    """Re-score candidates with a cross-encoder against the ORIGINAL query.
    Returns (reranked_list-or-None, diag). None means degraded to RRF order."""
    from retrieval import config
    started = time.time()
    diag = {"stage": "5_rerank", "model": config.RERANKER_MODEL,
            "n_scored": 0, "fallback": False, "elapsed_s": None}

    if reranker is None:
        reranker = _load_reranker()

    if reranker is None:
        diag["fallback"] = True
        diag["elapsed_s"] = round(time.time() - started, 2)
        return None, diag

    pairs = [(original_query, index_docs_by_id[c["chunk_id"]])
             for c in candidates if c["chunk_id"] in index_docs_by_id]
    if not pairs:
        diag["fallback"] = True
        diag["elapsed_s"] = round(time.time() - started, 2)
        return None, diag
    try:
        scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        eprint("[stage5] reranker predict failed (%s) - RRF order stands" % exc)
        diag["fallback"] = True
        diag["elapsed_s"] = round(time.time() - started, 2)
        return None, diag

    scored = []
    for c, s in zip(candidates, scores):
        d = dict(c)
        d["rerank_score"] = float(s)
        scored.append(d)
    scored.sort(key=lambda d: -d["rerank_score"])
    diag["n_scored"] = len(scored)
    diag["elapsed_s"] = round(time.time() - started, 2)
    return scored, diag


# ---------------------------------------------------------------------------
# Stage 5b - conditional MMR
# ---------------------------------------------------------------------------

def mmr(query_vec, cand_vecs, k, lam=None):
    """Maximal Marginal Relevance over the candidate vectors already
    computed at Stage 2. Returns the ORDER of kept indices (0-based)."""
    from retrieval import config
    lam = lam if lam is not None else config.MMR_LAMBDA

    Q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    V = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    sim_q = V @ Q
    sim_v = V @ V.T

    chosen = [int(np.argmax(sim_q))]
    while len(chosen) < min(k, len(sim_q)):
        worst = -1.0
        pick = None
        for i in range(len(sim_q)):
            if i in chosen:
                continue
            div = max(sim_v[i][j] for j in chosen)
            score = lam * sim_q[i] - (1.0 - lam) * div
            if score > worst:
                worst, pick = score, i
        chosen.append(pick)
    return chosen
