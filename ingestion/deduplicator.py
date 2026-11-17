"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 6 - Deduplication
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 After chunking and BEFORE chunks reach the vector index, this pass removes
 genuine redundancy WITHOUT removing meaningful version history:

   - EXACT duplicates (identical text from overlap or double-extraction):
     caught by hashing normalized chunk text and demoting every copy but
     the most authoritative one.
   - NEAR-duplicates (the same fact stated differently across sources -
     e.g. the gazette's wording vs CPD's paraphrase of the same fact):
     caught by pairwise embedding cosine similarity above a threshold,
     then resolved by the AUTHORITY RANKING:

         official gazette > official translation >
         reputable secondary source > advocacy report

     The lower-authority version is DEMOTED / cross-referenced, never
     deleted - it may still be useful context. Demotion means
     retrievable=False + superseded_by=<winner chunk_id>, so the pair
     stays fully auditable in dedup_report.json.

 DISTINCT from version filtering: `legal_status` / `in_force` /
 `effective_date` filtering preserves genuinely different VERSIONS of the
 law over time; dedup removes accidental redundancy that adds noise
 without adding information. These are separate steps and must stay
 separate - so this stage REFUSES to merge chunks from different law
 layers or different instruments (base text vs overlay, Act vs Rules),
 even when their texts look similar. Near-dup also never pairs two chunks
 of the SAME document: a statute repeats its own wording on purpose
 (s.26(b) and s.27(b) both read "14 days, in case of other workers"), and
 each copy is a separate provision with its own citation.

 Bangla-script chunks are scored with the character n-gram TF-IDF scorer,
 not MiniLM: MiniLM is English-only and maps all Bangla text to nearly the
 same vector, which would demote almost every transcribed chunk.

 Scaling note (from the plan): pairwise similarity is O(n^2) across
 chunks. At this corpus's size (thousands of chunks, not millions) that's
 fine - the similarity matrix is computed with numpy, not Python loops.
 If the corpus ever grows, switch to minHash/LSH blocking first and run
 embedding similarity only WITHIN candidate blocks.

 The near-dup scorer uses the small cached MiniLM model - it is ONLY a
 similarity judge here, never the retrieval embedder (Stage 7 embeds the
 index with BGE-M3). If the model cannot load, the stage falls back to a
 TF-IDF cosine scorer rather than skipping the gate.

 Used by: run_pipeline.py (after Stage 5 tagging).
============================================================================
"""

import hashlib
import re

import numpy as np

from ingestion.config import (NEARDUP_SIM_THRESHOLD, NEARDUP_MIN_TOKENS,
                              DEDUP_MODEL)
from ingestion.common import normalize

# authority ranking (lower number wins) - mirrors doc_metadata.authority_rank
AUTHORITY_ORDER = {"official_gazette": 0, "official_translation": 1,
                   "unofficial_translation": 2, "reputable_secondary": 3,
                   "advocacy_report": 4, None: 9}


def _rank(c):
    return AUTHORITY_ORDER.get(c.get("authority_rank"), 9)


def _text_hash(c):
    # normalized lowercase text -> stable identity for exact dedup
    return hashlib.sha1(normalize(c.get("text", "")).lower().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Near-dup scoring backends
# ---------------------------------------------------------------------------
def _embed_matrix(texts, model_name=DEDUP_MODEL):
    """Dense-embed all candidate texts with the small cached model.
    Returns (matrix, backend_name)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32), "embedding:%s" % model_name


def _tfidf_matrix(texts):
    """Offline fallback scorer: TF-IDF character n-grams + L2 norm. Weaker
    than embedding similarity for paraphrase, but keeps the gate running
    with zero downloads; threshold semantics are the same."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    m = vec.fit_transform(texts)                   # sparse rows
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    return (m.toarray().astype(np.float32) / norms[:, None]), "tfidf-char3-5"


def _is_bangla(text, share=0.3):
    """True when Bangla script dominates the text's letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if "ঀ" <= ch <= "৿") / len(letters) >= share


def _pair_allowed(a, b):
    """Version-history protection: near-dup must never merge different law
    layers or different instruments - that is version filtering's job, and
    the two steps must stay separate (see module docstring)."""
    if (a.get("text_layer") or "") != (b.get("text_layer") or ""):
        return False                               # base vs overlay, etc.
    if (a.get("source_act") or "") != (b.get("source_act") or ""):
        return False                               # Act vs Rules vs CPD ...
    if (a.get("amends_section") or "") != (b.get("amends_section") or ""):
        return False                               # same wording, different
                                                   # provision amended (the
                                                   # 2026 Act's ss.58/59/62)
    if a.get("doc") == b.get("doc"):
        return False                               # one document never repeats
                                                   # itself: near-identical
                                                   # chunks inside it are
                                                   # DIFFERENT provisions (s.26(b)
                                                   # vs s.27(b), s.294 vs s.295)
    return True


def _demote(loser, winner, status, sim=None):
    """Demote-and-cross-reference: the duplicate stays in every output file
    for transparency, but retrievable=False keeps it out of the index."""
    loser["retrievable"] = False
    loser["dedup_status"] = status
    loser["superseded_by"] = winner["chunk_id"]
    if sim is not None:
        loser["dedup_similarity"] = round(float(sim), 3)


# ---------------------------------------------------------------------------
# The dedup pass
# ---------------------------------------------------------------------------
def dedup(children):
    """Runs exact-hash dedup, then embedding near-dup, over all retrievable
    children (in place). Returns (report, updated_children)."""
    live = [c for c in children
            if c.get("retrievable", True) and c.get("kind") != "front_matter"]

    # ---- exact duplicates: hash groups -------------------------------------
    groups = {}
    for c in live:
        groups.setdefault(_text_hash(c), []).append(c)
    n_exact = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda c: (_rank(c), -(c.get("est_tokens") or 0)))
        winner = members[0]
        for dup in members[1:]:
            _demote(dup, winner, "exact_duplicate")
            n_exact += 1

    # ---- near-duplicates: pairwise similarity over survivors ----------------
    cand = [c for c in live if c.get("retrievable", True)
            and (c.get("est_tokens") or 0) >= NEARDUP_MIN_TOKENS]
    pairs, n_near = [], 0
    backends = []
    # MiniLM is English-only: Bangla text (hand transcripts of the Bangla
    # gazettes) tokenizes to near-identical vectors and every chunk would
    # look like a duplicate of every other. Bangla-script chunks are scored
    # with the character n-gram scorer instead, which works on any script.
    # (Cross-script pairs never qualify anyway: _pair_allowed requires the
    # same source_act.)
    bangla = [c for c in cand if _is_bangla(c["text"])]
    latin = [c for c in cand if not _is_bangla(c["text"])]
    for group, use_embedding in ((latin, True), (bangla, False)):
        if len(group) < 2:
            continue
        texts = [c["text"] for c in group]
        if use_embedding:
            try:
                mat, backend = _embed_matrix(texts)
            except Exception as exc:               # no model / no network
                print("  [dedup] embedding scorer unavailable (%s) - "
                      "falling back to TF-IDF cosine" % exc)
                mat, backend = _tfidf_matrix(texts)
        else:
            mat, backend = _tfidf_matrix(texts)
        backends.append("%s (%d chunks)" % (backend, len(group)))
        n_near += _near_dup_pass(group, mat, pairs)

    report = {"n_children": len(children),
              "n_scored": len(cand),
              "backend": "; ".join(backends) or None,
              "threshold": NEARDUP_SIM_THRESHOLD,
              "n_exact_dups": n_exact,
              "n_near_dups": n_near,
              "n_demoted": n_exact + n_near,
              "pairs": pairs[:200]}                # cap the audit trail size
    return report, children


def _near_dup_pass(cand, mat, pairs):
    """Pairwise cosine over one scored group; demotes the lower-authority
    side of every allowed pair at/above the threshold. Returns the count."""
    n_near = 0
    block = 512                                   # rows per similarity block
    for lo in range(0, len(cand), block):         # O(n^2) but numpy-fast
        hi = min(lo + block, len(cand))
        sims = mat[lo:hi] @ mat.T                 # (block, n) cosine matrix
        for r in range(hi - lo):
            i = lo + r
            for j in range(i + 1, len(cand)):
                sim = float(sims[r, j])
                if sim < NEARDUP_SIM_THRESHOLD:
                    continue                      # below threshold: not a pair
                a, b = cand[i], cand[j]
                if not (a.get("retrievable", True)
                        and b.get("retrievable", True)):
                    continue                      # already demoted this pass
                if not _pair_allowed(a, b):
                    continue                      # version history - keep both
                winner, loser = (a, b) if (_rank(a), -len(a["text"])) <= \
                    (_rank(b), -len(b["text"])) else (b, a)
                _demote(loser, winner, "near_duplicate", sim)
                n_near += 1
                pairs.append({"winner": winner["chunk_id"],
                              "loser": loser["chunk_id"],
                              "similarity": round(sim, 3),
                              "winner_doc": winner["doc"],
                              "loser_doc": loser["doc"]})
    return n_near
