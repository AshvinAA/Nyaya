"""
============================================================================
 Nyaya - Retrieval Pipeline :: index read registry (thin wrapper over the
 existing ingestion outputs)
============================================================================
 This module is the retrieval-side read path against the existing index. It is
 THIN and is the ONLY place that knows about Chroma / Qdrant specifics, so
 swapping stores later (dense-only Chroma -> dense+sparse Qdrant named vectors)
 changes only this module and Stage 2's second-list call — nothing downstream.

 It knows how to:
   - run a dense similarity query against the existing Chroma collection
     `nyaya_children` (db/chroma_db), with the filterable metadata fields as a
     Chroma where-filter
   - read a chunk's stored metadata by chunk_id (for Stage 4 dedup authority
     lookup, Stage 7 overlay detection, trace recording)
   - fetch a parent (prose section text or table JSON) from the parent store
     (ingestion_output/parent_store.json) by parent_id — the same IDs the index
     points at (ingestion Stage 4 guarantee)
   - read ingestion_outputs the retrieval stages need as inputs: parent_store.json,
     and (for trace joins) the chunk metadata that ingestion Stage 7 wrote into
     the Chroma documents/metadatas

 Objective of this file: give every retrieval stage a uniform way to query the
 index and resolve parents, without any stage importing Chroma / json directly.
 When the sparse upgrade lands, this file grows a second query path; Stages 2-7
 do not change.

 Verified against the real index in this checkout:
   - Chroma collection `nyaya_children` (db/chroma_db) exists with 170 chunks.
   - Every chunk carries a non-null parent_id (so Stage 6's "parent missing"
     path is a bug alarm, not a normal case — but we still handle it loudly).
   - Metadata fields are flat scalars (str/int/float/bool); lists were joined to
     "; "-separated strings at write time (indexer._flat_metadata) — we split
     them back where needed (e.g. topic_tags).
   - section_path uses a non-ASCII separator stored by Chroma; we normalise it
     when reading for overlap checks.
   - parent_store.json is NOT present in this checkout (ingestion has not been
     re-run here). resolve_parent / parent_store loading therefore fail loudly
     until ingestion is re-run — this is expected, not a bug in this file.
============================================================================
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from ingestion import config as ic
from ingestion.common import normalize

# ---------------------------------------------------------------------------
# Globals / lazy caches
# ---------------------------------------------------------------------------

# Lazy Chroma client + collection — opened on first use, not at import, so a
# retrieval run that only needs search doesn't pay to open the parent store.
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection: Optional[chromadb.Collection] = None

# Lazy BGE-M3 embedding function for query-time embedding.
#
# Chroma's default embedding function is MiniLM (384-dim), but the index was
# written by ingestion Stage 7 with BGE-M3 (1024-dim, model
# BAAI/bge-m3). A query embedding MUST come from the same model the index was
# written with, or the dimensions won't match and Chroma rejects the query
# (InvalidArgumentError: Collection expecting embedding with dimension of 1024,
# got 384).
#
# We use chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
# with the SAME model ingestion used (config.EMBED_MODEL = BAAI/bge-m3).
_bge_m3_ef: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None


def bge_m3_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return the BGE-M3 embedding function Chroma should use to embed queries.

    The index was written by ingestion Stage 7 with BGE-M3
    (config.EMBED_MODEL = BAAI/bge-m3, 1024-dim). Queries must be embedded with
    the SAME model, or Chroma rejects them for dimension mismatch.

    We use chromadb's SentenceTransformerEmbeddingFunction wrapping the same
    model ingestion used, so the query embedding dimension matches the stored
    vectors (1024).
    """
    global _bge_m3_ef
    if _bge_m3_ef is None:
        _bge_m3_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=ic.EMBED_MODEL,
        )
    return _bge_m3_ef

# Lazy parent-store dict — loaded once from ingestion_output/parent_store.json
# and cached. NOT present in this checkout until ingestion is re-run.
_parent_store: Optional[Dict[str, Any]] = None
_parent_store_path: str = ic.OUT_DIR + "/" + "parent_store.json"


def chroma_client() -> chromadb.PersistentClient:
    """Return the lazily-opened Chroma PersistentClient over db/chroma_db.

    Opens on first call and caches forever in this process. The collection
    `nyaya_children` (cosine space) already exists from a prior ingestion run;
    retrieval does not create it.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=ic.DB_DIR)
    return _chroma_client


def chroma_collection() -> chromadb.Collection:
    """Return the lazily-opened Chroma collection `nyaya_children`.

    This is the child-chunk index: 1024-dim BGE-M3 dense vectors, cosine space,
    with filterable metadata written by ingestion Stage 7.
    """
    global _chroma_collection
    if _chroma_collection is None:
        _chroma_collection = chroma_client().get_collection(name=ic.COLLECTION)
    return _chroma_collection


def parent_store() -> Dict[str, Any]:
    """Return the lazily-loaded parent store (parent_id -> parent record).

    Loaded once from ingestion_output/parent_store.json and cached. The parent
    store is the DOCUMENT STORE of the architecture: parents are keyed by the
    same IDs the child index points at (ingestion Stage 4 guarantee).

    Raises a loud, descriptive error if the file is missing — this means
    ingestion has not been re-run in this checkout, which is expected here but
    is a real prerequisite before Stage 6 (parent resolution) can be exercised.
    """
    global _parent_store
    if _parent_store is None:
        if not os.path.exists(_parent_store_path):
            raise FileNotFoundError(
                f"parent_store.json not found at {_parent_store_path}. "
                "Re-run the ingestion pipeline ( ingestion/run_pipeline.py ) so "
                "Stage 4 writes the parent store before any retrieval stage that "
                "resolves parents (Stage 6) is exercised."
            )
        with open(_parent_store_path, "r", encoding="utf-8") as fh:
            _parent_store = json.load(fh)
    return _parent_store


# ---------------------------------------------------------------------------
# Dense similarity query (Stage 2's current path)
# ---------------------------------------------------------------------------

def dense_search(
    query_text: str,
    filter_params: Optional[Dict[str, Any]] = None,
    k: int = 10,
    retain_vectors: bool = False,
) -> List[Dict[str, Any]]:
    """Run a dense BGE-M3 similarity query against nyaya_children.

    Parameters
    ----------
    query_text : str
        The query to embed and search with (a Stage 1 variant, or the original).
    filter_params : dict, optional
        Chroma where-filter applied INSIDE the search (not post-hoc). This is
        the applicability control:
          - default   -> in_force=True AND legal_status!="superseded"
          - historical -> filter swapped out (None) so superseded text can surface.
        When None, no metadata filter is applied.
    k : int
        Number of candidates to return (per-list depth; config.SEARCH_PER_LIST_K
        is the default starting point, 10).
    retain_vectors : bool
        If True, also fetch the dense embedding vector for each returned chunk
        and attach it as "dense_vector" on the result. This is what Stage 5b MMR
        needs (it reuses the vectors already computed here — it does not
        re-embed). When False, vectors are not fetched (cheaper; used when MMR
        will not run, e.g. single-aspect queries).

    Returns
    -------
    List of dicts, one per returned chunk, in rank order (1-based rank attached):
        {
          "chunk_id": str,
          "score": float,          # cosine similarity (dense)
          "rank": int,            # 1-based position in this list
          "metadata": dict,       # the stored metadata (filterable fields +
                                  # parent_id, section_path, topic_tags, ...)
          "text": str | None,     # the bare generation text Chroma stored
                                  # ("" when not requested / not present)
          "dense_vector": list[float] | None,   # only when retain_vectors=True
        }
    """
    col = chroma_collection()

    # Chroma where-filter: metadata values are flat scalars in the store.
    # Chroma where clauses can have only ONE operator per dict, so we build one
    # dict per operator and combine them with {'$and': [...]}. Booleans are stored
    # as bools; legal_status is a string inequality ($ne).
    where = None
    if filter_params:
        clauses: list = []
        if filter_params.get("in_force") is not None:
            clauses.append({"in_force": bool(filter_params["in_force"])})
        ls_not = filter_params.get("legal_status_not")
        if ls_not:
            clauses.append({"legal_status": {"$ne": str(ls_not)}})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

    # The index was written by ingestion Stage 7 with BGE-M3 (1024-dim).
    # Chroma's default query embedder (MiniLM, 384-dim) would reject with
    # InvalidArgumentError (dimension mismatch: got 384, expected 1024). We embed
    # the query ourselves with the SAME BGE-M3 model ingestion used, and pass the
    # query embedding explicitly via query_embeddings.
    ef = bge_m3_embedding_function()
    query_emb = ef([query_text])

    include = ["metadatas", "documents"]
    if retain_vectors:
        include.append("embeddings")

    result = col.query(
        query_embeddings=query_emb,
        n_results=k,
        where=where,
        include=include,
    )

    # Chroma returns lists keyed by 'ids', 'distances', 'metadatas', 'documents',
    # 'embeddings' — each a list-of-lists (one inner list per query text). We have
    # one query text, so we take index 0 of each. When querying by query_embeddings
    # (not query_texts), Chroma may return distances=None — we guard the [0] access
    # against NoneTypes.
    def _first(lst):
        inner = lst[0] if lst else []
        return inner if inner is not None else []

    ids = _first(result.get("ids") or [[]])
    distances = _first(result.get("distances") or [[]])
    metadatas = _first(result.get("metadatas") or [[]])
    documents = _first(result.get("documents") or [[]])
    embeddings = (
        _first(result.get("embeddings") or [[]]) if retain_vectors else [None] * len(ids)
    )

    # IMPORTANT: when querying by query_embeddings (not query_texts), Chroma may
    # return distances=None (it knows the cosine similarity from the vector store but
    # doesn't always return it in this path). We treat a missing distance as None
    # rather than failing — the rank still comes from Chroma's result order.

    # IMPORTANT: the vectors in `embeddings` are the VECTOR STORE's vectors for the
    # RETRIEVED CHUNKS (BGE-M3, 1024-dim) — NOT the query embedding we computed.
    # When retain_vectors=True, these are what Stage 5b MMR needs (chunk-to-chunk
    # similarity against the BGE-M3 dense vectors already computed in Stage 2). They
    # are NOT the query vector.

    n = len(ids)
    out: List[Dict[str, Any]] = []
    for rank in range(1, n + 1):
        idx = rank - 1
        cid = ids[idx] if idx < len(ids) else None
        dist = distances[idx] if distances and idx < len(distances) else None
        meta = metadatas[idx] if idx < len(metadatas) else {}
        doc = documents[idx] if idx < len(documents) else ""
        vec = embeddings[idx] if idx < len(embeddings) else None
        if cid is None:
            continue
        entry: Dict[str, Any] = {
            "chunk_id": cid,
            "score": float(dist) if dist is not None else None,
            "rank": rank,
            "metadata": _unflatten_metadata(meta) if meta else {},
            "text": doc if doc else "",
        }
        if retain_vectors and vec is not None:
            entry["dense_vector"] = [float(v) for v in vec]
        out.append(entry)
    return out


def _unflatten_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse ingestion Stage 7's _flat_metadata into usable Python values.

    Chroma can only store str/int/float/bool, so ingestion joined lists to
    "; "-separated strings and cast everything to scalar. Here we restore:
      - topic_tags  -> list[str]  (split on "; ")
      - est_tokens  -> int
      - page        -> int
      - retrievable -> bool
    Everything else is left as stored (str).
    """
    out: Dict[str, Any] = dict(meta or {})
    for key in ("topic_tags",):
        v = out.get(key, "")
        if isinstance(v, str) and v.strip():
            out[key] = [s.strip() for s in v.split(";") if s.strip()]
        else:
            out[key] = []
    for key, cast in (("est_tokens", int), ("page", int)):
        v = out.get(key, "")
        try:
            out[key] = cast(v) if v not in ("", None) else -1
        except (ValueError, TypeError):
            out[key] = -1
    rv = out.get("retrievable")
    if isinstance(rv, str):
        out["retrievable"] = rv.strip().lower() not in ("0", "false", "")
    return out


# ---------------------------------------------------------------------------
# Chunk metadata read (used by Stage 4 dedup + Stage 7 overlay detection)
# ---------------------------------------------------------------------------

def chunk_metadata(chunk_id: str) -> Optional[Dict[str, Any]]:
    """Read the stored metadata for a single chunk by chunk_id.

    Uses Chroma's get (by IDs) rather than a query, so we get the exact chunk
    (not a similarity search). Returns the unflattened metadata dict, or None if
    the chunk_id is not in the index.

    A missing chunk_id is a bug alarm — ingestion Stage 7 guarantees every id in
    the index exists — but we return None rather than crashing, and let the caller
    (Stage 4 / Stage 7) treat it as a loud problem.
    """
    col = chroma_collection()
    result = col.get(ids=[chunk_id], include=["metadatas"])
    if not result.get("ids"):
        return None
    meta = result["metadatas"][0] if result.get("metadatas") else {}
    return _unflatten_metadata(meta) if meta else {}


# ---------------------------------------------------------------------------
# Parent resolution (Stage 6)
# ---------------------------------------------------------------------------

def resolve_parent(
    child_or_parent_id: Dict[str, Any] | str,
) -> Dict[str, Any] | None:
    """Fetch a parent record from the parent store by parent_id.

    Accepts either a child dict that carries a 'parent_id' key, or a bare
    parent_id string.

    Returns one of:
      - {parent_id, kind, text|table_json, ...}   (found parent)
      - None                                        (parent missing — bug alarm)

    Parent kinds (from ingestion Stage 4):
      - "prose"   -> parent record has 'text' (the full section) and 'pages'
      - "table"   -> parent record has 'table_json' (the full structured JSON)
      - "self"    -> parent record has 'text' (the child itself; sentence-fallback
                     chunks with no real section context, honestly labeled)

    A missing parent is impossible by construction (ingestion Stage 4 guarantees
    every retrievable child has a parent_id in the store). Its occurrence is a bug
    alarm — but we return None rather than crashing, so the caller (Stage 6) can
    label the chunk "[bare chunk — parent missing]" and trace the event.
    """
    if isinstance(child_or_parent_id, dict):
        pid = child_or_parent_id.get("parent_id")
    else:
        pid = child_or_parent_id

    if not pid:
        return None

    store = parent_store()
    return store.get(pid, None)


def parent_text(parent_id: str) -> str | Dict[str, Any] | None:
    """Convenience: return the LLM-facing content of a parent by parent_id.

    - prose / self parent  -> the 'text' field (str)
    - table parent         -> the 'table_json' dict
    - missing              -> None

    Used by the trace (to record a preview of what the resolved context contains)
    and by eval faithfulness / parent-resolution ablation checks.
    """
    p = resolve_parent(parent_id)
    if p is None:
        return None
    if p.get("kind") == "table":
        return p.get("table_json") or {}
    return p.get("text") or ""


# ---------------------------------------------------------------------------
# Sparse query path (future — only when the upgrade lands)
# ---------------------------------------------------------------------------

def sparse_search(
    query_text: str,
    filter_params: Optional[Dict[str, Any]] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Sparse half of hybrid search — ONLY when the Qdrant sparse index exists.

    When config.SPARSE_AVAILABLE is False (the current default — dense-only Chroma),
    this raises NotImplementedError with a message that points at the sparse upgrade
    path. Stage 2 calls this only when SPARSE_AVAILABLE is True, so the raise never
    fires in the current dense-only run.

    When the upgrade lands, this becomes a Qdrant named-sparse-vector query using
    BGE-M3's learned lexical weights (NOT BM25). The return shape matches
    dense_search's so Stage 3's fusion is unchanged.
    """
    raise NotImplementedError(
        "sparse_search is not available in the current dense-only build. "
        "Stand up BGE-M3 sparse named vectors in Qdrant (ingestion Stage 7 change), "
        "set config.SPARSE_AVAILABLE = True, then this function becomes the sparse "
        "half of hybrid search. Retrieval Stages 3-7 need no change."
    )


# ---------------------------------------------------------------------------
# Ingestion artifact helpers (trace joins, eval)
# ---------------------------------------------------------------------------

def chunk_document(chunk_id: str) -> str:
    """Return the bare generation text Chroma stored as the document for a child.

    This is the LLM-facing text — the hierarchical prefix was prepended only for
    the embedding vector (indexer.build_embed_text), never for the LLM. The reranker
    (Stage 5) scores this text; the trace records a preview of it.

    Returns the stored document string, or "" if the chunk is not in the index.
    """
    col = chroma_collection()
    result = col.get(ids=[chunk_id], include=["documents"])
    docs = result.get("documents", [[]])[0] if result.get("documents") else []
    return docs[0] if docs else ""


# ---------------------------------------------------------------------------
# Overlay / overlap helpers (used by Stage 7)
# ---------------------------------------------------------------------------

def section_number_of(metadata: Dict[str, Any]) -> Optional[str]:
    """Return the section_number a chunk's metadata reports, normalised to a string.

    Used by Stage 7 to detect base+overlay pairs that cover the SAME provision.
    Returns None when the chunk has no section_number (e.g. a table chunk, a
    fallback piece) — such chunks can't form a dual-layer provision pair.
    """
    sn = metadata.get("section_number")
    if sn is None:
        return None
    return str(sn)


def text_layer_of(metadata: Dict[str, Any]) -> Optional[str]:
    """Return the text_layer a chunk's metadata reports.

    Returns one of 'consolidated_base', 'amendment_overlay', or None.
    Used by Stage 7 to decide whether a chunk is the base side or the overlay side
    of a dual-layer pair.
    """
    return metadata.get("text_layer") or None


def same_provision(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> bool:
    """Return True when two chunks' metadata indicate they cover the same provision.

    The detection is deliberately conservative and uses only metadata ingestion
    already attached (not re-parsed text):
      1. Same section_number  (the primary signal — e.g. both are section 23)
      2. Same section_path     (fallback, when section_number is missing but the
                                path still identifes the provision)

    This is what lets Stage 7 surface a base chunk and an overlay chunk that both
    cover section 23 (for example) as a dual_layer pair, without re-deriving the
    section from text.

    Returns False when either chunk has no section identifier (can't establish they
    cover the same provision), or when they clearly cover different provisions.
    """
    sn_a = section_number_of(meta_a)
    sn_b = section_number_of(meta_b)
    if sn_a and sn_b:
        return sn_a == sn_b

    # Fallback: same section_path (both missing section_number, but same path)
    path_a = meta_a.get("section_path", "")
    path_b = meta_b.get("section_path", "")
    if path_a and path_b:
        # Normalise the path separator — Chroma may have stored a non-ASCII char
        # where ingestion used " · ". Treat any run of non-alphanumeric separators
        # as equivalent.
        norm_a = re.sub(r"[^\w\s]+", " ", path_a.lower())
        norm_b = re.sub(r"[^\w\s]+", " ", path_b.lower())
        # Strip common bookkeeping words for a looser but still meaningful match
        stopwords = {"section", "chapter", "sub-section", "clause", "page", "table", "front", "matter"}
        tokens_a = [t for t in norm_a.split() if t not in stopwords]
        tokens_b = [t for t in norm_b.split() if t not in stopwords]
        return tokens_a == tokens_b

    return False


# ---------------------------------------------------------------------------
# Batch helpers (used by the eval harness / trace joins)
# ---------------------------------------------------------------------------

def chunk_metadata_batch(chunk_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return {chunk_id: metadata_or_None} for a batch of chunk_ids.

    Single Chroma get call for the whole batch (more efficient than one call per
    id). Missing ids map to None.
    """
    if not chunk_ids:
        return {}
    col = chroma_collection()
    result = col.get(ids=chunk_ids, include=["metadatas"])
    ids = result.get("ids", [])
    metas = result.get("metadatas", []) or []
    by_id = dict(zip(ids, metas))
    return {
        cid: (_unflatten_metadata(by_id[cid]) if cid in by_id and by_id[cid] else None)
        for cid in chunk_ids
    }


# ---------------------------------------------------------------------------
# Guarantees this registry relies on (from ingestion — do not re-validate here)
# ---------------------------------------------------------------------------

#   - every id in the index has a non-empty node + chunk_id (Stage 3 gate +
#     Stage 5 chunk_id mint)
#   - every retrievable child has a parent_id pointing into the parent store
#     (Stage 4 guarantee) — a missing parent_id is a bug alarm, not a handled
#     case
#   - the index contains only retrievable=true, non-front-matter, non-demoted
#     chunks — so the Stage 2 filter is the ENTIRE applicability control; the
#     registry doesn't need to re-check retrievable on read
#   - metadata fields are flat scalars in Chroma (str/int/float/bool); lists
#     were joined to "; "-separated strings at write time (indexer._flat_metadata)
#     — we split them back for some uses (topic_tags)
#
# We do NOT re-validate these here. If they're violated, the wrong thing happens
# downstream and the trace + eval catch it; the registry's job is to read what's
# there, honestly.
