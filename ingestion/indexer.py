"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 7 - Embedding & index write
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Final retrievable children are embedded and written to the vector index.

 Embed-time context injection (the "embed with context, generate on bare
 text" rule): a single legal clause is short and heavily anaphoric -
 "said worker", "such employer", "the amount referred to in sub-section
 (1)". Embedded bare, a 30-word clause matches poorly against a real
 query. So the text that gets EMBEDDED is not the text that gets
 generated on:

   embedding text  = hierarchical path prefix + chunk content
                     (build_embed_prefix from Stage 5 + " - " + chunk text)
   generation text = the bare chunk (or its parent) - stored as the
                     document, so the LLM never sees the prefix as law text

 The embedding model is BGE-M3 (multilingual; natively dense+sparse). The
 index here is written DENSE-ONLY: the demo's vector store is Chroma,
 which has no named sparse-vector support - the plan's dense+sparse hybrid
 (Qdrant named sparse vectors holding BGE-M3's learned lexical weights) is
 the retrieval-stage upgrade path, and swapping stores means changing only
 this module. If BGE-M3 cannot be loaded (no network, no disk), the stage
 falls back to the small cached MiniLM model with a loud warning rather
 than failing the run - embeddings stay comparable only per build.

 Where things land (per the plan):
   vector store   Chroma collection `nyaya_children` under db/ - child
                  embeddings alongside metadata as FILTERABLE attributes
                  (source_act, legal_status, in_force, authority_rank, ...)
   document store parent chunks and structured table JSON keyed by the
                  same ids - already written by Stage 4 (parent_store.json)

 Re-runs rebuild the collection from scratch (delete + create): the index
 is a pure function of the corpus and the pipeline, never incrementally
 stale.

 Used by: run_pipeline.py (last stage).
============================================================================
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingestion.config import DB_DIR, COLLECTION, EMBED_MODEL, EMBED_BATCH
from ingestion.metadata_tagger import build_embed_prefix

# metadata fields stored as filterable attributes (flat scalars only)
FILTERABLE_FIELDS = ["doc", "kind", "node", "source_act", "source_type",
                     "legal_status", "in_force", "authority_rank",
                     "translation_status", "text_layer", "effective_date",
                     "section_number", "section_path", "parent_id"]


def build_embed_text(c):
    """Embedding text = context prefix + chunk content (see module docstring).
    The generation text stays c['text'] - the prefix never reaches the LLM."""
    return "%s — %s" % (build_embed_prefix(c), c.get("text", ""))


def _flat_metadata(c):
    """Chroma metadata values must be str/int/float/bool - flatten every
    field into that shape (None -> '', lists -> joined strings)."""
    meta = {}
    for f in FILTERABLE_FIELDS:
        v = c.get(f)
        if v is None:
            v = ""
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v)
        meta[f] = v
    meta["page"] = c.get("page") if isinstance(c.get("page"), int) else -1
    meta["est_tokens"] = int(c.get("est_tokens") or 0)
    meta["retrievable"] = bool(c.get("retrievable", True))
    meta["topic_tags"] = "; ".join(c.get("topic_tags") or [])
    meta["chunk_id"] = c.get("chunk_id", "")
    return meta


def embed_and_index(children, db_dir=DB_DIR, collection_name=COLLECTION,
                    model_name=EMBED_MODEL, batch_size=EMBED_BATCH):
    """Embed all retrievable children and (re)build the Chroma collection.
    Returns the manifest dict for index_manifest.json."""
    # only indexable chunks: passed validation, not demoted by dedup,
    # not front matter (front matter is never an answer candidate)
    indexable = [c for c in children
                 if c.get("retrievable", True)
                 and c.get("kind") != "front_matter"]

    backend = model_name
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:                       # BGE-M3 unavailable:
        backend = "sentence-transformers/all-MiniLM-L6-v2"   # cached fallback
        print("  [index] %s unavailable (%s) - falling back to %s"
              % (model_name, exc, backend))
        model = SentenceTransformer(backend)

    dim = model.get_sentence_embedding_dimension()
    emb_texts = [build_embed_text(c) for c in indexable]
    print("  [index] embedding %d children with %s (dim=%d)..."
          % (len(indexable), backend, dim))
    vecs = model.encode(emb_texts, batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)

    # ---- write the index (full rebuild: delete + create + add) -------------
    client = chromadb.PersistentClient(path=db_dir)
    try:
        client.delete_collection(collection_name)  # rebuild from scratch
    except Exception:
        pass                                       # first run: nothing to drop
    col = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    if indexable:
        col.upsert(                                # ids -> vectors + documents
            ids=[c["chunk_id"] for c in indexable],
            embeddings=vecs.tolist(),
            documents=[c["text"] for c in indexable],   # generation text (bare)
            metadatas=[_flat_metadata(c) for c in indexable])

    manifest = {"n_children_in": len(children),
                "n_indexed": len(indexable),
                "excluded": len(children) - len(indexable),
                "model": backend,
                "requested_model": model_name,
                "dim": dim,
                "db_dir": db_dir,
                "collection": collection_name,
                "collection_count": col.count(),
                "filterable_fields": FILTERABLE_FIELDS}
    return manifest
