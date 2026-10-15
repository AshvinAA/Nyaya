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

 TODO: implement against the existing dense-only Chroma collection + parent store
 once Stages 2 and 6 are being built. Right now this is the interface shell with
 each function documented.
============================================================================
"""

# ---------------------------------------------------------------------------
# Openers / clients (lazy — loaded on first use, not at import)
# ---------------------------------------------------------------------------
# _chroma_client() -> chromadb.PersistentClient  (path = db/chroma_db)
#   opens the existing collection `nyaya_children` (cosine space)
#   the collection already exists from a prior ingestion run (or will, once
#   ingestion is re-run); retrieval does not create it
#
# _parent_store() -> dict  (parent_id -> parent record)
#   loads ingestion_output/parent_store.json once and caches it
#   parents are keyed by the same ids children's parent_id points at
#
# Both are lazy so a retrieval run that only needs search (no parent
# resolution yet, e.g. an early Stage 2-3 test) doesn't pay for loading the
# parent store.

# ---------------------------------------------------------------------------
# Dense similarity query (Stage 2's current path)
# ---------------------------------------------------------------------------
# dense_search(query_text, filter=None, k=SEARCH_PER_LIST_K) -> list of dicts
#   runs a Chroma dense query against nyaya_children with the given where-filter
#   (DEFAULT_FILTER or HISTORICAL_FILTER from config, depending on the Stage 1
#   historical flag)
#   returns per-variant ranked list: [{chunk_id, score, rank, metadata}, ...]
#   (metadata includes the filterable fields + parent_id + section_path, so
#    downstream stages don't need a second lookup to read authority / layer)
#
# The filter is the applicability control:
#   default  -> in_force=true AND legal_status != superseded
#   historical -> filter swapped out entirely (Stage 7 labeling takes over)
#
# Dense scores are cosine similarities from BGE-M3; they're comparable across
# dense lists from the same model, but Stage 3 fuses by RANK, not raw score,
# because dense and sparse scores aren't on the same scale. So even now
# (dense-only), dense_search returns rank-positioned results, and Stage 3
# fuses by rank — don't fuse by raw cosine just because there's one mode.

# ---------------------------------------------------------------------------
# Chunk metadata read (used by Stage 4 dedup + Stage 7 overlay detection)
# ---------------------------------------------------------------------------
# chunk_metadata(chunk_id) -> dict | None
#   reads the stored metadata for a chunk from the Chroma collection by id
#   (the metadata Chroma stored alongside the vector — the filterable fields +
#    parent_id + topic_tags + chunk_id)
#   used by:
#     - Stage 4: authority_rank lookup for near-dup collapse resolution
#     - Stage 4: text_layer / source_act for the hard cross-layer guard
#     - Stage 7: text_layer for base-vs-overlay pair detection
#
# If a chunk_id isn't found in the index, that's a bug alarm (the ingestion
# index write guarantees the id exists); return None and let the calling stage
# treat it as a loud problem, not a silent skip.

# ---------------------------------------------------------------------------
# Parent resolution (Stage 6)
# ---------------------------------------------------------------------------
# resolve_parent(child_chunk_or_metadata) -> parent record | None
#   given a child (or its metadata with parent_id), fetches the parent from the
#   parent store by parent_id:
#     - prose parent  -> {parent_id, kind="prose", text (full section), ...}
#     - table parent  -> {parent_id, kind="table", table_json (full structured
#                        JSON), ...}
#     - self parent   -> {parent_id, kind="self", text (the child itself),
#                        ...}  (sentence-fallback chunks; honest, not pretending
#                        structure exists)
#     - missing       -> None  (should be impossible by construction — ingestion
#                        Stage 4 guarantees the pointer; occurrence = bug alarm)
#
# The returned parent record is what Stage 6 hands toward generation (or what
# the trace records as the resolved parent ref). The LLM never sees the bare
# child chunk.

# ---------------------------------------------------------------------------
# Sparse query path (future — only when the upgrade lands)
# ---------------------------------------------------------------------------
# sparse_search(query_text, filter=None, k=SEARCH_PER_LIST_K) -> list of dicts
#   WHEN Qdrant named sparse vectors exist (BGE-M3 lexical weights), this runs
#   the sparse half of hybrid search. Until then, SPARSE_AVAILABLE=False and
#   Stage 2 only calls dense_search.
#
# Adding this function is the ONLY retrieval-code change the sparse upgrade
# needs (plus Stage 2 calling it to produce a second list per variant, and
# Stage 3 fusing across more lists — which it already does generally). No
# downstream stage changes.

# ---------------------------------------------------------------------------
# Ingestion artifact helpers (trace joins, eval)
# ---------------------------------------------------------------------------
# These are read-only conveniences for the trace / eval layer, not for the
# retrieval stages themselves:
#
#   chunk_document(chunk_id) -> str | None
#     the bare generation text Chroma stored as the document for this child
#     (the LLM-facing text — the prefix never reaches the LLM; this is the
#      bare chunk text). Used by the trace's final_context_set preview and by
#      eval faithfulness checks.
#
#   parent_text(parent_id) -> str | dict | None
#     convenience: prose parent -> text; table parent -> table_json; missing ->
#     None. Used by the trace to record what the resolved context actually
#     contains (preview), and by eval faithfulness / parent-resolution ablation.

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
#     — the registry may need to split them back for some uses (e.g. topic_tags)
