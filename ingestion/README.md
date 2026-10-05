# Nyaya — Ingestion Pipeline (`ingestion/`)

Implementation of the architecture in [`INGESTION_PIPELINE.md`](../INGESTION_PIPELINE.md):
turns the raw legal corpus in `docs/` into a retrieval-ready, structure-preserving index.

## File map — one file per stage/job

| File | Job | Stage |
|---|---|---|
| `config.py` | every tunable knob in one place | all |
| `common.py` | shared helpers (normalize, token estimates, ids, JSON dump) | all |
| `doc_metadata.py` | doc-level authority/applicability registry + table summarizer | 5 |
| `page_router.py` | classify every page (prose/table/mixed/empty/visual) + front-matter detection | 1 |
| `prose_chunker.py` | hierarchical, structure-aware prose chunking (Chapter→Section→Clause) | 2a |
| `table_extractor.py` | structured table extraction, never flattening; inline validation + retry | 2b |
| `document_intake.py` | per-document driver wiring page routing → chunking | 1+2 |
| `validation_gate.py` | structural validation gate (continuity, hierarchy, furniture, node IDs) | 3 |
| `parent_builder.py` | parents (sections, table JSON) + child pointers; parent store | 4 |
| `metadata_tagger.py` | per-chunk tagging (path, topics, authority) + embed-prefix builder | 5 |
| `deduplicator.py` | exact-hash + embedding near-dup, resolved by authority rank | 6 |
| `indexer.py` | embed-time context injection + BGE-M3 + Chroma write | 7 |
| `run_pipeline.py` | CLI driver running stages 1–7 end to end | — |

(The original single-file demo is kept at `../legacy/ingestion_stage1_2.py`.)

## Run

```bash
venv/Scripts/python ingestion/run_pipeline.py            # demo: 30 pages/doc
venv/Scripts/python ingestion/run_pipeline.py --full     # whole corpus
venv/Scripts/python ingestion/run_pipeline.py --stages 123   # skip parents/dedup/embed
venv/Scripts/python ingestion/run_pipeline.py --no-embed # chunks only, no vector DB
venv/Scripts/python ingestion/run_pipeline.py --strict   # Stage 3: review blocks docs
```

## Outputs (`ingestion_output/` + `db/`)

- `routing_report.json` — Stage 1 manifest, one record per routed page
- `prose_chunks.json` — Stage 2a chunks, fully tagged (Stages 3–5 fields)
- `tables_extracted.json` — Stage 2b structured tables with summaries
- `flags_manual_review.json` — quarantined tables, visual-routing pages, blocked docs
- `doc_metadata.json` — document-level metadata registry
- `validation_report.json` — Stage 3 per-document pass/fail + pass rates
- `parent_store.json` — Stage 4 document store (parents keyed by id)
- `dedup_report.json` — Stage 6 demotion audit trail (winner/loser/similarity)
- `index_manifest.json` — Stage 7 index build summary
- `db/chroma_db` — Chroma collection `nyaya_children` (dense BGE-M3 vectors,
  flat filterable metadata); parents live in `parent_store.json` keyed by id

## Design notes

- **Route before chunking** — prose and tables fail differently; Stage 1 decides per page.
- **Respect structure, verify it** — Stage 3 turns "the parser walks the real hierarchy"
  into a checked invariant; failures block a document from the index, loudly.
- **Never flatten tables** — summary embeds, JSON generates (Stage 2b/4).
- **Embed with context, generate on bare text** — Stage 7 prepends the hierarchical
  path prefix to the embedding, never to the LLM's context.
- **Dedup noise, preserve history** — Stage 6 refuses to merge different law layers
  or instruments; version filtering is a separate concern.
- The vector index is dense-only on Chroma; the plan's dense+sparse hybrid
  (BGE-M3 sparse via Qdrant named vectors) is the retrieval-stage upgrade path.
