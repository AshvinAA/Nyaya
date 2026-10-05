"""
============================================================================
 Nyaya - Ingestion Pipeline package
============================================================================
 Turns the raw legal corpus in docs/ into a retrieval-ready,
 structure-preserving index. One module per pipeline job:

     config.py           every tunable knob, in one place
     common.py           shared helpers (normalize, ids, JSON dump)
     doc_metadata.py     doc-level authority/applicability registry
     page_router.py      intake: classify every page BEFORE chunking
     prose_chunker.py    prose path: hierarchical structure-aware chunking
     table_extractor.py  table path: structured extraction, never flattening
     document_intake.py  per-document driver wiring routing -> chunkers
     validation_gate.py  structural validation gate (bad parses never index)
     parent_builder.py   parent-child indexing (embed small, generate large)
     metadata_tagger.py  per-chunk metadata tagging + embed prefix builder
     deduplicator.py     exact-hash + near-dup dedup by authority
     indexer.py          context injection + BGE-M3 embedding + Chroma write
     run_pipeline.py     CLI driver: runs the whole pipeline end to end

 Run it (from the project root):
     venv/Scripts/python ingestion/run_pipeline.py           # demo: 30 pages/doc
     venv/Scripts/python ingestion/run_pipeline.py --full    # whole corpus
     venv/Scripts/python ingestion/run_pipeline.py --stages 123   # no embedding
     venv/Scripts/python ingestion/run_pipeline.py --help    # all flags

 Architecture: see INGESTION_PIPELINE.md at the repo root.
============================================================================
"""

__version__ = "1.1.0"
