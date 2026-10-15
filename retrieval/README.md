# Nyaya — Retrieval Pipeline (`retrieval/`)

Implementation of the architecture in [`RETRIEVAL_PIPELINE.md`](../RETRIEVAL_PIPELINE.md):
takes a text query (English or Bangla) and returns a final, labeled,
parent-resolved, deduplicated context set ready for the generation step —
with a per-query JSON trace as the evaluation substrate.

## Status

**Designed, not built.** The full architecture is specified in
`RETRIEVAL_PIPELINE.md`. This package does not exist yet — the files below
are the planned one-file-per-stage structure, annotated with each file's
objective. No retrieval code runs yet; the only runnable pipeline today is
ingestion (`ingestion/run_pipeline.py`).

## File map — one file per stage/job

| File | Job | Stage |
|---|---|---|
| `config.py` | every retrieval tunable knob in one place | all |
| `trace.py` | the per-query JSON trace object + writer (the evaluation substrate) | all (observability) |
| `registry.py` | thin wrapper around the existing index: run dense queries against Chroma `nyaya_children`, read chunk metadata, fetch parents from `parent_store.json` | Stage 2 / Stage 6 read path |
| `stage1_expand.py` | multi-query expansion: LLM rewrites the query into 2-3 formal-register variants; one structured envelope (variants + multi_aspect + historical) | 1 |
| `stage2_search.py` | hybrid search per variant with metadata pre-filtering (default: in_force=true AND legal_status≠superseded; historical: swap filter) — dense-only until the sparse upgrade lands | 2 |
| `stage3_fuse.py` | RRF fusion across variant (and later dense+sparse) ranked lists by rank position, k=60, fused pool capped to ~30 | 3 |
| `stage4_dedup.py` | retrieval-time near-dup collapse of the fused pool by authority_rank; hard guard: never collapse across text_layer/source_act; runs before reranking | 4 |
| `stage5_rerank.py` | BGE-Reranker-v2-M3 cross-encoder re-score vs the ORIGINAL query (not the variants); final top-k ~5; fallback: RRF order stands | 5 |
| `stage5b_mmr.py` | optional MMR (lambda≈0.6) only when Stage 1 flagged multi_aspect; uses dense vectors already computed in Stage 2 | 5b |
| `stage6_resolve.py` | parent/table resolution: fetch parent section text or full table JSON per surviving child via parent_id; token budgeting with truncation labeling; LLM never sees bare children | 6 |
| `stage7_overlay.py` | overlay-vs-base resolution: when both consolidated_base and amendment_overlay cover the same provision, surface both with explicit labels, never silently pick one; record dual_layer pairs in trace | 7 |
| `orchestrator.py` | wires Stages 1-7 into one `retrieve(query) -> (context_set, trace)` entry point; every stage records into the trace; every fallback is logged | — |
| `cli.py` | small demo script to run a single query end to end and print/inspect the trace before trusting it | — |

(The original single-file demo predecessor is kept at `../legacy/ingestion_stage1_2.py`
for the ingestion side only — retrieval has no legacy code; an earlier retrieval
experiment was deliberately cleaned out of this repo's history.)

## Output

- `retrieval_traces/<query_id>.json` — one trace per query (per-stage ranked lists,
  fusion scores, dedup decisions, rerank scores, mmr applied?, final context set
  with labels + parent ids, dual_layer pairs, timings, fallbacks triggered).
- The final context set is returned in memory to the caller (the generation stage
  built later); traces are the persisted artifact.

## Design notes

- **ASR is deferred, deliberately** — build and evaluate retrieval against typed/clean
  text first; voice (Groq Whisper + TTS) is a later I/O layer. The reserved `asr` field
  in the Stage 1 envelope exists for when voice lands; do not build ASR-specific logic now.
- **Expand before you search** — Stage 1 bridges colloquial-vs-legal register; the
  variants widen the net, the reranker judges relevance to the original query.
- **One query-understanding point** — Stage 1's envelope (variants + multi_aspect +
  historical) is the only place query intent is interpreted; everything downstream
  consumes the envelope, never re-guesses.
- **Filter at the source** — `in_force`/`legal_status` filtering happens inside
  Stage 2's search, not as a post-hoc check; correctness filters degrade last.
- **Fuse by rank, not raw score** — dense and sparse scores aren't comparable; RRF
  sidesteps that. Write Stage 3 for the general case from the start; dense-only just
  means fewer lists, not a different fusion.
- **Dedup twice, for different reasons** — ingestion-time dedup cleans the corpus once;
  retrieval-time dedup stops multi-query fan-out from spending scarce final-k slots on
  the same fact per-query; both inherit the hard guard: never merge across text_layer
  / source_act.
- **Rerank against the real query** — variants widen the net; the reranker judges the
  original question.
- **MMR is conditional, not default** — only for genuinely multi-aspect queries, flagged
  upstream by Stage 1.
- **Never return a bare child chunk** — parent/table resolution is mandatory before
  generation.
- **Never silently resolve a legal conflict** — overlay vs base gets surfaced with
  labels, not collapsed.
- **Degrade loudly** — every stage has a named fallback; convenience degrades, correctness
  filters don't; everything lands in the trace.
- **Measure every stage** — each stage exists because an ablation row says so; the
  `eval/` harness consumes the trace.

## Sparse upgrade path

The demo index is dense-only Chroma today. The plan's target is dense+sparse (BGE-M3
learned lexical weights in Qdrant named vectors — **not BM25**). When dense-only
recall@k on the test set falls short — watch the **table/numeric lookup category first**,
the predicted dense-only failure mode — stand up the sparse index as an ingestion-side
change to Stage 7 only. Retrieval Stages 2 (add a second list per variant) and 3
(already general fusion) absorb it with no downstream redesign.
