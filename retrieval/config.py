"""
============================================================================
 Nyaya - Retrieval Pipeline :: shared configuration
============================================================================
 Every tunable knob of the retrieval pipeline lives in THIS one file.

 The pipeline has one source of truth for each number so that:
   - an experiment that changes a threshold knows it changed it everywhere
   - the eval harness can sweep these constants
   - no stage module hardcodes a value another module already defines

 Values mirror RETRIEVAL_PIPELINE.md, the agreed architecture document.

 Note: the demo index is dense-only Chroma today. The config below is written
 for the general dense+sparse design from the start — dense-only just means
 Stage 2 produces one list per variant instead of two, and Stage 3 fuses fewer
 lists. Nothing downstream cares.

 Objective of this file: hold every retrieval constant so the eval harness's
 ablation config can turn stages off / reduce them by flipping these knobs
 rather than rewriting stage logic.

 Verified against the real index in this checkout:
   - collection: nyaya_children (db/chroma_db), 170 chunks, cosine space.
   - metadata fields are flat scalars in Chroma; we reuse ingestion's
     FILTERABLE_FIELDS for the where-filter document.
============================================================================
"""

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# RETRIEVAL_DIR = os.path.dirname(__file__)   # this package's own dir

# Where per-query JSON traces are written (one file per query).
# Already in .gitignore as retrieval_traces/.
import os

# Where per-query JSON traces are written (one file per query).
# Already in .gitignore as retrieval_traces/.
TRACE_DIR = "retrieval_traces"

# ---------------------------------------------------------------------------
# Runtime / model selection (tunable — not a given)
# ---------------------------------------------------------------------------
# Stage 1 expansion model: the demo LLM stand-in is ollama/llama3.2 (already
# in requirements.txt, already pulled). It is weak at Bangla; Bangla->formal-
# legal-English register shifting is the riskiest LLM-dependent step. Model
# choice is a tunable, decided by measurement against the register-gap queries
# in the test set, not by preference. The production LLM (Groq Llama 3.3 70B /
# Gemini Flash per the plans) is NOT integrated yet — no code exists that
# talks to either, so this stays as the dev stand-in until evaluated.
EXPAND_MODEL = "ollama/llama3.2"   # LangChain Ollama chat model; the dev stand-in.
   # Bangla->formal-legal-English rewrite is the riskiest LLM-dependent step.

# Stage 5 reranker: BGE-Reranker-v2-M3 (cross-encoder). NOT yet in
# requirements.txt — adding it is a separate decision (see TODO below).
# The reranker scores (original_query, candidate_text) pairs directly; it is
# the pipeline's CPU bottleneck on CPU, which is why the fused pool is capped
# at ~30 before it (Stage 4 / config.FUSED_POOL_CAP).
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Stage 6 / Stage 4 near-dup scorer: can reuse the cached MiniLM from
# ingestion Stage 6 (already in requirements.txt) for the small fused pool,
# OR embed the pool with BGE-M3. Do NOT use the cross-encoder here — that's
# Stage 5's job and it's the most expensive call.

# ---------------------------------------------------------------------------
# Stage 1 — Multi-query expansion
# ---------------------------------------------------------------------------
MAX_VARIANTS = 3          # LLM rewrites into 2-3 formal-register variants
                         # (the envelope's variants list is capped here)

# Fallback when expansion fails (LLM unavailable / malformed JSON):
# run with the original query as the sole variant, all flags false.
# The filter stays default-STRICT — never widen legality filters on a
# degraded run. Logged in the trace, never silent.
EXPAND_FALLBACK_VARIANTS = ("<original query used as sole variant>",)

# Reserved field in the envelope for future ASR transcript-cleanup signals.
# Deliberately absent now (RETRIEVAL_PIPELINE.md Flagged follow-ups).
# ENVELOPE_ASR_FIELD = "asr"   # do not build against this yet

# ---------------------------------------------------------------------------
# Stage 2 — Hybrid search, metadata pre-filtering
# ---------------------------------------------------------------------------
# Per-list depth: how many candidates each variant x mode returns before
# fusion. Starting point from the doc.
SEARCH_PER_LIST_K = 10

# Default filter (applicability control): the index already contains only
# chunks that passed the ingestion gate (retrievable=true, no front matter,
# no demoted duplicates). This filter is the ENTIRE applicability control:
#   in_force = true  AND  legal_status != superseded
# The 2025 Ordinance survives by design: in_force=true,
# legal_status=pending_ratification — it competes normally, Stage 7 labels it.
DEFAULT_FILTER = {
    "in_force": True,
    "legal_status_not": "superseded",
}

# Historical relaxation: when Stage 1 flagged historical, SWAP the filter
# entirely (drop in_force / legal_status constraints), not loosen at the edges.
# Partial relaxation ("allow superseded but still require in_force") is
# incoherent — a superseded provision is by definition not in force.
HISTORICAL_FILTER = {
    # no in_force / legal_status constraints; Stage 7 labeling takes over
    # the job of saying which text governed when.
}

# Dense-only until the sparse upgrade lands. When sparse exists (BGE-M3 lexical
# weights in Qdrant named vectors), Stage 2 produces a second list per variant
# and Stage 3 fuses across both — no downstream change.
SPARSE_AVAILABLE = False   # TODO: flip when Qdrant sparse index exists

# ---------------------------------------------------------------------------
# Stage 3 — RRF fusion
# ---------------------------------------------------------------------------
RRF_K = 60                # standard constant: score = sum(1 / (k + rank))
FUSED_POOL_CAP = 30       # cap the fused pool before Stage 4 (enough to
                          # survive dedup + reranking with a real final-k,
                          # small enough to keep the cross-encoder cheap)

# ---------------------------------------------------------------------------
# Stage 4 — Retrieval-time dedup
# ---------------------------------------------------------------------------
# Retrieval-time near-dup threshold — SEPARATE from ingestion's 0.92.
# The population is different (fused-pool candidates, not the whole corpus),
# and the purpose is different (per-query slot conservation, not corpus cleanup).
# This is a tunable to be set and swept against the test set's duplicate-rate
# metric; start conservative.
RETRIEVAL_NEAR_DUP_THRESHOLD = 0.92   # TODO: tune against test set

# Reuse ingestion's authority ranking (official_gazette > ... > advocacy).
# Imported from ingestion.doc_metadata or ingestion.deduplicator — do NOT
# re-declare it here. The ranking is the same; the application is per-query.

# Hard guard (inherited from ingestion): never collapse a pair across a
# different text_layer or source_act. A base provision and its 2025-overlay
# are near-identical by construction, and Stage 7 exists to surface both with
# labels. If this stage collapsed them, it would destroy the overlay rule.

# ---------------------------------------------------------------------------
# Stage 5 — Reranking
# ---------------------------------------------------------------------------
FINAL_K = 5               # final top-k handed to generation (after rerank,
                          # and after MMR if applied)

# Reranker scores against the ORIGINAL query (not the expanded variants).
# The variants exist to widen the net; the reranker judges relevance to what
# the user actually asked.

# Fallback: if the reranker cannot be loaded, the RRF order stands and
# reranker="fallback_rrf" goes in the trace. Retrieval never hard-fails
# because an optional re-scorer is down.

# ---------------------------------------------------------------------------
# Stage 5b — Optional MMR (conditional, not default)
# ---------------------------------------------------------------------------
MMR_LAMBDA = 0.6          # diversity vs relevance tradeoff
                          # (against BGE-M3 dense vectors already computed
                          # in Stage 2 — do not recompute)

# MMR runs ONLY if Stage 1 flagged the query multi_aspect. On a single-aspect
# legal question, MMR can reduce precision by demoting the most relevant chunk
# for diversity that wasn't needed. This conditional trigger is itself an eval
# item: measure (a) how often the flag fires and whether correctly, and
# (b) the precision cost MMR would have imposed on single-aspect queries had
# it run — that delta is the ablation row that justifies keeping it off by
# default.

# ---------------------------------------------------------------------------
# Stage 6 — Parent/table resolution
# ---------------------------------------------------------------------------
# Token budgeting: assemble parents ordered by rerank score, include WHOLE
# parents until the budget is reached, mark any overflow parent as truncated
# IN ITS LABEL so generation knows the text is partial — never silently cut.
# The exact budget is a tunable the eval harness sweeps (against the target
# LLM's context window). Start with a reasonable value; the doc leaves the
# exact number open as a flagged follow-up.
PARENT_TOKEN_BUDGET = 4000   # TODO: tune against target LLM context window

# Parent store location: parent_store.json written by ingestion Stage 4,
# keyed by the same IDs the child index points at. Path is relative to the
# project root (the orchestrator is run from the project root, like
# ingestion/run_pipeline.py).
PARENT_STORE_PATH = "ingestion_output/parent_store.json"   # TODO: confirm path

# Table parent = full structured table_json (the LLM reasons over clean
# row/column data, never the flattened summary that got embedded).

# ---------------------------------------------------------------------------
# Stage 7 — Overlay vs. base resolution
# ---------------------------------------------------------------------------
# Explicit label strings attached to chunks for the generation prompt.
LABEL_OVERLAY_PENDING = "[current text — ratification pending]"
LABEL_BASE_SETTLED = "[settled law since 2018]"

# When both a consolidated_base provision and an amendment_overlay provision
# cover the same topic:
#   - do NOT discard either (the reranker may order them; this stage may not
#     drop one)
#   - tag both explicitly for the generation prompt with the labels above
#   - record the pair as dual_layer=true in the trace (both chunk ids)
#   - let evaluation measure conflict-handling directly

# Open design question (carried from RETRIEVAL_PIPELINE.md / ingestion Stage 0):
# the exact prompt template that turns these dual-labeled chunks into a coherent,
# correctly-caveated answer belongs to the GENERATION stage, not retrieval.
# Retrieval's obligation is only to keep both layers present and clearly labeled.
# Flag here so it isn't lost between "retrieval done" and "generation prompt
# designed."

# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
# Every query writes one JSON trace to TRACE_DIR. The trace schema is defined
# in trace.py — this file should not redefine it. This section is just a
# reminder of what the trace must carry (mirrors RETRIEVAL_PIPELINE.md):
#
#   query_id, original, variants, flags (multi_aspect / historical),
#   per-stage ranked lists (chunk_id, score, rank),
#   fusion scores, dedup decisions (collapsed pairs + why),
#   rerank scores, mmr applied?, final context set (ids + labels + parent ids),
#   dual_layer pairs, timings per stage, fallbacks triggered
#
# This is not optional instrumentation: it is the substrate the eval harness
# consumes (RAGAS context metrics from the final context set; stage-level
# ablations from the per-stage lists) and the only way to debug a wrong answer
# down to the stage that caused it.

# TODO: add retrieval-specific dependencies to requirements.txt:
#   - langchain-ollama / ollama (already there for the dev LLM stand-in)
#   - sentence-transformers (already there; BGE-M3 + MiniLM + reranker)
#   - chromadb / langchain-chroma (already there; dense read path)
#   - BGE-Reranker-v2-M3 (sentence-transformers hub; NOT yet listed)
#   - (Qdrant client + named sparse vectors: ONLY when the sparse upgrade lands)
