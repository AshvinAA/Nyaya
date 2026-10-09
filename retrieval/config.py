"""
============================================================================
 Nyaya - Retrieval Pipeline :: shared configuration
============================================================================
 Every tunable knob of the retrieval pipeline lives in THIS one file
 (mirrors ingestion/config.py's one-source-of-truth approach), so the
 RAGAS evaluation harness can sweep any constant from a single place.

 Values mirror RETRIEVAL_PIPELINE.md, the agreed architecture document.

 Paths are project-root relative; run from the project root:
     python retrieval/run_pipeline.py --query "what is the legal wage?"
============================================================================
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Folders / stores (shared with ingestion)
# ---------------------------------------------------------------------------
DB_DIR = os.path.join(PROJECT_ROOT, "db", "chroma_db")
COLLECTION = "nyaya_children"             # dense-only child index (ingestion Stage 7)
PARENT_STORE = os.path.join(PROJECT_ROOT, "ingestion_output", "parent_store.json")
TRACE_DIR = os.path.join(PROJECT_ROOT, "retrieval_traces")

# ---------------------------------------------------------------------------
# Stage 1 - Multi-query expansion
# ---------------------------------------------------------------------------
# The demo LLM is llama3.2 via a local Ollama server (requirements.txt). It is
# deliberately weak at Bangla - Bangla->formal-legal-English register shifting
# is the riskiest LLM-dependent step in this pipeline, and rewrite quality on
# the Bangla test set is a measured eval item, not an assumption.
OLLAMA_HOST = "http://localhost:11434"
EXPAND_MODEL = "llama3.2"
MAX_VARIANTS = 3                 # official: 2-3 formal-register paraphrases
EXPAND_TIMEOUT_S = 30            # per LLM call
# the structured envelope Stage 1 must return (per RETRIEVAL_PIPELINE.md)
EXPAND_KEYS = ("original", "variants", "multi_aspect", "historical")

# ---------------------------------------------------------------------------
# Stage 2 - Filtered dense search
# ---------------------------------------------------------------------------
N_RESULTS_VARIANT = 10           # per-list depth (the plan's "per-list depth of 10")
# Default applicability filter: in_force AND not superseded, per Stage 2.
# The 2025 Ordinance survives this by design (pending_ratification, in_force).
# When Stage 1 flags historical, the constraints are SWAPPED OUT entirely and
# Stage 7's labeling takes over - never partially relaxed.
DEFAULT_FILTER = {"in_force": True, "legal_status": {"$ne": "superseded"}}
HISTORICAL_FILTER = {}           # swapped, not loosened at the edges

# ---------------------------------------------------------------------------
# Stage 3 - Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
RRF_K = 60                       # standard constant
RRF_POOL_CAP = 30                # fused pool cap before Stage 4

# ---------------------------------------------------------------------------
# Stage 4 - Retrieval-time dedup
# ---------------------------------------------------------------------------
# Same authority ranking as ingestion/deduplicator.py; resolve pairs per-query.
AUTHORITY_ORDER = {"official_gazette": 0, "official_translation": 1,
                   "unofficial_translation": 2, "reputable_secondary": 3,
                   "advocacy_report": 4, None: 9}
# Hard guard inherited from ingestion: NEVER collapse a pair across a
# different text_layer or source_act - a base provision and its 2025
# Ordinance overlay are near-identical by construction, and Stage 7 exists
# to surface both with labels.
DEDUP_CROSSES_LAYERS_GUARD = True
# near-dup scorer: cached, small, and only a similarity judge here
DEDUP_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEDUP_SIM_THRESHOLD = 0.92       # same threshold semantics as ingestion Stage 6

# ---------------------------------------------------------------------------
# Stage 5 - Reranking
# ---------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/mmarco-minilm-l6-v2"  # multilingual cross-encoder
RERANK_POOL_CAP = 25             # never spend the cross-encoder on a duplicate
FINAL_K = 5                      # final context size

# ---------------------------------------------------------------------------
# Stage 5b - Optional MMR (conditional, not default)
# ---------------------------------------------------------------------------
MMR_LAMBDA = 0.6                 # relevance/diversity balance

# ---------------------------------------------------------------------------
# Stage 6 - Parent/table resolution
# ---------------------------------------------------------------------------
PARENT_TOKEN_BUDGET = 4000       # order by rerank score, include until budget
