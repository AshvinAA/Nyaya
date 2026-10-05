"""
============================================================================
 Nyaya - Ingestion Pipeline :: shared configuration
============================================================================
 Every tunable knob of the ingestion pipeline lives in THIS one file.

 The pipeline has one source of truth for each number so that:
   - an experiment that changes a threshold knows it changed it everywhere
   - the RAGAS evaluation harness (Stage 7+) can sweep these constants
   - no stage module hardcodes a value another module already defines

 Values mirror INGESTION_PIPELINE.md, the agreed architecture document.

 Run it:
     venv/Scripts/python ingestion/run_pipeline.py            # demo: 30 pages/doc
     venv/Scripts/python ingestion/run_pipeline.py --full     # whole corpus
============================================================================
"""

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
DOCS_DIR = "docs"                   # folder holding the raw corpus PDFs
OUT_DIR = "ingestion_output"        # folder where stage outputs are written
DB_DIR = "db/chroma_db"             # Chroma persistence dir (Stage 7)
COLLECTION = "nyaya_children"       # collection of child chunks

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEFAULT_MAX_PAGES = 30              # per-document page cap so a demo run is quick
EMBED_MODEL = "BAAI/bge-m3"         # multilingual dense+sparse model (Stage 7)
EMBED_BATCH = 32                    # chunks per forward pass of the encoder

# ---------------------------------------------------------------------------
# Stage 1 - Intake & content-type routing
# ---------------------------------------------------------------------------
TABLE_PAGE_RATIO = 0.5              # page is "table" when tables cover >= 50% area
MIXED_PAGE_RATIO = 0.08             # page is "mixed" when tables cover >= 8% area
EMPTY_PAGE_WORDS = 15               # fewer words + no tables = cover/blank page
FRONT_MATTER_MAX_LINES = 25         # leading lines scanned for a gazette masthead

# ---------------------------------------------------------------------------
# Stage 2a - Prose path (hierarchical chunking)
# ---------------------------------------------------------------------------
MIN_LEAF_TOKENS = 12                # a leaf/heading shorter than this merges
                                    # FORWARD into the next node ("PRELIMINARY"
                                    # becomes the prefix of the next chunk)
TOKEN_CAP = 350                     # token-cap fallback for oversized leaves
                                    # (the plan suggests 256-384 as start)
OVERLAP_RATIO = 0.15                # 15% overlap between split pieces

# ---------------------------------------------------------------------------
# Stage 3 - Structural validation gate
# ---------------------------------------------------------------------------
VALIDATION_STRICT = False           # True = also treat review items as failures
                                    # (CLI: --strict)
PARENTLESS_FAIL_RATE = 0.5          # a doc where MORE than this fraction of
                                    # leaves are parentless clauses has a
                                    # broken hierarchy parse -> hard failure
POLLUTION_MAX_TOKENS = 10           # page furniture is short (page numbers,
                                    # running headers); longer text is content
POLLUTION_MIN_PAGES = 3             # a short text repeated on at least this
                                    # many pages of one doc is furniture

# ---------------------------------------------------------------------------
# Stage 4 - Parent-child indexing
# ---------------------------------------------------------------------------
CHILD_EST_TOKEN_CAP = 120           # a leaf over this size is too big to be
                                    # a child -> it becomes its own PARENT and a
                                    # split-sibling set is used instead
MIN_PARENT_EST_TOKENS = 360         # smaller sections skip parent assembly
                                    # (the children already carry their full
                                    # hierarchy in path metadata)
MAX_PARENT_CHILDREN = 40            # safety valve - never assemble a parent
                                    # with more leaves than this
PARENT_TOKEN_SOFT_CAP = 5000        # hard cap on a parent's character length
                                    # (guards the LLM context window later)

# ---------------------------------------------------------------------------
# Stage 6 - Deduplication
# ---------------------------------------------------------------------------
NEARDUP_MIN_TOKENS = 6              # chunks shorter than this skip near-dup
                                    # checks (short legal-marker phrases like
                                    # "Given under my hand" dup-legally on purpose)
NEARDUP_SIM_THRESHOLD = 0.92        # cosine similarity at/above which two
                                    # chunks are the same fact restated
DEDUP_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small cached model
                                    # used ONLY to score near-dup pairs here;
                                    # the retrieval index itself uses the
                                    # BGE-M3 model in Stage 7
# Scaling note (from the plan): pairwise similarity is O(n^2) across chunks.
# At this corpus's size (thousands of chunks, not millions) that is fine.
# If the corpus ever grows, switch to minHash/LSH blocking first and run
# embedding similarity only WITHIN candidate blocks.
