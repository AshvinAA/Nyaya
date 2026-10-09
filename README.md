# Nyaya

**A voice-first RAG system for Bangladesh's Readymade Garment (RMG) workers — and a technical demonstration of a properly-engineered, rigorously-evaluated retrieval-augmented generation pipeline.**

---

## The Problem

RMG factory jobs in Bangladesh are legally free to apply for, but middlemen (*dalals*) routinely extort workers at the factory gate — demanding bribes of several thousand taka just to get an application in front of HR, or lying about starting wages to pocket the difference.

This preys specifically on workers who can't verify the truth for themselves: no smartphone, no internet access, no way to check what the law actually says or what a factory is actually hiring for. Existing job platforms assume exactly the digital literacy and access that the most vulnerable rural migrant workers don't have.

## The Solution

A worker calls a toll-free number from any basic phone and speaks in natural, colloquial Bangla. The system:

1. **Understands intent** — distinguishing a legal question (*"is this fee legal?"*) from a job search (*"who's hiring lockstitch operators in Gazipur?"*), or both at once.
2. **Answers legal questions with grounded, citable accuracy** — retrieving the actual relevant clause from Bangladesh's Labour Act, Labour Rules, and wage gazettes, rather than generating a plausible-sounding but unverified answer.
3. **Connects workers directly to verified, compliant factories** — bypassing the gatekeeper entirely by pushing the worker's information straight to a named factory HR contact once an interview is confirmed, removing the gate-fee leverage point altogether.

The core mechanism is **removing information asymmetry**: a dalal's power comes entirely from being the only accessible source of information a worker has. Nyaya makes the legal wage, the real hiring process, and a legitimate point of contact something any worker can access directly, for free, from anywhere.

## What This Project Is Right Now

This is explicitly scoped as a **demo and proof-of-pipeline**, not a production deployment. The goal is to prove two things simultaneously:

- **The impact claim** — that an AI pipeline like this can meaningfully attack a real, large-scale socioeconomic problem in Bangladesh: not by replacing enforcement or policy, but by giving workers direct, verifiable access to information and a legitimate hiring channel that currently exists only through corrupt intermediaries.
- **The technical claim** — that this is a properly engineered, rigorously evaluated RAG system, not a thin wrapper around an LLM call. Every architectural decision exists because the corpus demanded it: legally consequential numeric tables, law that is actively amended and contested, and a real register gap between written legal English/Bangla and spoken colloquial Bangla.

The two claims reinforce each other rather than compete — the corpus is genuinely messy and high-stakes in exactly the ways that justify the pipeline's engineering, not as a checklist of impressive-sounding techniques.

⚠️ **Only the ingestion side of this claim exists in code today.** See [Current implementation status](#current-implementation-status) for the honest breakdown — it is the most accurate section of this README.

## Architecture

The corpus is real primary sources in `docs/`: the Bangladesh Labour Act 2006 (English, up to the 2018 amendments), the Labour Rules 2015 (unofficial English translation), the 2015 wage gazette and 2025 amendment gazette (Bangla originals), the Legal 500 Bangladesh employment guide, ILO Committee on Freedom of Association materials, and the CPD 2023 RMG wage-revision report.

The full pipeline is designed across two architecture documents, and each stage is implemented only where this repo says so:

| | Document | Status |
|---|---|---|
| Ingestion (Stages 0–7) | [INGESTION_PIPELINE.md](INGESTION_PIPELINE.md) | **Stages 1–7 implemented end to end** |
| Retrieval (Stages 1–7) | [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) | **Designed and specified; not yet built** |
| Evaluation (RAGAS + LangSmith + ablation table) | (spec lives inside the two docs above) | **Not started** |

### Ingestion — what actually runs today

One module per stage, driven by [`ingestion/run_pipeline.py`](ingestion/run_pipeline.py):

- **Stage 1 — content-type routing.** Every PDF page is classified (prose / table / mixed / empty / *visual_routing_required*) **before any chunking happens**. Bangla-gazette pages whose text extraction collapses into garbage (transliterated junk, repeated glyph spam, legacy Bijoy-font mojibake) are caught by statistical heuristics and flagged for manual handling instead of being silently ingested. Gazette mastheads are extracted as `kind=front_matter` blocks with `retrievable: false` — document identity feeds document-level metadata and is never an answer candidate.
- **Stage 2a — hierarchical prose chunking.** Walks the legal document's own structure (Chapter → Section → Sub-section → Clause → proviso), never cutting across a clause boundary. Documents without legal numbering fall back to sentence grouping with a token cap and overlap. Bare headings below a minimum token count (12) merge *forward* into the next node, never standing alone as useless candidates.
- **Stage 2b — structured table extraction.** Tables become structured JSON (col/row labels, coerced numeric cells, caption capture) — never flattened prose. Each table gets a rule-based natural-language summary: **the summary is what gets embedded, the JSON is what the LLM would reason over**. Unreadable grids retry once with text-strategy settings, then quarantine to the manual-review queue — never the index.
- **Stage 3 — structural validation gate.** Section-number continuity, hierarchy integrity (no parentless leaves, no double parents), header/footer pollution, empty-header and flattened-cell table checks, non-empty node IDs. A silently mis-parsed chunk is treated as more dangerous than an obviously crude one, so a document that fails the gate is **blocked from the index**, loudly, not quietly ingested wrong.
- **Stage 4 — parent-child indexing.** Small, precise children are embedded and searched; their full parent sections (all of Section 23, including the misconduct exceptions that qualify the compensation right in 23(3)) are what the LLM actually reasons over. Oversized leaves become their own parent with split-sibling children; everything is keyed by shared IDs in `parent_store.json`.
- **Stage 5 — metadata tagging.** Every chunk carries `source_act`, `source_type`, `text_layer`, `legal_status`, `in_force`, `authority_rank`, `translation_status`, `section_path`, `topic_tags`, `effective_date` — the fields semantic similarity alone can't express. This is the machinery that can always tell the actual wage gazette from CPD's *proposed* wages, and ratified law from the pending-ratification 2025 Ordinance (`in_force: true` + `legal_status: pending_ratification` — two separate facts, never collapsed).
- **Stage 6 — deduplication.** Exact-hash dedup + near-dup via embedding similarity (threshold 0.92, small cached MiniLM scorer), resolved by authority ranking: `official gazette > official translation > reputable secondary > advocacy`. Losers are demoted/cross-referenced (`retrievable: false`, `superseded_by`), never deleted. The pass **refuses to merge anything across a different `text_layer` or `source_act`** — dedup removes noise, version filtering is a separate concern, and the base-text vs. 2025-overlay distinction survives by construction.
- **Stage 7 — embedding & index write.** Every child is embedded with **context injection**: the hierarchical path prefix becomes part of the embedding vector, never part of the generation text — the LLM never sees the prefix as if it were law. Model: **BGE-M3** (dense, 1024-dim). Rebuilt from scratch on every run, never incrementally stale.

The metadata registry in `ingestion/doc_metadata.py` is the code-side implementation of Stage 0's *layering rule*: the 2025 Ordinance enters as a `text_layer: amendment_overlay` chunk set, tagged `legal_status: pending_ratification` — designed so retrieval surfaces overlay and base **together with explicit labels**, never silently picks one. A physical consolidated base text has not been built yet (see status below).

📄 Full architecture decisions and rationale: [INGESTION_PIPELINE.md](INGESTION_PIPELINE.md) · implementation details: [`ingestion/README.md`](ingestion/README.md)

### Retrieval — designed in full, not yet coded

[RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) specifies the full retrieval architecture: multi-query expansion (colloquial-Bangla → formal-legal-English register bridging) → hybrid dense+sparse search with `in_force`/`legal_status` metadata pre-filtering → reciprocal rank fusion → retrieval-time dedup (distinct from the ingestion-time dedup, with the same hard cross-layer guard) → BGE-Reranker-v2-M3 reranking against the *original* query → conditional MMR for genuinely multi-aspect queries → parent/table resolution → overlay-vs-base dual-labeling, never silent conflict resolution. Every query produces a per-stage JSON trace as the evaluation substrate, and every stage has a defined, loudly-logged fallback ("convenience features may degrade; legal-applicability filters may not").

**No code in this repo implements any of that yet.** The demo index is dense-only Chroma; the sparse half of the hybrid design (BGE-M3's learned lexical weights in Qdrant named vectors — *not* BM25) is the first upgrade on the path, and the retrieval design deliberately swaps it in without changing any downstream code.

### Cross-lingual design (planned)

The legal corpus is ingested in English (official translations where they exist). The live path will be Bangla input → multi-query expansion against the English corpus → English retrieval → the LLM reasons over English context and generates directly in Bangla — no English-then-translate hop — guarded by a number/entity-preservation check, since wage figures surviving a cross-lingual generation step is the single data point that matters most.

## Current implementation status

The honest audit, verified against the code and the artifacts of a real run in `ingestion_output/`:

**✅ Working end to end — ingestion Stages 1–7.** The `ingestion/` package (14 modules, one per stage) runs the full corpus through routing, dual-path chunking, the validation gate, parent-child indexing, metadata tagging, authority-resolved dedup, and BGE-M3 embedding into Chroma. The committed `index_manifest.json` shows a real build: **192 children in → 170 indexed** (22 excluded as front matter or deduplicated), 1024-dim BGE-M3, Chroma collection `nyaya_children`, flat filterable metadata. The committed `validation_report.json` and `dedup_report.json` show the gate and dedup pass actually ran and are auditable — 12 exact + 10 near-duplicate demotions, 1 table quarantined to manual review.

**⚠️ Partially implemented:**
- **Stage 0 (corpus consolidation)** exists only as the code-side metadata registry (`text_layer`, `legal_status`, `in_force` per document), not as a physically consolidated base text. The Act with ratified amendments is represented by the English Act PDF as the base document; a true hand-consolidated merged text is not yet built.
- The **dense-only demo index** (Chroma) is short of the plan's dense+sparse hybrid target (BGE-M3 sparse in Qdrant named vectors) — acknowledged in both the ingestion and retrieval docs, deliberately structured so the upgrade changes only ingestion Stage 7.

**🚧 Designed but not implemented:**
- **The entire retrieval pipeline** — multi-query expansion, RRF fusion, retrieval-time dedup, reranking, MMR, parent resolution, overlay labeling, the per-query trace. An earlier retrieval experiment (Ollama llama3.2 chat, hybrid search, a reranker draft) existed in this repo's history and was deliberately cleaned out; the surviving artifact is the strategy document [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md).
- **Voice (Bangla ASR/TTS)** — explicitly deferred until the text pipeline is solid and evaluated. Not a stretch goal; a deliberate scope line.

**❌ Not started:**
- **The evaluation harness** — no RAGAS or LangSmith integration, no hand-verified test query set (the input everything downstream depends on), no ablation config, no custom metrics (number/entity preservation, conflict-handling correctness, advocacy-vs-primary-law discrimination), no judge-model validation for Bangla output. This is explicitly a named prerequisite in the plans: the test set is to be built *before* retrieval Stage 1, since every stage is measured against it.
- Factory/vacancy matching (PostgreSQL structured data), LangGraph orchestration, and the LLM generation stage itself.

**The currently implemented system scope is: PDFs in → validated, structure-preserving, deduplicated, metadata-tagged child chunks + parent store + vector index out.** No query path, no generation, no answers yet. Everything described in the problem and solution sections above is the target this pipeline is being built toward, in that order.

## Tech stack

As actually used in code (`requirements.txt`), with planned-but-not-yet-integrated pieces marked:

**Implemented today:** Python 3 + LangChain community connectors · `pdfplumber` (extraction, table detection) · `sentence-transformers` (BGE-M3 for Stage 7 embeddings; MiniLM as the dedup scorer and the loud fallback embedder) · `langchain-chroma`/`chromadb` (dense-only vector store, collection `nyaya_children`) · `langchain-text-splitters` (token-cap fallback) · `langchain-huggingface` · `python-dotenv` · optional local Ollama client with `langchain-ollama` (chat model: **`llama3.2` — the current dev stand-in, deliberately weak at Bangla; already flagged in the retrieval docs as the riskiest LLM-dependent step**).

**Designed, not yet integrated:** Qdrant (sparse named vectors for the dense+sparse hybrid) · BGE-Reranker-v2-M3 (cross-encoder reranking; the earlier draft was removed in this repo's history) · a production LLM replacing the local dev stand-in (Groq Llama 3.3 70B / Gemini Flash per the plans — *not verified against code, since no such integration exists yet*) · LangGraph (orchestration) · PostgreSQL (factory/vacancy structured data) · Groq Whisper (ASR) + TTS · RAGAS + LangSmith (evaluation).

## Setup / how to run

The only runnable pipeline right now is **ingestion**.

```bash
# 1. Create and use a Python venv
python -m venv venv
source venv/Scripts/activate          # Git Bash / Windows: venv/Scripts/python

# 2. Install pinned dependencies
venv/Scripts/python -m pip install -r requirements.txt

# 3. (Optional) the demo LLM for later stages — not needed for ingestion itself
ollama pull llama3.2

# 4. Run the ingestion pipeline end to end, from the project root
venv/Scripts/python ingestion/run_pipeline.py            # demo: 30 pages/doc
venv/Scripts/python ingestion/run_pipeline.py --full     # whole corpus
venv/Scripts/python ingestion/run_pipeline.py --stages 123   # skip parents/dedup/embed
venv/Scripts/python ingestion/run_pipeline.py --no-embed # chunks only, no vector DB
venv/Scripts/python ingestion/run_pipeline.py --strict   # Stage 3: review items block docs
```

 Outputs land in `ingestion_output/` (per-stage JSON reports: routing, prose chunks, extracted tables, quarantine flags, doc metadata, validation, parent store, dedup audit, index manifest) and the Chroma index under `db/chroma_db/`. A committed set of outputs from a previous full run is in `ingestion_output/` for inspection without re-running.

 The single-file predecessor (`legacy/ingestion_stage1_2.py`) is kept for reference only — superseded by the `ingestion/` package.

## Roadmap

Grounded in the actual gaps found in the audit, in dependency order:

1. **Build the hand-verified test query set first** (single-provision lookups, table/numeric, register-gap, multi-aspect vs. single-aspect control pairs, historical, dual-layer/overlay-conflict, advocacy-vs-primary-law) — the plans make this the prerequisite for both retrieval and evaluation, not the other way around.
2. **Implement retrieval Stages 1–7** against the dense-only demo index, exactly as [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) specifies — expansion → fuse → dedup → rerank → parent resolution → overlay labeling, with per-query JSON traces.
3. **Add the evaluation harness** (RAGAS core metrics + the custom metrics RAGAS can't cover + LangSmith trace inspection), with the config-object ablation mechanism — measure every stage, not assume it; validate the RAGAS judge itself on Bangla output before trusting any Bangla ablation row.
4. **Stand up the sparse half of hybrid search** (BGE-M3 lexical weights in Qdrant named vectors) when dense-only recall on the test set falls short — retrieval code is designed to absorb it with no downstream change.
5. **Build the generation stage** and resolve the carried-open prompt-design question for dual-labeled overlay-vs-base chunks.
6. **Hand-consolidate the base text** (Act + ratified amendments) so Stage 0 stops living only in the metadata registry.
7. **Only then: voice.** Bangla ASR/TTS layered on top of a pipeline that is already independently validated on text — the plan's deliberate deferral, not an oversight.
8. Factory/vacancy structured matching (PostgreSQL) and the direct-to-HR application path — the part of the concept that removes the dalal's leverage entirely.

## Scope note

**This is a demo and proof-of-pipeline, not production software and not legal advice.** Ingestion is real and working; retrieval is specified; evaluation, generation, and voice are pending. The 2025 amendment ordinance's ratification status is genuinely unresolved and is tracked explicitly in the ingestion metadata (`legal_status: pending_ratification`) so any future generation answers "this is the operative text, and its ratification is pending" — not settled law. Workers facing live disputes should be routed to human/NGO support.
