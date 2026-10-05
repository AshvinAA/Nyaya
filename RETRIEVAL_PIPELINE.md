# Nyaya — Retrieval Pipeline Architecture

**Plan and architecture document for the retrieval layer, to be read alongside [`INGESTION_PIPELINE.md`](INGESTION_PIPELINE.md).**

---

## Scope note — read this first

**Bangla ASR / voice integration is explicitly out of scope for this stage.** We are deliberately deferring the full voice-to-RAG-to-voice pipeline until the core RAG method (ingestion → retrieval → generation → evaluation) is solid and measured on its own. Right now, "query" means **typed or transcribed text** — English or Bangla — not live audio.

This matters for how retrieval should be built and tested right now:

- Query variants below are developed and evaluated against a **typed test query set**, not live ASR output.
- Any mention of "the Bangla query" in this document means **Bangla text input**, not a voice transcript — there is no ASR-specific noise (misheard numbers, dialect variation, audio artifacts) to handle yet, and we do not pre-optimize for it.
- Voice (Groq Whisper ASR + TTS) gets layered on top of this pipeline later, per the project roadmap — it is an I/O layer added once retrieval + generation are independently validated on text.

Do not add ASR-specific logic, error-correction, or assumptions into the retrieval pipeline at this stage. If something in this design *will* need revisiting once real ASR transcripts enter the picture, it is listed in **Flagged follow-ups** at the end rather than built for now.

---

## Where this fits

This document covers **retrieval only** — what happens between "a query arrives as text" and "a final, labeled context set is handed to the generation step." It assumes the ingestion pipeline (`INGESTION_PIPELINE.md`) has already produced:

- Embedded child chunks (BGE-M3) with `parent_id` pointers in the vector store, alongside their metadata. The declared `filterable_fields` written by ingestion Stage 7 are: `doc`, `kind`, `node`, `source_act`, `source_type`, `legal_status`, `in_force`, `authority_rank`, `translation_status`, `text_layer`, `effective_date`, `section_number`, `section_path`, `parent_id`. (`topic_tags`, `page`, `est_tokens`, `retrievable`, `chunk_id` are stored on every chunk too, but are not part of the declared filter set.)
- Parent chunks (prose sections) and structured table JSON in the document store (`parent_store.json`), keyed by the same IDs as the children.
- Ingestion-time dedup already applied: exact and near-duplicate chunks were demoted (`retrievable: false`) and **excluded from the index** — with the explicit guard that nothing was merged across a different `text_layer` or `source_act`.

**Current implementation status (read honestly):** the demo index is **dense-only** (Chroma, cosine, collection `nyaya_children`). The ingestion spec's target is dense + sparse (BGE-M3's learned lexical weights in Qdrant named sparse vectors — **not BM25**), and the sparse upgrade is a documented ingestion-side change that swaps no retrieval code below: every stage after Stage 2 consumes ranked lists and doesn't care how many retrieval modes produced them. Until the sparse index exists, Stage 3 fuses across *variants only* (2–3 lists instead of 4–6) and the rest of the pipeline is unchanged.

---

## Pipeline Overview

```
Query (text — English or Bangla)
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 1: Multi-query expansion            │
│ LLM rewrites into 2-3 formal-register     │
│ variants. One structured output:          │
│   variants + multi_aspect flag            │
│   + historical flag                       │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 2: Hybrid search (BGE-M3)           │
│ Dense + sparse, per variant               │
│ Metadata pre-filter: in_force=true,       │
│ legal_status ≠ superseded (relaxed only   │
│ when Stage 1 flagged historical)          │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 3: RRF fusion                       │
│ Merge ranked lists across variants ×      │
│ dense/sparse modes                        │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 4: Retrieval-time dedup             │
│ Near-dup collapse by authority_rank       │
│ (distinct from ingestion-time dedup —     │
│ see note below)                           │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 5: Reranking (BGE-Reranker-v2-M3)   │
│ Cross-encoder re-score vs. ORIGINAL query │
│ (not the expanded variants)               │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 5b: Optional MMR                    │
│ Only applied if Stage 1 flagged the query │
│ as multi-aspect — not run by default      │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 6: Parent/table resolution          │
│ Fetch parent chunk or table JSON for each │
│ surviving child — LLM never sees bare     │
│ child chunks                              │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 7: Overlay vs. base resolution      │
│ Apply text_layer precedence; BOTH base    │
│ and overlay surfaced with explicit labels │
│ when they cover the same provision —      │
│ never silently pick one                   │
└───────────────────────────────────────────┘
        │
        ▼
Final context set → generation (text output for now, Bangla or English —
                     TTS is a later, separate layer)
```

---

## Stage 1: Multi-query expansion

The LLM rewrites the input query into 2–3 paraphrased variants, shifting toward the corpus's formal legal register. This bridges the gap between how a worker phrases a question colloquially and how the Labour Act itself is worded — the Act says "recruitment fee" or "placement charge," not "dalal wants money."

**One stage, one structured output.** This stage is the pipeline's single query-understanding point, so everything downstream needs from it comes out of one JSON envelope:

```
{
  "original":      "<the query as received>",
  "variants":      ["<formal-register paraphrase 1>", "<...2>", "<...3>"],
  "multi_aspect":  true | false,     // enables Stage 5b (MMR)
  "historical":    true | false      // relaxes Stage 2's filter
}
```

- `multi_aspect` is set when generating variants surfaces that the question has genuinely distinct aspects (e.g., "is this fee legal AND what's the real wage"). That flag is what conditionally enables Stage 5b — MMR must not run by default on single-aspect queries, where it hurts precision rather than helps.
- `historical` is set when the question is explicitly about law *over time* ("what did the law say before 2018?", "what changed in the 2025 Ordinance?"). Stage 2 consumes it; nothing else in the pipeline guesses at history.
- A reserved `asr` field will eventually carry transcript-cleanup signals — **deliberately absent now** (see Flagged follow-ups).

**Model choice is a tunable, not a given.** The demo LLM is llama3.2 via Ollama (`requirements.txt`); it is weak at Bangla, and Bangla→formal-legal-English register shifting is the riskiest LLM-dependent step in this pipeline. Rewrite quality on the Bangla test set is a measured eval item, not an assumption (see Evaluation hooks). The fallback path if expansion fails is defined in the degradation policy below.

**Deferred for later:** once real ASR is in the loop, this stage may also need to absorb transcript-cleanup duties (garbled numbers, dialect normalization) before expansion. Not needed now — typed/clean text input only.

---

## Stage 2: Hybrid search, with metadata pre-filtering

Each query variant runs against the BGE-M3 index using both dense and sparse representations (sparse per the upgrade path above; dense-only until then).

**Metadata filtering happens here, not as an afterthought.** The index only contains chunks that already passed the ingestion gate — `retrievable: true`, no front matter, no demoted duplicates — so the filter below is the *entire* applicability control:

- **Default filter:** `in_force = true` AND `legal_status ≠ superseded`.
- **Historical relaxation:** when Stage 1 flagged `historical`, the default filter is **swapped**, not loosened at the edges: the `in_force` / `legal_status` constraints are dropped entirely and Stage 7's labeling takes over the job of saying which text governed when. Partial relaxation ("allow superseded but still require in_force") would be incoherent — a superseded provision is by definition not in force.
- **The 2025 Ordinance survives the default filter by design:** it is `in_force: true` with `legal_status: pending_ratification`, so it competes normally in default mode — and Stage 7 labels it. The filter's job is applicability, not ratification status; conflating them would silently hide the current operative text.

This is the retrieval-side half of the ingestion doc's Stage 0 overlay rule.

---

## Stage 3: RRF fusion

With 2–3 variants × 2 retrieval modes (dense/sparse), expect 4–6 ranked lists per query. Merge by rank position (not raw score, since dense and sparse scores aren't on the same scale):

```
score(chunk) = Σ 1 / (k + rank_in_list)   — summed across all lists the chunk appears in
```

with the standard constant **k = 60** and a per-list depth of 10 as starting points.

Two properties worth making explicit:

- **Same-ID fan-out merges for free.** If the same chunk surfaces from four of six lists, RRF sums four contributions — that is a *relevance signal*, not a bug. The double-counting problem Stage 4 addresses is a different one (distinct chunks, same fact).
- **Consensus beats luck.** Chunks that rank well across multiple variants and modes surface to the top — genuine relevance rather than a single lucky embedding match.

The fused pool is capped (top ~30 by RRF score) before Stage 4 — enough to survive dedup and reranking with a real final-k, small enough to keep the cross-encoder cheap.

---

## Stage 4: Retrieval-time dedup — NOT the same as ingestion-time dedup

Ingestion-time dedup (ingestion doc, Stage 6) is a **static, one-time** corpus cleanup. This is a **dynamic, per-query** problem, and the two must never be conflated.

**What this stage actually does:**

1. **Distinct-ID near-dup collapse.** Multi-query fan-out can surface two *different* chunk IDs carrying the same fact from different angles — and the final context budget is small (top-k of ~5), so a duplicate burns a slot that should carry new evidence. Ingestion dedup removed same-layer, same-act duplicates before indexing; what can still co-occur here are pairs ingestion *refused to merge* (different `source_act` or `text_layer`) plus threshold stragglers. Resolve by the same authority ranking as ingestion (`authority_rank`: official gazette > official translation > reputable secondary > advocacy), applied per-query instead of once over the corpus.
2. **Hard guard, inherited from ingestion:** **never collapse a pair across a different `text_layer` or `source_act`.** A base-text provision and its 2025-Ordinance overlay are near-identical by construction — and Stage 7 exists precisely to *surface both with labels*. If this stage collapsed them, it would destroy the very thing the overlay rule protects. When in doubt, leave both in and let the reranker and the small final-k do the trimming.
3. **Runs before reranking on purpose:** the cross-encoder is the most expensive call in the pipeline — never spend it re-scoring a duplicate.

Exact-ID duplicates need no handling at this stage at all: RRF already merged them into one entry.

---

## Stage 5: Reranking

**BGE-Reranker-v2-M3** re-scores the deduplicated, fused candidate set directly against the **original** input query (not the Stage 1 expanded variants) using a cross-encoder — more expensive, more accurate than embedding similarity alone. Produces the final top-k (~5).

**Why against the original:** the expanded variants exist to *widen the net* in search; the reranker's job is to judge relevance to what the user actually asked. Re-ranking against a variant would optimize for the paraphrase, not the question.

Practical notes: the reranker is multilingual (Bangla queries score natively — no translation step), it is the pipeline's latency bottleneck on CPU (hence the Stage 4 pool cap of ~25 into it), and if the model cannot be loaded, the RRF order stands and the failure is recorded in the trace — retrieval never hard-fails because an optional re-scorer is down.

---

## Stage 5b: Optional MMR (conditional, not default)

Only runs if Stage 1 flagged the query `multi_aspect`. Maximal Marginal Relevance (lambda ≈ 0.6 against the BGE-M3 dense vectors already computed in Stage 2) diversifies the top-k to avoid returning several near-identical chunks about one narrow aspect when the question needed coverage of two different things.

**Do not run on every query** — on a single-aspect legal question, MMR can reduce precision by deliberately demoting the most relevant chunk in favor of diversity that wasn't needed. This conditional trigger is itself an eval item: the harness must measure (a) how often the flag fires and whether it fires *correctly*, and (b) the precision cost MMR would have imposed on single-aspect queries had it run — that delta is the ablation row that justifies keeping it off by default.

---

## Stage 6: Parent/table resolution

For each surviving top-k **child** chunk, fetch from the document store via the child's `parent_id`:

- Its **parent section** (prose) — e.g., matched on Section 23(3) but handing the LLM all of Section 23, including the misconduct exceptions in 23(1) that qualify the compensation right in 23(3). Parents are stored in `parent_store.json` keyed by the same IDs the index points at; self-parented chunks (sentence-fallback pieces with no section context) resolve to themselves, honestly.
- Its **full structured table JSON** (tables) — the LLM reasons over clean row/column data, never a flattened table-as-prose summary.

The LLM never sees bare, context-stripped child chunks.

**Token budgeting is the open issue here, not resolution mechanics:** parents were built with a soft cap and a part-splitting rule at ingestion, but assembling 3–5 full sections plus a table JSON can still overflow a small local LLM's context. The starting policy: order parents by rerank score, include whole parents until the budget is reached, and mark any overflow parent as truncated *in its label* so generation knows the text is partial — never silently cut. The exact budget is a tunable the RAGAS harness sweeps (see Flagged follow-ups).

---

## Stage 7: Overlay vs. base resolution

Implements the ingestion doc's Stage 0 layering rule — "overlay supersedes base, but both are surfaced with labels, never silently pick one" — at the point where the context set is finalized.

If the fetched parents include both a base-text provision (`text_layer: consolidated_base`) and an overlay provision (`text_layer: amendment_overlay`, e.g. the 2025 Ordinance) covering the same topic:

- **Do not discard either.** The reranker may order them; this stage may not drop one.
- Tag both explicitly for the generation prompt, e.g.:
  - `[current text — ratification pending]` for the overlay
  - `[settled law since 2018]` for the base
- This lets generation produce something like: *"under a 2025 amendment currently in effect but not yet ratified, X — this replaces the previous rule that said Y"* — rather than silently picking one version and stating it as uncontested fact.

The retrieval trace records the pair (`dual_layer: true`, with both chunk IDs), so evaluation can measure conflict-handling directly instead of inferring it from answers.

**Open design question, not yet resolved**: the exact prompt template that turns these dual-labeled chunks into a coherent, correctly-caveated answer belongs to the generation stage, not retrieval. Flagging it here so it isn't lost between "retrieval is done" and "generation prompt is designed."

---

## Failure and degradation policy

Every stage has a defined fallback, and every fallback is **logged, never silent** — the same fails-loud ethos as the ingestion gate:

| Failure | Fallback |
|---|---|
| Stage 1 LLM unavailable / malformed JSON | Run with the original query as the sole variant; all flags `false` (filter stays default-strict — never widen legality filters on a degraded run) |
| Sparse mode unavailable | Dense-only lists (current default anyway; no code change) |
| Reranker load failure | RRF order stands; `reranker: "fallback_rrf"` in the trace |
| MMR requested but vectors unavailable | Skip MMR, note in trace — diversity is optional, correctness filters are not |
| Parent ID missing from document store | Return the child with a `[bare chunk — parent missing]` label and a trace warning; this should be impossible by construction (Stage 4 ingestion guarantees the pointer), so its occurrence is a bug alarm, not a handled case |

The asymmetry is deliberate: **convenience features may degrade; legal-applicability filters may not.**

---

## Observability: the retrieval trace

Every query writes a JSON trace to `retrieval_traces/` — the retrieval-side twin of the ingestion pipeline's run reports:

```
query_id, original, variants, flags (multi_aspect / historical),
per-stage ranked lists (chunk_id, score, rank),
fusion scores, dedup decisions (collapsed pairs + why),
rerank scores, mmr applied?, final context set (ids + labels + parent ids),
dual_layer pairs, timings per stage, fallbacks triggered
```

This is not optional instrumentation: it is the substrate the evaluation harness consumes (RAGAS context metrics are computed from the final context set; stage-level ablations are computed from the per-stage lists) and the only way to debug a wrong answer down to the stage that caused it.

---

## Evaluation hooks & the ablation table

The project's technical claim is a measured pipeline, so every stage ships with the metric that would catch it failing, and an ablation row that proves it earns its complexity:

| Stage | Primary metric | Ablation row (remove/reduce the stage → measure) |
|---|---|---|
| 1 — Multi-query | Recall@k of the fused pool vs. single-query baseline; Bangla rewrite quality spot-check set | Variants off → recall delta (register bridging value) |
| 2 — Hybrid + filter | Recall@k; filter-correctness unit checks (no superseded text in default-mode traces, Ordinance present) | Dense-only vs. +sparse (the upgrade path's justification) |
| 3 — RRF | nDCG@k vs. single best list | RRF off (use one dense list) |
| 4 — Retrieval dedup | Duplicate rate in final context; precision delta | Dedup off → wasted slots / double-counted evidence |
| 5 — Rerank | nDCG@k / MRR vs. RRF order | Reranker off (RRF order straight to final-k) |
| 5b — MMR | Aspect-coverage recall on the multi-aspect subset; precision delta on single-aspect subset | MMR forced on / forced off |
| 6 — Parent resolution | Faithfulness & answer correctness (RAGAS) with parents vs. bare children | Bare children to the LLM |
| 7 — Overlay labels | Conflict-handling correctness on the base-vs-overlay test items | Labels off → does the answer state pending law as settled? |

The typed Bangla/English test query set — with gold `section_number` labels where the answer is a specific provision — is the input this whole table depends on, and building it comes *before* building Stage 1, since every stage above is measured against it.

---

## Output of this pipeline

A final, small set of **labeled, parent-resolved, deduplicated context chunks** — ready to hand to the generation step, with every per-query decision recorded in the trace. No ASR, no TTS, no voice-specific handling anywhere in this document. Text in, labeled context out.

---

## Design Principles Recap

1. **ASR is deferred, deliberately** — build and evaluate retrieval against typed/clean text first; voice is a later I/O layer, not a current concern.
2. **Expand before you search** — bridge colloquial-vs-legal register with multi-query, not by hoping embeddings generalize across it.
3. **One query-understanding point** — Stage 1's single structured envelope (variants + flags) is the only place query intent is interpreted; everything downstream consumes the envelope, never re-guesses.
4. **Filter at the source** — `in_force` / `legal_status` filtering happens inside hybrid search, not as a post-hoc check; and the filters may degrade *last*, never first.
5. **Fuse by rank, not raw score** — dense and sparse scores aren't comparable; RRF sidesteps that.
6. **Dedup twice, for different reasons** — ingestion-time dedup cleans the corpus once; retrieval-time dedup stops multi-query fan-out from spending scarce final-k slots on the same fact per-query — and inherits ingestion's guard: never merge across `text_layer` / `source_act`.
7. **Rerank against the real query** — the expanded variants exist to widen the net in search; the reranker's job is to judge relevance to what the user actually asked.
8. **MMR is conditional, not default** — only for genuinely multi-aspect queries, flagged upstream.
9. **Never return a bare child chunk** — parent/table resolution is mandatory before generation.
10. **Never silently resolve a legal conflict** — overlay vs. base ambiguity gets surfaced with labels, not collapsed.
11. **Degrade loudly** — every stage has a named fallback; convenience degrades, correctness filters don't; everything lands in the trace.
12. **Measure every stage** — each stage exists because an ablation row says so; if the row doesn't justify it, the stage goes.

---

## Flagged follow-ups (revisit, don't build now)

- **ASR-specific handling** — transcript noise, number normalization, dialect variation entering Stage 1; the reserved `asr` field in the envelope exists for this. Revisit when voice lands.
- **Generation prompt template for dual-layer chunks** — carried open question from ingestion Stage 0 and Stage 7 above; belongs to the generation stage.
- **Bangla rewrite quality of the demo LLM** — llama3.2 is weak at Bangla; if the Bangla spot-check set shows poor register shifting, options are a stronger local model or a two-step translate-then-expand path. Decided by measurement, not preference.
- **Sparse index upgrade trigger** — when dense-only recall@k on the test set falls short of the hybrid target, stand up Qdrant named sparse vectors (BGE-M3 lexical weights); retrieval code is written to absorb it with no redesign.
- **Parent token budget** — exact context budget and truncation-labeling policy for Stage 6; sweep via RAGAS once the harness exists.
- **MMR trigger precision** — collect real multi-aspect queries over time and re-check how often the Stage 1 flag fires correctly.
