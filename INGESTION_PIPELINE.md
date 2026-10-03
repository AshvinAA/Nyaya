# Nyaya — Ingestion Pipeline Architecture

**Plan and architecture document for turning the raw legal corpus into a retrieval-ready, structure-preserving index.**

---

## Pipeline Overview

```
Raw documents (docs/)
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 0: Corpus consolidation &           │
│          version resolution               │
│ (ratified base + labeled overlay layers)  │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ Stage 1: Intake & content-type routing    │
│ (prose vs. table, before any chunking)    │
└───────────────────────────────────────────┘
        │                          │
        ▼                          ▼
┌───────────────────────┐  ┌────────────────────────┐
│ Stage 2a: Prose path  │  │ Stage 2b: Table path   │
│ Hierarchical,         │  │ Structured extraction, │
│ structure-aware       │  │ never flattening       │
│ chunking              │  │ (vision path deferred) │
└───────────────────────┘  └────────────────────────┘
        │                          │
        └───────────┬──────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ Stage 3: Structural validation gate       │
│ (mis-parsed chunks never reach the index) │
└───────────────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ Stage 4: Parent-child indexing            │
│ (embed small, generate on large)          │
└───────────────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ Stage 5: Metadata tagging                 │
│ (path, dates, authority, applicability)   │
└───────────────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ Stage 6: Deduplication                    │
│ (exact hash + near-dup by authority)      │
└───────────────────────────────────────────┘
                    ▼
┌───────────────────────────────────────────┐
│ Stage 7: Embedding (BGE-M3, with context  │
│ injection) & index write                  │
└───────────────────────────────────────────┘
```

**Documents in scope:** the official 2015 Labour Act gazette, the EPZ Act, the Labour Rules, the CPD wage-revision report, the Legal 500 guide, and the wage gazettes. The **consolidated base text produced in Stage 0 (Act + ratified amendments) becomes the primary prose ingestion source**, the 2025 Ordinance enters as a **labeled overlay layer**, and the raw gazettes remain in the corpus as tagged historical versions.

---

## Stage 0: Corpus Consolidation and Version Resolution

**The problem:** amendments are *deltas*, not standalone law. The 2018 amendment act and the 2025 Ordinance don't contain the current text — they contain instructions that modify it ("in section 23, after sub-section (2), insert…"). Ingesting the three gazettes side by side means the index holds three partially-overlapping versions of the same sections, with nothing that says which text is the law *today*. A worker asking about overtime pay must never retrieve the pre-2018 clause.

**The consolidation rule: only ratified law merges into the base.**

- **(a) Consolidated base text — ratified amendments only.** Maintain one manually consolidated base text: the Labour Act 2006 with the **ratified** amendment acts applied (2013 and 2018). This base is the single source of truth and the primary prose ingestion source. The corpus is small and the stakes are high: a one-time human consolidation effort eliminates an entire class of retrieval errors and removes version ambiguity from every downstream stage.
- **The 2025 Ordinance is an overlay layer, never merged while unratified.** Because its ratification is genuinely unresolved, there may be no single canonical "current" text to consolidate *into* yet — and merging an unratified ordinance into the source of truth would recreate exactly the ambiguity Stage 0 exists to eliminate. Instead, the Ordinance is ingested as a separate, clearly-labeled overlay chunk set carrying its own `in_force` / `legal_status` tags.
- **(b) Automated versioning — if consolidation is deferred.** Ingest each version separately, attach amendment-mapping metadata (`in_force`, `applies_from` / `applies_to`, `amended_by`, `text_layer`) and enforce a **"current version wins" filter** at retrieval: retrieval defaults to `in_force = true`; superseded versions are retrievable only for explicitly historical questions ("what did the law say before 2018?", "what changed in the 2025 Ordinance?").

**Layering rule at retrieval:** where the overlay and the base cover the same provision, the overlay text supersedes the base for answering — and *both are surfaced with labels*. The system says "this is the operative text; its ratification is pending," never silently picks one.

**Resolution path:** when the Ordinance's ratification resolves, the overlay is either merged into the consolidated base or dropped — and that merge **re-triggers Stage 3 validation** like any new document. Consolidation is a process with triggers, not a one-time editing task.

**Two different fields, both required:**

- `legal_status` (`ratified` / `pending_ratification` / `superseded`) is about **authority** — is this provision settled law?
- `in_force` is about **applicability** — is this the text that governs *today*?

The 2025 Ordinance makes the distinction concrete: as an overlay it is `in_force: true` (operative from promulgation) while `legal_status: pending_ratification`; the consolidated base is `in_force: true` and `legal_status: ratified`. The system must be able to say "this is the current text, and its ratification is pending" — not collapse the two facts into one flag.

---

## Stage 1: Document Intake and Content-Type Routing

Every raw document enters through a router that classifies each page or section as **either prose or table** *before any chunking happens*.

**Why this decision matters:** the two content types have fundamentally different failure modes if mishandled —

- **Prose** loses meaning when cut at the wrong boundary.
- **Tables** lose meaning when flattened into text *at all*.

**How routing works:**

- **Text-extractable PDFs:** layout-detection tools (e.g., unstructured.io's partition functions, or pdfplumber's table detection) identify table regions on a page versus body text.
- **Bangla-script gazette PDFs:** standard text extraction failed entirely (pdftotext returned garbage; the real content only appeared after rendering to an image). Here, routing must happen **visually**: render the page, then classify by layout — a dense grid of short cells vs. flowing paragraphs — rather than by extracted text content.

---

## Stage 2a: The Prose Path — Hierarchical, Structure-Aware Chunking

Prose content is split along the document's own logical hierarchy rather than by a fixed token count:

```
Chapter → Section → Sub-section → Clause
```

The chunker walks the document's actual numbering and headings, producing **one chunk per leaf node** — typically a clause or sub-section, the smallest unit that still stands on its own. **A chunk boundary never cuts across two different clauses.**

**Fallback:** only if a single leaf node is still too long after the structural split do we fall back to token-based splitting *within* it — around **256–384 tokens with 15–20% overlap** as a reasonable starting point. This is exactly the kind of parameter the RAGAS evaluation harness should sweep over empirically once it exists, rather than treating as fixed.

**Failure mode this prevents:** a worker's question about Section 23 retrieving a chunk that starts mid-sentence in Section 22.

---

## Stage 2b: The Table Path — Structured Extraction, Not Flattening

Tables never become flowing text:

| Table type | Handling |
|---|---|
| **Text-based tables** (e.g., the CPD report's 40+ wage/survey tables) | Extracted with camelot-py or pdfplumber into **structured JSON** — explicit row labels, column labels, and a data dictionary, not a paragraph |
| **Gazette wage tables** (Bangla-script OCR difficulty) | Small enough and high-stakes enough to warrant a **one-time manual transcription** into structured JSON, human-verified — rather than an automated pipeline you'd have to separately verify anyway. The data dictionary includes a Bangla→English gloss for column headers so the LLM can read the table correctly |
| **Unparseable / scanned tables** | **Deferred — see below** |

**Deferred: the vision/ColPali path (YAGNI for now).** The original plan included rendering unparseable tables as images, embedding them with a vision-capable model (ColPali or Jina-CLIP), and having the LLM read the table visually at generation time. Looking at the actual corpus, that path currently has **zero real inputs**: the CPD report's tables are text-extractable, and the gazette wage tables — the only genuinely OCR-hard tables — go to manual transcription anyway. Building it now means adding a second, multi-vector index type that no document currently needs. **Decision:** keep it documented as a fallback and activate it only if a specific document proves it's necessary. The RAGAS ablation table is the arbiter of whether it ever earns its complexity.

**Summary-embeds, JSON-generates:** each extracted table also gets a short natural-language summary generated alongside it (e.g., *"This table shows CPD's proposed minimum wage by grade, broken into basic, housing, food, medical, and transport allowances"*). The **summary is what gets embedded** for semantic search; the **full structured JSON is what gets handed to the LLM** once that summary is retrieved. This is the same principle as parent-child indexing, applied to tables instead of legal clauses.

---

## Stage 3: Structural Validation Gate

A structure-aware chunker that silently mis-parses is *worse* than a naive fixed-size splitter: the naive splitter at least announces its crudeness, while a broken hierarchy parser produces chunks that *look* authoritative and are trusted all the way into the index. So every structured parse must pass a validation gate before its chunks proceed:

- **Section-number continuity:** the section numbers extracted from the Act must form the expected continuous sequence. A missing Section 24 is a parse failure, not missing law. Expected counts and ranges are recorded once from the gazette's own table of contents during Stage 0.
- **Hierarchy integrity:** every clause's parent section must exist; sub-section/clause numbering must fit the document's own pattern; no leaf node may be parentless or attached to two parents.
- **Header/footer pollution check:** page furniture (running headers, page numbers, gazette stamps) must not survive into chunk text.
- **Table spot-checks:** extracted JSON row/column counts must match the source table's shape, and a random sample of cells is checked against the source document.
- **Output:** a per-document validation report. Documents that fail are **blocked from the index** and queued for human review — the gate fails loud, never silently.

This is cheap to build (pure assertions over data the pipeline already produces) and converts the riskiest assumption in the pipeline — "the parser walks the real structure" — into a checked invariant.

**The gate re-runs on every ingestion — it is not a one-time step.** Any corpus change re-triggers validation, most importantly Stage 0 consolidation edits: when the 2025 Ordinance's status resolves and the overlay is merged into (or dropped from) the consolidated base, the merged text passes through this same gate before it can reach the index.

**Validation pass-rate is an evaluation artifact.** Report per-document validation pass-rates alongside the RAGAS retrieval/generation scores — a concrete, measurable corpus-quality signal that pairs naturally with the ablation table in the evaluation writeup.

---

## Stage 4: Parent-Child Indexing

For both content types, what gets embedded and searched is deliberately **smaller and more precise** than what gets handed to the LLM for generation:

| | Child (embedded, searched) | Parent (returned to the LLM) |
|---|---|---|
| **Prose** | A single clause — e.g., Section 23(3), the compensation entitlement. Small and specific, so semantic search matches it precisely against a narrow query. | The full surrounding section — e.g., all of Section 23, including the misconduct exceptions in 23(1) that qualify the compensation right in 23(3). |
| **Tables** | The table's natural-language summary. | The full structured table JSON. |

**Why this exists:** legal text is full of exactly the pattern where a right is granted in one sub-section and qualified in the next. A retriever matching only on the small chunk still finds the right needle; the LLM generating the answer then sees the full haystack around that needle — so it doesn't miss the exception that changes the answer.

---

## Stage 5: Metadata Tagging

Every chunk — prose or table — carries metadata that isn't used for semantic matching but for **filtering and prioritization that semantic similarity alone can't express**:

| Field | Purpose |
|---|---|
| `source_act` | Which law (Labour Act vs. EPZ Act vs. wage gazette) — prevents citing the wrong act for the wrong worker type |
| `text_layer` | Which layer this chunk came from: `consolidated_base` (Act + ratified amendments) vs. `amendment_overlay` (the unratified 2025 Ordinance) — drives the overlay-supersedes-base rule from Stage 0 |
| `section_path` | Full hierarchical path (Chapter → Section → Sub-section → Clause), filterable |
| `topic_tags` | Thin, manually-curated topic vocabulary (`termination`, `notice_period`, `maternity`, …) layered on top of `section_path` — workers ask by colloquial topic, not by the Act's structural path, and the two aren't the same shape. Kept deliberately small; Bangla→English register bridging stays the job of multi-query expansion at retrieval, not of these tags |
| `section_number` | Precise citation; overlapping-coverage detection during corpus QA |
| `effective_date` | Which version this text belongs to, since the Act has layered amendments (2015 gazette → 2018 amendments → 2025 Ordinance) |
| `in_force` | **Applicability:** whether this is the text that governs today — deliberately distinct from `legal_status` (see Stage 0) |
| `legal_status` | `ratified` / `pending_ratification` / `superseded` — specifically addresses the 2025 Ordinance's unresolved ratification status |
| `translation_status` | `official` (the 2015 gazette) vs. `unofficial` (mccibd.org) — lets us prefer the authoritative source when two versions disagree |
| `source_type` | `primary_law` vs. `policy_advocacy` — critically separates the actual wage gazette from CPD's proposed (but not adopted) wage figures, so the retriever can never confuse the two |
| `authority_rank` | Authority tier assigned during dedup (official gazette > official translation > reputable secondary > advocacy) — lets retrieval prefer the authoritative version when several sources carry the same fact |

---

## Stage 6: Deduplication

After chunking and *before* chunks reach the vector index, a dedup pass removes genuine redundancy **without removing meaningful version history**:

- **Exact duplicates** (identical text from overlap or double-extraction): caught by **hashing normalized chunk text** and dropping exact matches.
- **Near-duplicates** (the same fact stated differently across sources — e.g., the gazette's wording vs. CPD's paraphrase of the same fact): caught by **pairwise embedding similarity above a threshold**, then resolved by the **authority ranking** behind `authority_rank`:

  ```
  official gazette > official translation > reputable secondary source > advocacy report
  ```

  The lower-authority version is **demoted / cross-referenced, not deleted** — it may still be useful context.

**Distinct from version filtering:** `legal_status` / `in_force` / `effective_date` filtering preserves genuinely different *versions of the law over time*; dedup removes accidental redundancy that adds noise without adding information. These are separate steps and must stay separate.

**Scaling note:** pairwise embedding similarity is O(n²) across chunks. At this corpus's size (thousands of chunks, not millions) that's fine. If the corpus ever grows, switch to minHash/LSH blocking first and run embedding similarity only *within* candidate blocks — same recall, a fraction of the comparisons.

---

## Stage 7: Embedding and Index Write

### Embed-time context injection

A single legal clause is short and heavily anaphoric — "said worker," "such employer," "the amount referred to in sub-section (1)." Embedded bare, a 30-word clause matches poorly against a real query. So the text that gets **embedded is not the text that gets generated on**:

- **Embedding text** = hierarchical path prefix + chunk content, e.g.

  > *Labour Act 2006 · Chapter II · Conditions of Service · Section 23 · Sub-section (3) · Compensation for injury —* followed by the clause text.

  The prefix anchors every child to its section's vocabulary, so a query sharing none of the clause's own words can still surface it.
- **Generation text** = the bare chunk (or its parent). The LLM must never see the prefix as if it were law text.
- Both variants are stored; the prefix is reconstructable from `section_path` metadata, so metadata stays the single source of truth.

*(Optional later upgrade: LLM-generated contextual summaries prepended per chunk, Anthropic-style contextual retrieval. Start with the deterministic path prefix and let the RAGAS harness decide whether the LLM-generated version earns its per-chunk cost.)*

### The embedding model

Final chunks (children, with parent-pointers and metadata attached) are embedded with a genuinely **multilingual model: BGE-M3**.

**Why BGE-M3:** it natively supports **dense, sparse, and multi-vector representations in one model** — providing the dense+sparse hybrid-search foundation the retrieval stage needs without stitching together two separate systems.

**One clarification on "sparse":** BGE-M3's sparse output is **learned lexical weights, not BM25**. Qdrant's named sparse vectors store them natively; a true BM25 side-channel is an optional addition, not a requirement of the design. (If Pinecone is chosen instead, verify its sparse-dense support for M3's output format first.)

### Where things land

| Store | Contents |
|---|---|
| **Vector store** (Qdrant / Pinecone) | Child chunk embeddings, alongside the metadata fields as **filterable attributes** |
| **Document store** | Parent chunks and structured table JSON, **keyed by the same IDs** |

---

## Design Principles Recap

1. **Consolidate before you ingest** — ratified amendments merge into one base text; unratified changes stay as labeled overlay layers and are never silently merged; authority (`legal_status`) and applicability (`in_force`) remain separate, explicit fields.
2. **Route before chunking** — prose and tables have different failure modes; treat them differently from the first step.
3. **Respect the document's own structure — and verify it** — a hierarchy parser without a validation gate is a liability; continuity checks make structure a checked invariant, not an assumption.
4. **Never flatten tables** — extract structure, embed a summary, hand the LLM the structured data.
5. **Embed with context, generate on bare text** — the hierarchical path prefix belongs in the embedding vector, never in the LLM's context window as if it were law.
6. **Embed small, generate large** — parent-child indexing decouples retrieval precision from generation context.
7. **Metadata carries what similarity can't** — path, legal status, applicability, authority.
8. **Dedup noise, preserve history** — remove accidental redundancy; keep genuinely different versions of the law.
9. **Sweep, don't guess** — chunk-size and overlap parameters are starting points to be tuned by the RAGAS harness, not constants.
10. **Defer speculative complexity** — every component must earn its place with a real input in this corpus; the ablation table is the arbiter (see the deferred vision path).
