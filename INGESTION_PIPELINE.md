# Nyaya — Ingestion Pipeline Architecture

**Plan and architecture document for turning the raw legal corpus into a retrieval-ready, structure-preserving index.**

---

## Pipeline Overview

```
Raw documents (docs/)
        │
        ▼
┌─────────────────────────────────────┐
│ Stage 1: Intake & content-type      │
│          routing (prose vs. table)  │
└─────────────────────────────────────┘
        │                          │
        ▼                          ▼
┌───────────────────────┐  ┌────────────────────────┐
│ Stage 2a: Prose path  │  │ Stage 2b: Table path   │
│ Hierarchical,         │  │ Structured extraction, │
│ structure-aware       │  │ never flattening       │
│ chunking              │  │                        │
└───────────────────────┘  └────────────────────────┘
        │                          │
        └───────────┬──────────────┘
                    ▼
┌─────────────────────────────────────┐
│ Stage 3: Parent-child indexing      │
│ (embed small, generate on large)    │
└─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────┐
│ Stage 4: Metadata tagging           │
│ (source, dates, legal status, ...)  │
└─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────┐
│ Stage 5: Deduplication              │
│ (exact hash + near-dup by authority)│
└─────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────┐
│ Stage 6: Embedding (BGE-M3) &       │
│ index write (vector store + doc     │
│ store)                              │
└─────────────────────────────────────┘
```

**Documents in scope:** the official 2015 Labour Act gazette, the EPZ Act, the Labour Rules, the CPD wage-revision report, the Legal 500 guide, and the wage gazettes.

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

Tables never become flowing text. Three sub-paths depending on extractability:

| Table type | Handling |
|---|---|
| **Text-based tables** (e.g., the CPD report's 40+ wage/survey tables) | Extracted with camelot-py or pdfplumber into **structured JSON** — explicit row labels, column labels, and a data dictionary, not a paragraph |
| **Unparseable / scanned tables** (merged cells, nested headers, anything a text extractor mangles) | Rendered as an image and embedded with a **vision-capable model** (ColPali or Jina-CLIP); the LLM reads the table visually at generation time rather than trusting a broken extraction |
| **Gazette wage tables** (Bangla-script OCR difficulty) | Small enough and high-stakes enough to warrant a **one-time manual transcription** into structured JSON, human-verified — rather than an automated pipeline you'd have to separately verify anyway |

**Summary-embeds, JSON-generates:** each extracted table also gets a short natural-language summary generated alongside it (e.g., *"This table shows CPD's proposed minimum wage by grade, broken into basic, housing, food, medical, and transport allowances"*). The **summary is what gets embedded** for semantic search; the **full structured JSON is what gets handed to the LLM** once that summary is retrieved. This is the same principle as parent-child indexing, applied to tables instead of legal clauses.

---

## Stage 3: Parent-Child Indexing

For both content types, what gets embedded and searched is deliberately **smaller and more precise** than what gets handed to the LLM for generation:

| | Child (embedded, searched) | Parent (returned to the LLM) |
|---|---|---|
| **Prose** | A single clause — e.g., Section 23(3), the compensation entitlement. Small and specific, so semantic search matches it precisely against a narrow query. | The full surrounding section — e.g., all of Section 23, including the misconduct exceptions in 23(1) that qualify the compensation right in 23(3). |
| **Tables** | The table's natural-language summary. | The full structured table JSON. |

**Why this exists:** legal text is full of exactly the pattern where a right is granted in one sub-section and qualified in the next. A retriever matching only on the small chunk still finds the right needle; the LLM generating the answer then sees the full haystack around that needle — so it doesn't miss the exception that changes the answer.

---

## Stage 4: Metadata Tagging

Every chunk — prose or table — carries metadata that isn't used for semantic matching but for **filtering and prioritization that semantic similarity alone can't express**:

| Field | Purpose |
|---|---|
| `source_act` | Which law (Labour Act vs. EPZ Act vs. wage gazette) — prevents citing the wrong act for the wrong worker type |
| `section_number` | Precise citation; lets us detect overlapping coverage during corpus QA |
| `effective_date` | Which version is current, since the Act has layered amendments (2015 gazette → 2018 amendments → 2025 Ordinance) |
| `legal_status` | `ratified` / `pending_ratification` / `superseded` — specifically addresses the 2025 Ordinance's unresolved ratification status |
| `translation_status` | `official` (the 2015 gazette) vs. `unofficial` (mccibd.org) — lets us prefer the authoritative source when two versions disagree |
| `source_type` | `primary_law` vs. `policy_advocacy` — critically separates the actual wage gazette from CPD's proposed (but not adopted) wage figures, so the retriever can never confuse the two |

---

## Stage 5: Deduplication

After chunking and *before* chunks reach the vector index, a dedup pass removes genuine redundancy **without removing meaningful version history**:

- **Exact duplicates** (identical text from overlap or double-extraction): caught by **hashing normalized chunk text** and dropping exact matches.
- **Near-duplicates** (the same fact stated differently across sources — e.g., the gazette's wording vs. CPD's paraphrase of the same fact): caught by **pairwise embedding similarity above a threshold**, then resolved by an **authority ranking**:

  ```
  official gazette > official translation > reputable secondary source > advocacy report
  ```

  The lower-authority version is **demoted / cross-referenced, not deleted** — it may still be useful context.

**Distinct from legal-status filtering:** `legal_status` / `effective_date` filtering preserves genuinely different *versions of the law over time*; dedup removes accidental redundancy that adds noise without adding information. These are separate steps and must stay separate.

---

## Stage 6: Embedding and Index Write

Final chunks (children, with parent-pointers and metadata attached) are embedded with a genuinely **multilingual model: BGE-M3**.

**Why BGE-M3:** it natively supports **dense, sparse, and multi-vector representations in one model** — which conveniently provides the dense+sparse hybrid-search foundation the retrieval stage needs, without stitching together two separate systems.

**Where things land:**

| Store | Contents |
|---|---|
| **Vector store** (Qdrant / Pinecone) | Child chunk embeddings, alongside the metadata fields as **filterable attributes** |
| **Document store** | Parent chunks and structured table JSON, **keyed by the same IDs** |

---

## Design Principles Recap

1. **Route before chunking** — prose and tables have different failure modes; treat them differently from the first step.
2. **Respect the document's own structure** — the legal corpus already has a hierarchy; use it instead of a fixed-size splitter.
3. **Never flatten tables** — extract structure, embed a summary, hand the LLM the structured data.
4. **Embed small, generate large** — parent-child indexing decouples retrieval precision from generation context.
5. **Metadata carries what similarity can't** — legal status, authority, effective dates.
6. **Dedup noise, preserve history** — remove accidental redundancy; keep genuinely different versions of the law.
7. **Sweep, don't guess** — chunk-size and overlap parameters are starting points to be tuned by the RAGAS harness, not constants.
