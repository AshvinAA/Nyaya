# Nyaya

**A voice-first, agentic RAG system for Bangladesh's Readymade Garment (RMG) workers — and a technical demonstration of a properly-engineered, rigorously-evaluated retrieval-augmented generation pipeline.**

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

- **The theoretical/impact claim** — that an AI pipeline like this can meaningfully attack a real, large-scale socioeconomic problem in Bangladesh: not by replacing enforcement or policy, but by giving workers direct, verifiable access to information and a legitimate hiring channel that currently exists only through corrupt intermediaries.
- **The technical claim** — that the builder has genuine proficiency in building a properly engineered, rigorously evaluated RAG system, not just a basic retrieve-and-generate wrapper around an LLM.

## Technical Design Highlights

Every advanced technique in the pipeline exists because the corpus is genuinely messy and high-stakes in exactly the ways that justify it (mixed scripts and translation quality, legally consequential numeric tables, contested and changing law, a real register gap between how the law is written and how people speak) — nothing is bolted on to look impressive.

### Ingestion Pipeline

- **Corpus consolidation with overlay layers** — a manually consolidated base (Labour Act 2006 + ratified 2013/2018 amendments) is the source of truth; the unratified 2025 Ordinance enters as a separately-labeled overlay, never silently merged; authority (`legal_status`) and applicability (`in_force`) are tracked as separate, explicit fields.
- **Structure-aware chunking** that follows the legal corpus's own hierarchy (Chapter → Section → Clause) rather than naive fixed-size splitting — with a **structural validation gate** so a mis-parsed document can never silently reach the index.
- **Parent-child indexing** so retrieval precision and generation context don't trade off against each other, with **embed-time context injection** (the hierarchical path is embedded, never shown to the LLM as law text).
- **A separate structured path for data tables** (wage grades, survey statistics) so they aren't destroyed by being flattened into prose.
- **Deduplication logic** that removes true redundancy while preserving meaningful version history across amended law.

📄 See [INGESTION_PIPELINE.md](INGESTION_PIPELINE.md) for the full ingestion architecture.

### Retrieval

Layered and justified — each technique solves a specific, named problem in this corpus, not a generic checklist:

- Hybrid (dense + sparse) search
- Multi-query expansion to bridge the gap between colloquial spoken Bangla and formal legal register
- Reciprocal rank fusion
- Reranking

### Memory

- Conversational history-awareness for multi-turn calls
- Lightweight entity/knowledge-graph memory so a returning caller's context persists across calls

### Evaluation

A full **RAGAS + LangSmith** harness measuring both retrieval metrics (context precision/recall) and generation metrics (faithfulness, answer correctness) — built early enough to produce a genuine **ablation table** showing what each architectural decision actually contributed, not a single final score asserted without comparison.

### Voice Pipeline

A real voice-to-RAG-to-voice pipeline in Bangla, with an explicitly reasoned **cross-lingual architecture**: English-language legal corpus, Bangla spoken input and output, with the LLM reasoning over English context and generating directly in Bangla — rather than a naive translate-everything approach.

### Guardrails & Safety

- A **faithfulness threshold** that routes low-confidence legal questions to a human/NGO fallback.
- **Strict isolation of sensitive worker data** (NID, phone number) from the RAG context window.
- **Metadata that tracks a provision's legal status explicitly** — Bangladesh's labour law currently includes a 2025 amendment ordinance whose ratification is still pending, and the system says so rather than stating it as settled law.

## The Corpus

The knowledge base is built from real primary sources in `docs/`, including:

- The Bangladesh Labour Act (English, up to the 2018 amendments)
- The Bangladesh Labour Rules 2015 (unofficial English translation)
- The 2015 wage gazette and 2025 amendment gazette
- The 2023 RMG minimum wage revision report
- Legal 500's Bangladesh employment guide and ILO Committee on Freedom of Association materials

## Why This Framing Works

The social-impact story and the technical engineering story **reinforce each other rather than compete**: the corpus is genuinely messy and high-stakes in exactly the ways that justify sophisticated RAG engineering — so every advanced technique in the pipeline has a concrete reason to exist.

## Status & Disclaimer

⚠️ **This is a demo and proof-of-pipeline, not production software and not legal advice.** Answers are grounded in the ingested legal corpus, but the 2025 amendment ordinance's ratification status is unresolved and is tracked explicitly in the system's metadata. Workers facing live disputes should be routed to human/NGO support.
