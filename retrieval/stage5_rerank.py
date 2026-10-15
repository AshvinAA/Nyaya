"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 5 - Reranking (BGE-Reranker-v2-M3)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 BGE-Reranker-v2-M3 re-scores the deduplicated, fused candidate set directly
 against the ORIGINAL input query (NOT the Stage 1 expanded variants) using a
 cross-encoder — more expensive, more accurate than embedding similarity alone.

 Produces the final top-k (~5).

 WHY AGAINST THE ORIGINAL: the expanded variants exist to WIDEN THE NET in
 search; the reranker's job is to judge relevance to what the user actually
 asked. Re-ranking against a variant would optimize for the paraphrase, not the
 question.

 Practical notes:
   - The reranker is multilingual (Bangla queries score natively — no
     translation step).
   - It is the pipeline's latency bottleneck on CPU (hence the Stage 4 pool cap
     of ~30 entering it).
   - If the model cannot be loaded, the RRF order stands and the failure is
     recorded in the trace (reranker="fallback_rrf") — retrieval never hard-fails
     because an optional re-scorer is down.

 Used by: orchestrator.py (Stage 5 of the retrieve() flow).

 Objective of this file: take the deduped pool from Stage 4, re-score each
 candidate against the ORIGINAL query with the cross-encoder, return the final
 top-k (~5) by rerank score — plus rerank scores and fallback status for the
 trace and the eval ablation row "reranker off -> RRF order straight to final-k."
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 5 contract
# ---------------------------------------------------------------------------
# rerank(deduped_pool, original_query, trace) -> final_topk
#   deduped_pool   : the Stage 4 deduped pool [{chunk_id, ..., metadata}, ...]
#   original_query : the ORIGINAL query string from the Stage 1 envelope
#                    (NOT the variants — variants widened the net; this judges
#                    the actual question)
#   trace          : the in-progress trace (so this stage records rerank scores,
#                    the reranker used or fallback, and any fallback triggered)
#   returns        : final_topk = [{chunk_id, rerank_score, rank}, ...]
#                    capped to config.FINAL_K (~5) by rerank score
#
# The final top-k is what Stage 5b (MMR, if applied) and Stage 6 (parent
# resolution) receive.

# ---------------------------------------------------------------------------
# Cross-encoder scoring vs the original query
# ---------------------------------------------------------------------------
# For each candidate in the deduped pool:
#   - score = cross_encoder.score(original_query, candidate_text)
#   - candidate_text = the bare chunk text (the generation text Chroma stored,
#     NOT the embedding prefix — the prefix was for the embedding vector, never
#     for the LLM; the reranker should score the text the LLM would actually read)
#
# The cross-encoder scores (query, passage) pairs directly — more accurate than
# embedding cosine for relevance, at higher cost. This is why the pool entering
# it is capped at ~30 (Stage 4 / fused-pool cap).
#
# Score against the ORIGINAL query, not the variants. The variants exist to
# widen the search net; the reranker's job is to judge relevance to the question
# the user actually asked. Re-ranking against a variant would optimize for the
# paraphrase, not the question.

# ---------------------------------------------------------------------------
# Multilingual note
# ---------------------------------------------------------------------------
# BGE-Reranker-v2-M3 is multilingual — Bangla queries score natively, no
# translation step. This is a property worth recording in the trace (the doc
# says to make it explicit), not a hidden assumption. Don't translate the query
# before reranking "to be safe" — that would add a translation hop the reranker
# doesn't need and could introduce number/entity errors.

# ---------------------------------------------------------------------------
# Final top-k
# ---------------------------------------------------------------------------
# Sort by rerank_score descending, take the top config.FINAL_K (~5). This is the
# final context set entering Stage 5b (MMR, if applied) and then Stage 6 (parent
# resolution). The FINAL_K is a tunable — small enough for a small local LLM's
# context, large enough to survive MMR diversification if it runs.

# ---------------------------------------------------------------------------
# Fallback (defined, loud — never silent)
# ---------------------------------------------------------------------------
# If the reranker model cannot be loaded (no network, no disk, model missing):
#   - the RRF order (from Stage 3 / Stage 4 order) STANDS as the final order
#   - trace records: reranker = "fallback_rrf", fallback_triggered = true,
#     reason
#   - retrieval does NOT hard-fail because an optional re-scorer is down
#
# This is the degradation policy asymmetry again: convenience features (reranking)
# may degrade; correctness filters (the Stage 2 applicability filter) may not.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search, fuse, or dedup — that's Stages 2/3/4 (it receives
#     the deduped pool).
#   - It does not apply MMR — that's Stage 5b (which receives this stage's
#     final top-k if multi_aspect is flagged).
#   - It does not resolve parents — that's Stage 6.
#   - It does not label overlays — that's Stage 7.
#   - It does not interpret query intent — that's Stage 1's envelope.
#   - It does NOT score against the variants. If it did, it would optimize for
#     the paraphrase, not the question — that's the explicit reason the doc says
#     to score against the original.
