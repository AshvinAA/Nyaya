"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 5b - Optional MMR (conditional, not default)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Maximal Marginal Relevance (lambda ~ 0.6 against the BGE-M3 dense vectors
 ALREADY COMPUTED in Stage 2) diversifies the top-k to avoid returning several
 near-identical chunks about one narrow aspect when the question needed coverage
 of two different things.

 ONLY runs if Stage 1 flagged the query multi_aspect. NOT run by default.

 Why conditional: on a single-aspect legal question, MMR can REDUCE PRECISION by
 deliberately demoting the most relevant chunk in favor of diversity that wasn't
 needed. The conditional trigger is itself an eval item:
   - (a) how often does the flag fire, and does it fire CORRECTLY (against the
         multi-aspect vs single-aspect control pairs in the test set)?
   - (b) what precision cost would MMR have imposed on single-aspect queries HAD
         it run — that delta is the ablation row that justifies keeping it off by
         default (MMR forced on / forced off).

 Used by: orchestrator.py (Stage 5b of the retrieve() flow, AFTER Stage 5
 rerank, BEFORE Stage 6 parent resolution).

 Objective of this file: if envelope["multi_aspect"] is true, apply MMR to the
 top-k using the dense vectors already computed in Stage 2 (don't recompute),
 producing a diversified top-k; if the flag is false or the vectors aren't
 available, skip and record why in the trace.
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 5b contract
# ---------------------------------------------------------------------------
# optional_mmr(final_topk_from_stage5, dense_vectors_by_id, envelope, trace)
#   -> mmr_topk | None
#   final_topk_from_stage5 : the Stage 5 final top-k [{chunk_id, rerank_score,
#                             rank}, ...]
#   dense_vectors_by_id    : the BGE-M3 dense vectors retained by Stage 2,
#                             keyed by chunk_id (so MMR can compute similarity
#                             against already-computed vectors — do NOT recompute
#                             embeddings here)
#   envelope         : the Stage 1 envelope (reads envelope["multi_aspect"])
#   trace            : the in-progress trace (records mmr_applied, the mmr_topk
#                       or the skip reason)
#   returns          : mmr_topk = [{chunk_id, ...}, ...] if applied, else None
#
# If mmr_applied is False, the caller (orchestrator) passes the Stage 5
# final_topk straight to Stage 6 — MMR is not the default path.

# ---------------------------------------------------------------------------
# MMR algorithm (lambda ~ 0.6)
# ---------------------------------------------------------------------------
# Start with the top reranked chunk. Then iteratively add the chunk that
# maximizes:
#   lambda * relevance_score - (1 - lambda) * max_similarity_to_already_selected
#
#   - relevance_score : the rerank score from Stage 5 (the chunk's relevance to
#     the original query)
#   - max_similarity_to_already_selected : the chunk's max cosine similarity to
#     any chunk already selected into the MMR set (against the BGE-M3 dense
#     vectors retained in Stage 2)
#   - lambda = config.MMR_LAMBDA (~0.6): trade off relevance vs diversity.
#
# The selected set is the diversified top-k. Order within the set is by
# selection order (or by rerank score — decide and record which; the doc doesn't
# pin the within-set ordering, so pick one and make it a tunable / trace field).

# ---------------------------------------------------------------------------
# Uses dense vectors already computed in Stage 2 (no recomputation)
# ---------------------------------------------------------------------------
# MMR needs chunk-to-chunk cosine similarity against the BGE-M3 dense vectors.
# Those vectors were already computed in Stage 2 (the dense search pass). Stage
# 2 must retain them (per-chunk, keyed by chunk_id) so Stage 5b can use them
# without a second encode. If Stage 2 didn't retain them (e.g. a future sparse-
# only path where dense vectors weren't kept), MMR can't run — skip and record
# the reason in the trace.
#
# Do NOT re-embed the candidates here. Recomputing would add cost for no benefit
# (the vectors already exist) and would risk a different model/settings than
# Stage 2 used (inconsistent similarity). The principle: reuse what Stage 2
# already computed.

# ---------------------------------------------------------------------------
# Conditional trigger (multi_aspect only)
# ---------------------------------------------------------------------------
# MMR runs ONLY if envelope["multi_aspect"] is True.
#
# If False: skip MMR entirely. The Stage 5 final top-k goes to Stage 6 as-is.
# Record mmr_applied=False in the trace (no reason needed — the flag wasn't set).
#
# If True: run MMR. Record mmr_applied=True in the trace, plus the mmr_topk.
#
# If True but vectors unavailable: skip MMR, record mmr_applied=False with
# reason="dense vectors not retained for MMR" in the trace. Diversity is
# OPTIONAL; correctness filters are not. This is the degradation policy again:
# a missing convenience feature (diversity) is logged, not fatal.

# ---------------------------------------------------------------------------
# Why NOT run by default
# ---------------------------------------------------------------------------
# On a single-aspect legal question (the common case for this corpus — "is this
# fee legal?", "what's my notice period?"), MMR can reduce precision by
# deliberately demoting the most relevant chunk in favor of diversity that the
# question didn't need. Running MMR by default would impose that precision cost
# on every query, including the ones where it hurts.
#
# The conditional trigger (Stage 1's multi_aspect flag) is the gate. The eval
# harness validates it:
#   - on the multi-aspect subset: does MMR improve aspect-coverage recall?
#   - on the single-aspect subset: what precision cost would MMR have imposed
#     had it run? (ablation row: MMR forced on / forced off)
#
# If the flag fires incorrectly (multi_aspect set on a single-aspect query, or
# missed on a genuine multi-aspect query), that's a Stage 1 quality problem, not
# an MMR problem — but MMR's conditional nature means a bad flag directly hurts
# precision or wastes a slot. The two are measured together.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search, fuse, dedup, or rerank — that's Stages 2/3/4/5
#     (it receives the Stage 5 final top-k).
#   - It does not resolve parents — that's Stage 6 (it passes the mmr_topk or
#     the Stage 5 top-k to Stage 6).
#   - It does not label overlays — that's Stage 7.
#   - It does not interpret query intent — that's Stage 1's envelope (it only
#     reads envelope["multi_aspect"]).
#   - It does NOT recompute embeddings — it uses Stage 2's retained dense
#     vectors; if those aren't available, it skips, it doesn't re-embed.
