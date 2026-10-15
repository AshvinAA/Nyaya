"""
============================================================================
 Nyaya - Retrieval Pipeline :: Stage 3 - RRF fusion
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 With 2-3 variants x 2 retrieval modes (dense/sparse), expect 4-6 ranked lists
 per query. Merge by RANK POSITION (not raw score, since dense and sparse scores
 aren't on the same scale):

   score(chunk) = sum( 1 / (k + rank_in_list) )   — summed across all lists the
                                                    chunk appears in

 with the standard constant k = 60 and a per-list depth of 10 as starting points
 (config.SEARCH_PER_LIST_K, config.RRF_K).

 Two properties worth making explicit:

   1. Same-ID fan-out merges for free. If the same chunk surfaces from four of
      six lists, RRF sums four contributions — that is a RELEVANCE SIGNAL, not
      a bug. The double-counting problem Stage 4 addresses is a DIFFERENT one
      (distinct chunks, same fact).
   2. Consensus beats luck. Chunks that rank well across multiple variants and
      modes surface to the top — genuine relevance rather than a single lucky
      embedding match.

 The fused pool is capped (top ~30 by RRF score) before Stage 4 — enough to
 survive dedup and reranking with a real final-k, small enough to keep the
 cross-encoder cheap.

 Dense-only right now means 2-3 lists instead of 4-6 — RRF's fusion signal is
 thinner than the design assumes, which is fine as long as we measure the
 pre-sparse recall@k baseline and watch the table/numeric lookup category
 specifically (the predicted dense-only weak spot). When sparse lands, Stage 3
 automatically fuses across more lists — no change to this file.

 Used by: orchestrator.py (Stage 3 of the retrieve() flow).

 Objective of this file: take the per-variant ranked lists from Stage 2, fuse
 them by RRF rank position into one capped fused pool, and return it (plus the
 constituent-list references and per-chunk fusion scores, for the trace and for
 the eval ablation row "RRF off -> use one dense list").
============================================================================
"""

# ---------------------------------------------------------------------------
# Stage 3 contract
# ---------------------------------------------------------------------------
# rrf_fuse(per_variant_lists, trace) -> fused_pool
#   per_variant_lists : list of ranked lists (one per variant; later one per
#                       variant x mode)
#   trace            : the in-progress trace (so this stage records the fused
#                       pool, per-chunk fusion scores, constituent-list refs,
#                       and any fallback)
#   returns          : fused_pool = [{chunk_id, rrf_score, rank}, ...]
#                       capped to config.FUSED_POOL_CAP (~30) by RRF score
#
# The fused pool is what Stage 4 (retrieval-time dedup) receives. It is capped
# BEFORE Stage 4 — enough to survive dedup and reranking with a real final-k,
# small enough to keep the cross-encoder (Stage 5) cheap.

# ---------------------------------------------------------------------------
# RRF scoring
# ---------------------------------------------------------------------------
# For each chunk that appears in ANY of the input lists:
#   rrf_score = sum( 1 / (config.RRF_K + rank_in_list) )  across the lists it
#               appears in
#
# rank_in_list is 1-based position in that list. A chunk that doesn't appear in
# a list contributes 0 from that list (not a penalty — it just doesn't add).
#
# k = 60 (config.RRF_K): the standard constant. With k=60, the first position
# contributes 1/61, the tenth contributes 1/70 — ranks are compressed enough
# that top-position consensus matters more than tail position, but low ranks
# still contribute a little (so a chunk that appears in many lists at modest
# ranks can still beat a chunk that appears in one list at rank 1).

# ---------------------------------------------------------------------------
# Same-ID fan-out is a feature, not a bug
# ---------------------------------------------------------------------------
# If the same chunk_id appears in multiple lists, RRF sums one contribution per
# list it appears in. This is the intended behavior: a chunk that multiple
# variants + modes agree on is more likely genuinely relevant than one that
# appeared in only one list. The double-counting problem Stage 4 addresses is
# DIFFERENT — distinct chunk_ids carrying the same fact (multi-query fan-out
# burning scarce final-k slots). Don't try to "fix" same-ID fan-out here; it's
# the consensus signal RRF exists to capture.

# ---------------------------------------------------------------------------
# Cap the fused pool
# ---------------------------------------------------------------------------
# After scoring, sort by rrf_score descending and take the top
# config.FUSED_POOL_CAP (~30). This cap is what keeps Stage 5's cross-encoder
# cheap (the reranker is the pipeline's CPU bottleneck). Don't grow the pool
# entering the reranker without reconsidering latency.

# ---------------------------------------------------------------------------
# Dense-only caveat (pre-sparse baseline)
# ---------------------------------------------------------------------------
# Right now (dense-only Chroma, no sparse mode), there are 2-3 variant lists
# instead of 4-6. RRF fuses fewer lists, so the consensus signal is thinner
# than the design assumes. This is FINE — it's the pre-sparse baseline, and it
# is explicitly what the eval harness measures recall@k against.
#
# The specific watch-point: the table/numeric lookup category. Dense embeddings
# are weakest at exact-keyword / numeric matches (wage figures, section numbers),
# which is exactly what sparse lexical weights are expected to catch. If the
# table/numeric recall@k on the test set is bad against the hybrid target, THAT
# is the actual trigger to build the sparse upgrade — not a vague "if it falls
# short." Stage 3 needs no change to absorb sparse; only Stage 2's second list
# and the eval's measured delta matter.

# ---------------------------------------------------------------------------
# What this file does NOT do
# ---------------------------------------------------------------------------
#   - It does not run search — that's Stage 2 (it receives ranked lists).
#   - It does not dedup — that's Stage 4 (the fused pool enters Stage 4).
#   - It does not rerank — that's Stage 5 (it receives the deduped pool).
#   - It does not resolve parents — that's Stage 6.
#   - It does not label overlays — that's Stage 7.
#   - It does not interpret query intent — that's Stage 1's envelope.
#
#   - It does NOT fuse by raw score. Dense and sparse scores aren't comparable;
#     RRF fuses by rank position. Even in dense-only, return rank-positioned
#     results to Stage 3 — don't special-case "fuse by cosine because there's
#     one mode."
