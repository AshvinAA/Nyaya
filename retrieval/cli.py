"""
retrieval demo / trace inspection CLI.

NOT the generation entry point — that's built much later. This CLI exists to run
a single query end to end against the retrieval pipeline and print/inspect the
trace BEFORE trusting the pipeline with real queries.

Usage:
    venv/Scripts/python -m retrieval.cli "<query text>" [--query-id q0001] [--no-llm]

What it does:
    1. Runs retrieve(query) -> (context_set, trace)
    2. Prints a human-readable summary of the trace:
       - Stage 1: variants + flags + fallback?
       - Stage 2: filter applied + per-variant hit counts
       - Stage 3: fused pool size + top fused chunks
       - Stage 4: deduped pool size + collapsed pairs / guard hits
       - Stage 5: final top-k + reranker used / fallback
       - Stage 5b: MMR applied? / skipped reason
       - Stage 6: resolved context (child -> parent kind + truncation labels)
       - Stage 7: dual_layer pairs + labels applied
       - timings per stage
       - overall fallbacks
    3. Writes the trace to retrieval_traces/<query_id>.json (always, even if
       the print is the main intent — the trace is the evaluation substrate)

Objective of this file: give a human a way to inspect a single retrieval
decision end to end, before the eval harness runs the full test set. A wrong
answer is debugged by reading this trace down to the stage that caused it.
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict

from retrieval.orchestrator import retrieve

logger = logging.getLogger("retrieval.cli")


def _print_summary(trace: Dict[str, Any]) -> None:
    stages = trace.get("stages", {})

    print("=" * 72)
    print("query:", trace.get("original"))
    print("query_id:", trace.get("query_id"))

    # -- Stage 1 ------------------------------------------------------------
    s1 = stages.get("1", {})
    print("-" * 72)
    print("Stage 1  | variants:", json.dumps(trace.get("variants", []), ensure_ascii=False))
    flags = trace.get("flags", {})
    print("         | multi_aspect=%s historical=%s fallback=%s"
          % (flags.get("multi_aspect"), flags.get("historical"),
             any(f.get("stage") == "1" for f in trace.get("overall_fallbacks", []))))

    # -- Stage 2 ------------------------------------------------------------
    s2 = stages.get("2", {})
    print("Stage 2  | filter=%s  lists=%d  dense_vectors_retained=%s"
          % (s2.get("filter_applied"), len(s2.get("variants", [])),
             s2.get("dense_vectors_retained")))

    # -- Stage 3 ------------------------------------------------------------
    s3 = stages.get("3", {})
    fused = s3.get("fused_pool", [])
    top = ", ".join("%s(%.3f)" % (e["chunk_id"], e["rrf_score"]) for e in fused[:5])
    print("Stage 3  | fused_pool=%d  top: %s" % (len(fused), top or "none"))

    # -- Stage 4 ------------------------------------------------------------
    s4 = stages.get("4", {})
    deduped = s4.get("deduped_pool", [])
    print("Stage 4  | deduped_pool=%d collapsed=%d guard_hits=%d scorer=%s"
          % (len(deduped), len(s4.get("collapsed_pairs", [])),
             len(s4.get("guard_hits", [])), s4.get("scorer_backends") or "none"))
    for pair in s4.get("collapsed_pairs", []):
        print("         |   collapse: %s beaten by %s (sim=%s)"
              % (pair["loser_id"], pair["winner_id"], pair["similarity"]))
    for hit in s4.get("guard_hits", []):
        print("         |   guard (NOT collapsed): %s vs %s — %s"
              % (hit["pair"][0], hit["pair"][1], hit["why"]))

    # -- Stage 5 ------------------------------------------------------------
    s5 = stages.get("5", {})
    topk = s5.get("final_topk", [])
    print("Stage 5  | reranker=%s fallback=%s" % (s5.get("reranker"), s5.get("fallback_triggered")))
    for item in topk:
        print("         |   rank %d: %s (rerank=%.6f)" % (item["rank"], item["chunk_id"], item["rerank_score"]))

    # -- Stage 5b -----------------------------------------------------------
    s5b = stages.get("5b", {})
    if s5b.get("mmr_applied"):
        print("Stage 5b | MMR applied:")
        for item in s5b.get("mmr_topk", []):
            print("         |   mmr_rank %d: %s (rerank=%.6f)"
                  % (item["mmr_rank"], item["chunk_id"], item["rerank_score"] or 0.0))
    else:
        print("Stage 5b | MMR skipped — %s" % s5b.get("reason_if_skipped"))

    # -- Stage 6 ------------------------------------------------------------
    s6 = stages.get("6", {})
    print("Stage 6  | parent resolution:")
    for entry in s6.get("resolved_context", []):
        print("         |   %s -> parent %s kind=%s est_tokens=%s%s"
              % (entry.get("child_id"), entry.get("parent_id"),
                 entry.get("parent_kind"), entry.get("est_tokens"),
                 "" if not entry.get("label") else "  label=%s" % entry["label"]))

    # -- Stage 7 ------------------------------------------------------------
    s7 = stages.get("7", {})
    print("Stage 7  | dual_layer_pairs=%d" % len(s7.get("dual_layer_pairs", [])))
    for pair in s7.get("dual_layer_pairs", []):
        print("         |   base=%s (%s)  overlay=%s (%s)"
              % (pair["base_chunk_id"], pair["base_label"],
                 pair["overlay_chunk_id"], pair["overlay_label"]))

    # -- Final context ---------------------------------------------------------
    print("-" * 72)
    for item in trace.get("final_context_set", []):
        print("FINAL   | %s  parent=%s  kind=%s  label=%s"
              % (item["chunk_id"], item.get("parent_id"),
                 item.get("parent_kind"), item.get("label") or "-"))

    # -- Timings + fallbacks --------------------------------------------------
    print("-" * 72)
    timings = trace.get("timings", {})
    print("timings :", ", ".join("s%s=%.3fs" % (k, v) for k, v in sorted(timings.items())))
    fallbacks = trace.get("overall_fallbacks", [])
    if fallbacks:
        print("fallbacks:")
        for f in fallbacks:
            print("  - stage %s: %s (used: %s)" % (f["stage"], f["reason"], f["used_instead"]))
    else:
        print("fallbacks: none")
    print("=" * 72)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one query through the retrieval pipeline and inspect the trace."
    )
    parser.add_argument("query", help="The query text (Bangla or English).")
    parser.add_argument(
        "--query-id", default=None,
        help="Stable trace id (matches eval/test_queries.json ids when evaluating).",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the Stage 1 LLM expansion (smoke-test mode; original query becomes the sole variant).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    query_id = args.query_id or "cli_%s" % str(abs(hash(args.query)) % 10_000_000)

    context_set, trace = retrieve(
        args.query,
        query_id,
        use_llm=not args.no_llm,
    )

    _print_summary(trace)

    if not context_set:
        print("\nWARNING: empty final context set — read the trace fallbacks above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
