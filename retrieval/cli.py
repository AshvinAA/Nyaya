"""
retrieval demo / trace inspection CLI.

NOT the generation entry point — that's built much later. This CLI exists to run
a single query end to end against the retrieval pipeline and print/inspect the
trace BEFORE trusting the pipeline with real queries.

Usage (once built):
    venv/Scripts/python -m retrieval.cli "<query text>" [--query-id q0001]

What it does (once Stages 1-7 + orchestrator exist):
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

TODO: implement once orchestrator.py exists. Right now this is the shell with the
intended behavior documented.
"""
