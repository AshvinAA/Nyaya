"""
============================================================================
 Nyaya - Retrieval Pipeline :: per-query trace (the evaluation substrate)
============================================================================
 Every query produces ONE JSON trace written to retrieval_traces/<query_id>.json.

 This is not optional instrumentation. It is the substrate the eval harness
 consumes:
   - RAGAS context metrics are computed from the final context set
   - stage-level ablations are computed from the per-stage ranked lists
   - a wrong answer is debugged down to the stage that caused it by reading
     the trace

 The trace schema + writer are implemented here (not just a shell) because every
 stage needs to append into the same trace object as it runs. The trace is built
 up incrementally by the orchestrator: Stage 1 writes its envelope, Stage 2 its
 ranked lists, ..., Stage 7 its overlay labels, then the writer flushes once at
 the end.

 Objective of this file: give every stage a uniform Trace API so the trace
 contract is enforced by code, not by hope. The eval harness depends on the
 finalised trace shape.

 Verified: TRACE_DIR comes from config (retrieval_traces, already in .gitignore).
============================================================================
"""

import json
import os
import time
from typing import Any, Dict, List

from retrieval import config as cfg


# ---------------------------------------------------------------------------
# Trace — the in-progress trace object the orchestrator builds stage by stage
# ---------------------------------------------------------------------------

class Trace:
    """One per-query trace, built up incrementally and frozen at finalize().

    Stages append to it as they run. The writer flushes the finalised dict to
    retrieval_traces/<query_id>.json. After finalize() the trace is immutable.

    Public API
    ---------
    Trace(query_id, original)
        new trace; stages empty, flags default False, fallbacks empty.
    add_variant(text)
        Stage 1: append one variant string (called in order).
    set_flags(multi_aspect, historical)
        Stage 1: set the envelope flags (the only query-understanding output).
    record_stage(stage_nb, record)
        Any stage: store its record under stages[str(stage_nb)].
        Raises after finalize().
    add_fallback(stage, reason, used)
        Any stage: record a fallback event (stage nb or name, reason, what was
        used instead). Collected into overall_fallbacks at finalize().
    finalize(context_set, timings)
        Freeze the trace: attach final_context_set, timings, overall_fallbacks,
        and return the final dict ready to write. After this, record_stage()
        raises.
    """

    def __init__(self, query_id: str, original: str) -> None:
        self.query_id = query_id
        self.original = original
        self._variants: List[str] = []
        self._multi_aspect: bool = False
        self._historical: bool = False
        self._stages: Dict[str, Any] = {}
        self._fallbacks: List[Dict[str, Any]] = []
        self._finalized: bool = False
        self._start = time.time()

    # -- variants / flags (Stage 1 envelope) ---------------------------------

    def add_variant(self, text: str) -> None:
        """Append one formal-register variant (Stage 1)."""
        if self._finalized:
            raise RuntimeError("trace already finalised")
        self._variants.append(text)

    def set_flags(self, multi_aspect: bool, historical: bool) -> None:
        """Set the Stage 1 envelope flags (the only query-understanding output)."""
        if self._finalized:
            raise RuntimeError("trace already finalised")
        self._multi_aspect = bool(multi_aspect)
        self._historical = bool(historical)

    # -- per-stage records ---------------------------------------------------

    def record_stage(self, stage_nb: int | str, record: Any) -> None:
        """Store a per-stage record (any stage). Raises after finalize()."""
        if self._finalized:
            raise RuntimeError("trace already finalised")
        self._stages[str(stage_nb)] = record

    # -- fallbacks -----------------------------------------------------------

    def add_fallback(self, stage: int | str, reason: str, used: str) -> None:
        """Record a fallback event (any stage). Collected at finalize()."""
        if self._finalized:
            raise RuntimeError("trace already finalised")
        self._fallbacks.append(
            {"stage": str(stage), "reason": reason, "used_instead": used}
        )

    # -- finalize ------------------------------------------------------------

    def finalize(
        self,
        context_set: List[Dict[str, Any]],
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        """Freeze the trace and return the final dict ready to write.

        Parameters
        ----------
        context_set : list
            The final labeled context set handed to generation (Stage 7 output).
        timings : dict
            Per-stage wall-clock seconds, keyed by stage number string.
            The orchestrator records each stage's elapsed time here.

        Returns
        -------
        dict  -- the final trace, ready for write_trace().
        """
        if self._finalized:
            raise RuntimeError("trace already finalised")
        self._finalized = True

        overall: List[Dict[str, Any]] = list(self._fallbacks)

        return {
            "query_id": self.query_id,
            "original": self.original,
            "variants": list(self._variants),
            "flags": {
                "multi_aspect": self._multi_aspect,
                "historical": self._historical,
            },
            "stages": dict(self._stages),
            "final_context_set": context_set,
            "timings": timings,
            "overall_fallbacks": overall,
            "_trace_version": 1,
            "_built_at": time.time() - self._start,
        }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_trace(trace: Dict[str, Any], query_id: str) -> str:
    """Pretty-print a finalised trace to retrieval_traces/<query_id>.json.

    Returns the written path. The trace must already be finalised (Trace.finalize()
    returns the dict passed here). This function only serializes; validation of the
    schema belongs in the eval harness's trace-invariance check.
    """
    os.makedirs(cfg.TRACE_DIR, exist_ok=True)
    path = os.path.join(cfg.TRACE_DIR, f"{query_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=1, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Trace-invariance sanity check (eval harness concern, provided here for
# convenience -- the harness calls this on every loaded trace file)
# ---------------------------------------------------------------------------

REQUIRED_STAGES = {"1", "2", "3", "4", "5", "5b", "6", "7"}

# Stage 5b is always recorded (even when MMR is skipped), so the trace must
# always contain a Stage 5b record with mmr_applied + (if False) reason_if_skipped.
PARTIAL_STAGE_CHECK = {
    "5b": lambda s: bool(s.get("mmr_applied"))
        and (s.get("mmr_applied") or bool(s.get("reason_if_skipped"))),
}


def assert_trace_wellformed(trace: Dict[str, Any]) -> list:
    """Return a list of problems found in a finalised trace (empty = OK).

    These are eval-time assertions on the written trace file, not runtime checks
    in the pipeline. The pipeline's job is to record honestly; this function's job
    is to verify the record is complete enough to measure against.

    Checks
    ------
    - all 7 stages present in trace["stages"]
    - final_context_set non-empty
    - if flags.multi_aspect is true, Stage 5b exists and recorded mmr_applied (and
      if False, a reason_if_skipped)
    - if flags.multi_aspect is false, Stage 5b still must exist (it is always recorded,
      even when skipped) with mmr_applied=False and a reason_if_skipped
    - if any stage's fallback_triggered is true, overall_fallbacks is non-empty and
      covers that stage
    - if dual_layer pairs exist in Stage 7, labels_applied covers both chunks of each
      pair
    """
    problems: list = []

    stages = trace.get("stages", {})
    missing = REQUIRED_STAGES - set(stages.keys())
    if missing:
        problems.append("missing stages: " + str(sorted(missing)))

    if not trace.get("final_context_set"):
        problems.append("final_context_set is empty (retrieval returned nothing)")

    flags = trace.get("flags", {})
    s5b = stages.get("5b", {})
    if flags.get("multi_aspect"):
        # When multi_aspect is flagged, Stage 5b MUST have run (mmr_applied=True).
        if not s5b.get("mmr_applied"):
            problems.append(
                "multi_aspect flagged but Stage 5b has no mmr_applied"
            )
    else:
        # When multi_aspect is NOT flagged, Stage 5b still runs (it is always
        # recorded), but must skip MMR and record why.
        if not s5b.get("mmr_applied") and not s5b.get("reason_if_skipped"):
            problems.append(
                "multi_aspect not flagged but Stage 5b skipped MMR with no reason_if_skipped"
            )

    fallbacks = trace.get("overall_fallbacks", []) or []
    fallback_stages = {f["stage"] for f in fallbacks}
    for snb, rec in stages.items():
        if rec.get("fallback_triggered"):
            if snb not in fallback_stages:
                problems.append(
                    "stage "
                    + str(snb)
                    + " reports fallback_triggered but not in overall_fallbacks"
                )

    s7 = stages.get("7", {})
    pairs = s7.get("dual_layer_pairs", []) or []
    labels = s7.get("labels_applied", {}) or {}
    for pair in pairs:
        for key in ("base_chunk_id", "overlay_chunk_id"):
            if key not in pair:
                problems.append("dual_layer pair missing " + key)
        if "base_chunk_id" in pair and pair["base_chunk_id"] not in labels:
            problems.append(
                "dual_layer pair base "
                + str(pair["base_chunk_id"])
                + " not in labels_applied"
            )
        if "overlay_chunk_id" in pair and pair["overlay_chunk_id"] not in labels:
            problems.append(
                "dual_layer pair overlay "
                + str(pair["overlay_chunk_id"])
                + " not in labels_applied"
            )

    return problems
