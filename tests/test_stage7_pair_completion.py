"""
============================================================================
 Nyaya - regression test :: Stage 7 pair completion
============================================================================
 WHY THIS FILE EXISTS
 --------------------
 On the eval set, a 2026-amendment query returned overlay chunks (s.56 amends
 s.264, s.81 amends s.326) but not the base chunks for s.264 / s.326, so Stage 7
 paired nothing and the conflict was never labelled. Pair completion pulls the
 base section in when an overlay is retrieved without it.

 These tests use a fake registry, so they run without Chroma or the embedding
 models. They check: the base section is added once, nothing is removed, an
 already-present base section is not refetched, non-section targets are skipped,
 the cap holds, the switch turns it off, and Stage 7 then pairs and labels the
 completed pair.

 HOW TO RUN (repo root)
     python tests/test_stage7_pair_completion.py
 Exits 0 on pass, 1 on failure.
============================================================================
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from retrieval import config as cfg  # noqa: E402
from retrieval.stage7_overlay import complete_base_pairs, resolve_overlay  # noqa: E402


class FakeRegistry:
    def __init__(self, base_by_section):
        self.base_by_section = base_by_section   # section -> (chunk_id, meta)
        self.calls = []

    def base_children_for_section(self, query_text, section_number, n=1):
        self.calls.append(section_number)
        hit = self.base_by_section.get(section_number)
        return [hit] if hit else []


class FakeTrace:
    def __init__(self):
        self.stages = {}

    def record_stage(self, nb, record):
        self.stages[str(nb)] = record


BASE_264 = ("base264", {"text_layer": "consolidated_base", "section_number": "264",
                        "parent_id": "par264", "doc": cfg.BASE_ACT_DOC})
BASE_326 = ("base326", {"text_layer": "consolidated_base", "section_number": "326",
                        "parent_id": "par326", "doc": cfg.BASE_ACT_DOC})
BASE_307A = ("base307A", {"text_layer": "consolidated_base", "section_number": "307A",
                          "parent_id": "par307A", "doc": cfg.BASE_ACT_DOC})
BASE_307B = ("base307B", {"text_layer": "consolidated_base", "section_number": "307B",
                          "parent_id": "par307B", "doc": cfg.BASE_ACT_DOC})


def overlay(cid, amends, status="ratified"):
    return (cid, {"text_layer": "amendment_overlay", "section_number": "99",
                  "amends_section": amends, "legal_status": status,
                  "parent_id": "par_" + cid, "doc": "bangladesh_labour_amendment_act_2026_bangla_gazette"})


def run(children_ids, meta_map, registry, enabled=True):
    old = cfg.PAIR_COMPLETION_ENABLED
    cfg.PAIR_COMPLETION_ENABLED = enabled
    try:
        children = [{"chunk_id": c, "rerank_score": 0.5, "rank": i + 1} for i, c in enumerate(children_ids)]
        before = [c["chunk_id"] for c in children]
        out, added_meta, rec = complete_base_pairs(children, meta_map, "q", registry)
    finally:
        cfg.PAIR_COMPLETION_ENABLED = old
    return before, out, added_meta, rec


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("ok  -", msg)


def main():
    # 1. overlay s.264 retrieved, base s.264 not -> base s.264 added, nothing removed
    o = overlay("ov56", "264")
    meta = {"ov56": o[1]}
    reg = FakeRegistry({"264": BASE_264})
    before, out, added, rec = run(["ov56"], meta, reg)
    check([c["chunk_id"] for c in out][:1] == ["ov56"] and "base264" in [c["chunk_id"] for c in out],
          "base s.264 is added after the overlay child")
    check(set(before).issubset({c["chunk_id"] for c in out}), "no child is removed")
    check(reg.calls == ["264"] and "base264" in added, "exactly one base lookup for s.264")

    # 2. base s.264 already present -> no lookup
    meta2 = {"ov56": o[1], "base264": BASE_264[1]}
    reg2 = FakeRegistry({"264": BASE_264})
    _, out2, _, rec2 = run(["ov56", "base264"], meta2, reg2)
    check(reg2.calls == [], "already-present base section is not refetched")
    check(len(out2) == 2, "no extra child when base is already present")

    # 3. non-section targets (wildcard, schedule) are skipped, never fetched
    meta3 = {"ovA": overlay("ovA", "*")[1], "ovB": overlay("ovB", "schedule_1")[1]}
    reg3 = FakeRegistry({})
    _, out3, _, rec3 = run(["ovA", "ovB"], meta3, reg3)
    check(reg3.calls == [] and len(rec3["skipped"]) == 2, "wildcard and schedule targets are skipped")

    # 4. multi-section amendment (307A;307B) -> both base sections fetched
    meta4 = {"ov318": overlay("ov318", "307A;307B")[1]}
    reg4 = FakeRegistry({"307A": BASE_307A, "307B": BASE_307B})
    _, out4, added4, _ = run(["ov318"], meta4, reg4)
    check(reg4.calls == ["307A", "307B"] and {"base307A", "base307B"} <= set(added4),
          "each section in a multi-section amendment is completed")

    # 5. cap: more than PAIR_COMPLETION_MAX sections -> only the cap is fetched
    secs = ["264", "326", "307A", "307B"]
    reg5 = FakeRegistry({"264": BASE_264, "326": BASE_326, "307A": BASE_307A, "307B": BASE_307B})
    meta5 = {"ovbig": overlay("ovbig", ";".join(secs))[1]}
    _, _, _, rec5 = run(["ovbig"], meta5, reg5)
    check(len(reg5.calls) == cfg.PAIR_COMPLETION_MAX, "completion stops at PAIR_COMPLETION_MAX")
    check(any(s["reason"] == "over PAIR_COMPLETION_MAX" for s in rec5["skipped"]), "over-cap sections are recorded")

    # 6. switch off -> no lookup at all
    reg6 = FakeRegistry({"264": BASE_264})
    _, out6, _, _ = run(["ov56"], {"ov56": o[1]}, reg6, enabled=False)
    check(reg6.calls == [] and len(out6) == 1, "PAIR_COMPLETION_ENABLED=False disables the step")

    # 7. end to end: after completion, Stage 7 pairs and labels both sides
    reg7 = FakeRegistry({"264": BASE_264})
    meta7 = {"ov56": o[1]}
    _, out7, added7, _ = run(["ov56"], meta7, reg7)
    meta7.update(added7)
    resolved = []
    for c in out7:
        m = meta7[c["chunk_id"]]
        resolved.append({"child_id": c["chunk_id"], "parent_id": m["parent_id"], "parent_kind": "prose",
                         "parent_text_or_json": "text", "truncation_label": None, "label": "",
                         "metadata": m, "rerank_score": c.get("rerank_score"), "rank": c.get("rank")})
    tr = FakeTrace()
    rec7 = resolve_overlay(resolved, tr)
    check(len(rec7["dual_layer_pairs"]) == 1, "Stage 7 pairs the completed base with its overlay")
    check(len(rec7["final_context_set"]) == 2, "both sides are kept in the final context set")
    labels = {x["chunk_id"]: x["label"] for x in rec7["final_context_set"]}
    check(all(labels.values()), "both sides carry a label")

    print("\nALL PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
