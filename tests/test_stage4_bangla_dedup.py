"""
============================================================================
 Nyaya - regression test :: Stage 4 near-dup collapse must survive Bangla
============================================================================
 WHY THIS FILE EXISTS
 --------------------
 Stage 4 once scored the whole fused pool with a single MiniLM similarity
 matrix. MiniLM is English-only, so Bangla text embeds to near-identical
 vectors: every Bangla chunk read as a duplicate of every other one. On a
 query against the RMG minimum-wage gazette, eight of ten pooled chunks
 collapsed at sim=1.0 and the pool shrank to two - the distinct wage-grade
 rows were thrown away before the reranker ever saw them. The grade rows
 differ only in their numbers, and those numbers ARE the answer.

 The fix routes by script, mirroring ingestion Stage 6: Bangla -> character
 n-grams, Latin -> embeddings, cross-script -> 0.0 (never collapse).

 This is the kind of bug that regresses silently. Collapsing _pool_similarity
 back to one matrix raises no error and breaks no English query - it just
 quietly starts eating Bangla wage tables again. Hence this test.

 HOW TO RUN (from the repo root; no pytest needed - the repo has none)
     python tests/test_stage4_bangla_dedup.py
 Exits 0 on pass, 1 on failure.

 The core checks need no network. The two checks that need the MiniLM weights
 are reported as SKIP when the model cannot be loaded, so this test stays
 useful offline and on a machine that cannot hold the model.
============================================================================
"""

import glob
import io
import os
import sys

# Repo root on sys.path, so this runs from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from ingestion.deduplicator import _is_bangla
from retrieval import config as cfg
from retrieval.stage4_dedup import _pool_similarity

THRESHOLD = float(cfg.RETRIEVAL_NEAR_DUP_THRESHOLD)
WAGE_DIR = os.path.join("docs", "transcriptions",
                        "rmg_minimum_wage_gazette_dec2023_bangla")

# Three English provisions: the first two are paraphrases of one another (they
# SHOULD still collapse - the fix must not blunt the path that worked), the
# third is unrelated. Kept in-source and ASCII so this file needs no console
# encoding; the Bangla side is read from the real transcriptions instead.
ENGLISH = [
    "Every woman worker shall be entitled to maternity benefit for eight "
    "weeks before and eight weeks after the day of her delivery.",
    "A woman worker is entitled to maternity benefit for a period of eight "
    "weeks preceding and eight weeks following the day of her delivery.",
    "No worker shall be employed for more than eight hours in any one day "
    "except as provided in section 108.",
]

failures = []
skips = []


def check(name, condition, detail=""):
    print("  %-7s %s%s" % ("ok" if condition else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not condition:
        failures.append(name)


def load_wage_rows(limit=8):
    """Real pipe-delimited Bangla table rows from the Dec-2023 wage gazette.

    Selected structurally (Bangla-dominant line, starts with a pipe, at least
    five cells) so this file carries no Bangla literals of its own and cannot
    drift from the gazette's wording.

    Returns (distinct, repeated):
      distinct  - unique rows, in document order. These carry different wage
                  figures and must NEVER collapse into one another.
      repeated  - a row that appears verbatim on more than one page (the wage
                  table's column header is reprinted on each continuation
                  page). Byte-identical rows SHOULD collapse, so these are the
                  positive control: they prove the Bangla path still dedups
                  real duplicates instead of having been neutered.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(WAGE_DIR, "p00[2-4].txt"))):
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if (line.startswith("|") and line.count("|") >= 5
                    and _is_bangla(line)):
                rows.append(line)

    seen, distinct, repeated = set(), [], []
    for r in rows:
        if r in seen:
            if not repeated:
                repeated = [r, r]
        else:
            seen.add(r)
            distinct.append(r)
    return distinct[:limit], repeated


def main():
    print("Stage 4 Bangla dedup regression (threshold=%s)" % THRESHOLD)

    bangla, repeated = load_wage_rows()
    if len(bangla) < 3:
        print("  FAIL    could not load wage rows from %s (found %d)"
              % (WAGE_DIR, len(bangla)))
        return 1
    nb, nl = len(bangla), len(ENGLISH)
    print("  pool:   %d Bangla gazette rows + %d English provisions" % (nb, nl))

    sims, backends = _pool_similarity(bangla + ENGLISH)
    print("  scorer: %s" % backends)

    bb = [(i, j) for i in range(nb) for j in range(i + 1, nb)]
    ll = [(i, j) for i in range(nb, nb + nl) for j in range(i + 1, nb + nl)]
    xs = [(i, j) for i in range(nb) for j in range(nb, nb + nl)]

    # ---- core checks: no network needed ---------------------------------
    worst = max(float(sims[i, j]) for i, j in bb)
    check("distinct Bangla rows never collapse",
          all(float(sims[i, j]) < THRESHOLD for i, j in bb),
          "worst pair %.4f vs threshold %.2f" % (worst, THRESHOLD))

    check("cross-script pairs score exactly 0.0",
          all(float(sims[i, j]) == 0.0 for i, j in xs),
          "%d pairs checked" % len(xs))

    check("Bangla group scored by the n-gram backend",
          bool(backends) and "tfidf" in backends, repr(backends))

    # Positive control: identical Bangla rows MUST still collapse. Without
    # this, "never collapse Bangla" could be passed by disabling the Bangla
    # path altogether, which would waste context slots on verbatim repeats.
    if len(repeated) == 2:
        dup_sims, _ = _pool_similarity(repeated)
        dup = float(dup_sims[0, 1])
        check("identical Bangla rows do still collapse",
              dup >= THRESHOLD, "repeated header pair %.4f" % dup)
    else:
        skips.append("positive control (no verbatim-repeated row in the "
                     "gazette pages read)")

    check("similarity matrix is square and symmetric",
          sims.shape == (nb + nl, nb + nl)
          and np.allclose(sims, sims.T, atol=1e-5))

    # ---- checks that need the MiniLM weights ----------------------------
    try:
        from retrieval.stage4_dedup import _embed_pool_texts
        emb_en = _embed_pool_texts(ENGLISH)
        emb_bn = _embed_pool_texts(bangla)
    except Exception as exc:
        skips.append("embedding-backed checks (%s: %s)"
                     % (type(exc).__name__, exc))
        emb_en = emb_bn = None

    if emb_en is not None:
        # The English path must still work: the paraphrase pair collapses,
        # the unrelated one does not.
        para = float(sims[nb + 0, nb + 1])
        unrel = float(sims[nb + 0, nb + 2])
        check("English paraphrases still collapse",
              para >= THRESHOLD, "paraphrase pair %.4f" % para)
        check("unrelated English does not collapse",
              unrel < THRESHOLD, "unrelated pair %.4f" % unrel)
        check("English group scored by the embedding backend",
              bool(backends) and "embedding" in backends, repr(backends))

        # Guard the guard: prove the routing is still load-bearing. If MiniLM
        # ever stops conflating these rows, this check fails and the routing
        # can be revisited deliberately instead of by accident.
        en = emb_bn @ emb_bn.T
        would = sum(1 for i, j in bb if float(en[i, j]) >= THRESHOLD)
        check("MiniLM would still collapse Bangla rows (routing is needed)",
              would > 0,
              "%d/%d pairs >= %.2f under the embedding scorer"
              % (would, len(bb), THRESHOLD))

    for s in skips:
        print("  SKIP    %s" % s)

    print("\n%s  (%d checks failed, %d skipped)"
          % ("FAILED" if failures else "PASSED", len(failures), len(skips)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
