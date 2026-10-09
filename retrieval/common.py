"""
============================================================================
 Nyaya - Retrieval Pipeline :: shared helpers
============================================================================
 Small dependency-light utilities used by every retrieval stage module.
 Nothing stage-specific belongs here.
============================================================================
"""

import json
import os
import sys

import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def safe_div(a, b):
    return a / b if b else 0.0


def est_tokens(text):
    """Rough token estimate mirroring ingestion/common.py semantics."""
    return max(1, round(len((text or "").split()) * 1.3))


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def cosine_matrix(Q, D):
    """Q: (n_q, d) float32, D: (n_d, d) float32, both row-normalized already.
    Returns (n_q, n_d) similarity matrix."""
    if len(D) == 0:
        return np.zeros((Q.shape[0], 0), dtype=np.float32)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return (Qn @ Dn.T).astype(np.float32)
