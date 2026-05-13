from __future__ import annotations

import numpy as np


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values. NaNs remain NaN."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = adj
    q[valid] = out
    return q
