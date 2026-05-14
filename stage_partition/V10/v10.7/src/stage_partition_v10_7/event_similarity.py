from __future__ import annotations

import numpy as np
import pandas as pd

from .spatial_composite import flatten_finite


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = flatten_finite(a, b)
    if len(x) < 2:
        return np.nan
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(np.dot(x, y) / den)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x, y = flatten_finite(a, b)
    if len(x) < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _same_sign_fraction(a: np.ndarray, b: np.ndarray) -> float:
    x, y = flatten_finite(a, b)
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.sign(x) == np.sign(y)))


def _top_overlap(profile_diff: pd.DataFrame, a: str, b: str, topn: int) -> float:
    if profile_diff.empty:
        return np.nan
    aa = profile_diff.loc[profile_diff["event_id"] == a].sort_values("rank_by_abs_diff").head(topn)["feature_index"].astype(int)
    bb = profile_diff.loc[profile_diff["event_id"] == b].sort_values("rank_by_abs_diff").head(topn)["feature_index"].astype(int)
    if len(aa) == 0 or len(bb) == 0:
        return np.nan
    return float(len(set(aa) & set(bb)) / max(1, min(len(set(aa)), len(set(bb)))))


def compute_event_similarity(profile_vectors: dict[str, np.ndarray], spatial_maps: dict[str, np.ndarray], profile_diff: pd.DataFrame) -> pd.DataFrame:
    pairs = [("H18", "H35"), ("H18", "H45"), ("H35", "H45"), ("H35", "H57"), ("H18", "H57")]
    rows = []
    for a, b in pairs:
        pv_a = profile_vectors.get(a)
        pv_b = profile_vectors.get(b)
        sm_a = spatial_maps.get(a)
        sm_b = spatial_maps.get(b)
        pcorr = _corr(pv_a, pv_b) if pv_a is not None and pv_b is not None else np.nan
        scorr = _corr(sm_a, sm_b) if sm_a is not None and sm_b is not None else np.nan
        rows.append({
            "comparison": f"{a}_vs_{b}",
            "event_a": a,
            "event_b": b,
            "profile_cosine_similarity": _cosine(pv_a, pv_b) if pv_a is not None and pv_b is not None else np.nan,
            "profile_pearson_correlation": pcorr,
            "profile_same_sign_fraction": _same_sign_fraction(pv_a, pv_b) if pv_a is not None and pv_b is not None else np.nan,
            "profile_top_feature_overlap_top5": _top_overlap(profile_diff, a, b, 5),
            "profile_top_feature_overlap_top10": _top_overlap(profile_diff, a, b, 10),
            "spatial_pattern_correlation": scorr,
            "spatial_cosine_similarity": _cosine(sm_a, sm_b) if sm_a is not None and sm_b is not None else np.nan,
            "spatial_same_sign_fraction": _same_sign_fraction(sm_a, sm_b) if sm_a is not None and sm_b is not None else np.nan,
            "interpretation_hint": _hint(pcorr, scorr, a, b),
        })
    return pd.DataFrame(rows)


def _hint(profile_corr: float, spatial_corr: float, a: str, b: str) -> str:
    vals = [v for v in (profile_corr, spatial_corr) if np.isfinite(v)]
    if not vals:
        return "insufficient_similarity_evidence"
    mean = float(np.nanmean(vals))
    if mean >= 0.6:
        return "similar_content_candidate_same_type_adjustment"
    if mean <= 0.2:
        return "low_similarity_candidate_distinct_content"
    return "moderate_similarity_unclear"
