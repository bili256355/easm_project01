from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .config import Settings
from .utils import day_index_to_month_day, trapezoid_integral


def add_width_to_peaks(local_peaks_df: pd.DataFrame, width: int) -> pd.DataFrame:
    df = local_peaks_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["detector_width", "candidate_id", "candidate_day", "month_day", "candidate_score", "candidate_prominence", "candidate_rank", "source_type"])
    df = df.rename(columns={"peak_day": "candidate_day", "peak_score": "candidate_score", "peak_prominence": "candidate_prominence", "peak_rank": "candidate_rank"})
    df.insert(0, "detector_width", int(width))
    df["candidate_id"] = [f"H_W{int(width):02d}_C{i:03d}" for i in range(1, len(df) + 1)]
    df["month_day"] = df["candidate_day"].astype(int).map(day_index_to_month_day)
    keep = ["detector_width", "candidate_id", "candidate_day", "month_day", "candidate_score", "candidate_prominence", "candidate_rank", "source_type"]
    return df[keep]


def build_full_curve_table(width_outputs: dict[int, dict[str, Any]]) -> pd.DataFrame:
    dfs = []
    from .main_method_detector import profile_to_dataframe
    for width, out in width_outputs.items():
        dfs.append(profile_to_dataframe(out["profile"], width))
    if not dfs:
        return pd.DataFrame(columns=["detector_width", "day", "score"])
    return pd.concat(dfs, ignore_index=True)


def build_candidate_catalog(width_outputs: dict[int, dict[str, Any]]) -> pd.DataFrame:
    dfs = []
    for width, out in width_outputs.items():
        dfs.append(add_width_to_peaks(out["local_peaks_df"], width))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _nearest_candidate(candidates: pd.DataFrame, width: int, target_day: int) -> dict[str, Any]:
    sub = candidates[candidates["detector_width"].astype(int) == int(width)].copy()
    if sub.empty:
        return {"nearest_candidate_day": np.nan, "nearest_candidate_score": np.nan, "nearest_candidate_distance": np.nan}
    sub["distance"] = (sub["candidate_day"].astype(int) - int(target_day)).abs()
    sub = sub.sort_values(["distance", "candidate_score", "candidate_day"], ascending=[True, False, True])
    row = sub.iloc[0]
    return {
        "nearest_candidate_day": int(row["candidate_day"]),
        "nearest_candidate_score": float(row["candidate_score"]),
        "nearest_candidate_distance": int(row["distance"]),
    }


def _candidates_in_range(candidates: pd.DataFrame, width: int, start: int, end: int) -> pd.DataFrame:
    sub = candidates[candidates["detector_width"].astype(int) == int(width)].copy()
    if sub.empty:
        return sub
    return sub[(sub["candidate_day"].astype(int) >= int(start)) & (sub["candidate_day"].astype(int) <= int(end))].copy()


def _candidates_between(candidates: pd.DataFrame, width: int, start: int, end: int) -> list[int]:
    sub = _candidates_in_range(candidates, width, start, end)
    if sub.empty:
        return []
    return sorted(sub["candidate_day"].astype(int).tolist())


def _window_auc(curve: pd.DataFrame, width: int, start: int, end: int) -> tuple[float, float, int | None]:
    sub = curve[(curve["detector_width"].astype(int) == int(width)) & (curve["day"].astype(int) >= int(start)) & (curve["day"].astype(int) <= int(end))].copy()
    if sub.empty:
        return np.nan, np.nan, None
    sub = sub.sort_values("day")
    auc = trapezoid_integral(sub["score"].to_numpy(), sub["day"].to_numpy())
    idx = int(sub["score"].astype(float).idxmax())
    return float(auc), float(sub.loc[idx, "score"]), int(sub.loc[idx, "day"])


def classify_h_role(inside: list[int], pre: list[int], post: list[int], width_sensitive: bool = False) -> str:
    if inside and not pre and not post:
        return "inside_window_main_candidate"
    if inside and pre:
        return "pre_and_inside_candidate"
    if inside and post:
        return "inside_and_post_candidate"
    if pre and post and not inside:
        return "pre_and_post_without_inside"
    if pre and not inside:
        return "pre_window_candidate_only"
    if post and not inside:
        return "post_window_candidate_only"
    return "window_absent"


def build_h_event_atlas(cfg: Settings, curve: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    widths = [int(w) for w in cfg.detector.widths]
    for win_id, center, start, end in cfg.windows.strong_windows:
        pre_start = max(0, int(start) - int(cfg.windows.pre_search_days))
        pre_end = int(start) - 1
        post_start = int(end) + 1
        post_end = int(end) + int(cfg.windows.post_search_days)
        for width in widths:
            inside = _candidates_between(candidates, width, start, end)
            pre = _candidates_between(candidates, width, pre_start, pre_end)
            post = _candidates_between(candidates, width, post_start, post_end)
            nearest = _nearest_candidate(candidates, width, center)
            auc, max_score, max_day = _window_auc(curve, width, start, end)
            role = classify_h_role(inside, pre, post)
            if role == "window_absent" and nearest.get("nearest_candidate_distance", np.nan) <= 10:
                role = "near_window_candidate_without_inside_marker"
            rows.append(
                {
                    "window_id": win_id,
                    "window_center": int(center),
                    "window_start": int(start),
                    "window_end": int(end),
                    "detector_width": int(width),
                    "h_inside_window_candidates": ";".join(map(str, inside)),
                    "h_pre_window_candidates": ";".join(map(str, pre)),
                    "h_post_window_candidates": ";".join(map(str, post)),
                    "window_auc_score": auc,
                    "window_max_score": max_score,
                    "window_max_score_day": max_day,
                    **nearest,
                    "h_role_class": role,
                    "h_event_package_summary": _format_event_package(inside, pre, post, role),
                }
            )
    return pd.DataFrame(rows)


def _format_event_package(inside: list[int], pre: list[int], post: list[int], role: str) -> str:
    return f"role={role}; pre={pre if pre else 'none'}; inside={inside if inside else 'none'}; post={post if post else 'none'}"


def build_width_stability_summary(cfg: Settings, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline_width = int(cfg.detector.baseline_width)
    baseline = candidates[candidates["detector_width"].astype(int) == baseline_width].copy()
    rows: list[dict[str, Any]] = []
    if baseline.empty:
        return pd.DataFrame(columns=["baseline_candidate_day", "matched_widths", "n_matched_widths", "min_day", "max_day", "max_shift_abs", "stable_under_width_flag", "matched_candidate_days_by_width", "interpretation_note"])

    for _, brow in baseline.sort_values("candidate_day").iterrows():
        bday = int(brow["candidate_day"])
        matched: dict[int, int] = {}
        for width in cfg.detector.widths:
            sub = candidates[candidates["detector_width"].astype(int) == int(width)].copy()
            if sub.empty:
                continue
            sub["abs_shift"] = (sub["candidate_day"].astype(int) - bday).abs()
            sub = sub.sort_values(["abs_shift", "candidate_score"], ascending=[True, False])
            best = sub.iloc[0]
            if int(best["abs_shift"]) <= cfg.reference.expected_match_tolerance_days:
                matched[int(width)] = int(best["candidate_day"])
        days = list(matched.values())
        max_shift_abs = max([abs(d - bday) for d in days], default=np.nan)
        stable = len(matched) >= max(3, len(cfg.detector.widths) - 1) and (np.isfinite(max_shift_abs) and max_shift_abs <= cfg.reference.expected_match_tolerance_days)
        rows.append(
            {
                "baseline_candidate_day": bday,
                "matched_widths": ";".join(map(str, sorted(matched.keys()))),
                "n_matched_widths": int(len(matched)),
                "min_day": int(min(days)) if days else np.nan,
                "max_day": int(max(days)) if days else np.nan,
                "max_shift_abs": float(max_shift_abs) if np.isfinite(max_shift_abs) else np.nan,
                "stable_under_width_flag": bool(stable),
                "matched_candidate_days_by_width": ";".join([f"w{w}:{d}" for w, d in sorted(matched.items())]),
                "interpretation_note": "width-stable baseline H candidate" if stable else "width-sensitive or weakly matched H candidate",
            }
        )
    return pd.DataFrame(rows)


def build_baseline_reproduction_audit(cfg: Settings, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline_width = int(cfg.detector.baseline_width)
    expected = list(cfg.reference.expected_h_candidates_width20)
    actual = sorted(candidates[candidates["detector_width"].astype(int) == baseline_width]["candidate_day"].astype(int).tolist())
    rows: list[dict[str, Any]] = []
    for e in expected:
        if actual:
            nearest = min(actual, key=lambda x: abs(x - e))
            off = abs(nearest - e)
        else:
            nearest = None
            off = None
        rows.append(
            {
                "expected_candidate_day": int(e),
                "nearest_actual_candidate_day": int(nearest) if nearest is not None else np.nan,
                "abs_offset": int(off) if off is not None else np.nan,
                "matched_within_tolerance": bool(off is not None and off <= cfg.reference.expected_match_tolerance_days),
            }
        )
    audit = pd.DataFrame(rows)
    return audit
