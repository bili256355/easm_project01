from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "v7_x"
OUTPUT_TAG = "H_Jw_pattern_similarity_window_audit_v7_x"


@dataclass(frozen=True)
class Segment:
    name: str
    start_day: int
    end_day: int


@dataclass(frozen=True)
class V7XConfig:
    version: str = VERSION
    output_tag: str = OUTPUT_TAG
    input_output_tag: str = "w45_H_Jw_baseline_sensitive_state_growth_v7_v"
    input_curve_filename: str = "w45_H_Jw_state_progress_curves_v7_v.csv"
    detector_start_day: int = 0
    detector_end_day: int = 70
    w45_start_day: int = 40
    w45_end_day: int = 48
    h_object_window_start_day: int = 32
    h_object_window_end_day: int = 37
    jw_object_window_start_day: int = 37
    jw_object_window_end_day: int = 49
    window_width_days: int = 20
    local_peak_min_distance_days: int = 3
    candidate_band_min_half_width_days: int = 2
    candidate_band_max_half_width_days: int = 10
    peak_floor_quantile: float = 0.35
    prominence_ratio_threshold: float = 0.5
    lag_min_days: int = -8
    lag_max_days: int = 8
    lag_min_overlap_days: int = 4
    near_equal_epsilon: float = 0.01
    growth_quantile: float = 0.75
    dynamic_range_floor: float = 1e-8


SEGMENTS: Tuple[Segment, ...] = (
    Segment("pre_early_25_29", 25, 29),
    Segment("early_30_39", 30, 39),
    Segment("core_40_45", 40, 45),
    Segment("late_46_53", 46, 53),
    Segment("search_35_53", 35, 53),
    Segment("broad_30_53", 30, 53),
)

BASELINES: Tuple[str, ...] = (
    "C0_full_stage",
    "C1_buffered_stage",
    "C2_immediate_pre",
)
FIELDS: Tuple[str, ...] = ("H", "Jw")
SIGNALS: Tuple[str, ...] = ("R_diff", "S_pattern")


def _ensure_dirs(v7_root: Path) -> Tuple[Path, Path, Path]:
    output_dir = v7_root / "outputs" / OUTPUT_TAG
    log_dir = v7_root / "logs" / OUTPUT_TAG
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, log_dir, fig_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_input_path(v7_root: Path, cfg: V7XConfig) -> Path:
    env_path = os.environ.get("V7X_INPUT_CURVE_CSV")
    if env_path:
        return Path(env_path)
    return v7_root / "outputs" / cfg.input_output_tag / cfg.input_curve_filename


def _load_curves(v7_root: Path, cfg: V7XConfig) -> pd.DataFrame:
    path = _default_input_path(v7_root, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find V7-v state progress curve CSV: {path}. "
            "Run V7-v first, or set V7X_INPUT_CURVE_CSV to the CSV path."
        )
    df = pd.read_csv(path)
    required = {"baseline_config", "field", "day", "R_diff", "S_pattern"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input curve CSV is missing required columns: {missing}")
    df = df[df["baseline_config"].isin(BASELINES) & df["field"].isin(FIELDS)].copy()
    if df.empty:
        raise ValueError("Input curve CSV contains no H/Jw rows for C0/C1/C2 baselines.")
    df["day"] = df["day"].astype(int)
    df = df[(df["day"] >= cfg.detector_start_day) & (df["day"] <= cfg.detector_end_day)].copy()
    if df.empty:
        raise ValueError("No curve rows remain in detector day range 0-70.")
    return df.sort_values(["baseline_config", "field", "day"]).reset_index(drop=True)


def _add_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (baseline, field), group in df.groupby(["baseline_config", "field"], sort=False):
        g = group.sort_values("day").copy()
        g["dR_diff"] = g["R_diff"].diff().fillna(0.0)
        g["dS_pattern"] = g["S_pattern"].diff().fillna(0.0)
        g["dRdiff_smooth3"] = g["dR_diff"].rolling(window=3, center=True, min_periods=1).mean()
        g["dSpattern_smooth3"] = g["dS_pattern"].rolling(window=3, center=True, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _slice_segment(df: pd.DataFrame, segment: Segment) -> pd.DataFrame:
    return df[(df["day"] >= segment.start_day) & (df["day"] <= segment.end_day)].copy()


def _first_last_gain(seg: pd.DataFrame, value_col: str) -> Tuple[float, float, float]:
    if seg.empty:
        return (np.nan, np.nan, np.nan)
    s = seg.sort_values("day")
    start = float(s[value_col].iloc[0])
    end = float(s[value_col].iloc[-1])
    return (start, end, end - start)


def _positive_sum(seg: pd.DataFrame, value_col: str) -> float:
    if seg.empty:
        return float("nan")
    values = pd.to_numeric(seg[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.maximum(values, 0.0).sum())


def _make_trajectory_curves(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "baseline_config",
        "field",
        "day",
        "R_diff",
        "S_pattern",
        "dR_diff",
        "dS_pattern",
        "dRdiff_smooth3",
        "dSpattern_smooth3",
    ]
    return df[keep].copy()


def _make_early_ramp_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for baseline in BASELINES:
        for field in FIELDS:
            sub = df[(df["baseline_config"] == baseline) & (df["field"] == field)]
            denom_seg = sub[(sub["day"] >= 30) & (sub["day"] <= 53)]
            denom_r = _positive_sum(denom_seg, "dR_diff")
            denom_s = _positive_sum(denom_seg, "dS_pattern")
            for segment in SEGMENTS:
                seg = _slice_segment(sub, segment)
                r0, r1, rgain = _first_last_gain(seg, "R_diff")
                s0, s1, sgain = _first_last_gain(seg, "S_pattern")
                pos_r = _positive_sum(seg, "dR_diff")
                pos_s = _positive_sum(seg, "dS_pattern")
                rows.append(
                    {
                        "baseline_config": baseline,
                        "field": field,
                        "segment": segment.name,
                        "start_day": segment.start_day,
                        "end_day": segment.end_day,
                        "Rdiff_start": r0,
                        "Rdiff_end": r1,
                        "Rdiff_gain": rgain,
                        "Spattern_start": s0,
                        "Spattern_end": s1,
                        "Spattern_gain": sgain,
                        "positive_Rdiff_growth_sum": pos_r,
                        "positive_Spattern_growth_sum": pos_s,
                        "Rdiff_growth_share_of_day30_53": pos_r / denom_r if denom_r and not np.isnan(denom_r) else np.nan,
                        "Spattern_growth_share_of_day30_53": pos_s / denom_s if denom_s and not np.isnan(denom_s) else np.nan,
                        "mean_Spattern": float(seg["S_pattern"].mean()) if not seg.empty else np.nan,
                        "interpretation": _segment_growth_interpretation(field, segment.name, rgain, pos_r, denom_r),
                    }
                )
    return pd.DataFrame(rows)


def _segment_growth_interpretation(field: str, segment_name: str, gain: float, pos_sum: float, denom: float) -> str:
    if np.isnan(gain):
        return "no_data"
    share = pos_sum / denom if denom and not np.isnan(denom) else np.nan
    if segment_name == "early_30_39" and field == "Jw" and gain > 0 and (np.isnan(share) or share >= 0.25):
        return "candidate_early_pattern_ramp"
    if gain > 0:
        return "positive_pattern_similarity_growth"
    if gain < 0:
        return "negative_or_reversing_pattern_similarity_growth"
    return "near_zero_gain"


def _make_segment_advantage(df: pd.DataFrame, cfg: V7XConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for baseline in BASELINES:
        base = df[df["baseline_config"] == baseline]
        for segment in SEGMENTS:
            seg = _slice_segment(base, segment)
            pivot = seg.pivot_table(index="day", columns="field", values="S_pattern", aggfunc="mean")
            if not {"H", "Jw"}.issubset(pivot.columns):
                rows.append(
                    {
                        "baseline_config": baseline,
                        "segment": segment.name,
                        "start_day": segment.start_day,
                        "end_day": segment.end_day,
                        "mean_H_minus_Jw_Spattern": np.nan,
                        "H_ahead_fraction": np.nan,
                        "Jw_ahead_fraction": np.nan,
                        "near_equal_fraction": np.nan,
                        "dominant_segment_relation": "no_data",
                    }
                )
                continue
            diff = (pivot["H"] - pivot["Jw"]).dropna()
            if diff.empty:
                relation = "no_data"
                h_frac = j_frac = near_frac = np.nan
                mean_diff = np.nan
            else:
                h_frac = float((diff > cfg.near_equal_epsilon).mean())
                j_frac = float((diff < -cfg.near_equal_epsilon).mean())
                near_frac = float((diff.abs() <= cfg.near_equal_epsilon).mean())
                mean_diff = float(diff.mean())
                if h_frac >= 0.7:
                    relation = "H_ahead"
                elif j_frac >= 0.7:
                    relation = "Jw_ahead"
                elif near_frac >= 0.5:
                    relation = "near_equal"
                else:
                    relation = "mixed"
            rows.append(
                {
                    "baseline_config": baseline,
                    "segment": segment.name,
                    "start_day": segment.start_day,
                    "end_day": segment.end_day,
                    "mean_H_minus_Jw_Spattern": mean_diff,
                    "H_ahead_fraction": h_frac,
                    "Jw_ahead_fraction": j_frac,
                    "near_equal_fraction": near_frac,
                    "dominant_segment_relation": relation,
                }
            )
    return pd.DataFrame(rows)


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _make_lag_alignment(df: pd.DataFrame, cfg: V7XConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for baseline in BASELINES:
        base = df[df["baseline_config"] == baseline]
        h = base[base["field"] == "H"].set_index("day")["S_pattern"].sort_index()
        j = base[base["field"] == "Jw"].set_index("day")["S_pattern"].sort_index()
        for segment in SEGMENTS:
            corr_rows = []
            seg_days = np.arange(segment.start_day, segment.end_day + 1)
            for lag in range(cfg.lag_min_days, cfg.lag_max_days + 1):
                h_days = seg_days + lag
                common_mask = np.array([(d in h.index and sd in j.index) for d, sd in zip(h_days, seg_days)])
                if int(common_mask.sum()) < cfg.lag_min_overlap_days:
                    corr = np.nan
                    n = int(common_mask.sum())
                else:
                    hv = np.array([h.loc[d] for d in h_days[common_mask]], dtype=float)
                    jv = np.array([j.loc[d] for d in seg_days[common_mask]], dtype=float)
                    corr = _corr_safe(hv, jv)
                    n = int(common_mask.sum())
                corr_rows.append({"lag": lag, "corr": corr, "n_overlap": n})
            valid = [r for r in corr_rows if not np.isnan(r["corr"])]
            if not valid:
                best = {"lag": np.nan, "corr": np.nan, "n_overlap": 0}
            else:
                best = max(valid, key=lambda r: r["corr"])
            corr0 = next((r["corr"] for r in corr_rows if r["lag"] == 0), np.nan)
            interpretation = _lag_interpretation(best["lag"], best["corr"], corr0)
            rows.append(
                {
                    "baseline_config": baseline,
                    "segment": segment.name,
                    "start_day": segment.start_day,
                    "end_day": segment.end_day,
                    "best_lag": best["lag"],
                    "best_corr": best["corr"],
                    "corr_lag0": corr0,
                    "n_overlap_best": best["n_overlap"],
                    "lag_interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def _lag_interpretation(lag: float, best_corr: float, corr0: float) -> str:
    if np.isnan(lag) or np.isnan(best_corr):
        return "lag_not_interpretable"
    if best_corr < 0.3:
        return "lag_not_interpretable_low_corr"
    if abs(lag) <= 1:
        return "near_zero_lag"
    if lag > 1:
        return "Jw_pattern_trajectory_earlier_than_H_in_segment"
    return "H_pattern_trajectory_earlier_than_Jw_in_segment"


def _window_score_1d(days: np.ndarray, values: np.ndarray, width: int) -> pd.DataFrame:
    half = max(2, int(width // 2))
    rows: List[dict] = []
    day_to_value = {int(d): float(v) for d, v in zip(days, values)}
    min_day, max_day = int(days.min()), int(days.max())
    for t in range(min_day, max_day + 1):
        left_days = [d for d in range(t - half, t) if d in day_to_value]
        right_days = [d for d in range(t, t + half) if d in day_to_value]
        if len(left_days) < 2 or len(right_days) < 2:
            score = np.nan
        else:
            left = np.array([day_to_value[d] for d in left_days], dtype=float)
            right = np.array([day_to_value[d] for d in right_days], dtype=float)
            n1, n2 = len(left), len(right)
            score = float((n1 * n2 / (n1 + n2)) * (np.nanmean(right) - np.nanmean(left)) ** 2)
        rows.append({"day": t, "score": score})
    return pd.DataFrame(rows)


def _local_peaks(score_df: pd.DataFrame, min_distance: int) -> pd.DataFrame:
    s = score_df.dropna(subset=["score"]).copy().sort_values("day")
    if s.empty:
        return pd.DataFrame(columns=["peak_day", "peak_score", "peak_rank", "peak_prominence"])
    vals = s["score"].to_numpy(dtype=float)
    days = s["day"].to_numpy(dtype=int)
    candidates: List[Tuple[int, float]] = []
    for i in range(len(vals)):
        left = vals[i - 1] if i > 0 else -np.inf
        right = vals[i + 1] if i < len(vals) - 1 else -np.inf
        if vals[i] >= left and vals[i] >= right and vals[i] > 0:
            candidates.append((int(days[i]), float(vals[i])))
    # Greedy non-maximum suppression by score.
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected: List[Tuple[int, float]] = []
    for day, score in candidates:
        if all(abs(day - d) >= min_distance for d, _ in selected):
            selected.append((day, score))
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    rows: List[dict] = []
    score_map = {int(r.day): float(r.score) for r in s.itertuples()}
    for rank, (day, score) in enumerate(selected, start=1):
        left_window = [score_map[d] for d in range(day - 10, day) if d in score_map]
        right_window = [score_map[d] for d in range(day + 1, day + 11) if d in score_map]
        left_min = min(left_window) if left_window else 0.0
        right_min = min(right_window) if right_window else 0.0
        prominence = score - max(left_min, right_min)
        rows.append(
            {
                "peak_day": day,
                "peak_score": score,
                "peak_rank": rank,
                "peak_prominence": float(prominence),
            }
        )
    return pd.DataFrame(rows)


def _candidate_band(score_df: pd.DataFrame, peak_day: int, peak_score: float, cfg: V7XConfig) -> Tuple[int, int, float, str, str]:
    valid_scores = score_df.dropna(subset=["score"])
    if valid_scores.empty or np.isnan(peak_score):
        return peak_day, peak_day, np.nan, "no_score", "no_score"
    qfloor = float(valid_scores["score"].quantile(cfg.peak_floor_quantile))
    pfloor = float(peak_score * cfg.prominence_ratio_threshold)
    floor = max(qfloor, pfloor)
    score_map = {int(r.day): float(r.score) for r in valid_scores.itertuples()}
    start = peak_day
    end = peak_day
    left_reason = "max_half_width"
    right_reason = "max_half_width"
    for d in range(peak_day - 1, peak_day - cfg.candidate_band_max_half_width_days - 1, -1):
        if d not in score_map:
            left_reason = "missing_score"
            break
        if score_map[d] < floor and abs(d - peak_day) >= cfg.candidate_band_min_half_width_days:
            left_reason = "below_floor"
            break
        start = d
    for d in range(peak_day + 1, peak_day + cfg.candidate_band_max_half_width_days + 1):
        if d not in score_map:
            right_reason = "missing_score"
            break
        if score_map[d] < floor and abs(d - peak_day) >= cfg.candidate_band_min_half_width_days:
            right_reason = "below_floor"
            break
        end = d
    return start, end, floor, left_reason, right_reason


def _overlap_days(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _classify_peak(peak_day: int, start: int, end: int, cfg: V7XConfig) -> str:
    if end < cfg.w45_start_day:
        return "early_pattern_candidate"
    if start > cfg.w45_end_day:
        return "late_pattern_candidate"
    if _overlap_days(start, end, cfg.w45_start_day, cfg.w45_end_day) > 0:
        return "W45_pattern_candidate"
    return "weak_candidate"


def _make_state_detector_peaks(df: pd.DataFrame, cfg: V7XConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    peak_rows: List[dict] = []
    score_rows: List[pd.DataFrame] = []
    for baseline in BASELINES:
        for field in FIELDS:
            sub = df[(df["baseline_config"] == baseline) & (df["field"] == field)].sort_values("day")
            for signal in SIGNALS:
                s = sub[["day", signal]].dropna().copy()
                if s.empty:
                    continue
                score_df = _window_score_1d(s["day"].to_numpy(), s[signal].to_numpy(), cfg.window_width_days)
                score_df["baseline_config"] = baseline
                score_df["field"] = field
                score_df["signal"] = signal
                score_rows.append(score_df[["baseline_config", "field", "signal", "day", "score"]])
                peaks = _local_peaks(score_df, cfg.local_peak_min_distance_days)
                for r in peaks.itertuples(index=False):
                    start, end, floor, left_reason, right_reason = _candidate_band(score_df, int(r.peak_day), float(r.peak_score), cfg)
                    peak_rows.append(
                        {
                            "baseline_config": baseline,
                            "field": field,
                            "signal": signal,
                            "peak_day": int(r.peak_day),
                            "peak_score": float(r.peak_score),
                            "peak_prominence": float(r.peak_prominence),
                            "peak_rank": int(r.peak_rank),
                            "candidate_band_start": int(start),
                            "candidate_band_end": int(end),
                            "support_floor": floor,
                            "left_stop_reason": left_reason,
                            "right_stop_reason": right_reason,
                            "overlap_with_W45_days": _overlap_days(start, end, cfg.w45_start_day, cfg.w45_end_day),
                            "overlap_with_H_object_window_day32_37": _overlap_days(start, end, cfg.h_object_window_start_day, cfg.h_object_window_end_day),
                            "overlap_with_Jw_object_window_day37_49": _overlap_days(start, end, cfg.jw_object_window_start_day, cfg.jw_object_window_end_day),
                            "classification": _classify_peak(int(r.peak_day), int(start), int(end), cfg),
                        }
                    )
    peak_df = pd.DataFrame(peak_rows)
    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return peak_df, score_df


def _segments_above_threshold(days: np.ndarray, values: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    above = np.isfinite(values) & (values > threshold)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    last_day: Optional[int] = None
    for d, ok in zip(days, above):
        if ok and start is None:
            start = int(d)
            last_day = int(d)
        elif ok:
            last_day = int(d)
        elif start is not None:
            segments.append((start, int(last_day)))
            start = None
            last_day = None
    if start is not None:
        segments.append((start, int(last_day)))
    return segments


def _weighted_center(days: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(weights.astype(float), 0.0)
    if weights.sum() <= 0:
        return float(np.nanmean(days)) if len(days) else np.nan
    return float(np.sum(days * weights) / np.sum(weights))


def _make_growth_windows(df: pd.DataFrame, cfg: V7XConfig) -> pd.DataFrame:
    rows: List[dict] = []
    signal_map = {
        "R_diff": "dRdiff_smooth3",
        "S_pattern": "dSpattern_smooth3",
    }
    for baseline in BASELINES:
        for field in FIELDS:
            sub = df[(df["baseline_config"] == baseline) & (df["field"] == field)].sort_values("day")
            for signal, dcol in signal_map.items():
                valid = sub[["day", dcol]].dropna()
                if valid.empty:
                    continue
                threshold = float(valid[dcol].quantile(cfg.growth_quantile))
                days = valid["day"].to_numpy(dtype=int)
                values = valid[dcol].to_numpy(dtype=float)
                segments = _segments_above_threshold(days, values, threshold)
                # Positive-growth shares over the canonical diagnostic interval.
                diag = sub[(sub["day"] >= 30) & (sub["day"] <= 53)]
                total_pos = float(np.maximum(diag[dcol].to_numpy(dtype=float), 0.0).sum()) if not diag.empty else np.nan
                shares = {}
                for seg in ("early_30_39", "core_40_45", "late_46_53"):
                    sdef = next(s for s in SEGMENTS if s.name == seg)
                    tmp = sub[(sub["day"] >= sdef.start_day) & (sub["day"] <= sdef.end_day)]
                    pos = float(np.maximum(tmp[dcol].to_numpy(dtype=float), 0.0).sum()) if not tmp.empty else np.nan
                    shares[seg] = pos / total_pos if total_pos and not np.isnan(total_pos) else np.nan
                for idx, (start, end) in enumerate(segments, start=1):
                    w = valid[(valid["day"] >= start) & (valid["day"] <= end)].copy()
                    if w.empty:
                        continue
                    peak_idx = int(w[dcol].idxmax())
                    peak_day = int(valid.loc[peak_idx, "day"]) if peak_idx in valid.index else int(w.loc[w[dcol].idxmax(), "day"])
                    integrated = float(np.maximum(w[dcol].to_numpy(dtype=float), 0.0).sum())
                    center = _weighted_center(w["day"].to_numpy(dtype=float), w[dcol].to_numpy(dtype=float))
                    rows.append(
                        {
                            "baseline_config": baseline,
                            "field": field,
                            "signal": signal,
                            "growth_window_id": f"GW{idx:03d}",
                            "growth_window_start": int(start),
                            "growth_window_center": center,
                            "growth_window_end": int(end),
                            "peak_growth_day": peak_day,
                            "threshold_q75": threshold,
                            "integrated_positive_growth": integrated,
                            "early_growth_share": shares["early_30_39"],
                            "core_growth_share": shares["core_40_45"],
                            "late_growth_share": shares["late_46_53"],
                            "classification": _classify_peak(peak_day, start, end, cfg),
                        }
                    )
    return pd.DataFrame(rows)


def _baseline_result_map(table: pd.DataFrame, filter_expr: pd.Series, value_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for baseline in BASELINES:
        sub = table[filter_expr & (table["baseline_config"] == baseline)]
        if sub.empty:
            out[baseline] = "no_data"
        else:
            out[baseline] = str(sub.iloc[0][value_col])
    return out


def _make_final_summary(
    ramp: pd.DataFrame,
    advantage: pd.DataFrame,
    lag: pd.DataFrame,
    peaks: pd.DataFrame,
    growth: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    # Jw early R_diff gain.
    jw_early = ramp[(ramp["field"] == "Jw") & (ramp["segment"] == "early_30_39")]
    h_early = ramp[(ramp["field"] == "H") & (ramp["segment"] == "early_30_39")]
    def _fmt_gain(sub: pd.DataFrame) -> str:
        return "; ".join(
            f"{r.baseline_config}: Rgain={r.Rdiff_gain:.4g}, Sgain={r.Spattern_gain:.4g}, Rshare={r.Rdiff_growth_share_of_day30_53:.3f}"
            for r in sub.itertuples()
        )
    rows.append(
        {
            "question": "Does Jw show an early day30-39 raw R_diff ramp?",
            "C0_result": _row_text(jw_early, "C0_full_stage", "Rdiff_gain"),
            "C1_result": _row_text(jw_early, "C1_buffered_stage", "Rdiff_gain"),
            "C2_result": _row_text(jw_early, "C2_immediate_pre", "Rdiff_gain"),
            "baseline_sensitivity": _same_sign_sensitivity(jw_early, "Rdiff_gain"),
            "allowed_statement": "Jw early pattern-similarity ramp is supported if R_diff gain is positive across C0/C1/C2; this is a pattern-similarity signal, not a profile-object peak.",
            "forbidden_statement": "Do not call it a confirmed Jw object-level transition window without detector/bootstrap support.",
            "evidence_detail": f"Jw: {_fmt_gain(jw_early)} | H: {_fmt_gain(h_early)}",
        }
    )

    for seg_name in ("early_30_39", "core_40_45", "late_46_53"):
        sub = advantage[advantage["segment"] == seg_name]
        rows.append(
            {
                "question": f"Which field is ahead in S_pattern during {seg_name}?",
                "C0_result": _row_text(sub, "C0_full_stage", "dominant_segment_relation"),
                "C1_result": _row_text(sub, "C1_buffered_stage", "dominant_segment_relation"),
                "C2_result": _row_text(sub, "C2_immediate_pre", "dominant_segment_relation"),
                "baseline_sensitivity": _categorical_sensitivity(sub, "dominant_segment_relation"),
                "allowed_statement": "Use only as segment-level pattern-progress relation.",
                "forbidden_statement": "Do not collapse segment-specific relations into a single whole-window lead.",
                "evidence_detail": "; ".join(
                    f"{r.baseline_config}: meanHminusJw={r.mean_H_minus_Jw_Spattern:.4g}, rel={r.dominant_segment_relation}"
                    for r in sub.itertuples()
                ),
            }
        )

    early_lag = lag[lag["segment"] == "early_30_39"]
    rows.append(
        {
            "question": "Does lag alignment support early Jw pattern leading?",
            "C0_result": _row_text(early_lag, "C0_full_stage", "lag_interpretation"),
            "C1_result": _row_text(early_lag, "C1_buffered_stage", "lag_interpretation"),
            "C2_result": _row_text(early_lag, "C2_immediate_pre", "lag_interpretation"),
            "baseline_sensitivity": _categorical_sensitivity(early_lag, "lag_interpretation"),
            "allowed_statement": "Early segment lag can indicate local phase offset if best correlations are interpretable.",
            "forbidden_statement": "Do not treat early-segment lag as whole-window phase lead.",
            "evidence_detail": "; ".join(
                f"{r.baseline_config}: lag={r.best_lag}, corr={r.best_corr:.3f}, interp={r.lag_interpretation}"
                for r in early_lag.itertuples()
            ),
        }
    )

    jw_early_peaks = peaks[(peaks["field"] == "Jw") & (peaks["classification"] == "early_pattern_candidate")]
    rows.append(
        {
            "question": "Does pattern-only state detector produce Jw early candidate bands?",
            "C0_result": _peak_result(jw_early_peaks, "C0_full_stage"),
            "C1_result": _peak_result(jw_early_peaks, "C1_buffered_stage"),
            "C2_result": _peak_result(jw_early_peaks, "C2_immediate_pre"),
            "baseline_sensitivity": "stable_if_early_candidates_exist_across_C0_C1_C2",
            "allowed_statement": "If present, call these pattern-similarity candidate bands only.",
            "forbidden_statement": "Do not replace V7-w profile-object windows with pattern-only candidates.",
            "evidence_detail": f"n_Jw_early_pattern_candidates={len(jw_early_peaks)}",
        }
    )
    return pd.DataFrame(rows)


def _row_text(df: pd.DataFrame, baseline: str, col: str) -> str:
    sub = df[df["baseline_config"] == baseline]
    if sub.empty:
        return "no_data"
    val = sub.iloc[0][col]
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "nan"
        return f"{float(val):.6g}"
    return str(val)


def _same_sign_sensitivity(df: pd.DataFrame, col: str) -> str:
    values = []
    for baseline in BASELINES:
        sub = df[df["baseline_config"] == baseline]
        if sub.empty:
            return "missing_baseline"
        v = float(sub.iloc[0][col])
        if np.isnan(v):
            return "nan_baseline"
        values.append(v)
    if all(v > 0 for v in values):
        return "positive_across_baselines"
    if all(v < 0 for v in values):
        return "negative_across_baselines"
    return "sign_sensitive"


def _categorical_sensitivity(df: pd.DataFrame, col: str) -> str:
    values = []
    for baseline in BASELINES:
        sub = df[df["baseline_config"] == baseline]
        if sub.empty:
            return "missing_baseline"
        values.append(str(sub.iloc[0][col]))
    if len(set(values)) == 1:
        return "stable_across_baselines"
    return "baseline_sensitive"


def _peak_result(peaks: pd.DataFrame, baseline: str) -> str:
    sub = peaks[peaks["baseline_config"] == baseline]
    if sub.empty:
        return "no_early_pattern_candidate"
    # Keep top-ranked early peak for concise summary.
    sub = sub.sort_values(["peak_rank", "peak_score"], ascending=[True, False])
    r = sub.iloc[0]
    return f"peak_day={int(r.peak_day)}, band={int(r.candidate_band_start)}-{int(r.candidate_band_end)}, signal={r.signal}"


def _make_figures(df: pd.DataFrame, score_df: pd.DataFrame, growth: pd.DataFrame, fig_dir: Path) -> None:
    if os.environ.get("V7X_SKIP_FIGURES", "0") == "1":
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on env
        warnings.warn(f"Skipping figures because matplotlib import failed: {exc}")
        return

    for baseline in BASELINES:
        sub = df[df["baseline_config"] == baseline]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for field in FIELDS:
            g = sub[sub["field"] == field]
            ax.plot(g["day"], g["R_diff"], label=f"{field} R_diff")
        ax.axvspan(40, 48, alpha=0.15, label="system W45")
        ax.axvspan(30, 39, alpha=0.08, label="early")
        ax.set_title(f"R_diff trajectory - {baseline}")
        ax.set_xlabel("day")
        ax.set_ylabel("R_diff")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"H_Jw_Rdiff_trajectory_{baseline}_v7_x.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for field in FIELDS:
            g = sub[sub["field"] == field]
            ax.plot(g["day"], g["S_pattern"], label=f"{field} S_pattern")
        ax.axvspan(40, 48, alpha=0.15, label="system W45")
        ax.axvspan(30, 39, alpha=0.08, label="early")
        ax.set_title(f"S_pattern trajectory - {baseline}")
        ax.set_xlabel("day")
        ax.set_ylabel("S_pattern")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"H_Jw_Spattern_trajectory_{baseline}_v7_x.png", dpi=160)
        plt.close(fig)

        ssub = score_df[score_df["baseline_config"] == baseline]
        if not ssub.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for (field, signal), g in ssub.groupby(["field", "signal"]):
                ax.plot(g["day"], g["score"], label=f"{field} {signal}")
            ax.axvspan(40, 48, alpha=0.15, label="system W45")
            ax.set_title(f"Pattern state detector scores - {baseline}")
            ax.set_xlabel("day")
            ax.set_ylabel("local before-after score")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"H_Jw_pattern_state_detector_scores_{baseline}_v7_x.png", dpi=160)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for field in FIELDS:
            g = sub[sub["field"] == field]
            ax.plot(g["day"], g["dRdiff_smooth3"], label=f"{field} dRdiff_smooth3")
        ax.axvspan(40, 48, alpha=0.15, label="system W45")
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"Pattern growth speed - {baseline}")
        ax.set_xlabel("day")
        ax.set_ylabel("dRdiff_smooth3")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"H_Jw_pattern_growth_speed_{baseline}_v7_x.png", dpi=160)
        plt.close(fig)


def _write_markdown_summary(path: Path, final_summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# V7-x H/Jw pattern-similarity trajectory window audit")
    lines.append("")
    lines.append("This targeted side audit reads V7-v state-progress curves and checks whether the Jw day30-39 early pattern signal is present in raw R_diff, S_pattern segment advantage, lag alignment, and pattern-only detector scores.")
    lines.append("")
    lines.append("## Key audit questions")
    lines.append("")
    for r in final_summary.itertuples(index=False):
        lines.append(f"### {r.question}")
        lines.append(f"- C0: {r.C0_result}")
        lines.append(f"- C1: {r.C1_result}")
        lines.append(f"- C2: {r.C2_result}")
        lines.append(f"- Baseline sensitivity: {r.baseline_sensitivity}")
        lines.append(f"- Allowed statement: {r.allowed_statement}")
        lines.append(f"- Forbidden statement: {r.forbidden_statement}")
        lines.append(f"- Evidence detail: {r.evidence_detail}")
        lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append("V7-x does not replace V7-w profile-object windows. A pattern-similarity candidate or ramp can support a targeted statement about pattern similarity, but it must not be upgraded to a confirmed object-level profile transition window without separate support.")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_H_Jw_pattern_similarity_window_audit_v7_x(v7_root: Path | str) -> None:
    v7_root = Path(v7_root)
    cfg = V7XConfig()
    output_dir, log_dir, fig_dir = _ensure_dirs(v7_root)
    input_path = _default_input_path(v7_root, cfg)

    curves = _load_curves(v7_root, cfg)
    curves = _add_derivatives(curves)

    trajectory = _make_trajectory_curves(curves)
    ramp = _make_early_ramp_audit(curves)
    advantage = _make_segment_advantage(curves, cfg)
    lag = _make_lag_alignment(curves, cfg)
    peaks, score_df = _make_state_detector_peaks(curves, cfg)
    growth = _make_growth_windows(curves, cfg)
    final_summary = _make_final_summary(ramp, advantage, lag, peaks, growth)

    trajectory.to_csv(output_dir / "H_Jw_pattern_similarity_trajectory_curves_v7_x.csv", index=False)
    ramp.to_csv(output_dir / "H_Jw_early_pattern_ramp_audit_v7_x.csv", index=False)
    advantage.to_csv(output_dir / "H_Jw_pattern_segment_advantage_v7_x.csv", index=False)
    lag.to_csv(output_dir / "H_Jw_pattern_lag_alignment_v7_x.csv", index=False)
    peaks.to_csv(output_dir / "H_Jw_pattern_state_detector_peaks_v7_x.csv", index=False)
    score_df.to_csv(output_dir / "H_Jw_pattern_state_detector_scores_v7_x.csv", index=False)
    growth.to_csv(output_dir / "H_Jw_pattern_growth_windows_v7_x.csv", index=False)
    final_summary.to_csv(output_dir / "H_Jw_pattern_similarity_window_audit_summary_v7_x.csv", index=False)

    _write_markdown_summary(output_dir / "H_Jw_pattern_similarity_window_audit_summary_v7_x.md", final_summary)
    _make_figures(curves, score_df, growth, fig_dir)

    run_meta = {
        "version": cfg.version,
        "output_tag": cfg.output_tag,
        "primary_goal": "targeted side audit of H/Jw early pattern-similarity signal from V7-v curves",
        "input_curve_csv": str(input_path),
        "input_is_existing_V7v_output": True,
        "fields": list(FIELDS),
        "baseline_configs": list(BASELINES),
        "signals": list(SIGNALS),
        "segments": [asdict(s) for s in SEGMENTS],
        "detector_day_range": [cfg.detector_start_day, cfg.detector_end_day],
        "state_detector": {
            "type": "one_dimensional_local_before_after_window_score",
            "window_width_days": cfg.window_width_days,
            "local_peak_min_distance_days": cfg.local_peak_min_distance_days,
            "candidate_band_max_half_width_days": cfg.candidate_band_max_half_width_days,
            "note": "This is a pattern-similarity targeted side audit, not a replacement for V7-w profile-object detection.",
        },
        "growth_detector": {
            "signals": ["dRdiff_smooth3", "dSpattern_smooth3"],
            "threshold_quantile": cfg.growth_quantile,
            "role": "diagnostic_only",
        },
        "external_references": {
            "system_W45": [cfg.w45_start_day, cfg.w45_end_day],
            "H_V7w_object_window": [cfg.h_object_window_start_day, cfg.h_object_window_end_day],
            "Jw_V7w_object_window": [cfg.jw_object_window_start_day, cfg.jw_object_window_end_day],
        },
        "not_done": [
            "No raw field recomputation",
            "No P/V/Je",
            "No profile-object window replacement",
            "No causality or pathway interpretation",
        ],
    }
    _write_json(output_dir / "run_meta.json", run_meta)
    _write_json(log_dir / "run_meta.json", run_meta)

    summary_payload = {
        "status": "success",
        "output_dir": str(output_dir),
        "n_curve_rows": int(len(curves)),
        "n_pattern_state_peaks": int(len(peaks)),
        "n_pattern_growth_windows": int(len(growth)),
        "key_outputs": [
            "H_Jw_pattern_similarity_trajectory_curves_v7_x.csv",
            "H_Jw_early_pattern_ramp_audit_v7_x.csv",
            "H_Jw_pattern_segment_advantage_v7_x.csv",
            "H_Jw_pattern_lag_alignment_v7_x.csv",
            "H_Jw_pattern_state_detector_peaks_v7_x.csv",
            "H_Jw_pattern_growth_windows_v7_x.csv",
            "H_Jw_pattern_similarity_window_audit_summary_v7_x.csv",
        ],
    }
    _write_json(output_dir / "summary.json", summary_payload)

    print("[V7-x] H/Jw pattern-similarity trajectory audit completed.")
    print(f"[V7-x] Output directory: {output_dir}")


__all__ = ["run_H_Jw_pattern_similarity_window_audit_v7_x"]
