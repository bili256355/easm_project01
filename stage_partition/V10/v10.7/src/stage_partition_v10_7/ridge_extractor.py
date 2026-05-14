from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_prominences
except Exception:  # pragma: no cover
    find_peaks = None
    peak_prominences = None


def _nearest_label(day: int, target_days: dict[str, int]) -> tuple[str, int, int]:
    if not target_days:
        return "", -999, 999
    rows = [(label, int(tday), abs(int(day) - int(tday))) for label, tday in target_days.items()]
    rows.sort(key=lambda x: (x[2], x[1]))
    return rows[0]


def extract_scale_local_maxima(
    energy_map: pd.DataFrame,
    target_days: dict[str, int],
    focus_day_min: int,
    focus_day_max: int,
    percentile_threshold: float,
    min_prominence_norm: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sigma, sub in energy_map.groupby("sigma", sort=True):
        s = sub.sort_values("day").reset_index(drop=True)
        s = s[(s["day"] >= int(focus_day_min)) & (s["day"] <= int(focus_day_max))].reset_index(drop=True)
        if s.shape[0] < 3:
            continue
        values = s["energy_norm_within_sigma"].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, 0.0)
        if find_peaks is not None:
            peaks, _ = find_peaks(values)
            if peak_prominences is not None and len(peaks):
                prominences, _, _ = peak_prominences(values, peaks)
            else:
                prominences = np.zeros(len(peaks), dtype=float)
        else:
            peaks = np.asarray([i for i in range(1, len(values) - 1) if values[i] >= values[i - 1] and values[i] >= values[i + 1]], dtype=int)
            prominences = np.asarray([max(0.0, values[i] - max(values[i - 1], values[i + 1])) for i in peaks], dtype=float)
        for idx, prom in zip(peaks, prominences):
            row = s.iloc[int(idx)]
            pct = float(row["energy_percentile_within_sigma"])
            if pct < float(percentile_threshold):
                continue
            if float(prom) < float(min_prominence_norm):
                continue
            label, tday, dist = _nearest_label(int(row["day"]), target_days)
            rows.append({
                "sigma": float(sigma),
                "day": int(row["day"]),
                "energy_raw": float(row["energy_raw"]),
                "energy_norm": float(row["energy_norm_within_sigma"]),
                "energy_percentile": pct,
                "energy_robust_z": float(row["energy_robust_z_within_sigma"]),
                "prominence_norm": float(prom),
                "boundary_risk_flag": bool(row["boundary_risk_flag"]),
                "nearest_target_label": label,
                "nearest_target_day": int(tday),
                "nearest_target_distance": int(dist),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "sigma", "day", "energy_raw", "energy_norm", "energy_percentile", "energy_robust_z",
            "prominence_norm", "boundary_risk_flag", "nearest_target_label", "nearest_target_day", "nearest_target_distance",
        ])
    return pd.DataFrame(rows).sort_values(["sigma", "day"]).reset_index(drop=True)


def link_scale_ridges(local_maxima: pd.DataFrame, ridge_link_radius_days: int) -> pd.DataFrame:
    if local_maxima is None or local_maxima.empty:
        return pd.DataFrame(columns=list(local_maxima.columns) + ["ridge_id"] if local_maxima is not None else ["ridge_id"])
    df = local_maxima.sort_values(["sigma", "day", "energy_norm"], ascending=[True, True, False]).reset_index(drop=True)
    ridges: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    next_id = 1
    for sigma in sorted(df["sigma"].unique()):
        sub = df[df["sigma"] == sigma].copy().sort_values(["energy_norm", "day"], ascending=[False, True])
        used_ridges: set[int] = set()
        for _, row in sub.iterrows():
            day = int(row["day"])
            candidates = []
            for rid, info in ridges.items():
                if rid in used_ridges:
                    continue
                if float(info["last_sigma"]) >= float(sigma):
                    continue
                dist = abs(day - int(info["last_day"]))
                if dist <= int(ridge_link_radius_days):
                    sigma_gap = float(sigma) - float(info["last_sigma"])
                    candidates.append((dist, sigma_gap, -float(info["last_energy_norm"]), rid))
            if candidates:
                candidates.sort()
                rid = candidates[0][3]
            else:
                rid = next_id
                next_id += 1
            used_ridges.add(rid)
            ridges[rid] = {
                "last_day": day,
                "last_sigma": float(sigma),
                "last_energy_norm": float(row["energy_norm"]),
            }
            d = row.to_dict()
            d["ridge_id"] = f"R{rid:03d}"
            rows.append(d)
    return pd.DataFrame(rows).sort_values(["ridge_id", "sigma", "day"]).reset_index(drop=True)


def _role_hint_for_ridge(day_min: int, day_max: int, sigma_min: float, sigma_max: float, persistence: float, max_norm: float, nearest_label: str, nearest_distance: int) -> str:
    day_range = int(day_max - day_min)
    if nearest_label == "H45" and nearest_distance <= 3 and persistence < 0.30:
        return "no_clear_scale_structure_near_H45"
    if day_range >= 12 or (day_min <= 22 and day_max >= 30):
        return "broad_prewindow_adjustment"
    if persistence >= 0.60 and day_range <= 6 and max_norm >= 0.65:
        return "stable_cross_scale_transition"
    if 0.30 <= persistence < 0.60 and sigma_min >= 3 and sigma_max <= 11 and day_range <= 8:
        return "medium_scale_local_bump"
    if persistence < 0.30:
        return "weak_low_persistence_fluctuation"
    return "candidate_scale_structure"


def summarize_ridge_families(ridges: pd.DataFrame, sigmas: tuple[float, ...], target_days: dict[str, int]) -> pd.DataFrame:
    if ridges is None or ridges.empty:
        return pd.DataFrame(columns=[
            "ridge_id", "day_min", "day_max", "day_center_weighted", "sigma_min", "sigma_max",
            "sigma_count", "persistence_fraction", "max_energy_norm", "mean_energy_norm", "max_energy_day",
            "max_energy_sigma", "nearest_target_label", "nearest_target_day", "nearest_target_distance", "role_hint"
        ])
    rows = []
    n_sigmas = max(1, len(sigmas))
    for rid, sub in ridges.groupby("ridge_id", sort=True):
        sub = sub.copy()
        weights = sub["energy_norm"].to_numpy(dtype=float)
        days = sub["day"].to_numpy(dtype=float)
        if np.nansum(weights) > 0:
            day_center = float(np.nansum(days * weights) / np.nansum(weights))
        else:
            day_center = float(np.nanmean(days))
        imax = int(sub["energy_norm"].astype(float).idxmax())
        max_row = sub.loc[imax]
        label, tday, dist = _nearest_label(int(round(day_center)), target_days)
        day_min = int(sub["day"].min())
        day_max = int(sub["day"].max())
        sigma_min = float(sub["sigma"].min())
        sigma_max = float(sub["sigma"].max())
        sigma_count = int(sub["sigma"].nunique())
        persistence = float(sigma_count / n_sigmas)
        max_norm = float(sub["energy_norm"].max())
        role = _role_hint_for_ridge(day_min, day_max, sigma_min, sigma_max, persistence, max_norm, label, int(dist))
        rows.append({
            "ridge_id": rid,
            "day_min": day_min,
            "day_max": day_max,
            "day_center_weighted": day_center,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "sigma_count": sigma_count,
            "persistence_fraction": persistence,
            "max_energy_norm": max_norm,
            "mean_energy_norm": float(sub["energy_norm"].mean()),
            "max_energy_day": int(max_row["day"]),
            "max_energy_sigma": float(max_row["sigma"]),
            "nearest_target_label": label,
            "nearest_target_day": int(tday),
            "nearest_target_distance": int(dist),
            "role_hint": role,
        })
    return pd.DataFrame(rows).sort_values(["day_center_weighted", "sigma_min"]).reset_index(drop=True)


def build_target_day_scale_response(
    energy_map: pd.DataFrame,
    ridges: pd.DataFrame,
    target_days: dict[str, int],
    target_radius_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, target_day in target_days.items():
        for sigma, sub in energy_map.groupby("sigma", sort=True):
            part = sub[(sub["day"] >= int(target_day) - int(target_radius_days)) & (sub["day"] <= int(target_day) + int(target_radius_days))].copy()
            if part.empty:
                rows.append({"target_label": label, "target_day": int(target_day), "sigma": float(sigma), "local_max_day_within_radius": np.nan})
                continue
            part = part.sort_values(["energy_norm_within_sigma", "energy_percentile_within_sigma", "day"], ascending=[False, False, True])
            best = part.iloc[0]
            nearby_ridges = pd.DataFrame()
            if ridges is not None and not ridges.empty:
                nearby_ridges = ridges[(ridges["sigma"].astype(float) == float(sigma)) & ((ridges["day"].astype(int) - int(best["day"])).abs() <= int(target_radius_days))]
            if not nearby_ridges.empty:
                nearby_ridges = nearby_ridges.sort_values(["energy_norm", "day"], ascending=[False, True])
                nearest_ridge_id = str(nearby_ridges.iloc[0]["ridge_id"])
                has_local_ridge = True
            else:
                nearest_ridge_id = ""
                has_local_ridge = False
            rows.append({
                "target_label": label,
                "target_day": int(target_day),
                "sigma": float(sigma),
                "local_max_day_within_radius": int(best["day"]),
                "energy_raw": float(best["energy_raw"]),
                "energy_norm": float(best["energy_norm_within_sigma"]),
                "energy_percentile": float(best["energy_percentile_within_sigma"]),
                "has_local_ridge_nearby": bool(has_local_ridge),
                "nearest_ridge_id": nearest_ridge_id,
            })
    return pd.DataFrame(rows)
