from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math
import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# V10.7_n: H_zonal_width background / target / preinformation audit
# =============================================================================
# Questions answered:
# Qn1: Is the H_zonal_width signal local to E2, or a broader pre-M background?
# Qn2: Is the P target rainband-only, position-only, or a position-rainband
#      north-south structural reorganization?
# Qn3: Does E2 / pre-M H_zonal_width carry weak preinformation value beyond
#      common-year/background explanations?
#
# Method boundary:
# - NOT causal inference.
# - NOT full W33 -> W45 mapping.
# - NOT full P/V/H/Je/Jw object-network audit.
# - NOT proof that transition windows represent most object information.
# - Does NOT control away P/V/Je/Jw.
# - Does NOT re-test scalarized-transition failure from V10.7_m.
# =============================================================================


@dataclass
class Settings:
    project_root: Path

    # statistics
    n_perm: int = 1000
    n_boot: int = 500
    n_random_windows: int = 1000
    group_frac: float = 0.30
    random_seed: int = 20260515
    progress: bool = False

    # HOTFIX01 performance controls.
    # full: preserves all original resampling semantics, with vectorized pairwise p/CI and faster LOOCV.
    # screen: uses screen_n_perm/screen_n_boot for experiment-A relation screens and skips formal delta resampling.
    # skip_heavy: runs compact A1 relation table, skips expensive A2 incremental-CV and A3 sliding-window screens.
    experiment_a_policy: str = "full"  # full | screen | skip_heavy
    screen_n_perm: int = 199
    screen_n_boot: int = 199

    # IO
    smoothed_fields_path_override: Path | None = None
    version: str = "v10.7_n"
    output_tag: str = "h_zonal_width_background_target_preinfo_v10_7_n"

    # core windows, day 0 = Apr 1
    e1_full: tuple[int, int] = (12, 23)
    e2_pre: tuple[int, int] = (27, 31)
    e2_post: tuple[int, int] = (34, 38)
    e2_full: tuple[int, int] = (27, 38)
    m_pre: tuple[int, int] = (40, 43)
    m_post: tuple[int, int] = (45, 48)
    m_full: tuple[int, int] = (40, 48)

    # broad background windows
    pre_e2_full: tuple[int, int] = (0, 26)
    e1_to_e2_full: tuple[int, int] = (12, 38)
    pre_m_full: tuple[int, int] = (0, 39)

    # negative-control windows
    pre_m_near_full: tuple[int, int] = (28, 39)
    post_m_full: tuple[int, int] = (49, 60)

    # sliding anchor settings
    source_anchor_len: int = 12
    source_anchor_pre_len: int = 5
    source_anchor_gap_len: int = 2
    source_anchor_post_len: int = 5
    source_anchor_min_start: int = 0
    source_anchor_max_end: int = 39

    # domains
    h_lat_range: tuple[float, float] = (15.0, 35.0)
    h_lon_range: tuple[float, float] = (110.0, 140.0)
    p_lat_range: tuple[float, float] = (15.0, 35.0)
    p_lon_range: tuple[float, float] = (110.0, 140.0)

    # modes
    modes: tuple[str, ...] = ("anomaly", "local_background_removed", "raw")
    primary_modes: tuple[str, ...] = ("anomaly", "local_background_removed")

    # H source focus
    h_main_metric: str = "H_zonal_width"

    # P transition metrics
    p_position_metric: str = "P_centroid_lat_transition"
    p_rainband_metrics: tuple[str, ...] = (
        "P_main_band_share_transition",
        "P_south_band_share_18_24_transition",
        "P_main_minus_south_transition",
    )
    p_all_transition_metrics: tuple[str, ...] = (
        "P_total_strength_transition",
        "P_centroid_lat_transition",
        "P_spread_lat_transition",
        "P_main_band_share_transition",
        "P_south_band_share_18_24_transition",
        "P_main_minus_south_transition",
    )

    def smoothed_fields_path(self) -> Path:
        if self.smoothed_fields_path_override is not None:
            return self.smoothed_fields_path_override
        return (
            self.project_root
            / "foundation"
            / "V1"
            / "outputs"
            / "baseline_a"
            / "preprocess"
            / "smoothed_fields.npz"
        )

    def output_root(self) -> Path:
        return (
            self.project_root
            / "stage_partition"
            / "V10"
            / "v10.7"
            / "outputs"
            / self.output_tag
        )

    def to_dict(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if isinstance(x, Path):
                return str(x)
            if isinstance(x, tuple):
                return [conv(v) for v in x]
            if isinstance(x, list):
                return [conv(v) for v in x]
            if isinstance(x, dict):
                return {str(k): conv(v) for k, v in x.items()}
            return x

        return conv(asdict(self))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(settings: Settings, msg: str) -> None:
    if settings.progress:
        print(f"[V10.7_n] {msg}", flush=True)


def clean_output_root(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for sub in ("tables", "figures", "run_meta"):
        (path / sub).mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _screen_settings(settings: Settings) -> Settings:
    """Return a lightweight settings copy for screening tables.

    This is explicit and opt-in through --experiment-a-policy screen. It does not
    change the default formal run, and the resulting tables carry policy columns.
    """
    return replace(
        settings,
        n_perm=max(0, int(settings.screen_n_perm)),
        n_boot=max(0, int(settings.screen_n_boot)),
    )




# HOTFIX02: explicit empty-table schemas.  Some execution policies (notably
# --experiment-a-policy skip_heavy) intentionally skip A2/A3.  Pandas then
# creates a truly empty DataFrame with no columns, which broke route decisions
# that still expected a "mode" column.  Keep schemas explicit so skipped
# outputs remain readable and downstream route logic can degrade gracefully.
INCREMENT_TABLE_COLUMNS = [
    "mode",
    "e2_feature",
    "broad_feature",
    "target_name",
    "n_valid_years",
    "cv_r2_broad",
    "cv_r2_e2",
    "cv_r2_broad_plus_e2",
    "delta_cv_r2_e2_given_broad",
    "delta_boot_ci_low",
    "delta_boot_ci_high",
    "delta_perm_p",
    "increment_support_flag",
    "increment_evaluation_policy",
]

RELATION_TABLE_MIN_COLUMNS = [
    "mode",
    "source_feature",
    "target_name",
    "primary_score",
    "support_flag",
]


def empty_increment_table(policy: str = "skipped") -> pd.DataFrame:
    df = pd.DataFrame(columns=INCREMENT_TABLE_COLUMNS)
    df["increment_evaluation_policy"] = pd.Series(dtype="object")
    return df


def empty_relation_table() -> pd.DataFrame:
    return pd.DataFrame(columns=RELATION_TABLE_MIN_COLUMNS)


def filter_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df is None or df.empty or "mode" not in df.columns:
        return pd.DataFrame()
    return df[df["mode"] == mode]

def safe_nanmean(a: np.ndarray, axis=None, keepdims: bool = False):
    arr = np.asarray(a, dtype=float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis, keepdims=keepdims)
    total = np.nansum(arr, axis=axis, keepdims=keepdims)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count > 0, mean, np.nan)


def safe_nanstd(a: np.ndarray, axis=None, keepdims: bool = False):
    arr = np.asarray(a, dtype=float)
    mu = safe_nanmean(arr, axis=axis, keepdims=True)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis, keepdims=keepdims)
    sq = np.where(valid, (arr - mu) ** 2, np.nan)
    var = safe_nanmean(sq, axis=axis, keepdims=keepdims)
    return np.sqrt(var)


def first_key(data: dict[str, Any], candidates: Iterable[str]) -> str | None:
    lower = {str(k).lower(): k for k in data.keys()}
    for c in candidates:
        if c in data:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_npz(settings: Settings) -> dict[str, Any]:
    path = settings.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields file: {path}")
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def coerce_1d(x: Any, fallback_len: int | None = None, name: str = "array") -> np.ndarray:
    if x is None:
        if fallback_len is None:
            raise ValueError(f"Cannot infer {name}: missing value and no fallback_len")
        return np.arange(fallback_len)
    arr = np.asarray(x)
    arr = np.ravel(arr)
    if arr.dtype.kind in "OUS":
        # Defensive conversion for object arrays containing scalars.
        arr = arr.astype(float)
    return arr


def normalize_field_dims(
    field: np.ndarray,
    years: np.ndarray | None,
    days: np.ndarray | None,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return field as (year, day, lat, lon), with lat/lon sorted ascending.

    The local data used by this project normally has explicit year/day/lat/lon
    coordinates. This routine is intentionally defensive because older archives
    sometimes differ in axis order.
    """
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D field (year, day, lat, lon in any order), got shape={arr.shape}")

    nlat = len(lat)
    nlon = len(lon)
    nyear = len(years) if years is not None else None
    nday = len(days) if days is not None else None

    sizes = list(arr.shape)
    axes = set(range(arr.ndim))

    def pick_axis(size: int | None, label: str, preferred: list[int] | None = None) -> int:
        if size is None:
            raise ValueError(f"Cannot pick {label} axis without coordinate length")
        candidates = [a for a in axes if sizes[a] == size]
        if preferred:
            candidates = [a for a in preferred if a in candidates] + [a for a in candidates if a not in preferred]
        if not candidates:
            raise ValueError(f"Could not infer {label} axis with size {size}; field shape={arr.shape}")
        axis = candidates[0]
        axes.remove(axis)
        return axis

    lat_axis = pick_axis(nlat, "lat")
    lon_axis = pick_axis(nlon, "lon")

    if nyear is not None:
        year_axis = pick_axis(nyear, "year")
    else:
        # remaining smaller axis is usually year
        year_axis = min(axes, key=lambda a: sizes[a])
        axes.remove(year_axis)
        years = np.arange(sizes[year_axis])

    if nday is not None:
        day_axis = pick_axis(nday, "day")
    else:
        day_axis = list(axes)[0]
        axes.remove(day_axis)
        days = np.arange(sizes[day_axis])

    out = np.transpose(arr, (year_axis, day_axis, lat_axis, lon_axis))
    years_arr = coerce_1d(years, out.shape[0], "years")
    days_arr = coerce_1d(days, out.shape[1], "days")
    lat_arr = coerce_1d(lat, out.shape[2], "lat")
    lon_arr = coerce_1d(lon, out.shape[3], "lon")

    lat_order = np.argsort(lat_arr)
    lon_order = np.argsort(lon_arr)
    lat_sorted = lat_arr[lat_order]
    lon_sorted = lon_arr[lon_order]
    out = out[:, :, lat_order, :][:, :, :, lon_order]

    audit = {
        "original_shape": list(arr.shape),
        "normalized_shape": list(out.shape),
        "axis_year": int(year_axis),
        "axis_day": int(day_axis),
        "axis_lat": int(lat_axis),
        "axis_lon": int(lon_axis),
        "lat_sorted_ascending": bool(np.all(np.diff(lat_sorted) >= 0)),
        "lon_sorted_ascending": bool(np.all(np.diff(lon_sorted) >= 0)),
    }
    return out, years_arr, days_arr, lat_sorted, lon_sorted, audit


def subset_domain(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0, lat1 = min(lat_range), max(lat_range)
    lon0, lon1 = min(lon_range), max(lon_range)
    lat_mask = (lat >= lat0) & (lat <= lat1)
    lon_mask = (lon >= lon0) & (lon <= lon1)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(f"Empty domain subset: lat_range={lat_range}, lon_range={lon_range}")
    return field[:, :, lat_mask, :][:, :, :, lon_mask], lat[lat_mask], lon[lon_mask]


def day_mask(days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    start, end = win
    days_arr = np.asarray(days)
    return (days_arr >= start) & (days_arr <= end)


def daily_anomaly(field: np.ndarray) -> np.ndarray:
    clim = safe_nanmean(field, axis=0, keepdims=True)
    return field - clim


def local_background_removed_field(field: np.ndarray) -> np.ndarray:
    # Remove same-day local spatial background inside the object domain.
    # This keeps local shape/position information but suppresses the domain-mean level.
    bg = safe_nanmean(field, axis=(2, 3), keepdims=True)
    lbr = field - bg
    # Then remove daily climatology of the local residual to keep the anomaly-oriented
    # interpretation consistent with V10.7_l/m main evidence modes.
    return daily_anomaly(lbr)


def build_mode_fields(field: np.ndarray, settings: Settings) -> dict[str, np.ndarray]:
    return {
        "raw": field.copy(),
        "anomaly": daily_anomaly(field),
        "local_background_removed": local_background_removed_field(field),
    }


def _active_weights(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=bool), np.full_like(arr, np.nan, dtype=float)
    vals = arr[finite]
    q = np.nanpercentile(vals, 75.0)
    active = finite & (arr >= q)
    weights = np.where(active, arr - q, 0.0)
    if np.nansum(weights) <= 0:
        # Fallback: positive shifted weights over finite cells.
        mn = np.nanmin(vals)
        weights = np.where(finite, arr - mn, 0.0)
    if np.nansum(weights) <= 0:
        weights = np.where(active, 1.0, 0.0)
    return active, weights


def build_h_daily_metrics(h_field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    ny, nd, nlat, nlon = h_field.shape
    lat2d = lat[:, None] * np.ones((nlat, nlon))
    lon2d = np.ones((nlat, nlon)) * lon[None, :]

    out = {k: np.full((ny, nd), np.nan) for k in (
        "H_strength",
        "H_centroid_lat",
        "H_west_extent_lon",
        "H_zonal_width",
        "H_north_edge_lat",
        "H_south_edge_lat",
    )}

    for i in range(ny):
        for j in range(nd):
            arr = h_field[i, j]
            finite = np.isfinite(arr)
            if not finite.any():
                continue
            out["H_strength"][i, j] = float(safe_nanmean(arr))
            active, weights = _active_weights(arr)
            wsum = np.nansum(weights)
            if wsum > 0:
                out["H_centroid_lat"][i, j] = float(np.nansum(weights * lat2d) / wsum)
            if active.any():
                active_lats = lat2d[active]
                active_lons = lon2d[active]
                out["H_west_extent_lon"][i, j] = float(np.nanmin(active_lons))
                out["H_zonal_width"][i, j] = float(np.nanmax(active_lons) - np.nanmin(active_lons))
                out["H_north_edge_lat"][i, j] = float(np.nanmax(active_lats))
                out["H_south_edge_lat"][i, j] = float(np.nanmin(active_lats))
    return out


def build_p_daily_metrics(p_field: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    ny, nd, nlat, nlon = p_field.shape
    out = {k: np.full((ny, nd), np.nan) for k in (
        "P_total_strength",
        "P_centroid_lat",
        "P_spread_lat",
        "P_main_band_share",
        "P_south_band_share_18_24",
        "P_main_minus_south",
    )}
    lat_arr = np.asarray(lat, dtype=float)
    main_mask = (lat_arr >= 28.0) & (lat_arr <= 34.0)
    south_mask = (lat_arr >= 18.0) & (lat_arr <= 24.0)
    for i in range(ny):
        for j in range(nd):
            arr = p_field[i, j]
            finite = np.isfinite(arr)
            if not finite.any():
                continue
            out["P_total_strength"][i, j] = float(safe_nanmean(arr))
            # Use positive precipitation contribution for distributional metrics.
            wfield = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
            if np.nansum(wfield) <= 0:
                # For anomaly-like fields that can be negative, use positive shifted values.
                mn = np.nanmin(arr[finite])
                wfield = np.where(finite, arr - mn, np.nan)
            lat_profile = safe_nanmean(wfield, axis=1)
            total = float(np.nansum(lat_profile))
            if total <= 0 or not np.isfinite(total):
                continue
            centroid = float(np.nansum(lat_profile * lat_arr) / total)
            out["P_centroid_lat"][i, j] = centroid
            out["P_spread_lat"][i, j] = float(np.sqrt(np.nansum(lat_profile * (lat_arr - centroid) ** 2) / total))
            main = float(np.nansum(lat_profile[main_mask]) / total) if main_mask.any() else np.nan
            south = float(np.nansum(lat_profile[south_mask]) / total) if south_mask.any() else np.nan
            out["P_main_band_share"][i, j] = main
            out["P_south_band_share_18_24"][i, j] = south
            out["P_main_minus_south"][i, j] = main - south
    return out


def window_mean_series(ts: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    mask = day_mask(days, win)
    if not mask.any():
        return np.full(ts.shape[0], np.nan)
    return safe_nanmean(ts[:, mask], axis=1)


def slope_series(ts: np.ndarray, days: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    mask = day_mask(days, win)
    if not mask.any():
        return np.full(ts.shape[0], np.nan)
    x = np.asarray(days[mask], dtype=float)
    y = np.asarray(ts[:, mask], dtype=float)
    out = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        yi = y[i]
        ok = np.isfinite(yi) & np.isfinite(x)
        if ok.sum() < 3:
            continue
        xx = x[ok] - np.nanmean(x[ok])
        yy = yi[ok] - np.nanmean(yi[ok])
        denom = np.nansum(xx ** 2)
        if denom > 0:
            out[i] = float(np.nansum(xx * yy) / denom)
    return out


def zscore_years(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(arr, np.nan, dtype=float)
    return (arr - mu) / sd


def build_source_features_for_mode(
    h_metrics: dict[str, np.ndarray],
    years: np.ndarray,
    days: np.ndarray,
    settings: Settings,
    mode: str,
) -> pd.DataFrame:
    hzw = h_metrics[settings.h_main_metric]
    rows: list[dict[str, Any]] = []

    def add_feature(source_family: str, window_name: str, info_form: str, feature_name: str, values: np.ndarray,
                    day_start: int | None = None, day_end: int | None = None, temporal_position: str | None = None):
        for yr, val in zip(years, values):
            rows.append({
                "year": int(yr) if np.isfinite(yr) else yr,
                "mode": mode,
                "source_feature": feature_name,
                "source_family": source_family,
                "window_name": window_name,
                "information_form": info_form,
                "day_start": day_start,
                "day_end": day_end,
                "temporal_position": temporal_position or source_family,
                "source_value": float(val) if np.isfinite(val) else np.nan,
            })

    # E2-local features.
    e2_pre = window_mean_series(hzw, days, settings.e2_pre)
    e2_post = window_mean_series(hzw, days, settings.e2_post)
    e2_mean = window_mean_series(hzw, days, settings.e2_full)
    e2_slope = slope_series(hzw, days, settings.e2_full)
    e2_trans = e2_post - e2_pre
    add_feature("E2_local", "E2", "pre_state", "H_zonal_width_E2_pre_state", e2_pre, *settings.e2_pre, "e2")
    add_feature("E2_local", "E2", "post_state", "H_zonal_width_E2_post_state", e2_post, *settings.e2_post, "e2")
    add_feature("E2_local", "E2", "window_mean", "H_zonal_width_E2_window_mean", e2_mean, *settings.e2_full, "e2")
    add_feature("E2_local", "E2", "signed_transition", "H_zonal_width_E2_signed_transition", e2_trans, settings.e2_full[0], settings.e2_full[1], "e2")
    add_feature("E2_local", "E2", "abs_transition", "H_zonal_width_E2_abs_transition", np.abs(e2_trans), settings.e2_full[0], settings.e2_full[1], "e2")
    add_feature("E2_local", "E2", "slope", "H_zonal_width_E2_slope", e2_slope, *settings.e2_full, "e2")

    # Broad background features.
    for name, win in (
        ("pre_E2", settings.pre_e2_full),
        ("E1_to_E2", settings.e1_to_e2_full),
        ("pre_M", settings.pre_m_full),
    ):
        mean_vals = window_mean_series(hzw, days, win)
        slope_vals = slope_series(hzw, days, win)
        add_feature("broad_background", name, "window_mean", f"H_zonal_width_{name}_mean", mean_vals, *win, "broad_pre_m")
        add_feature("broad_background", name, "slope", f"H_zonal_width_{name}_slope", slope_vals, *win, "broad_pre_m")

    # Time-order controls.
    for name, win, pos in (
        ("E1", settings.e1_full, "early_pre"),
        ("E2", settings.e2_full, "e2"),
        ("pre_M_near", settings.pre_m_near_full, "near_pre_m"),
        ("pre_M", settings.pre_m_full, "broad_pre_m"),
        ("post_M", settings.post_m_full, "post_m"),
    ):
        mean_vals = window_mean_series(hzw, days, win)
        slope_vals = slope_series(hzw, days, win)
        add_feature("time_order_control", name, "window_mean", f"H_zonal_width_{name}_mean", mean_vals, *win, pos)
        add_feature("time_order_control", name, "slope", f"H_zonal_width_{name}_slope", slope_vals, *win, pos)

    return pd.DataFrame(rows)


def build_sliding_features_for_mode(
    h_metrics: dict[str, np.ndarray],
    years: np.ndarray,
    days: np.ndarray,
    settings: Settings,
    mode: str,
) -> pd.DataFrame:
    hzw = h_metrics[settings.h_main_metric]
    rows: list[dict[str, Any]] = []
    starts = range(settings.source_anchor_min_start, settings.source_anchor_max_end - settings.source_anchor_len + 2)
    for start in starts:
        end = start + settings.source_anchor_len - 1
        if end > settings.source_anchor_max_end:
            continue
        pre = (start, start + settings.source_anchor_pre_len - 1)
        post = (end - settings.source_anchor_post_len + 1, end)
        full = (start, end)
        pre_vals = window_mean_series(hzw, days, pre)
        post_vals = window_mean_series(hzw, days, post)
        mean_vals = window_mean_series(hzw, days, full)
        slope_vals = slope_series(hzw, days, full)
        trans_vals = post_vals - pre_vals
        feature_map = {
            "pre_state": pre_vals,
            "post_state": post_vals,
            "window_mean": mean_vals,
            "signed_transition": trans_vals,
            "slope": slope_vals,
        }
        for info_form, values in feature_map.items():
            for yr, val in zip(years, values):
                rows.append({
                    "year": int(yr) if np.isfinite(yr) else yr,
                    "mode": mode,
                    "window_start": start,
                    "window_end": end,
                    "window_name": f"sliding_{start:03d}_{end:03d}",
                    "information_form": info_form,
                    "source_feature": f"H_zonal_width_sliding_{info_form}",
                    "source_value": float(val) if np.isfinite(val) else np.nan,
                })
    return pd.DataFrame(rows)


def build_p_transition_targets_for_mode(
    p_metrics: dict[str, np.ndarray],
    p_field: np.ndarray,
    years: np.ndarray,
    days: np.ndarray,
    lat: np.ndarray,
    settings: Settings,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    base: dict[str, np.ndarray] = {}
    for metric in (
        "P_total_strength",
        "P_centroid_lat",
        "P_spread_lat",
        "P_main_band_share",
        "P_south_band_share_18_24",
        "P_main_minus_south",
    ):
        pre = window_mean_series(p_metrics[metric], days, settings.m_pre)
        post = window_mean_series(p_metrics[metric], days, settings.m_post)
        base[f"{metric}_transition"] = post - pre

    # Basic targets.
    family_map = {
        "P_total_strength_transition": "P_strength",
        "P_centroid_lat_transition": "P_position",
        "P_spread_lat_transition": "P_spread",
        "P_main_band_share_transition": "P_rainband_component",
        "P_south_band_share_18_24_transition": "P_rainband_component",
        "P_main_minus_south_transition": "P_rainband_component",
    }
    for name, vals in base.items():
        for yr, val in zip(years, vals):
            rows.append({
                "year": int(yr) if np.isfinite(yr) else yr,
                "mode": mode,
                "target_name": name,
                "target_family": family_map.get(name, "P_component"),
                "target_value": float(val) if np.isfinite(val) else np.nan,
            })

    # Composite targets, pre-registered signs.
    z_centroid = zscore_years(base["P_centroid_lat_transition"])
    z_main = zscore_years(base["P_main_band_share_transition"])
    z_south = zscore_years(base["P_south_band_share_18_24_transition"])
    z_main_minus = zscore_years(base["P_main_minus_south_transition"])
    z_total = zscore_years(base["P_total_strength_transition"])

    composites = {
        "P_position_only": z_centroid,
        "P_rainband_composite": z_main - z_south + z_main_minus,
        "P_rainband_main_minus_south": z_main_minus,
        "P_NS_reorganization_index": z_centroid + z_main - z_south + z_main_minus,
        "P_total_strength_z": z_total,
    }
    composite_family = {
        "P_position_only": "P_position_only",
        "P_rainband_composite": "P_rainband_only",
        "P_rainband_main_minus_south": "P_rainband_only",
        "P_NS_reorganization_index": "P_NS_reorganization",
        "P_total_strength_z": "P_strength",
    }
    for name, vals in composites.items():
        for yr, val in zip(years, vals):
            rows.append({
                "year": int(yr) if np.isfinite(yr) else yr,
                "mode": mode,
                "target_name": name,
                "target_family": composite_family[name],
                "target_value": float(val) if np.isfinite(val) else np.nan,
            })

    # Components audit table.
    for i, yr in enumerate(years):
        comp_rows.append({
            "year": int(yr) if np.isfinite(yr) else yr,
            "mode": mode,
            "z_P_centroid_lat_transition": z_centroid[i],
            "z_P_main_band_share_transition": z_main[i],
            "z_P_south_band_share_18_24_transition": z_south[i],
            "z_P_main_minus_south_transition": z_main_minus[i],
            "P_NS_reorganization_index": composites["P_NS_reorganization_index"][i],
            "P_rainband_composite": composites["P_rainband_composite"][i],
        })

    # Latitudinal profile contrast target.
    pre_mask = day_mask(days, settings.m_pre)
    post_mask = day_mask(days, settings.m_post)
    if pre_mask.any() and post_mask.any():
        trans_field = safe_nanmean(p_field[:, post_mask, :, :], axis=1) - safe_nanmean(p_field[:, pre_mask, :, :], axis=1)
        profile = safe_nanmean(trans_field, axis=2)  # (year, lat)
        north_mask = (lat >= 30.0) & (lat <= 32.0)
        south_mask = (lat >= 18.0) & (lat <= 25.0)
        north_vals = safe_nanmean(profile[:, north_mask], axis=1) if north_mask.any() else np.full(len(years), np.nan)
        south_vals = safe_nanmean(profile[:, south_mask], axis=1) if south_mask.any() else np.full(len(years), np.nan)
        contrast = north_vals - south_vals
        for yr, val, nval, sval in zip(years, contrast, north_vals, south_vals):
            rows.append({
                "year": int(yr) if np.isfinite(yr) else yr,
                "mode": mode,
                "target_name": "P_profile_contrast_30_32_minus_18_25",
                "target_family": "P_lat_profile_contrast",
                "target_value": float(val) if np.isfinite(val) else np.nan,
            })
            profile_rows.append({
                "year": int(yr) if np.isfinite(yr) else yr,
                "mode": mode,
                "P_profile_30_32": float(nval) if np.isfinite(nval) else np.nan,
                "P_profile_18_25": float(sval) if np.isfinite(sval) else np.nan,
                "P_profile_contrast_30_32_minus_18_25": float(val) if np.isfinite(val) else np.nan,
            })

    return pd.DataFrame(rows), pd.DataFrame(comp_rows), pd.DataFrame(profile_rows)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    xx = x[ok]
    yy = y[ok]
    if np.nanstd(xx) <= 0 or np.nanstd(yy) <= 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = pd.Series(x[ok]).rank().to_numpy(dtype=float)
    ry = pd.Series(y[ok]).rank().to_numpy(dtype=float)
    return pearson_r(rx, ry)


def high_low_diff(source: np.ndarray, target: np.ndarray, group_frac: float) -> float:
    ok = np.isfinite(source) & np.isfinite(target)
    if ok.sum() < 6:
        return np.nan
    s = source[ok]
    t = target[ok]
    order = np.argsort(s)
    ngrp = max(3, int(math.floor(len(s) * group_frac)))
    ngrp = min(ngrp, len(s) // 2)
    if ngrp < 2:
        return np.nan
    low = t[order[:ngrp]]
    high = t[order[-ngrp:]]
    return float(np.nanmean(high) - np.nanmean(low))


def permutation_p_corr(source: np.ndarray, target: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p for |Pearson r|.

    HOTFIX01: vectorized over permutations. This preserves the same null definition
    as the original loop implementation, but avoids Python-level loops over 5000
    permutations for every source-target pair.
    """
    ok = np.isfinite(source) & np.isfinite(target)
    if ok.sum() < 5:
        return np.nan
    s = np.asarray(source[ok], dtype=float)
    t = np.asarray(target[ok], dtype=float)
    obs = abs(pearson_r(s, t))
    if not np.isfinite(obs):
        return np.nan
    n = len(s)
    if n_perm <= 0:
        return np.nan

    # Generate independent random keys; argsort rows gives random permutations.
    order = np.argsort(rng.random((int(n_perm), n)), axis=1)
    sp = s[order]
    tc = t - np.nanmean(t)
    spc = sp - np.nanmean(sp, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(spc * spc, axis=1) * np.sum(tc * tc))
    with np.errstate(invalid="ignore", divide="ignore"):
        r_perm = np.sum(spc * tc[None, :], axis=1) / denom
    count = int(np.sum(np.abs(r_perm[np.isfinite(r_perm)]) >= obs))
    return float((count + 1) / (int(n_perm) + 1))


def bootstrap_ci_diff(source: np.ndarray, target: np.ndarray, group_frac: float, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap CI for high-minus-low target difference.

    HOTFIX01: vectorized bootstrap. It preserves the original resample-with-
    replacement and high/low grouping semantics, but removes the Python loop over
    bootstrap draws.
    """
    ok = np.isfinite(source) & np.isfinite(target)
    if ok.sum() < 6:
        return np.nan, np.nan
    s = np.asarray(source[ok], dtype=float)
    t = np.asarray(target[ok], dtype=float)
    n = len(s)
    n_boot = int(n_boot)
    if n_boot <= 0:
        return np.nan, np.nan
    ngrp = max(1, int(math.floor(n * group_frac)))
    if ngrp * 2 > n:
        return np.nan, np.nan

    idx = rng.integers(0, n, size=(n_boot, n))
    sb = s[idx]
    tb = t[idx]
    order = np.argsort(sb, axis=1)
    tb_sorted = np.take_along_axis(tb, order, axis=1)
    vals = np.nanmean(tb_sorted[:, -ngrp:], axis=1) - np.nanmean(tb_sorted[:, :ngrp], axis=1)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def evaluate_pairwise_relation(
    source: np.ndarray,
    target: np.ndarray,
    settings: Settings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ok = np.isfinite(source) & np.isfinite(target)
    n = int(ok.sum())
    r = pearson_r(source, target)
    sr = spearman_r(source, target)
    p = permutation_p_corr(source, target, settings.n_perm, rng)
    diff = high_low_diff(source, target, settings.group_frac)
    ci_low, ci_high = bootstrap_ci_diff(source, target, settings.group_frac, settings.n_boot, rng)
    ci_excludes_zero = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0))
    support_flag = bool(np.isfinite(p) and p <= 0.10 and ci_excludes_zero)
    primary_score = 0.0
    if np.isfinite(p):
        primary_score += max(0.0, -math.log10(max(p, 1.0 / (settings.n_perm + 1))))
    if np.isfinite(sr):
        primary_score += abs(sr)
    if support_flag:
        primary_score += 1.0
    return {
        "n_valid_years": n,
        "pearson_r": r,
        "spearman_r": sr,
        "perm_p": p,
        "high_low_diff": diff,
        "boot_ci_low": ci_low,
        "boot_ci_high": ci_high,
        "ci_excludes_zero": ci_excludes_zero,
        "support_flag": support_flag,
        "primary_score": primary_score if np.isfinite(primary_score) else np.nan,
    }


def make_feature_target_relation_table(
    sources: pd.DataFrame,
    targets: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
    source_value_col: str = "source_value",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_keys = [c for c in ["mode", "source_feature", "source_family", "information_form", "window_name", "temporal_position", "day_start", "day_end"] if c in sources.columns]
    for key_vals, sdf in sources.groupby(source_keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        source_meta = dict(zip(source_keys, key_vals))
        mode = source_meta.get("mode")
        tdf_mode = targets[targets["mode"] == mode]
        s_year = sdf[["year", source_value_col]].rename(columns={source_value_col: "source_value"})
        for (target_name, target_family), tdf in tdf_mode.groupby(["target_name", "target_family"], dropna=False):
            merged = s_year.merge(tdf[["year", "target_value"]], on="year", how="inner")
            stats = evaluate_pairwise_relation(
                merged["source_value"].to_numpy(dtype=float),
                merged["target_value"].to_numpy(dtype=float),
                settings,
                rng,
            )
            rows.append({**source_meta, "target_name": target_name, "target_family": target_family, **stats})
    return pd.DataFrame(rows)


def fit_linear_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return X1 @ coef


def loo_cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    """OLS leave-one-out CV R2 using the hat-matrix identity.

    HOTFIX01 replaces the original refit-for-each-left-out-year loop. The
    numerical target is the same ordinary least squares LOOCV quantity, but this
    computes it with a single fit per design matrix:
        e_LOO_i = e_i / (1 - h_ii)
    where h_ii is the leverage from the OLS hat matrix.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    n = len(y)
    if n < X.shape[1] + 4 or np.nanstd(y) <= 0:
        return np.nan
    X1 = np.column_stack([np.ones(n), X])
    try:
        xtx_inv = np.linalg.pinv(X1.T @ X1)
        beta = xtx_inv @ X1.T @ y
        yhat = X1 @ beta
        resid = y - yhat
        h = np.sum((X1 @ xtx_inv) * X1, axis=1)
        denom = 1.0 - h
        bad = np.abs(denom) < 1e-10
        if np.any(bad):
            return np.nan
        loo_pred = y - resid / denom
    except Exception:
        return np.nan
    sse = float(np.nansum((y - loo_pred) ** 2))
    sst = float(np.nansum((y - np.nanmean(y)) ** 2))
    if sst <= 0:
        return np.nan
    return 1.0 - sse / sst


def incremental_cv_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    settings: Settings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compare E2-local information against broad-background information.

    HOTFIX01: keeps the original model definitions and resampling logic, but
    speeds up the LOOCV calculations via the optimized loo_cv_r2() above and
    avoids recomputing the broad-only baseline inside every permutation.
    """
    rows: list[dict[str, Any]] = []
    e2_candidates = source_df[source_df["source_family"] == "E2_local"]
    broad_candidates = source_df[source_df["source_family"] == "broad_background"]
    target_names = [
        "P_NS_reorganization_index",
        "P_profile_contrast_30_32_minus_18_25",
        "P_position_only",
        "P_rainband_composite",
    ]
    for mode in settings.primary_modes:
        e2_mode = e2_candidates[e2_candidates["mode"] == mode]
        broad_mode = broad_candidates[broad_candidates["mode"] == mode]
        t_mode = target_df[(target_df["mode"] == mode) & (target_df["target_name"].isin(target_names))]
        for e2_name, e2s in e2_mode.groupby("source_feature"):
            e2v = e2s[["year", "source_value"]].rename(columns={"source_value": "e2_value"})
            for broad_name, bds in broad_mode.groupby("source_feature"):
                bdv = bds[["year", "source_value"]].rename(columns={"source_value": "broad_value"})
                xb = e2v.merge(bdv, on="year", how="inner")
                for target_name, tdf in t_mode.groupby("target_name"):
                    yv = tdf[["year", "target_value"]]
                    merged = xb.merge(yv, on="year", how="inner")
                    ok = np.isfinite(merged["e2_value"]) & np.isfinite(merged["broad_value"]) & np.isfinite(merged["target_value"])
                    m = merged[ok]
                    if len(m) < 8:
                        continue
                    y = m["target_value"].to_numpy(float)
                    e2 = m[["e2_value"]].to_numpy(float)
                    broad = m[["broad_value"]].to_numpy(float)
                    both = m[["broad_value", "e2_value"]].to_numpy(float)
                    r2_broad = loo_cv_r2(broad, y)
                    r2_e2 = loo_cv_r2(e2, y)
                    r2_both = loo_cv_r2(both, y)
                    delta = r2_both - r2_broad if np.isfinite(r2_both) and np.isfinite(r2_broad) else np.nan

                    deltas: list[float] = []
                    perm_deltas: list[float] = []
                    n = len(m)
                    # HOTFIX01: full mode preserves the formal delta resampling; screen mode
                    # keeps CV-R2/delta rankings but avoids the very expensive resampling loops.
                    if settings.experiment_a_policy == "full":
                        # Bootstrap keeps the original paired-resampling semantics.
                        for _ in range(int(settings.n_boot)):
                            idx = rng.integers(0, n, size=n)
                            db = loo_cv_r2(broad[idx], y[idx])
                            dt = loo_cv_r2(both[idx], y[idx])
                            if np.isfinite(db) and np.isfinite(dt):
                                deltas.append(dt - db)
                        # The broad-only baseline does not depend on the permuted E2 vector.
                        db_perm_base = loo_cv_r2(broad, y)
                        for _ in range(int(settings.n_perm)):
                            ep = rng.permutation(e2.ravel()).reshape(-1, 1)
                            bp = np.column_stack([broad.ravel(), ep.ravel()])
                            dt = loo_cv_r2(bp, y)
                            if np.isfinite(db_perm_base) and np.isfinite(dt):
                                perm_deltas.append(dt - db_perm_base)
                    ci_low = float(np.nanpercentile(deltas, 2.5)) if len(deltas) >= 5 else np.nan
                    ci_high = float(np.nanpercentile(deltas, 97.5)) if len(deltas) >= 5 else np.nan
                    if np.isfinite(delta) and perm_deltas:
                        p_delta = float((np.sum(np.asarray(perm_deltas) >= delta) + 1) / (len(perm_deltas) + 1))
                    else:
                        p_delta = np.nan
                    rows.append({
                        "mode": mode,
                        "e2_feature": e2_name,
                        "broad_feature": broad_name,
                        "target_name": target_name,
                        "n_valid_years": len(m),
                        "cv_r2_broad": r2_broad,
                        "cv_r2_e2": r2_e2,
                        "cv_r2_broad_plus_e2": r2_both,
                        "delta_cv_r2_e2_given_broad": delta,
                        "delta_boot_ci_low": ci_low,
                        "delta_boot_ci_high": ci_high,
                        "delta_perm_p": p_delta,
                        "increment_support_flag": bool(
                            (np.isfinite(delta) and delta > 0 and np.isfinite(p_delta) and p_delta <= 0.10)
                            if settings.experiment_a_policy == "full"
                            else (np.isfinite(delta) and delta > 0)
                        ),
                        "increment_evaluation_policy": settings.experiment_a_policy,
                    })
    if not rows:
        return empty_increment_table(settings.experiment_a_policy)
    return pd.DataFrame(rows, columns=INCREMENT_TABLE_COLUMNS)


def relation_family_summary(relation_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if relation_df.empty:
        return pd.DataFrame()
    agg = relation_df.groupby(group_cols, dropna=False).agg(
        mean_primary_score=("primary_score", "mean"),
        max_primary_score=("primary_score", "max"),
        n_support=("support_flag", "sum"),
        mean_abs_spearman=("spearman_r", lambda x: float(np.nanmean(np.abs(x)))),
        n_rows=("primary_score", "size"),
    ).reset_index()
    return agg


def composite_component_dominance(components_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    component_cols = [
        "z_P_centroid_lat_transition",
        "z_P_main_band_share_transition",
        "z_P_south_band_share_18_24_transition",
        "z_P_main_minus_south_transition",
    ]
    for mode, df in components_df.groupby("mode"):
        comp = df["P_NS_reorganization_index"].to_numpy(float)
        for col in component_cols:
            x = df[col].to_numpy(float)
            r = pearson_r(x, comp)
            rows.append({
                "mode": mode,
                "component": col,
                "corr_with_P_NS_reorganization_index": r,
                "component_std": float(np.nanstd(x)),
                "dominance_flag": bool(np.isfinite(r) and abs(r) > 0.90),
            })
    return pd.DataFrame(rows)


def leave_one_year_out_relation(source: pd.DataFrame, target: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_sources = source[source["source_feature"].isin([
        "H_zonal_width_E2_slope",
        "H_zonal_width_E2_signed_transition",
        "H_zonal_width_E2_post_state",
        "H_zonal_width_pre_M_mean",
        "H_zonal_width_pre_M_slope",
    ])]
    primary_targets = target[target["target_name"].isin([
        "P_NS_reorganization_index",
        "P_profile_contrast_30_32_minus_18_25",
        "P_position_only",
        "P_rainband_composite",
    ])]
    for (mode, sname), sdf in primary_sources.groupby(["mode", "source_feature"]):
        tmode = primary_targets[primary_targets["mode"] == mode]
        for tname, tdf in tmode.groupby("target_name"):
            base = sdf[["year", "source_value"]].merge(tdf[["year", "target_value"]], on="year", how="inner")
            base_stats = evaluate_pairwise_relation(base["source_value"].to_numpy(float), base["target_value"].to_numpy(float), settings, rng)
            for yr in sorted(base["year"].dropna().unique()):
                sub = base[base["year"] != yr]
                if len(sub) < 6:
                    continue
                stats = evaluate_pairwise_relation(sub["source_value"].to_numpy(float), sub["target_value"].to_numpy(float), settings, rng)
                rows.append({
                    "mode": mode,
                    "source_feature": sname,
                    "target_name": tname,
                    "dropped_year": yr,
                    "base_pearson_r": base_stats["pearson_r"],
                    "r_after_drop": stats["pearson_r"],
                    "delta_r_after_drop": stats["pearson_r"] - base_stats["pearson_r"] if np.isfinite(stats["pearson_r"]) and np.isfinite(base_stats["pearson_r"]) else np.nan,
                    "support_after_drop": stats["support_flag"],
                })
    return pd.DataFrame(rows)


def decade_block_sensitivity(source: pd.DataFrame, target: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_sources = source[source["source_feature"].isin([
        "H_zonal_width_E2_slope",
        "H_zonal_width_E2_signed_transition",
        "H_zonal_width_E2_post_state",
        "H_zonal_width_pre_M_mean",
        "H_zonal_width_pre_M_slope",
    ])]
    primary_targets = target[target["target_name"].isin([
        "P_NS_reorganization_index",
        "P_profile_contrast_30_32_minus_18_25",
        "P_position_only",
        "P_rainband_composite",
    ])]
    for (mode, sname), sdf in primary_sources.groupby(["mode", "source_feature"]):
        tmode = primary_targets[primary_targets["mode"] == mode]
        for tname, tdf in tmode.groupby("target_name"):
            base = sdf[["year", "source_value"]].merge(tdf[["year", "target_value"]], on="year", how="inner")
            if base.empty:
                continue
            years = np.asarray(sorted(base["year"].dropna().unique()), dtype=int)
            if len(years) < 15:
                continue
            base_stats = evaluate_pairwise_relation(base["source_value"].to_numpy(float), base["target_value"].to_numpy(float), settings, rng)
            start_year = int(np.nanmin(years) // 10 * 10)
            end_year = int(np.nanmax(years))
            for block_start in range(start_year, end_year + 1, 10):
                block_end = block_start + 9
                sub = base[~((base["year"] >= block_start) & (base["year"] <= block_end))]
                if len(sub) < 8:
                    continue
                stats = evaluate_pairwise_relation(sub["source_value"].to_numpy(float), sub["target_value"].to_numpy(float), settings, rng)
                rows.append({
                    "mode": mode,
                    "source_feature": sname,
                    "target_name": tname,
                    "block_start_year": block_start,
                    "block_end_year": block_end,
                    "n_removed": int(len(base) - len(sub)),
                    "base_pearson_r": base_stats["pearson_r"],
                    "r_after_block_removed": stats["pearson_r"],
                    "delta_r_after_block_removed": stats["pearson_r"] - base_stats["pearson_r"] if np.isfinite(stats["pearson_r"]) and np.isfinite(base_stats["pearson_r"]) else np.nan,
                    "support_after_block_removed": stats["support_flag"],
                })
    return pd.DataFrame(rows)


def decide_routes(
    local_rel: pd.DataFrame,
    increment_df: pd.DataFrame,
    target_rel: pd.DataFrame,
    time_rel: pd.DataFrame,
    comp_dom: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in settings.primary_modes:
        # Qn1 local vs background
        rel_m = filter_mode(local_rel, mode)
        inc_m = filter_mode(increment_df, mode)
        local_score = np.nanmean(rel_m[rel_m["source_family"] == "E2_local"]["primary_score"])
        broad_score = np.nanmean(rel_m[rel_m["source_family"] == "broad_background"]["primary_score"])
        inc_support = bool((inc_m["increment_support_flag"].sum() if "increment_support_flag" in inc_m else 0) > 0)
        if np.isfinite(local_score) and np.isfinite(broad_score) and local_score > broad_score and inc_support:
            decision = "E2_local_supported"
            allowed = "E2-anchored H_zonal_width information remains informative beyond tested broad-background features."
            nxt = "Use E2-local source form in mechanism candidate checks."
        elif np.isfinite(broad_score) and (not np.isfinite(local_score) or broad_score >= local_score) and not inc_support:
            decision = "broad_background_supported"
            allowed = "H_zonal_width signal is more consistent with broad pre-M background / memory than a purely local E2 transition."
            nxt = "Shift route wording toward broader pre-M H background, not E2-only transition."
        else:
            decision = "mixed_or_unclear_local_background"
            allowed = "Evidence is mixed between E2-local and broad-background H_zonal_width information."
            nxt = "Keep anomaly/local-background interpretations separated."
        rows.append({
            "question_id": "Qn1_local_vs_background",
            "mode": mode,
            "decision": decision,
            "direct_evidence": f"mean E2_local score={local_score:.3f}; mean broad score={broad_score:.3f}; any increment support={inc_support}",
            "derived_judgment": decision,
            "allowed_interpretation": allowed,
            "forbidden_interpretation": "Do not claim E2 transition uniquely represents H information.",
            "next_step": nxt,
        })

        # Qn2 target redefinition
        tr_m = filter_mode(target_rel, mode)
        fam = relation_family_summary(tr_m, ["target_family"])
        top_family = None if fam.empty else str(fam.sort_values(["mean_primary_score", "n_support"], ascending=False).iloc[0]["target_family"])
        dom_m = filter_mode(comp_dom, mode)
        composite_dominated = bool(dom_m["dominance_flag"].any()) if not dom_m.empty else False
        if top_family in ("P_NS_reorganization", "P_lat_profile_contrast") and not composite_dominated:
            decision = "target_ns_reorganization_supported"
            allowed = "Target is best treated as P north-south structural reorganization, not rainband-only."
        elif top_family == "P_position_only" or top_family == "P_position":
            decision = "target_position_only_or_position_dominant"
            allowed = "Target appears dominated by P position / centroid information."
        elif top_family in ("P_rainband_only", "P_rainband_component"):
            decision = "target_rainband_supported"
            allowed = "Target can still be described as rainband-related, with position checks retained."
        else:
            decision = "target_unclear_or_general_background"
            allowed = "Target specificity is unclear; avoid narrow target wording."
        rows.append({
            "question_id": "Qn2_target_redefinition",
            "mode": mode,
            "decision": decision,
            "direct_evidence": f"top target family={top_family}; composite dominated={composite_dominated}",
            "derived_judgment": decision,
            "allowed_interpretation": allowed,
            "forbidden_interpretation": "Do not state target is rainband-only if position/composite is stronger.",
            "next_step": "Use selected target definition for any later mechanism-narrow audit.",
        })

        # Qn3 weak preinformation value
        tm = filter_mode(time_rel, mode)
        temporal = relation_family_summary(tm, ["temporal_position"])
        top_pos = None if temporal.empty else str(temporal.sort_values(["mean_primary_score", "n_support"], ascending=False).iloc[0]["temporal_position"])
        post_score = np.nanmean(tm[tm["temporal_position"] == "post_m"]["primary_score"])
        pre_score = np.nanmean(tm[tm["temporal_position"].isin(["e2", "near_pre_m", "broad_pre_m"])]["primary_score"])
        if top_pos in ("e2", "near_pre_m", "broad_pre_m") and (not np.isfinite(post_score) or pre_score > post_score):
            decision = "weak_preinformation_candidate"
            allowed = "Pre-M H_zonal_width information is stronger than the post-M negative control in this audit."
        elif top_pos == "post_m" or (np.isfinite(post_score) and np.isfinite(pre_score) and post_score >= pre_score):
            decision = "common_year_background_likely"
            allowed = "Post-M or common-year information is comparable to pre-M information; preinformation wording is not supported."
        else:
            decision = "preinformation_unclear"
            allowed = "Temporal-position evidence is inconclusive."
        rows.append({
            "question_id": "Qn3_preinformation_value",
            "mode": mode,
            "decision": decision,
            "direct_evidence": f"top temporal position={top_pos}; mean pre score={pre_score:.3f}; mean post score={post_score:.3f}",
            "derived_judgment": decision,
            "allowed_interpretation": allowed,
            "forbidden_interpretation": "Do not state H causes P or that E2 has causal lead.",
            "next_step": "If candidate survives, move to narrow V/Je/Jw mechanism support; otherwise shift to common-background framing.",
        })
    return pd.DataFrame(rows)


def _bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path, top_n: int = 20) -> None:
    if df.empty or x not in df or y not in df:
        return
    plot_df = df.sort_values(y, ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df[x].astype(str), plot_df[y].astype(float))
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(route_df: pd.DataFrame, settings: Settings, output_root: Path) -> None:
    lines = [
        "# V10.7_n summary: H_zonal_width background / target / weak-preinformation audit",
        "",
        "This summary is not a causal interpretation. It separates direct outputs, derived judgments, allowed interpretations, and forbidden interpretations.",
        "",
    ]
    for _, row in route_df.iterrows():
        lines.extend([
            f"## {row['question_id']} / {row['mode']}",
            "",
            f"- Direct outputs: {row['direct_evidence']}",
            f"- Derived judgment: {row['derived_judgment']}",
            f"- Allowed interpretation: {row['allowed_interpretation']}",
            f"- Forbidden interpretation: {row['forbidden_interpretation']}",
            f"- Next step: {row['next_step']}",
            "",
        ])
    (output_root / "summary_h_zonal_width_background_target_preinfo_v10_7_n.md").write_text("\n".join(lines), encoding="utf-8")


def run_h_zonal_width_background_target_preinfo_v10_7_n(settings: Settings) -> dict[str, Any]:
    rng = np.random.default_rng(settings.random_seed)
    output_root = settings.output_root()
    clean_output_root(output_root)

    _log(settings, "loading smoothed fields")
    data = load_npz(settings)
    lat_key = first_key(data, ("lat", "latitude", "lats"))
    lon_key = first_key(data, ("lon", "longitude", "lons"))
    year_key = first_key(data, ("year", "years"))
    day_key = first_key(data, ("day", "days", "day_index"))
    h_key = first_key(data, ("z500_smoothed", "z500", "h", "H", "z500_field"))
    p_key = first_key(data, ("precip_smoothed", "precip", "P", "p", "precipitation"))
    if lat_key is None or lon_key is None:
        raise KeyError(f"Cannot find lat/lon in smoothed_fields keys={list(data.keys())}")
    if h_key is None or p_key is None:
        raise KeyError(f"Cannot find z500/precip fields in smoothed_fields keys={list(data.keys())}")

    years_raw = data[year_key] if year_key else None
    days_raw = data[day_key] if day_key else None
    lat_raw = coerce_1d(data[lat_key], name="lat")
    lon_raw = coerce_1d(data[lon_key], name="lon")

    h_norm, years, days, lat, lon, h_norm_audit = normalize_field_dims(data[h_key], years_raw, days_raw, lat_raw, lon_raw)
    p_norm, years_p, days_p, lat_p, lon_p, p_norm_audit = normalize_field_dims(data[p_key], years_raw, days_raw, lat_raw, lon_raw)
    if len(years) != len(years_p) or len(days) != len(days_p):
        raise ValueError("H and P normalized year/day dimensions do not match")

    h_sub, h_lat, h_lon = subset_domain(h_norm, lat, lon, settings.h_lat_range, settings.h_lon_range)
    p_sub, p_lat, p_lon = subset_domain(p_norm, lat_p, lon_p, settings.p_lat_range, settings.p_lon_range)

    input_audit = pd.DataFrame([
        {"field_name": "H", "detected_key": h_key, "shape": str(h_sub.shape), "lat_range": str((float(h_lat.min()), float(h_lat.max()))), "lon_range": str((float(h_lon.min()), float(h_lon.max()))), "domain_lat_range": str(settings.h_lat_range), "domain_lon_range": str(settings.h_lon_range), "n_years": len(years), "n_days": len(days), "status": "ok", **{f"norm_{k}": v for k, v in h_norm_audit.items()}},
        {"field_name": "P", "detected_key": p_key, "shape": str(p_sub.shape), "lat_range": str((float(p_lat.min()), float(p_lat.max()))), "lon_range": str((float(p_lon.min()), float(p_lon.max()))), "domain_lat_range": str(settings.p_lat_range), "domain_lon_range": str(settings.p_lon_range), "n_years": len(years), "n_days": len(days), "status": "ok", **{f"norm_{k}": v for k, v in p_norm_audit.items()}},
    ])
    write_dataframe(input_audit, output_root / "tables" / "input_audit_v10_7_n.csv")

    h_modes = build_mode_fields(h_sub, settings)
    p_modes = build_mode_fields(p_sub, settings)

    all_source_features: list[pd.DataFrame] = []
    all_sliding_features: list[pd.DataFrame] = []
    all_targets: list[pd.DataFrame] = []
    all_components: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []

    for mode in settings.modes:
        _log(settings, f"building daily metrics for mode={mode}")
        h_metrics = build_h_daily_metrics(h_modes[mode], h_lat, h_lon)
        p_metrics = build_p_daily_metrics(p_modes[mode], p_lat)
        all_source_features.append(build_source_features_for_mode(h_metrics, years, days, settings, mode))
        all_sliding_features.append(build_sliding_features_for_mode(h_metrics, years, days, settings, mode))
        targets, components, profiles = build_p_transition_targets_for_mode(p_metrics, p_modes[mode], years, days, p_lat, settings, mode)
        all_targets.append(targets)
        all_components.append(components)
        all_profiles.append(profiles)

    source_features = pd.concat(all_source_features, ignore_index=True)
    sliding_features = pd.concat(all_sliding_features, ignore_index=True)
    targets = pd.concat(all_targets, ignore_index=True)
    components = pd.concat(all_components, ignore_index=True)
    profiles = pd.concat(all_profiles, ignore_index=True)

    write_dataframe(source_features, output_root / "tables" / "h_zonal_width_source_features_v10_7_n.csv")
    write_dataframe(sliding_features, output_root / "tables" / "h_zonal_width_sliding_window_features_v10_7_n.csv")
    write_dataframe(targets, output_root / "tables" / "p_target_candidates_v10_7_n.csv")
    write_dataframe(components, output_root / "tables" / "p_ns_reorganization_index_components_v10_7_n.csv")
    write_dataframe(profiles, output_root / "tables" / "p_lat_profile_contrast_v10_7_n.csv")

    _log(settings, f"experiment A: local-vs-background relations (policy={settings.experiment_a_policy})")
    local_bg_sources = source_features[source_features["source_family"].isin(["E2_local", "broad_background"])]
    main_targets = targets[targets["target_name"].isin([
        "P_NS_reorganization_index",
        "P_profile_contrast_30_32_minus_18_25",
        "P_position_only",
        "P_rainband_composite",
        "P_rainband_main_minus_south",
        "P_centroid_lat_transition",
        "P_main_minus_south_transition",
    ])]

    if settings.experiment_a_policy == "screen":
        local_relation_settings = _screen_settings(settings)
    else:
        local_relation_settings = settings
    _log(settings, "experiment A1: compact local/broad pairwise table")
    local_bg_rel = make_feature_target_relation_table(local_bg_sources, main_targets, local_relation_settings, rng)
    local_bg_rel["experiment_a_policy"] = settings.experiment_a_policy
    write_dataframe(local_bg_rel, output_root / "tables" / "h_zonal_width_local_vs_background_relation_v10_7_n.csv")

    if settings.experiment_a_policy == "skip_heavy":
        _log(settings, "experiment A2/A3 skipped by policy=skip_heavy")
        increment_df = empty_increment_table(settings.experiment_a_policy)
        sliding_rel = empty_relation_table()
    else:
        _log(settings, "experiment A2: local-vs-background incremental CV")
        increment_settings = settings if settings.experiment_a_policy == "full" else replace(settings, n_perm=0, n_boot=0)
        increment_df = incremental_cv_table(local_bg_sources, targets, increment_settings, rng)

        _log(settings, "experiment A3: sliding-window background rank")
        sliding_settings = settings if settings.experiment_a_policy == "full" else _screen_settings(settings)
        sliding_rel = make_feature_target_relation_table(sliding_features, main_targets, sliding_settings, rng)
        sliding_rel["experiment_a_policy"] = settings.experiment_a_policy

    write_dataframe(increment_df, output_root / "tables" / "h_zonal_width_background_increment_v10_7_n.csv")
    write_dataframe(sliding_rel, output_root / "tables" / "h_zonal_width_sliding_background_rank_v10_7_n.csv")

    _log(settings, "experiment B: P target redefinition")
    selected_sources = source_features[source_features["source_feature"].isin([
        "H_zonal_width_E2_slope",
        "H_zonal_width_E2_signed_transition",
        "H_zonal_width_E2_post_state",
        "H_zonal_width_E2_window_mean",
        "H_zonal_width_pre_M_mean",
        "H_zonal_width_pre_M_slope",
    ])]
    target_rel = make_feature_target_relation_table(selected_sources, targets, settings, rng)
    write_dataframe(target_rel, output_root / "tables" / "p_target_redefinition_v10_7_n.csv")
    target_family_summary = relation_family_summary(target_rel, ["mode", "target_family"])
    write_dataframe(target_family_summary, output_root / "tables" / "p_target_family_route_decision_v10_7_n.csv")
    comp_dom = composite_component_dominance(components)
    write_dataframe(comp_dom, output_root / "tables" / "p_ns_composite_component_dominance_v10_7_n.csv")

    _log(settings, "experiment C: weak preinformation and temporal controls")
    time_sources = source_features[source_features["source_family"] == "time_order_control"]
    time_rel = make_feature_target_relation_table(time_sources, targets, settings, rng)
    write_dataframe(time_rel, output_root / "tables" / "time_order_negative_control_v10_7_n.csv")

    loyo = leave_one_year_out_relation(source_features, targets, settings, rng)
    write_dataframe(loyo, output_root / "tables" / "year_influence_preinformation_v10_7_n.csv")
    decade = decade_block_sensitivity(source_features, targets, settings, rng)
    write_dataframe(decade, output_root / "tables" / "decade_block_sensitivity_v10_7_n.csv")

    _log(settings, "route decisions")
    route_df = decide_routes(local_bg_rel, increment_df, target_rel, time_rel, comp_dom, settings)
    write_dataframe(route_df, output_root / "tables" / "route_decision_v10_7_n.csv")

    # Figures: simple score summaries.
    _log(settings, "writing figures")
    local_sum = relation_family_summary(local_bg_rel[local_bg_rel["mode"].isin(settings.primary_modes)], ["mode", "source_family"])
    if not local_sum.empty:
        local_sum["label"] = local_sum["mode"].astype(str) + " / " + local_sum["source_family"].astype(str)
        _bar_plot(local_sum, "label", "mean_primary_score", "Local E2 vs broad H_zonal_width background", output_root / "figures" / "local_vs_background_h_zonal_width_score_v10_7_n.png")
    if not target_family_summary.empty:
        target_family_summary["label"] = target_family_summary["mode"].astype(str) + " / " + target_family_summary["target_family"].astype(str)
        _bar_plot(target_family_summary, "label", "mean_primary_score", "P target family redefinition", output_root / "figures" / "p_target_redefinition_score_v10_7_n.png")
    temporal_sum = relation_family_summary(time_rel[time_rel["mode"].isin(settings.primary_modes)], ["mode", "temporal_position"])
    if not temporal_sum.empty:
        temporal_sum["label"] = temporal_sum["mode"].astype(str) + " / " + temporal_sum["temporal_position"].astype(str)
        _bar_plot(temporal_sum, "label", "mean_primary_score", "Time-order negative control", output_root / "figures" / "time_order_negative_control_v10_7_n.png")
    if not comp_dom.empty:
        comp_plot = comp_dom.copy()
        comp_plot["label"] = comp_plot["mode"].astype(str) + " / " + comp_plot["component"].astype(str)
        _bar_plot(comp_plot, "label", "corr_with_P_NS_reorganization_index", "P_NS composite component dominance", output_root / "figures" / "p_ns_reorganization_components_v10_7_n.png")
    if not loyo.empty:
        loyo_plot = loyo.groupby(["mode", "source_feature", "target_name"], dropna=False).agg(max_abs_delta_r=("delta_r_after_drop", lambda x: float(np.nanmax(np.abs(x))))).reset_index()
        loyo_plot["label"] = loyo_plot["mode"].astype(str) + " / " + loyo_plot["source_feature"].astype(str) + " / " + loyo_plot["target_name"].astype(str)
        _bar_plot(loyo_plot, "label", "max_abs_delta_r", "Maximum leave-one-year-out influence", output_root / "figures" / "year_influence_preinformation_v10_7_n.png")

    run_meta = {
        "version": settings.version,
        "task": settings.output_tag,
        "created_at_utc": now_utc(),
        "settings": settings.to_dict(),
        "input_file": str(settings.smoothed_fields_path()),
        "parent_versions": ["V10.7_l", "V10.7_m"],
        "base_logic_inherited_from": [
            "V10.7_l H/P metric construction style",
            "V10.7_m relation scoring and permutation/bootstrap style",
        ],
        "questions_answered": {
            "Qn1": "local E2 H_zonal_width information vs broad pre-M background",
            "Qn2": "P target redefinition: position-only vs rainband-only vs NS reorganization",
            "Qn3": "weak preinformation value vs common-year/background explanation",
        },
        "mode_definitions": {
            "raw": "raw smoothed domain field",
            "anomaly": "field minus daily climatology across years",
            "local_background_removed": "field minus same-day domain mean, then daily climatology removed from the local residual",
        },
        "method_boundary": [
            "not causal inference",
            "not full W33-to-W45 mapping",
            "not full P/V/H/Je/Jw object-network audit",
            "not proof that transition windows represent most object information",
            "does not control away P/V/Je/Jw",
            "does not re-test scalarized transition failure",
            "tests local-vs-background H_zonal_width information",
            "tests P target redefinition",
            "tests weak preinformation value, not causality",
        ],
        "forbidden_interpretations": [
            "Do not state H causes P",
            "Do not state H controls W45",
            "Do not state E2 is the unique source window unless evidence supports it",
            "Do not state transition window represents most H information",
            "Do not state P target is rainband-only if position/composite is stronger",
            "Do not interpret scalarized transition failure as strength-class failure",
        ],
        "outputs": {
            "tables": sorted([p.name for p in (output_root / "tables").glob("*.csv")]),
            "figures": sorted([p.name for p in (output_root / "figures").glob("*.png")]),
        },
    }
    write_json(run_meta, output_root / "run_meta" / "run_meta_v10_7_n.json")
    write_summary(route_df, settings, output_root)
    _log(settings, f"done: {output_root}")
    return run_meta
