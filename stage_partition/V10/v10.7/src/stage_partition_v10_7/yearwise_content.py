from __future__ import annotations

import numpy as np
import pandas as pd

from .event_content_config import EventContentConfig
from .event_content_io import SpatialFieldData
from .spatial_composite import _domain_subset, _event_diff, flatten_finite


def compute_yearwise_content(spatial: SpatialFieldData | None, cfg: EventContentConfig, climatological_maps: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if spatial is None or not spatial.has_year_dim:
        return pd.DataFrame(), pd.DataFrame(), {"yearwise_status": "skipped_no_year_dimension"}
    sub, lat, lon, _ = _domain_subset(spatial, cfg.lat_range, cfg.lon_range, "full_domain_context")
    years = spatial.years if spatial.years is not None else np.arange(sub.shape[0])
    rows = []
    for ev in cfg.event_windows:
        yearly = _event_diff(sub, ev)
        clim = climatological_maps.get(ev.event_id)
        for i in range(yearly.shape[0]):
            m = yearly[i]
            x, y = flatten_finite(m, clim) if clim is not None else (np.array([]), np.array([]))
            corr = np.nan
            same = np.nan
            if len(x) >= 3 and np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
                corr = float(np.corrcoef(x, y)[0, 1])
                same = float(np.mean(np.sign(x) == np.sign(y)))
            rows.append({
                "year": int(years[i]) if np.asarray(years).dtype.kind in "iu" else str(years[i]),
                "year_index": int(i),
                "event_id": ev.event_id,
                "field_change_norm": float(np.sqrt(np.nanmean(m ** 2))) if np.isfinite(m).any() else np.nan,
                "field_abs_mean": float(np.nanmean(np.abs(m))) if np.isfinite(m).any() else np.nan,
                "pattern_corr_to_climatological_event": corr,
                "same_sign_fraction_to_climatological_event": same,
            })
    df = pd.DataFrame(rows)
    sum_rows = []
    if not df.empty:
        for ev, g in df.groupby("event_id"):
            vals = g["field_change_norm"].to_numpy(dtype=float)
            corr = g["pattern_corr_to_climatological_event"].to_numpy(dtype=float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            cv = std / mean if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
            frac_pos = float(np.nanmean(corr > 0)) if np.isfinite(corr).any() else np.nan
            median_corr = float(np.nanmedian(corr)) if np.isfinite(corr).any() else np.nan
            if np.isfinite(median_corr) and median_corr >= 0.5 and frac_pos >= 0.7:
                cls = "yearwise_consistent"
            elif np.isfinite(median_corr) and median_corr >= 0.25 and frac_pos >= 0.55:
                cls = "moderately_consistent"
            elif np.isfinite(median_corr):
                cls = "yearwise_unstable"
            else:
                cls = "not_available"
            sum_rows.append({
                "event_id": ev,
                "n_years": int(len(g)),
                "mean_change_norm": float(mean) if np.isfinite(mean) else np.nan,
                "std_change_norm": float(std) if np.isfinite(std) else np.nan,
                "cv_change_norm": float(cv) if np.isfinite(cv) else np.nan,
                "median_pattern_corr": median_corr,
                "fraction_positive_pattern_corr": frac_pos,
                "yearwise_consistency_class": cls,
            })
    return df, pd.DataFrame(sum_rows), {"yearwise_status": "computed"}
