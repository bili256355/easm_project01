from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from stage_partition_v6.safe_stats import safe_nanmean, safe_daily_energy, build_all_nan_mask


FIELDS = ["P", "V", "H", "Je", "Jw"]


@dataclass
class FieldStateResult:
    field: str
    raw_matrix: np.ndarray
    state_matrix: np.ndarray
    valid_day_mask: np.ndarray
    valid_day_index: np.ndarray
    feature_table: pd.DataFrame
    scale_table: pd.DataFrame
    empty_feature_audit: pd.DataFrame
    meta: dict


def _zscore_features_along_day(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    mean, _ = safe_nanmean(x, axis=0, return_valid_count=True)
    centered = x - mean[None, :]
    var, _ = safe_nanmean(np.square(centered), axis=0, return_valid_count=True)
    std = np.sqrt(var)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    z = centered / std[None, :]
    return z, mean, std


def build_field_state(
    profiles: dict,
    field: str,
    *,
    standardize: bool = True,
    trim_invalid_days: bool = True,
    shared_valid_day_index: np.ndarray | None = None,
) -> FieldStateResult:
    if field not in FIELDS:
        raise ValueError(f"Unsupported field={field!r}; expected one of {FIELDS}")
    if field not in profiles:
        raise KeyError(f"profiles is missing field {field!r}")

    obj = profiles[field]
    seasonal, _ = safe_nanmean(obj.raw_cube, axis=0, return_valid_count=True)
    raw = np.asarray(seasonal, dtype=float)

    if standardize:
        state, feat_mean, feat_std = _zscore_features_along_day(raw)
        state_expression = "field_only_raw_smoothed_zscore"
    else:
        state = raw.copy()
        feat_mean = np.full(raw.shape[1], np.nan, dtype=float)
        feat_std = np.full(raw.shape[1], np.nan, dtype=float)
        state_expression = "field_only_raw_smoothed_unstandardized"

    if shared_valid_day_index is not None:
        shared_valid_day_index = np.asarray(shared_valid_day_index, dtype=int)
        local_finite = np.all(np.isfinite(state[shared_valid_day_index, :]), axis=1)
        valid_day_index = shared_valid_day_index[local_finite] if trim_invalid_days else shared_valid_day_index
        valid_day_mask = np.zeros(raw.shape[0], dtype=bool)
        valid_day_mask[valid_day_index] = True
    else:
        valid_day_mask = np.all(np.isfinite(state), axis=1) if trim_invalid_days else np.ones(raw.shape[0], dtype=bool)
        valid_day_index = np.where(valid_day_mask)[0].astype(int)

    scale_rows = []
    empty_rows = []
    for j, latv in enumerate(obj.lat_grid):
        col = raw[:, j]
        zcol = state[:, j]
        finite_col = np.isfinite(col)
        scale_rows.append(
            {
                "field": field,
                "feature_index": int(j),
                "lat_value": float(latv),
                "raw_mean_day": float(feat_mean[j]) if np.isfinite(feat_mean[j]) else np.nan,
                "raw_std_day": float(feat_std[j]) if np.isfinite(feat_std[j]) else np.nan,
                "z_mean_day": float(np.nanmean(zcol)) if np.isfinite(zcol).any() else np.nan,
                "z_std_day": float(np.nanstd(zcol)) if np.isfinite(zcol).any() else np.nan,
            }
        )
        empty_rows.append(
            {
                "field": field,
                "feature_index": int(j),
                "lat_value": float(latv),
                "all_nan_flag": bool(build_all_nan_mask(col, axis=0)),
                "n_valid_days": int(finite_col.sum()),
                "n_missing_days": int(raw.shape[0] - finite_col.sum()),
            }
        )

    meta = {
        "field": field,
        "n_days": int(raw.shape[0]),
        "n_features": int(raw.shape[1]),
        "n_valid_days": int(valid_day_index.size),
        "n_invalid_days": int(raw.shape[0] - valid_day_index.size),
        "valid_day_index": valid_day_index.astype(int).tolist(),
        "invalid_day_index": np.where(~valid_day_mask)[0].astype(int).tolist(),
        "state_expression_name": state_expression,
        "standardize": bool(standardize),
        "trim_invalid_days": bool(trim_invalid_days),
        "uses_shared_joint_valid_day_index": bool(shared_valid_day_index is not None),
        "field_energy_raw": safe_daily_energy(raw),
        "field_energy_state": safe_daily_energy(state),
    }

    feature_table = pd.DataFrame(
        [
            {"field": field, "feature_index": int(j), "lat_value": float(latv)}
            for j, latv in enumerate(obj.lat_grid)
        ]
    )

    return FieldStateResult(
        field=field,
        raw_matrix=raw,
        state_matrix=state,
        valid_day_mask=valid_day_mask,
        valid_day_index=valid_day_index,
        feature_table=feature_table,
        scale_table=pd.DataFrame(scale_rows),
        empty_feature_audit=pd.DataFrame(empty_rows),
        meta=meta,
    )


def build_field_state_matrix_for_year_indices(
    profiles: dict,
    field: str,
    year_indices: np.ndarray | list[int] | None,
    *,
    standardize: bool = True,
    trim_invalid_days: bool = True,
    shared_valid_day_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a lightweight field-only state matrix for a year subset/resample.

    This is used by V7-b bootstrap and leave-one-year-out timing checks. It mirrors
    build_field_state() but avoids producing large feature audit tables in each
    resampling iteration.

    Returns
    -------
    matrix_valid : np.ndarray
        day-by-feature matrix after valid-day trimming.
    valid_day_index : np.ndarray
        Original day indices corresponding to matrix_valid rows.
    meta : dict
        Small metadata block for audit/debugging.
    """
    if field not in FIELDS:
        raise ValueError(f"Unsupported field={field!r}; expected one of {FIELDS}")
    if field not in profiles:
        raise KeyError(f"profiles is missing field {field!r}")

    obj = profiles[field]
    cube = np.asarray(obj.raw_cube, dtype=float)
    n_years = int(cube.shape[0])
    if year_indices is None:
        selected = np.arange(n_years, dtype=int)
    else:
        selected = np.asarray(year_indices, dtype=int)
        if selected.size == 0:
            raise ValueError("year_indices is empty")
        if selected.min() < 0 or selected.max() >= n_years:
            raise IndexError(
                f"year_indices out of range for n_years={n_years}: "
                f"min={selected.min()}, max={selected.max()}"
            )
    sample_cube = cube[selected, :, :]
    seasonal, _ = safe_nanmean(sample_cube, axis=0, return_valid_count=True)
    raw = np.asarray(seasonal, dtype=float)

    if standardize:
        state, _, _ = _zscore_features_along_day(raw)
        expression = "field_only_raw_smoothed_zscore_year_resampled"
    else:
        state = raw.copy()
        expression = "field_only_raw_smoothed_unstandardized_year_resampled"

    if shared_valid_day_index is not None:
        shared_valid_day_index = np.asarray(shared_valid_day_index, dtype=int)
        if trim_invalid_days:
            local_finite = np.all(np.isfinite(state[shared_valid_day_index, :]), axis=1)
            valid_day_index = shared_valid_day_index[local_finite]
        else:
            valid_day_index = shared_valid_day_index
    else:
        if trim_invalid_days:
            valid_day_index = np.where(np.all(np.isfinite(state), axis=1))[0].astype(int)
        else:
            valid_day_index = np.arange(state.shape[0], dtype=int)

    matrix_valid = state[valid_day_index, :]
    meta = {
        "field": field,
        "n_years_available": n_years,
        "n_years_selected": int(selected.size),
        "unique_year_indices_selected": sorted(set(int(x) for x in selected.tolist())),
        "n_days": int(raw.shape[0]),
        "n_features": int(raw.shape[1]),
        "n_valid_days": int(valid_day_index.size),
        "state_expression_name": expression,
        "standardize": bool(standardize),
        "trim_invalid_days": bool(trim_invalid_days),
        "uses_shared_joint_valid_day_index": bool(shared_valid_day_index is not None),
    }
    return matrix_valid, valid_day_index.astype(int), meta
