from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .settings import CORE_FIELDS, REQUIRED_INPUTS


def validate_input_contract(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    missing = set(REQUIRED_INPUTS) - set(arrays)
    if missing:
        raise KeyError(f"输入数组缺失：{sorted(missing)}")

    report: Dict[str, Any] = {
        "field_shapes": {},
        "field_dtypes": {},
        "checks": {},
    }

    reference_shape = None
    for field_name in CORE_FIELDS:
        arr = np.asarray(arrays[field_name])
        if arr.ndim != 4:
            raise ValueError(
                f"{field_name} 维度错误：当前 shape={arr.shape}，要求 (year, day, lat, lon)。"
            )
        report["field_shapes"][field_name] = list(arr.shape)
        report["field_dtypes"][field_name] = str(arr.dtype)
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"主场 shape 不一致：参考 {reference_shape}，但 {field_name} 为 {arr.shape}。"
            )

    assert reference_shape is not None
    n_year, _n_day, n_lat, n_lon = reference_shape
    lat = np.asarray(arrays["lat"])
    lon = np.asarray(arrays["lon"])
    years = np.asarray(arrays["years"])

    if lat.ndim != 1:
        raise ValueError(f"lat 必须为一维，当前 shape={lat.shape}")
    if lon.ndim != 1:
        raise ValueError(f"lon 必须为一维，当前 shape={lon.shape}")
    if years.ndim != 1:
        raise ValueError(f"years 必须为一维，当前 shape={years.shape}")

    if len(lat) != n_lat:
        raise ValueError(f"lat 长度 {len(lat)} 与主场纬向维 {n_lat} 不一致。")
    if len(lon) != n_lon:
        raise ValueError(f"lon 长度 {len(lon)} 与主场经向维 {n_lon} 不一致。")
    if len(years) != n_year:
        raise ValueError(f"years 长度 {len(years)} 与主场年份维 {n_year} 不一致。")

    report["field_shapes"]["lat"] = list(lat.shape)
    report["field_shapes"]["lon"] = list(lon.shape)
    report["field_shapes"]["years"] = list(years.shape)
    report["field_dtypes"]["lat"] = str(lat.dtype)
    report["field_dtypes"]["lon"] = str(lon.dtype)
    report["field_dtypes"]["years"] = str(years.dtype)

    report["checks"] = {
        "main_fields_are_4d": True,
        "main_fields_same_shape": True,
        "lat_matches": True,
        "lon_matches": True,
        "years_matches": True,
        "main_shape_contract": "(year, day, lat, lon)",
    }
    return report


def summarize_finite_status(arrays: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for name, arr in arrays.items():
        if np.asarray(arr).dtype.kind not in "iuf":
            continue
        arr64 = np.asarray(arr, dtype=np.float64)
        total = int(arr64.size)
        nan_count = int(np.isnan(arr64).sum())
        inf_count = int(np.isinf(arr64).sum())
        summary[name] = {
            "total_count": float(total),
            "nan_count": float(nan_count),
            "inf_count": float(inf_count),
            "nan_fraction": float(nan_count / total if total else 0.0),
            "inf_fraction": float(inf_count / total if total else 0.0),
        }
    return summary
