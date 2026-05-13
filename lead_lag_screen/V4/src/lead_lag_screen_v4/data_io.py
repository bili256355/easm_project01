from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .settings import REGION_SPECS, LeadLagScreenV4Settings

CORE_FIELDS = ("precip", "u200", "z500", "v850")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 npz 文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {str(k): np.asarray(data[k]) for k in data.files}


def _field_from_bundle(bundle: Dict[str, np.ndarray], field: str, suffix: str) -> np.ndarray:
    candidates = [f"{field}_{suffix}", field]
    for key in candidates:
        if key in bundle:
            return np.asarray(bundle[key], dtype=np.float64)
    raise KeyError(f"bundle 缺少 {field}_{suffix} 或 {field}")


def load_field_anomalies(settings: LeadLagScreenV4Settings) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Load smooth5 anomaly fields.

    Preferred input:
        foundation/V1/outputs/baseline_smooth5_a/preprocess/anomaly_fields.npz
    Fallback:
        smoothed_fields.npz - daily_climatology.npz
    """
    pre = settings.preprocess_dir
    anom_path = pre / "anomaly_fields.npz"
    smoothed_path = pre / "smoothed_fields.npz"
    clim_path = pre / "daily_climatology.npz"
    meta: Dict[str, object] = {
        "preprocess_dir": str(pre),
        "used_path": None,
        "fallback_used": False,
    }

    if anom_path.exists():
        bundle = read_npz(anom_path)
        fields = {name: _field_from_bundle(bundle, name, "anom") for name in CORE_FIELDS}
        lat = np.asarray(bundle["lat"], dtype=np.float64)
        lon = np.asarray(bundle["lon"], dtype=np.float64)
        years = np.asarray(bundle["years"])
        meta["used_path"] = str(anom_path)
        meta["npz_keys"] = sorted(bundle.keys())
        return fields, lat, lon, years, meta

    if smoothed_path.exists() and clim_path.exists():
        sm = read_npz(smoothed_path)
        cl = read_npz(clim_path)
        fields = {}
        for name in CORE_FIELDS:
            sm_arr = _field_from_bundle(sm, name, "smoothed")
            cl_arr = _field_from_bundle(cl, name, "clim")
            fields[name] = sm_arr - cl_arr[None, :, :, :]
        lat = np.asarray(sm["lat"], dtype=np.float64)
        lon = np.asarray(sm["lon"], dtype=np.float64)
        years = np.asarray(sm["years"])
        meta["used_path"] = str(smoothed_path) + " minus " + str(clim_path)
        meta["fallback_used"] = True
        meta["smoothed_keys"] = sorted(sm.keys())
        meta["climatology_keys"] = sorted(cl.keys())
        return fields, lat, lon, years, meta

    raise FileNotFoundError(
        "未找到 5日场 anomaly 输入。需要以下之一：\n"
        f"1) {anom_path}\n"
        f"2) {smoothed_path} 与 {clim_path}\n"
        "注意：上传工程包可能不含 preprocess 大文件；请在本地完整工程中运行。"
    )


def mask_between(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.isfinite(arr) & (arr >= lower) & (arr <= upper)


def extract_object_subfield(
    fields: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    object_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = REGION_SPECS[object_name]
    lat_mask = mask_between(lat, *spec.lat_range)
    lon_mask = mask_between(lon, *spec.lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"{object_name} 纬度范围 {spec.lat_range} 在坐标中没有命中。")
    if not np.any(lon_mask):
        raise ValueError(f"{object_name} 经度范围 {spec.lon_range} 在坐标中没有命中。")
    field = np.asarray(fields[spec.source_field], dtype=np.float64)
    sub = field[:, :, lat_mask, :][:, :, :, lon_mask]
    return sub, np.asarray(lat[lat_mask], dtype=np.float64), np.asarray(lon[lon_mask], dtype=np.float64)


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None
