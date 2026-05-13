from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .metadata import VARIABLE_ORDER
from .settings import IndexValidityV1BSettings

CORE_FIELDS = ("precip", "u200", "z500", "v850")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 npz 文件：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {str(k): np.asarray(data[k]) for k in data.files}


def _field_from_bundle(bundle: Dict[str, np.ndarray], field: str, suffix: str) -> np.ndarray:
    for key in (f"{field}_{suffix}", field):
        if key in bundle:
            return np.asarray(bundle[key], dtype=np.float64)
    raise KeyError(f"bundle 缺少 {field}_{suffix} 或 {field}")


def load_fields_for_mode(settings: IndexValidityV1BSettings) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Load smooth5 fields for the selected index-validity data mode.

    Main/default mode is ``smoothed``: load ``smoothed_fields.npz`` and use
    ``index_values_smoothed.csv``. Anomaly mode is available only as an
    auxiliary audit: load ``anomaly_fields.npz`` or compute smoothed-minus-clim.
    """
    mode = settings.data_mode
    anom_path = settings.anomaly_fields_bundle
    smoothed_path = settings.smoothed_fields_bundle
    clim_path = settings.climatology_bundle
    meta: Dict[str, object] = {"fallback_used": False, "used_path": None, "notes": [], "data_mode": mode}

    if mode == "smoothed":
        if not smoothed_path.exists():
            raise FileNotFoundError(
                "未找到 smooth5 smoothed field 输入。index_validity 主审计需要：\n"
                f"{smoothed_path}\n"
                "注意：上传工程包可能不含 preprocess 大文件，请在本地完整工程中运行。"
            )
        bundle = read_npz(smoothed_path)
        fields = {name: _field_from_bundle(bundle, name, "smoothed") for name in CORE_FIELDS}
        lat = np.asarray(bundle["lat"], dtype=np.float64)
        lon = np.asarray(bundle["lon"], dtype=np.float64)
        years = np.asarray(bundle["years"]).astype(int)
        meta.update({"used_path": str(smoothed_path), "npz_keys": sorted(bundle.keys())})
        return fields, lat, lon, years, meta

    if mode == "anomaly":
        if anom_path.exists():
            bundle = read_npz(anom_path)
            fields = {name: _field_from_bundle(bundle, name, "anom") for name in CORE_FIELDS}
            lat = np.asarray(bundle["lat"], dtype=np.float64)
            lon = np.asarray(bundle["lon"], dtype=np.float64)
            years = np.asarray(bundle["years"]).astype(int)
            meta.update({"used_path": str(anom_path), "npz_keys": sorted(bundle.keys())})
            return fields, lat, lon, years, meta

        if smoothed_path.exists() and clim_path.exists():
            sm = read_npz(smoothed_path)
            cl = read_npz(clim_path)
            fields = {}
            for name in CORE_FIELDS:
                fields[name] = _field_from_bundle(sm, name, "smoothed") - _field_from_bundle(cl, name, "clim")[None, :, :, :]
            lat = np.asarray(sm["lat"], dtype=np.float64)
            lon = np.asarray(sm["lon"], dtype=np.float64)
            years = np.asarray(sm["years"]).astype(int)
            meta.update({
                "fallback_used": True,
                "used_path": f"{smoothed_path} minus {clim_path}",
                "smoothed_keys": sorted(sm.keys()),
                "climatology_keys": sorted(cl.keys()),
            })
            return fields, lat, lon, years, meta

        raise FileNotFoundError(
            "未找到 smooth5 anomaly field 输入。anomaly 辅助审计需要以下之一：\n"
            f"1) {anom_path}\n"
            f"2) {smoothed_path} 与 {clim_path}\n"
            "注意：上传工程包可能不含 preprocess 大文件，请在本地完整工程中运行。"
        )

    raise ValueError(f"Unsupported data_mode={mode!r}; expected 'smoothed' or 'anomaly'.")


def load_index_table_for_mode(settings: IndexValidityV1BSettings) -> pd.DataFrame:
    path = settings.selected_index_path
    if not path.exists():
        raise FileNotFoundError(f"找不到 {settings.data_mode} index 表：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"year", "day", *VARIABLE_ORDER}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少列：{missing}")
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    return df


# Backward-compatible aliases for older imports. Do not use for new code.
def load_anomaly_fields(settings: IndexValidityV1BSettings):
    return load_fields_for_mode(settings)


def load_index_anomalies(settings: IndexValidityV1BSettings) -> pd.DataFrame:
    return load_index_table_for_mode(settings)


def mask_between(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.isfinite(arr) & (arr >= lower) & (arr <= upper)


def subset_field(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_mask = mask_between(lat, *lat_range)
    lon_mask = mask_between(lon, *lon_range)
    if not np.any(lat_mask):
        raise ValueError(f"纬度范围 {lat_range} 没有命中格点。")
    if not np.any(lon_mask):
        raise ValueError(f"经度范围 {lon_range} 没有命中格点。")
    return field[:, :, lat_mask, :][:, :, :, lon_mask], np.asarray(lat[lat_mask], dtype=np.float64), np.asarray(lon[lon_mask], dtype=np.float64)


def samples_from_year_day(field: np.ndarray, years: np.ndarray, sample_df: pd.DataFrame) -> np.ndarray:
    year_to_i = {int(y): i for i, y in enumerate(years.astype(int))}
    yi = []
    di = []
    for row in sample_df.itertuples(index=False):
        y = int(getattr(row, "year"))
        d = int(getattr(row, "day")) - 1
        if y in year_to_i and 0 <= d < field.shape[1]:
            yi.append(year_to_i[y])
            di.append(d)
    if not yi:
        return np.empty((0,) + field.shape[2:], dtype=np.float64)
    return field[np.asarray(yi, dtype=int), np.asarray(di, dtype=int), :, :]
