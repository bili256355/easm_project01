from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        raise ValueError("smooth_window 必须为奇数。")
    kernel = np.ones(window, dtype=np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    numer = np.convolve(filled, kernel, mode="same")
    denom = np.convolve(valid, kernel, mode="same")
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def merge_segments(segments: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def nanmean_no_warn(arr: np.ndarray, axis=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr)
    count = np.sum(valid, axis=axis)
    summed = np.nansum(arr, axis=axis)
    out_shape = np.asarray(summed).shape
    out = np.full(out_shape, np.nan, dtype=np.float64)
    np.divide(summed, count, out=out, where=count > 0)
    return out


def nanstd_no_warn(arr: np.ndarray, axis=None, ddof: int = 0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mean = nanmean_no_warn(arr, axis=axis)
    if axis is not None:
        expanded = np.expand_dims(mean, axis=axis)
    else:
        expanded = mean
    diff2 = np.where(np.isfinite(arr), (arr - expanded) ** 2, np.nan)
    valid = np.isfinite(diff2)
    count = np.sum(valid, axis=axis)
    denom = count - ddof
    summed = np.nansum(diff2, axis=axis)
    out_shape = np.asarray(summed).shape
    out = np.full(out_shape, np.nan, dtype=np.float64)
    np.divide(summed, denom, out=out, where=denom > 0)
    return np.sqrt(out)


def safe_zscore_cube(cube: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    mu = nanmean_no_warn(flat, axis=0)
    sd = nanstd_no_warn(flat, axis=0)
    sd = np.where(sd < eps, np.nan, sd)
    return (arr - mu) / sd


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
