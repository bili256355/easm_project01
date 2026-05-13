# -*- coding: utf-8 -*-
"""Figure helpers for P/V850 latitudinal object-change audit v1_a."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .t3_p_v_latitudinal_object_change_settings import PVLatitudinalObjectChangeSettings


def _import_plotting(use_cartopy: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ccrs = None
    cfeature = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs  # type: ignore
            import cartopy.feature as cfeature  # type: ignore
        except Exception:
            ccrs = None
            cfeature = None
    return plt, ccrs, cfeature


def _sorted_lat_lon_data(lat: np.ndarray, lon: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    return lat[lat_order], lon[lon_order], data[np.ix_(lat_order, lon_order)]


def _symmetric_vlim(arrs: Iterable[np.ndarray], q: float = 98.0) -> float:
    vals = []
    for a in arrs:
        aa = np.asarray(a, dtype=float)
        vals.append(np.abs(aa[np.isfinite(aa)]))
    if not vals:
        return 1.0
    x = np.concatenate(vals)
    if x.size == 0:
        return 1.0
    v = float(np.nanpercentile(x, q))
    return v if v > 0 else 1.0


def _draw_boxes(ax, settings: PVLatitudinalObjectChangeSettings, ccrs=None) -> None:
    boxes = ["north_northeast", "main_meiyu", "south_china_scs"]
    for name in boxes:
        if name not in settings.regions:
            continue
        b = settings.regions[name]
        xs = [b.lon_min, b.lon_max, b.lon_max, b.lon_min, b.lon_min]
        ys = [b.lat_min, b.lat_min, b.lat_max, b.lat_max, b.lat_min]
        kwargs = dict(linewidth=0.8, linestyle="--")
        if ccrs is not None:
            ax.plot(xs, ys, transform=ccrs.PlateCarree(), **kwargs)
        else:
            ax.plot(xs, ys, **kwargs)
        if ccrs is not None:
            ax.text(b.lon_min, b.lat_max, name, transform=ccrs.PlateCarree(), fontsize=6)
        else:
            ax.text(b.lon_min, b.lat_max, name, fontsize=6)


def plot_window_map_chain(
    maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
    out_path: Path,
    title: str,
    use_diverging: bool = False,
) -> None:
    plt, ccrs, cfeature = _import_plotting(settings.use_cartopy)
    arrs = [maps[w] for w in settings.window_order if w in maps]
    if use_diverging:
        v = _symmetric_vlim(arrs)
        vmin, vmax = -v, v
        cmap = "RdBu_r"
    else:
        finite = np.concatenate([a[np.isfinite(a)] for a in arrs if np.any(np.isfinite(a))])
        vmin = float(np.nanpercentile(finite, 2)) if finite.size else None
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else None
        cmap = "viridis"
    n = len(settings.window_order)
    proj = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(4.0 * n, 4.0))
    axes = []
    im = None
    for i, w in enumerate(settings.window_order, start=1):
        ax = fig.add_subplot(1, n, i, projection=proj) if proj is not None else fig.add_subplot(1, n, i)
        la, lo, dat = _sorted_lat_lon_data(lat, lon, maps[w])
        if proj is not None:
            im = ax.pcolormesh(lo, la, dat, transform=ccrs.PlateCarree(), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.coastlines(linewidth=0.5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            except Exception:
                pass
            ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
        else:
            im = ax.pcolormesh(lo, la, dat, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        _draw_boxes(ax, settings, ccrs)
        ax.set_title(w)
        axes.append(ax)
    fig.suptitle(title)
    if im is not None:
        fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_map(
    target_map: np.ndarray,
    ref_map: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
    out_path: Path,
    title: str,
) -> None:
    plot_window_map_chain({"delta": target_map - ref_map}, lat, lon, _single_window_settings(settings, "delta"), out_path, title, True)


def _single_window_settings(settings: PVLatitudinalObjectChangeSettings, window_name: str) -> PVLatitudinalObjectChangeSettings:
    # Shallow mutate a new settings object for simple plotting.
    import copy
    s = copy.copy(settings)
    s.window_order = [window_name]
    return s


def plot_profile_chain(
    profile_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt, _, _ = _import_plotting(False)
    sectors = list(settings.lon_sectors.keys())
    n = len(sectors)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, sector in zip(axes, sectors):
        for w in settings.window_order:
            g = profile_df[(profile_df["sector"] == sector) & (profile_df["window"] == w)].sort_values("lat")
            if not g.empty:
                ax.plot(g["lat"], g["value"], label=w)
        ax.set_title(sector)
        ax.set_ylabel(ylabel)
        ax.axhline(0, linewidth=0.5)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Latitude")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_profile_delta_chain(
    delta_df: pd.DataFrame,
    settings: PVLatitudinalObjectChangeSettings,
    value_delta_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt, _, _ = _import_plotting(False)
    sectors = list(settings.lon_sectors.keys())
    comps = list(settings.comparisons.keys())
    fig, axes = plt.subplots(len(sectors), 1, figsize=(8, 2.7 * len(sectors)), sharex=True)
    if len(sectors) == 1:
        axes = [axes]
    for ax, sector in zip(axes, sectors):
        for comp in comps:
            g = delta_df[(delta_df["sector"] == sector) & (delta_df["comparison"] == comp)].sort_values("lat")
            if not g.empty:
                ax.plot(g["lat"], g[value_delta_col], label=comp)
        ax.set_title(sector)
        ax.axhline(0, linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Latitude")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_p_v_object_panel(
    p_maps: Dict[str, np.ndarray],
    v_maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: PVLatitudinalObjectChangeSettings,
    comparison: str,
    out_path: Path,
) -> None:
    if comparison not in settings.comparisons:
        return
    target, ref = settings.comparisons[comparison]
    plt, ccrs, cfeature = _import_plotting(settings.use_cartopy)
    proj = ccrs.PlateCarree() if ccrs is not None else None
    p_delta = p_maps[target] - p_maps[ref]
    v_delta = v_maps[target] - v_maps[ref]
    p_vlim = _symmetric_vlim([p_delta])
    v_vlim = _symmetric_vlim([v_delta])
    fig = plt.figure(figsize=(10, 8))
    for i, (name, dat, vlim) in enumerate([("P change", p_delta, p_vlim), ("V850 change", v_delta, v_vlim)], start=1):
        ax = fig.add_subplot(2, 1, i, projection=proj) if proj is not None else fig.add_subplot(2, 1, i)
        la, lo, d = _sorted_lat_lon_data(lat, lon, dat)
        if proj is not None:
            im = ax.pcolormesh(lo, la, d, transform=ccrs.PlateCarree(), shading="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            ax.coastlines(linewidth=0.5)
            try:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            except Exception:
                pass
            ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
        else:
            im = ax.pcolormesh(lo, la, d, shading="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        _draw_boxes(ax, settings, ccrs)
        ax.set_title(f"{name}: {target} - {ref}")
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
    fig.suptitle(f"P/V850 object transition panel: {comparison}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
