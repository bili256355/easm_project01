# -*- coding: utf-8 -*-
"""Figure helpers for transition-chain report v1_b."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .t3_v_to_p_transition_chain_settings import RegionSpec, TransitionChainReportSettings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except Exception:  # pragma: no cover
    plt = None
    Rectangle = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None


def _extent(lon: np.ndarray, lat: np.ndarray) -> List[float]:
    return [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]


def _symmetric_limits(arrays: Iterable[np.ndarray], min_abs: float = 1.0e-12) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if np.isfinite(a).any():
            vals.append(float(np.nanpercentile(np.abs(a), 98)))
    vmax = max(vals) if vals else 1.0
    vmax = max(vmax, min_abs)
    return -vmax, vmax


def _sequential_limits(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        if np.isfinite(a).any():
            vals.append((float(np.nanpercentile(a, 2)), float(np.nanpercentile(a, 98))))
    if not vals:
        return 0.0, 1.0
    vmin = min(v[0] for v in vals)
    vmax = max(v[1] for v in vals)
    if abs(vmax - vmin) <= 1.0e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def _add_region_boxes(ax, regions: Dict[str, RegionSpec], use_cartopy: bool) -> None:
    if Rectangle is None:
        return
    for name in ["north_northeast", "main_meiyu", "south_china_scs"]:
        if name not in regions:
            continue
        spec = regions[name]
        kwargs = dict(fill=False, linewidth=0.9, linestyle="--", edgecolor="black")
        if use_cartopy and ccrs is not None:
            kwargs["transform"] = ccrs.PlateCarree()
        ax.add_patch(Rectangle((spec.lon_min, spec.lat_min), spec.lon_max - spec.lon_min, spec.lat_max - spec.lat_min, **kwargs))
        try:
            if use_cartopy and ccrs is not None:
                ax.text(spec.lon_min, spec.lat_max, name, fontsize=6, transform=ccrs.PlateCarree())
            else:
                ax.text(spec.lon_min, spec.lat_max, name, fontsize=6)
        except Exception:
            pass


def _make_axis(fig, nrows: int, ncols: int, i: int, settings: TransitionChainReportSettings):
    use_cartopy = settings.use_cartopy and ccrs is not None
    if use_cartopy:
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        if cfeature is not None:
            ax.add_feature(cfeature.BORDERS, linewidth=0.25)
    else:
        ax = fig.add_subplot(nrows, ncols, i + 1)
    return ax, use_cartopy


def _pcolormesh(ax, lon, lat, arr, vmin, vmax, cmap, use_cartopy):
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
    if use_cartopy and ccrs is not None:
        kwargs["transform"] = ccrs.PlateCarree()
    return ax.pcolormesh(lon, lat, arr, **kwargs)


def save_chain_maps(
    maps: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: TransitionChainReportSettings,
    out_path: Path,
    title_prefix: str,
    cmap: str = "viridis",
    diverging: bool = False,
) -> None:
    if plt is None or not settings.make_figures:
        return
    ordered = [maps[w] for w in settings.window_order if w in maps]
    vmin, vmax = _symmetric_limits(ordered) if diverging else _sequential_limits(ordered)
    fig = plt.figure(figsize=(3.2 * len(settings.window_order), 3.2))
    last = None
    for i, w in enumerate(settings.window_order):
        ax, use_cartopy = _make_axis(fig, 1, len(settings.window_order), i, settings)
        arr = maps[w]
        last = _pcolormesh(ax, lon, lat, arr, vmin, vmax, cmap, use_cartopy)
        ax.set_title(f"{title_prefix}\n{w}", fontsize=8)
        ax.set_extent(_extent(lon, lat)) if use_cartopy else None
        _add_region_boxes(ax, settings.regions, use_cartopy)
    if last is not None:
        fig.colorbar(last, ax=fig.axes, orientation="horizontal", fraction=0.045, pad=0.08)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_single_map(
    arr: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    settings: TransitionChainReportSettings,
    out_path: Path,
    title: str,
    cmap: str = "RdBu_r",
    diverging: bool = True,
) -> None:
    if plt is None or not settings.make_figures:
        return
    vmin, vmax = _symmetric_limits([arr]) if diverging else _sequential_limits([arr])
    fig = plt.figure(figsize=(5.0, 4.0))
    ax, use_cartopy = _make_axis(fig, 1, 1, 0, settings)
    im = _pcolormesh(ax, lon, lat, arr, vmin, vmax, cmap, use_cartopy)
    ax.set_title(title, fontsize=9)
    ax.set_extent(_extent(lon, lat)) if use_cartopy else None
    _add_region_boxes(ax, settings.regions, use_cartopy)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_transition_panel(
    p_delta: np.ndarray,
    v_delta: np.ndarray,
    support_delta_by_component: Dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    settings: TransitionChainReportSettings,
    out_path: Path,
    comparison: str,
) -> None:
    if plt is None or not settings.make_figures:
        return
    comps = list(settings.v_components)
    ncols = len(comps)
    nrows = 3
    p_lim = _symmetric_limits([p_delta])
    v_lim = _symmetric_limits([v_delta])
    s_lim = _symmetric_limits([support_delta_by_component[c] for c in comps])
    fig = plt.figure(figsize=(4.1 * ncols, 9.5))
    axes = []
    for j, comp in enumerate(comps):
        ax, use_cartopy = _make_axis(fig, nrows, ncols, j, settings)
        im = _pcolormesh(ax, lon, lat, p_delta, p_lim[0], p_lim[1], "RdBu_r", use_cartopy)
        ax.set_title(f"P change\n{comparison}\n{comp}", fontsize=8)
        ax.set_extent(_extent(lon, lat)) if use_cartopy else None
        _add_region_boxes(ax, settings.regions, use_cartopy)
        axes.append((ax, im))
    for j, comp in enumerate(comps):
        ax, use_cartopy = _make_axis(fig, nrows, ncols, ncols + j, settings)
        im = _pcolormesh(ax, lon, lat, v_delta, v_lim[0], v_lim[1], "RdBu_r", use_cartopy)
        ax.set_title(f"V850 change\n{comparison}\n{comp}", fontsize=8)
        ax.set_extent(_extent(lon, lat)) if use_cartopy else None
        _add_region_boxes(ax, settings.regions, use_cartopy)
        axes.append((ax, im))
    for j, comp in enumerate(comps):
        ax, use_cartopy = _make_axis(fig, nrows, ncols, 2 * ncols + j, settings)
        im = _pcolormesh(ax, lon, lat, support_delta_by_component[comp], s_lim[0], s_lim[1], "RdBu_r", use_cartopy)
        ax.set_title(f"Support R² change\n{comparison}\n{comp}", fontsize=8)
        ax.set_extent(_extent(lon, lat)) if use_cartopy else None
        _add_region_boxes(ax, settings.regions, use_cartopy)
        axes.append((ax, im))
    # One colorbar per row, attached to row axes.
    for row in range(nrows):
        row_axes = [axes[row * ncols + j][0] for j in range(ncols)]
        im = axes[row * ncols][1]
        fig.colorbar(im, ax=row_axes, orientation="horizontal", fraction=0.05, pad=0.08)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
