from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _maybe_cartopy(use_cartopy: bool):
    if not use_cartopy:
        return False, None, None
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        return True, ccrs, cfeature
    except Exception:
        return False, None, None


def _sort_lat_lon(lat: np.ndarray, lon: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.asarray(arr, dtype=float)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        out = out[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        out = out[:, ::-1]
    return lat, lon, out


def _plot_map(ax, lon, lat, arr, title, cartopy_available, ccrs, cfeature, extent):
    lat, lon, arr = _sort_lat_lon(lat, lon, arr)
    vmax = np.nanpercentile(np.abs(arr), 98) if np.any(np.isfinite(arr)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if cartopy_available:
        ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, arr, transform=ccrs.PlateCarree(), shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35)
        ax.add_feature(cfeature.LAND, alpha=0.10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35)
        gl.top_labels = False
        gl.right_labels = False
    else:
        im = ax.pcolormesh(lon, lat, arr, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.set_title(title, fontsize=10)
    return im


def select_figure_targets(index_metrics: pd.DataFrame, family_guardrail: pd.DataFrame, max_figures: int) -> pd.DataFrame:
    targets = []
    # Always include all T3 indices.
    for row in index_metrics[index_metrics["window"].eq("T3")].itertuples(index=False):
        targets.append((row.window, row.index_name, "T3_all_indices", 0))
    # Include every window-family best index.
    for row in family_guardrail.itertuples(index=False):
        targets.append((row.window, row.best_index, "window_family_best_index", 1))
    # Include all high-risk/not-supported indices.
    bad = index_metrics[index_metrics["representativeness_tier"].isin(["high_risk", "not_supported"])]
    for row in bad.itertuples(index=False):
        targets.append((row.window, row.index_name, f"{row.representativeness_tier}_index", 2))
    df = pd.DataFrame(targets, columns=["window", "index_name", "reason", "priority"])
    if df.empty:
        return df
    df = df.drop_duplicates(["window", "index_name"]).sort_values(["priority", "window", "index_name"])
    return df.head(max_figures).reset_index(drop=True)


def plot_selected_composites(
    figure_payloads: Dict[Tuple[str, str], Dict[str, object]],
    targets: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    use_cartopy_if_available: bool,
    display_extent: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cartopy_available, ccrs, cfeature = _maybe_cartopy(use_cartopy_if_available)
    rows = []
    for row in targets.itertuples(index=False):
        key = (row.window, row.index_name)
        if key not in figure_payloads:
            continue
        obj = figure_payloads[key]
        lat = np.asarray(obj["lat"], dtype=float)
        lon = np.asarray(obj["lon"], dtype=float)
        high = np.asarray(obj["high"], dtype=float)
        low = np.asarray(obj["low"], dtype=float)
        diff = np.asarray(obj["diff"], dtype=float)
        band_lines = obj.get("band_lines", []) or []
        proj_kw = {"projection": ccrs.PlateCarree()} if cartopy_available else {}
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), subplot_kw=proj_kw, constrained_layout=True)
        for ax, arr, title in zip(axes, [high, low, diff], ["High composite", "Low composite", "High - Low"]):
            im = _plot_map(ax, lon, lat, arr, title, cartopy_available, ccrs, cfeature, display_extent)
            fig.colorbar(im, ax=ax, shrink=0.72)
            if cartopy_available:
                for b in band_lines:
                    ax.plot([display_extent[0], display_extent[1]], [float(b), float(b)], transform=ccrs.PlateCarree(), linewidth=0.45, alpha=0.45)
            else:
                for b in band_lines:
                    ax.axhline(float(b), linewidth=0.45, alpha=0.45)
        fname = f"{row.window}__{row.index_name}__composite_map.png".replace("/", "_")
        fig.suptitle(
            f"{row.window} | {row.index_name} | {obj.get('tier')} | score={float(obj.get('overall_score', np.nan)):.3f}\n{obj.get('expected_meaning','')}",
            fontsize=11,
        )
        fig.savefig(output_dir / fname, dpi=dpi)
        plt.close(fig)
        rows.append({
            "window": row.window,
            "index_name": row.index_name,
            "reason": row.reason,
            "figure_path": str(output_dir / fname),
            "cartopy_used": bool(cartopy_available),
        })
    status = "cartopy_used" if cartopy_available else "cartopy_unavailable_used_plain_matplotlib"
    return pd.DataFrame(rows), status
