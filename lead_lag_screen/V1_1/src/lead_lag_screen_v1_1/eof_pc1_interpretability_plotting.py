from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .eof_pc1_interpretability_core import EOFResult, area_weighted_map_mean, feature_grid, region_mask
from .eof_pc1_interpretability_settings import EOFPC1InterpretabilitySettings


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _maybe_cartopy(use_cartopy: bool):
    if not use_cartopy:
        return None, None
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        return ccrs, cfeature
    except Exception:
        return None, None


def _plot_map(ax, lon: np.ndarray, lat: np.ndarray, data: np.ndarray, title: str, use_cartopy: bool, cmap: str = "RdBu_r"):
    ccrs, cfeature = _maybe_cartopy(use_cartopy)
    if ccrs is not None:
        ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        mesh = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), shading="auto", cmap=cmap)
    else:
        mesh = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    ax.set_title(title, fontsize=9)
    return mesh


def plot_loading_modes(result: EOFResult, settings: EOFPC1InterpretabilitySettings, use_cartopy: bool = True) -> Path:
    plt = _import_matplotlib()
    ccrs, _ = _maybe_cartopy(use_cartopy)
    n = min(3, result.eofs_unweighted.shape[0])
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), subplot_kw=subplot_kw, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes, start=1):
        grid = feature_grid(result, i)
        mesh = _plot_map(ax, result.lon, result.lat, grid, f"{result.field_name} EOF{i} loading", use_cartopy)
        fig.colorbar(mesh, ax=ax, shrink=0.72)
    out = settings.figure_dir / f"{result.field_name}_EOF_loading_modes_1_3.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _window_mean_map(result: EOFResult, window: Tuple[int, int]) -> np.ndarray:
    mask = (result.days >= window[0]) & (result.days <= window[1])
    return np.nanmean(result.anomaly_field[:, mask, :, :], axis=(0, 1))


def _recon_delta_map(result: EOFResult, target_win: Tuple[int, int], ref_win: Tuple[int, int], modes: list[int]) -> np.ndarray:
    n_lat, n_lon = len(result.lat), len(result.lon)
    delta = np.zeros(n_lat * n_lon, dtype=float)
    for mode in modes:
        pc_col = f"{result.field_name}_PC{mode}"
        mt = result.pcs[(result.pcs["day"] >= target_win[0]) & (result.pcs["day"] <= target_win[1])][pc_col].mean()
        mr = result.pcs[(result.pcs["day"] >= ref_win[0]) & (result.pcs["day"] <= ref_win[1])][pc_col].mean()
        feat = np.full(n_lat * n_lon, np.nan, dtype=float)
        feat[result.valid_feature_mask] = result.eofs_unweighted[mode - 1, :]
        delta += (float(mt) - float(mr)) * np.nan_to_num(feat, nan=0.0)
    return delta.reshape(n_lat, n_lon)


def plot_observed_vs_reconstruction(result: EOFResult, settings: EOFPC1InterpretabilitySettings, comp_name: str, use_cartopy: bool = True) -> Path:
    plt = _import_matplotlib()
    ccrs, _ = _maybe_cartopy(use_cartopy)
    target_name, ref_name = settings.reconstruction_comparisons[comp_name]
    target_win = settings.windows[target_name]
    ref_win = settings.windows[ref_name]
    obs = _window_mean_map(result, target_win) - _window_mean_map(result, ref_win)
    pc1 = _recon_delta_map(result, target_win, ref_win, [1])
    pc123 = _recon_delta_map(result, target_win, ref_win, list(range(1, min(3, result.eofs_unweighted.shape[0]) + 1)))
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), subplot_kw=subplot_kw, constrained_layout=True)
    vmax = np.nanpercentile(np.abs(obs), 98)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else None
    for ax, data, title in zip(axes, [obs, pc1, pc123], [f"Observed {comp_name}", "PC1 recon", "PC1-3 recon"]):
        mesh = _plot_map(ax, result.lon, result.lat, data, title, use_cartopy)
        if vmax is not None:
            mesh.set_clim(-vmax, vmax)
        fig.colorbar(mesh, ax=ax, shrink=0.72)
    out = settings.figure_dir / f"{result.field_name}_observed_vs_PC_reconstruction_{comp_name}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_figures(p_eof: EOFResult, v_eof: EOFResult, settings: EOFPC1InterpretabilitySettings, use_cartopy: bool = True) -> list[Path]:
    settings.figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(plot_loading_modes(p_eof, settings, use_cartopy=use_cartopy))
    paths.append(plot_loading_modes(v_eof, settings, use_cartopy=use_cartopy))
    for comp in settings.reconstruction_comparisons:
        paths.append(plot_observed_vs_reconstruction(p_eof, settings, comp, use_cartopy=use_cartopy))
        paths.append(plot_observed_vs_reconstruction(v_eof, settings, comp, use_cartopy=use_cartopy))
    return paths
