# -*- coding: utf-8 -*-
"""Figures for P/V850 offset-correspondence audit v1_b."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .t3_p_v_offset_correspondence_settings import PVOffsetCorrespondenceSettings


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _sector_order(settings: PVOffsetCorrespondenceSettings) -> List[str]:
    return list(settings.lon_sectors.keys())


def plot_p_clim_bands_vs_v_clim_structure(
    p_profile: pd.DataFrame,
    v_profile: pd.DataFrame,
    p_clim_df: pd.DataFrame,
    v_clim_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
    out_path: Path,
) -> None:
    plt = _mpl()
    sectors = _sector_order(settings)
    windows = settings.window_order
    fig, axes = plt.subplots(len(sectors), 1, figsize=(12, 2.8 * len(sectors)), sharex=True)
    if len(sectors) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(windows)))
    for ax, sector in zip(axes, sectors):
        for color, window in zip(colors, windows):
            pg = p_profile[(p_profile["window"] == window) & (p_profile["sector"] == sector)]
            vg = v_profile[(v_profile["window"] == window) & (v_profile["sector"] == sector)]
            if pg.empty or vg.empty:
                continue
            ax.plot(pg["lat"], pg["value"], color=color, lw=1.6, label=f"P {window}")
            # Scale positive V to the P-axis range for location comparison only.
            vpos = np.maximum(vg["value"].to_numpy(dtype=float), 0.0)
            if np.nanmax(vpos) > 0 and np.nanmax(pg["value"].to_numpy(dtype=float)) > 0:
                v_scaled = vpos / np.nanmax(vpos) * np.nanmax(pg["value"].to_numpy(dtype=float))
                ax.plot(vg["lat"], v_scaled, color=color, lw=1.0, ls="--", alpha=0.65)
            b = p_clim_df[(p_clim_df["window"] == window) & (p_clim_df["sector"] == sector) & p_clim_df["p_clim_peak_lat"].notna()]
            ax.scatter(b["p_clim_peak_lat"], b["p_clim_peak_value"], color=color, s=18, marker="o")
            vc = v_clim_df[(v_clim_df["window"] == window) & (v_clim_df["sector"] == sector)]
            if not vc.empty:
                vc0 = vc.iloc[0]
                for x, ls in [(vc0.get("v_positive_peak_lat"), ":"), (vc0.get("v_positive_centroid_lat"), "-."), (vc0.get("v_positive_north_edge"), "--")]:
                    if np.isfinite(x):
                        ax.axvline(float(x), color=color, lw=0.7, ls=ls, alpha=0.35)
        ax.set_title(f"{sector}: P climatological bands vs scaled V850 positive structure")
        ax.set_ylabel("P mean (V850+ scaled)")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Latitude")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_p_change_vs_v_change(
    p_delta: pd.DataFrame,
    v_delta: pd.DataFrame,
    p_change_df: pd.DataFrame,
    v_change_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
    out_dir: Path,
) -> List[Path]:
    plt = _mpl()
    out_paths: List[Path] = []
    sectors = _sector_order(settings)
    for comp in settings.comparisons.keys():
        fig, axes = plt.subplots(len(sectors), 3, figsize=(15, 2.7 * len(sectors)), sharex=False)
        if len(sectors) == 1:
            axes = np.array([axes])
        for i, sector in enumerate(sectors):
            pg = p_delta[(p_delta["comparison"] == comp) & (p_delta["sector"] == sector)]
            vg = v_delta[(v_delta["comparison"] == comp) & (v_delta["sector"] == sector)]
            if pg.empty or vg.empty:
                continue
            pcol = "P_delta" if "P_delta" in pg.columns else [c for c in pg.columns if c.endswith("_delta")][0]
            vcol = "V850_delta" if "V850_delta" in vg.columns else [c for c in vg.columns if c.endswith("_delta")][0]
            lat_v = vg["lat"].to_numpy(float)
            vvals = vg[vcol].to_numpy(float)
            order = np.argsort(lat_v)
            lat_v = lat_v[order]
            vvals = vvals[order]
            grad = np.gradient(vvals, lat_v)

            ax = axes[i, 0]
            ax.axhline(0, color="k", lw=0.7)
            ax.plot(pg["lat"], pg[pcol], color="tab:blue", lw=1.4)
            b = p_change_df[(p_change_df["comparison"] == comp) & (p_change_df["sector"] == sector) & p_change_df["p_change_peak_lat"].notna()]
            for _, r in b.iterrows():
                marker = "^" if r["p_change_type"] == "positive" else "v"
                ax.scatter(r["p_change_peak_lat"], r["p_change_peak_value"], marker=marker, s=25, color="tab:blue")
            ax.set_title(f"{sector}: P delta")
            ax.grid(True, alpha=0.2)

            ax = axes[i, 1]
            ax.axhline(0, color="k", lw=0.7)
            ax.plot(vg["lat"], vg[vcol], color="tab:orange", lw=1.4)
            vc = v_change_df[(v_change_df["comparison"] == comp) & (v_change_df["sector"] == sector)]
            if not vc.empty:
                vc0 = vc.iloc[0]
                for x, color in [(vc0.get("v_change_peak_lat"), "red"), (vc0.get("v_change_trough_lat"), "purple")]:
                    if np.isfinite(x):
                        ax.axvline(float(x), color=color, lw=0.9, ls="--", alpha=0.75)
            ax.set_title("V850 delta")
            ax.grid(True, alpha=0.2)

            ax = axes[i, 2]
            ax.axhline(0, color="k", lw=0.7)
            ax.plot(lat_v, grad, color="tab:green", lw=1.4)
            if not vc.empty and np.isfinite(vc.iloc[0].get("v_gradient_change_peak_lat")):
                ax.axvline(float(vc.iloc[0].get("v_gradient_change_peak_lat")), color="tab:green", ls="--", lw=0.9)
            ax.set_title("d(V850 delta)/dlat")
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"P change peaks vs V850 change structures: {comp}")
        for ax in axes[-1, :]:
            ax.set_xlabel("Latitude")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"P_change_peaks_vs_V_change_structure_{comp}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_p_highlat_v_north_edge_chain(
    p_highlat_corr: pd.DataFrame,
    v_clim_df: pd.DataFrame,
    p_profile_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
    out_path: Path,
) -> None:
    plt = _mpl()
    sector = "full_easm_lon"
    fig, ax1 = plt.subplots(figsize=(11, 5))
    windows = settings.window_order
    # Plot several high-latitude profile values averaged by latband.
    bands = [(35, 40), (40, 45), (45, 50), (50, 55), (55, 60)]
    for lo, hi in bands:
        vals = []
        for w in windows:
            g = p_profile_df[(p_profile_df["window"] == w) & (p_profile_df["sector"] == sector) & (p_profile_df["lat"] >= lo) & (p_profile_df["lat"] < hi)]
            vals.append(float(g["value"].mean()) if not g.empty else np.nan)
        ax1.plot(windows, vals, marker="o", label=f"P {lo}-{hi}N")
    ax1.set_ylabel("P mean by high-lat band")
    ax1.grid(True, alpha=0.2)
    ax2 = ax1.twinx()
    vg = v_clim_df[v_clim_df["sector"] == sector]
    edge = [float(vg[vg["window"] == w]["v_positive_north_edge"].iloc[0]) if not vg[vg["window"] == w].empty else np.nan for w in windows]
    cent = [float(vg[vg["window"] == w]["v_positive_centroid_lat"].iloc[0]) if not vg[vg["window"] == w].empty else np.nan for w in windows]
    ax2.plot(windows, edge, color="k", marker="s", lw=2, label="V+ north edge")
    ax2.plot(windows, cent, color="gray", marker="d", lw=1.5, label="V+ centroid")
    ax2.set_ylabel("V850 positive structure latitude")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_p_south_retention_vs_v_retreat(
    v_clim_df: pd.DataFrame,
    p_profile_df: pd.DataFrame,
    settings: PVOffsetCorrespondenceSettings,
    out_path: Path,
) -> None:
    plt = _mpl()
    sector = "full_easm_lon"
    windows = settings.window_order
    fig, ax1 = plt.subplots(figsize=(11, 5))
    bands = [(10, 20), (20, 25), (35, 60)]
    for lo, hi in bands:
        vals = []
        for w in windows:
            g = p_profile_df[(p_profile_df["window"] == w) & (p_profile_df["sector"] == sector) & (p_profile_df["lat"] >= lo) & (p_profile_df["lat"] < hi)]
            vals.append(float(g["value"].mean()) if not g.empty else np.nan)
        ax1.plot(windows, vals, marker="o", label=f"P {lo}-{hi}N")
    ax1.set_ylabel("P mean by lat band")
    ax1.grid(True, alpha=0.2)
    ax2 = ax1.twinx()
    vg = v_clim_df[v_clim_df["sector"] == sector]
    edge = [float(vg[vg["window"] == w]["v_positive_north_edge"].iloc[0]) if not vg[vg["window"] == w].empty else np.nan for w in windows]
    ax2.plot(windows, edge, color="k", marker="s", lw=2, label="V+ north edge")
    ax2.set_ylabel("V850 positive north edge")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
