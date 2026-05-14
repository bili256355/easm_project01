from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None


def plot_profile_diff_panel(profile_diff: pd.DataFrame, out_path: Path) -> None:
    if profile_diff.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ev, g in profile_diff.groupby("event_id"):
        gg = g.sort_values("feature_index")
        xcol = "feature_coord_if_available" if gg["feature_coord_if_available"].notna().any() else "feature_index"
        ax.plot(gg[xcol].astype(float), gg["diff"].astype(float), marker="o", label=str(ev))
    ax.axhline(0, linewidth=0.8)
    ax.set_title("H W045 event profile differences (post - pre)")
    ax.set_xlabel("H profile latitude feature" if profile_diff["feature_coord_if_available"].notna().any() else "feature index")
    ax.set_ylabel("profile diff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_feature_contribution_top(feature_contrib: pd.DataFrame, out_path: Path) -> None:
    if feature_contrib.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Average contribution across sigma per event and show top features.
    g = feature_contrib.groupby(["event_id", "feature_index"], as_index=False)["energy_contribution_fraction"].mean()
    events = list(g["event_id"].drop_duplicates())
    fig, axes = plt.subplots(len(events), 1, figsize=(10, max(3, 2.4 * len(events))), sharex=True)
    if len(events) == 1:
        axes = [axes]
    for ax, ev in zip(axes, events):
        gg = g.loc[g["event_id"] == ev].sort_values("feature_index")
        ax.bar(gg["feature_index"].astype(int), gg["energy_contribution_fraction"].astype(float))
        ax.set_ylabel(str(ev))
    axes[-1].set_xlabel("feature index")
    fig.suptitle("H event feature contribution (mean across selected sigma)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_spatial_panel(diff_maps: dict[str, np.ndarray], lat: list[float] | np.ndarray | None, lon: list[float] | np.ndarray | None, out_path: Path) -> str:
    if not diff_maps or lat is None or lon is None:
        return "skipped_missing_spatial_maps"
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    events = [e for e in ("H18", "H35", "H45", "H57") if e in diff_maps]
    vals = np.concatenate([np.ravel(diff_maps[e]) for e in events]) if events else np.array([])
    finite = vals[np.isfinite(vals)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if ccrs is not None:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 3.2 * len(events)))
        for i, ev in enumerate(events, start=1):
            ax = fig.add_subplot(len(events), 1, i, projection=proj)
            m = ax.pcolormesh(lon_arr, lat_arr, diff_maps[ev], transform=proj, shading="auto", vmin=-vmax, vmax=vmax)
            ax.coastlines(linewidth=0.6)
            if cfeature is not None:
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([float(np.nanmin(lon_arr)), float(np.nanmax(lon_arr)), float(np.nanmin(lat_arr)), float(np.nanmax(lat_arr))], crs=proj)
            ax.set_title(f"{ev} H-field composite diff (post - pre)")
            fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return "cartopy"
    # Fallback plot; status recorded by caller.
    fig, axes = plt.subplots(len(events), 1, figsize=(10, 3.0 * len(events)))
    if len(events) == 1:
        axes = [axes]
    for ax, ev in zip(axes, events):
        im = ax.imshow(diff_maps[ev], origin="lower", aspect="auto", vmin=-vmax, vmax=vmax,
                       extent=[float(np.nanmin(lon_arr)), float(np.nanmax(lon_arr)), float(np.nanmin(lat_arr)), float(np.nanmax(lat_arr))])
        ax.set_title(f"{ev} H-field composite diff (fallback non-cartopy)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return "matplotlib_fallback_no_cartopy"


def plot_similarity_heatmap(similarity: pd.DataFrame, out_path: Path) -> None:
    if similarity.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(similarity["comparison"].astype(str))
    vals = similarity[["profile_pearson_correlation", "spatial_pattern_correlation"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(labels))))
    im = ax.imshow(vals, aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["profile corr", "spatial corr"])
    fig.colorbar(im, ax=ax)
    ax.set_title("H18/H35/H45/H57 content similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
