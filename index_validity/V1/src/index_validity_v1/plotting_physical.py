\
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _try_cartopy():
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
        return ccrs, cfeature
    except Exception:
        return None, None


def _map_panel(ax, lon, lat, data, title, use_cartopy: bool, ccrs=None, cfeature=None):
    Lon, Lat = np.meshgrid(lon, lat)
    if use_cartopy and ccrs is not None:
        im = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(), shading="auto")
        ax.coastlines(linewidth=0.7)
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        except Exception:
            pass
        ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
        try:
            gl.top_labels = False
            gl.right_labels = False
        except Exception:
            pass
    else:
        im = ax.pcolormesh(Lon, Lat, data, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title)
    return im


def plot_physical_figures(composites: Dict[str, Dict[str, object]], output_dir: Path, dpi: int = 180, use_cartopy_if_available: bool = True) -> tuple[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ccrs, cfeature = _try_cartopy() if use_cartopy_if_available else (None, None)
    cartopy_status = "available" if ccrs is not None else "not_available_plain_matplotlib_maps_used"
    count = 0

    for name, obj in composites.items():
        lat = np.asarray(obj["lat"], dtype=float)
        lon = np.asarray(obj["lon"], dtype=float)
        high = np.asarray(obj["high"], dtype=float)
        low = np.asarray(obj["low"], dtype=float)
        diff = np.asarray(obj["diff"], dtype=float)
        meta = obj["meta"]
        check_type = str(meta["physical_check_type"])
        family = str(obj["family"])

        if "map" in check_type:
            use_cart = ccrs is not None
            if use_cart:
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)
            else:
                fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
            for ax, data, title in zip(axes, [high, low, diff], ["High composite", "Low composite", "High - Low"]):
                im = _map_panel(ax, lon, lat, data, f"{name}: {title}", use_cart, ccrs, cfeature)
                fig.colorbar(im, ax=ax, shrink=0.78)
            fig.suptitle(f"Physical map check: {name} ({family})")
            fig.savefig(output_dir / f"map_{name}.png", dpi=dpi)
            plt.close(fig)
            count += 1

        if "profile" in check_type:
            high_profile = np.asarray(obj["high_profile"], dtype=float)
            low_profile = np.asarray(obj["low_profile"], dtype=float)
            diff_profile = np.asarray(obj["diff_profile"], dtype=float)
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
            axes[0].plot(high_profile, lat, label="High composite")
            axes[0].plot(low_profile, lat, label="Low composite")
            axes[0].set_xlabel("Lon-mean field value")
            axes[0].set_ylabel("Latitude")
            axes[0].set_title(f"{name}: high vs low")
            axes[0].legend(fontsize=8)
            axes[1].plot(diff_profile, lat, label="High - Low")
            axes[1].axvline(0, linewidth=0.8)
            axes[1].set_xlabel("Difference")
            axes[1].set_title(f"{name}: difference")
            for ax in axes:
                for line in meta.get("band_lines", []):
                    ax.axhline(float(line), linewidth=0.6, alpha=0.5)
                ax.grid(True, linewidth=0.25, alpha=0.4)
            fig.suptitle(f"Physical lat-profile check: {name} ({family})")
            fig.savefig(output_dir / f"profile_{name}.png", dpi=dpi)
            plt.close(fig)
            count += 1

    return count, cartopy_status
