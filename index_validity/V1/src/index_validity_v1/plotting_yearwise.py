\
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .index_metadata import VARIABLE_ORDER


def plot_yearwise_figures(index_values: pd.DataFrame, yearwise_audit: pd.DataFrame, output_dir: Path, dpi: int = 180) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in VARIABLE_ORDER:
        pivot = index_values.pivot(index="year", columns="day", values=name).sort_index(axis=0).sort_index(axis=1)
        years = pivot.index.to_numpy()
        days = pivot.columns.to_numpy()
        arr = pivot.to_numpy(dtype=float)
        mu = np.nanmean(arr, axis=0)
        p10 = np.nanpercentile(arr, 10, axis=0)
        p90 = np.nanpercentile(arr, 90, axis=0)

        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for i in range(arr.shape[0]):
            ax.plot(days, arr[i, :], linewidth=0.5, alpha=0.35)
        ax.fill_between(days, p10, p90, alpha=0.20, label="10-90% range")
        ax.plot(days, mu, linewidth=2.0, label="multi-year mean")
        ax.set_title(f"Spaghetti: {name}")
        ax.set_xlabel("Day of Apr-Sep season")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(output_dir / f"spaghetti_{name}.png", dpi=dpi)
        plt.close(fig)
        count += 1

        dev = arr - mu[None, :]
        vmax = np.nanpercentile(np.abs(dev), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        im = ax.imshow(
            dev,
            aspect="auto",
            origin="lower",
            extent=[days.min(), days.max(), years.min(), years.max()],
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu_r",
        )
        ax.set_title(f"Year-day deviation heatmap: {name}")
        ax.set_xlabel("Day of Apr-Sep season")
        ax.set_ylabel("Year")
        fig.colorbar(im, ax=ax, label=f"{name} - daily multi-year mean")
        fig.savefig(output_dir / f"year_day_heatmap_{name}.png", dpi=dpi)
        plt.close(fig)
        count += 1

        sub = yearwise_audit[yearwise_audit["index_name"] == name].sort_values("year")
        fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), constrained_layout=True, sharex=True)
        for ax, col, title in zip(
            axes,
            ["offset_abs_mean", "roughness_mean_abs", "curvature_mean_abs"],
            ["Mean absolute seasonal offset", "Mean absolute day-to-day jump", "Mean absolute curvature"],
        ):
            ax.plot(sub["year"], sub[col], marker="o", linewidth=1.0)
            flagged_col = {
                "offset_abs_mean": "flag_large_offset",
                "roughness_mean_abs": "flag_large_roughness",
                "curvature_mean_abs": "flag_large_curvature",
            }[col]
            flagged = sub[sub[flagged_col]]
            if not flagged.empty:
                ax.scatter(flagged["year"], flagged[col], s=35, marker="x", label="flagged")
                ax.legend(fontsize=8)
            ax.set_ylabel(col)
            ax.set_title(title)
        axes[-1].set_xlabel("Year")
        fig.suptitle(f"Yearwise shape metrics: {name}")
        fig.savefig(output_dir / f"year_metrics_{name}.png", dpi=dpi)
        plt.close(fig)
        count += 1

    return count
