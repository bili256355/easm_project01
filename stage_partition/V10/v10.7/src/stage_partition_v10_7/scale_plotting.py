from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt


def _pivot_energy(energy_map: pd.DataFrame) -> tuple[np.ndarray, list[float], list[int]]:
    pivot = energy_map.pivot_table(index="sigma", columns="day", values="energy_norm_within_sigma", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    arr = pivot.to_numpy(dtype=float)
    sigmas = [float(x) for x in pivot.index.tolist()]
    days = [int(x) for x in pivot.columns.tolist()]
    return arr, sigmas, days


def plot_scale_energy_map(energy_map: pd.DataFrame, target_days: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    arr, sigmas, days = _pivot_energy(energy_map)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(arr, aspect="auto", origin="lower", extent=[min(days), max(days), min(sigmas), max(sigmas)])
    ax.set_title("H W045 Gaussian derivative scale-energy map (V10.7_b)")
    ax.set_xlabel("day index (day 0 = Apr 1)")
    ax.set_ylabel("Gaussian sigma (days)")
    fig.colorbar(im, ax=ax, label="within-sigma normalized energy")
    for label, day in target_days.items():
        ax.axvline(int(day), linestyle="--", linewidth=1)
        ax.text(int(day), max(sigmas), label, rotation=90, va="top", ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scale_ridge_overlay(energy_map: pd.DataFrame, ridges: pd.DataFrame, ridge_summary: pd.DataFrame, target_days: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    arr, sigmas, days = _pivot_energy(energy_map)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(arr, aspect="auto", origin="lower", extent=[min(days), max(days), min(sigmas), max(sigmas)])
    fig.colorbar(im, ax=ax, label="within-sigma normalized energy")
    if ridges is not None and not ridges.empty:
        for ridge_id, sub in ridges.groupby("ridge_id", sort=True):
            sub = sub.sort_values("sigma")
            ax.plot(sub["day"].to_numpy(dtype=float), sub["sigma"].to_numpy(dtype=float), marker="o", linewidth=1, markersize=3, label=str(ridge_id))
    for label, day in target_days.items():
        ax.axvline(int(day), linestyle="--", linewidth=1)
        ax.text(int(day), max(sigmas), label, rotation=90, va="top", ha="right")
    ax.set_title("H W045 scale-energy map with linked ridge families (V10.7_b)")
    ax.set_xlabel("day index")
    ax.set_ylabel("Gaussian sigma (days)")
    if ridges is not None and not ridges.empty and ridges["ridge_id"].nunique() <= 12:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_day_scale_response(target_response: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, sub in target_response.groupby("target_label", sort=False):
        sub = sub.sort_values("sigma")
        ax.plot(sub["sigma"].to_numpy(dtype=float), sub["energy_norm"].to_numpy(dtype=float), marker="o", label=str(label))
    ax.set_title("Target-day local scale response around H19/H35/H45/H57 (V10.7_b)")
    ax.set_xlabel("Gaussian sigma (days)")
    ax.set_ylabel("local max normalized energy within target radius")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_derivative_panel(energy_map: pd.DataFrame, selected_sigmas: tuple[float, ...], target_days: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 5))
    available = sorted(float(x) for x in energy_map["sigma"].unique())
    for sigma in selected_sigmas:
        closest = min(available, key=lambda x: abs(x - float(sigma)))
        sub = energy_map[energy_map["sigma"].astype(float) == float(closest)].sort_values("day")
        ax.plot(sub["day"].to_numpy(dtype=int), sub["energy_norm_within_sigma"].to_numpy(dtype=float), label=f"sigma={closest:g}")
    for label, day in target_days.items():
        ax.axvline(int(day), linestyle="--", linewidth=1)
        ax.text(int(day), 1.0, label, rotation=90, va="top", ha="right")
    ax.set_title("H derivative energy curves at selected scales (V10.7_b)")
    ax.set_xlabel("day index")
    ax.set_ylabel("within-sigma normalized derivative energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
