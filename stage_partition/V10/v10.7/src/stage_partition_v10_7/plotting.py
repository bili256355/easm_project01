from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .config import Settings


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required for V10.7_a figures") from e


def plot_h_fullseason_by_width(cfg: Settings, curve: pd.DataFrame, candidates: pd.DataFrame, out_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 6))
    for width in cfg.detector.widths:
        sub = curve[curve["detector_width"].astype(int) == int(width)].sort_values("day")
        if not sub.empty:
            ax.plot(sub["day"], sub["score"], linewidth=1.4, label=f"width={width}")
        cand = candidates[candidates["detector_width"].astype(int) == int(width)]
        if not cand.empty:
            y = np.interp(cand["candidate_day"].astype(float), sub["day"].astype(float), sub["score"].astype(float)) if not sub.empty else cand["candidate_score"]
            ax.scatter(cand["candidate_day"], y, s=16)
    for win_id, center, start, end in cfg.windows.strong_windows:
        ax.axvspan(start, end, alpha=0.08)
        ax.axvline(center, linewidth=0.8, linestyle="--")
        ax.text(center, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, win_id, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("V10.7_a H detector score curves by detector_width")
    ax.set_xlabel("day index (Apr 1 = 0)")
    ax.set_ylabel("ruptures.Window detector score")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_h_candidate_raster(cfg: Settings, candidates: pd.DataFrame, out_path: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if not candidates.empty:
        for width in cfg.detector.widths:
            sub = candidates[candidates["detector_width"].astype(int) == int(width)]
            if not sub.empty:
                sizes = 20 + 25 * (sub["candidate_rank"].max() + 1 - sub["candidate_rank"]) / max(1, sub["candidate_rank"].max())
                ax.scatter(sub["candidate_day"], [width] * len(sub), s=sizes)
    for win_id, center, start, end in cfg.windows.strong_windows:
        ax.axvspan(start, end, alpha=0.08)
        ax.axvline(center, linewidth=0.8, linestyle="--")
        ax.text(center, max(cfg.detector.widths) + 1, win_id, rotation=90, va="bottom", ha="center", fontsize=8)
    ax.set_yticks(list(cfg.detector.widths))
    ax.set_xlabel("candidate day")
    ax.set_ylabel("detector_width")
    ax.set_title("V10.7_a H candidate days by detector_width")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_h_window_panels(cfg: Settings, curve: pd.DataFrame, candidates: pd.DataFrame, out_path: Path) -> None:
    plt = _import_matplotlib()
    fig, axes = plt.subplots(len(cfg.windows.strong_windows), 1, figsize=(13, 12), sharex=False)
    if len(cfg.windows.strong_windows) == 1:
        axes = [axes]
    for ax, (win_id, center, start, end) in zip(axes, cfg.windows.strong_windows):
        lo = max(0, start - 25)
        hi = min(182, end + 25)
        for width in cfg.detector.widths:
            sub = curve[(curve["detector_width"].astype(int) == int(width)) & (curve["day"].astype(int) >= lo) & (curve["day"].astype(int) <= hi)].sort_values("day")
            if not sub.empty:
                ax.plot(sub["day"], sub["score"], linewidth=1.2, label=f"w={width}")
            cand = candidates[(candidates["detector_width"].astype(int) == int(width)) & (candidates["candidate_day"].astype(int) >= lo) & (candidates["candidate_day"].astype(int) <= hi)]
            if not cand.empty and not sub.empty:
                y = np.interp(cand["candidate_day"].astype(float), sub["day"].astype(float), sub["score"].astype(float))
                ax.scatter(cand["candidate_day"], y, s=14)
        ax.axvspan(start, end, alpha=0.10)
        ax.axvline(center, linestyle="--", linewidth=0.8)
        ax.set_title(f"{win_id}: H score curves and candidates")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("score")
    axes[-1].set_xlabel("day index")
    axes[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
