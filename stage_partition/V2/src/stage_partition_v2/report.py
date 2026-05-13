from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _plot_series(series: pd.Series, path: Path, title: str) -> None:
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index.to_numpy(), series.to_numpy(), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('day')
    ax.set_ylabel('score')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_backend_scores(mw_scores: pd.Series, rw_profile: pd.Series, output_root: Path) -> None:
    if not rw_profile.empty:
        _plot_series(rw_profile, output_root / 'ruptures_profile.png', 'ruptures.Window profile')


def plot_window_catalog(catalog: pd.DataFrame, output_root: Path) -> None:
    if catalog.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for _, row in catalog.iterrows():
        color = 'tab:blue' if row['window_type'] == 'primary' else 'tab:orange'
        ax.hlines(y=1, xmin=row['start_day'], xmax=row['end_day'], colors=color, linewidth=4)
        ax.scatter([row['center_day']], [1], color='black', s=12)
    ax.set_yticks([1])
    ax.set_yticklabels(['ruptures'])
    ax.set_xlabel('day')
    ax.set_title('Window catalog overview')
    fig.tight_layout()
    fig.savefig(output_root / 'window_catalog_overview.png', dpi=150)
    plt.close(fig)
