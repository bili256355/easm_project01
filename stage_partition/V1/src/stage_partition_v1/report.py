
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import save_json


def plot_general_score_curve(score_df: pd.DataFrame, candidate_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(score_df['day_index'], score_df['score_raw'], label='score_raw')
    ax.plot(score_df['day_index'], score_df['score_smooth'], label='score_smooth')
    for _, row in candidate_df.iterrows():
        ax.axvspan(row['start_day'], row['end_day'], alpha=0.15)
        ax.axvline(row['peak_day'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('day_index')
    ax.set_ylabel('score')
    ax.set_title('general score curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'general_score_curve.png', dpi=150)
    plt.close(fig)


def plot_candidate_overview(candidate_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if candidate_df.empty:
        ax.text(0.5, 0.5, 'no candidates', ha='center', va='center')
    else:
        y = np.arange(len(candidate_df))
        ax.hlines(y, candidate_df['start_day'], candidate_df['end_day'])
        ax.scatter(candidate_df['peak_day'], y)
        ax.set_yticks(y)
        ax.set_yticklabels(candidate_df['window_id'])
    ax.set_xlabel('day_index')
    ax.set_title('candidate windows overview')
    fig.tight_layout()
    fig.savefig(output_dir / 'candidate_windows_overview.png', dpi=150)
    plt.close(fig)


def plot_yearwise_anchor_density(yearwise_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if yearwise_df.empty or yearwise_df['anchor_day'].dropna().empty:
        ax.text(0.5, 0.5, 'no yearwise anchors', ha='center', va='center')
    else:
        for window_id, sub in yearwise_df.groupby('window_id'):
            vals = sub['anchor_day'].dropna().to_numpy(dtype=np.float64)
            if vals.size:
                ax.hist(vals, bins=20, alpha=0.4, label=window_id)
        ax.legend()
    ax.set_xlabel('anchor_day')
    ax.set_title('yearwise anchor density')
    fig.tight_layout()
    fig.savefig(output_dir / 'yearwise_anchor_density.png', dpi=150)
    plt.close(fig)


def write_general_layer_summary(candidate_df: pd.DataFrame, decision_df: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        'candidate_count': int(len(candidate_df)),
        'candidate_source_counts': candidate_df['candidate_source'].value_counts(dropna=False).to_dict() if not candidate_df.empty else {},
        'general_status_counts': decision_df['general_status'].value_counts(dropna=False).to_dict() if not decision_df.empty else {},
        'failure_mode_counts': decision_df['failure_mode'].value_counts(dropna=False).to_dict() if not decision_df.empty else {},
    }
    save_json(output_dir / 'general_layer_summary.json', summary)
