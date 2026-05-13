from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import List

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE_ORDER = ['C0_full_stage', 'C1_buffered_stage', 'C2_immediate_pre']


def _stage_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_object_csv(v8_root: Path) -> Path:
    env = os.environ.get('V8_HJW_STATE_OBJECT_CSV', '').strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f'V8_HJW_STATE_OBJECT_CSV does not exist: {p}')
    default = v8_root / 'outputs' / 'state_relation_v8_a' / 'per_window' / 'W045' / 'object_state_curves_W045.csv'
    if default.exists():
        return default
    raise FileNotFoundError(
        f'Cannot find default object state curve csv: {default}. '
        'Please set V8_HJW_STATE_OBJECT_CSV to the old result object_state_curves csv.'
    )


def _resolve_output_dir(v8_root: Path, label: str) -> Path:
    env = os.environ.get('V8_HJW_STATE_PLOT_OUTPUT_DIR', '').strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = v8_root / 'outputs' / 'state_relation_v8_a' / f'figures_h_jw_state_curves_{label}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_label(object_csv: Path) -> str:
    env = os.environ.get('V8_HJW_STATE_PLOT_LABEL', '').strip()
    if env:
        return env
    parent = object_csv.parent.name
    if parent:
        return parent.replace(' ', '_')
    return 'custom'


def _load_and_validate(object_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(object_csv)
    required = ['object', 'baseline_config', 'day', 'S_dist', 'S_pattern']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {object_csv}: {missing}')
    if 'window_id' not in df.columns:
        df['window_id'] = os.environ.get('V8_HJW_STATE_WINDOW', 'W045')
    return df


def _subset_object(df: pd.DataFrame, object_name: str, baseline: str) -> pd.DataFrame:
    sub = df[(df['object'] == object_name) & (df['baseline_config'] == baseline)].copy()
    if sub.empty:
        raise ValueError(f'No rows found for object={object_name}, baseline={baseline}')
    sub = sub.sort_values('day').reset_index(drop=True)
    return sub


def _compute_delta(df_h: pd.DataFrame, df_jw: pd.DataFrame, ycol: str) -> pd.DataFrame:
    merged = pd.merge(
        df_h[['day', ycol]],
        df_jw[['day', ycol]],
        on='day',
        how='inner',
        suffixes=('_H', '_Jw')
    )
    merged['delta'] = merged[f'{ycol}_H'] - merged[f'{ycol}_Jw']
    return merged.sort_values('day').reset_index(drop=True)


def _save_curve_plot(df_h: pd.DataFrame, df_jw: pd.DataFrame, baseline: str, ycol: str, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df_h['day'], df_h[ycol], label='H')
    ax.plot(df_jw['day'], df_jw[ycol], label='Jw')
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel('Day index')
    ax.set_ylabel(ycol)
    ax.set_title(f'{label}: H vs Jw | {baseline} | {ycol}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_delta_plot(df_delta: pd.DataFrame, baseline: str, ycol: str, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df_delta['day'], df_delta['delta'], label=f'Δ{ycol}(H-Jw)')
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel('Day index')
    ax.set_ylabel(f'Δ{ycol}')
    ax.set_title(f'{label}: H-Jw delta | {baseline} | {ycol}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_plot_h_jw_state_curves_from_object_csv_v8_a(v8_root: Path | None = None) -> Path:
    v8_root = Path(v8_root) if v8_root is not None else (_stage_root_from_this_file() / 'V8')
    object_csv = _resolve_object_csv(v8_root)
    label = _resolve_label(object_csv)
    out_dir = _resolve_output_dir(v8_root, label)
    df = _load_and_validate(object_csv)

    manifest: List[dict] = []
    for baseline in BASELINE_ORDER:
        df_h = _subset_object(df, 'H', baseline)
        df_jw = _subset_object(df, 'Jw', baseline)
        for ycol in ['S_dist', 'S_pattern']:
            state_png = out_dir / f'{label}__H_Jw_state_curves__{baseline}__{ycol}.png'
            _save_curve_plot(df_h, df_jw, baseline, ycol, label, state_png)
            manifest.append({
                'label': label,
                'baseline_config': baseline,
                'plot_type': 'state_curves',
                'quantity': ycol,
                'file_name': state_png.name,
                'source_object_csv': str(object_csv),
            })
            delta_df = _compute_delta(df_h, df_jw, ycol)
            delta_png = out_dir / f'{label}__H_minus_Jw_delta__{baseline}__{ycol}.png'
            _save_delta_plot(delta_df, baseline, ycol, label, delta_png)
            manifest.append({
                'label': label,
                'baseline_config': baseline,
                'plot_type': 'delta_curve',
                'quantity': ycol,
                'file_name': delta_png.name,
                'source_object_csv': str(object_csv),
            })
    pd.DataFrame(manifest).to_csv(out_dir / f'plot_manifest_{label}.csv', index=False, encoding='utf-8-sig')
    return out_dir


if __name__ == '__main__':
    out = run_plot_h_jw_state_curves_from_object_csv_v8_a()
    print(f'Plot output written to: {out}')
