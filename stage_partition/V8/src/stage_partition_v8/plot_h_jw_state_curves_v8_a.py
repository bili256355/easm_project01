from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Dict, List

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_TAG = 'state_relation_v8_a'
PLOT_TAG = 'plot_h_jw_state_curves_v8_a'
DEFAULT_WINDOW = 'W045'
BASELINE_ORDER = ['C0_full_stage', 'C1_buffered_stage', 'C2_immediate_pre']
BRANCH_TO_COL = {'dist': 'S_dist', 'pattern': 'S_pattern'}


def _stage_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_state_output_root(v8_root: Path) -> Path:
    env = os.environ.get('V8_STATE_PLOT_INPUT_DIR', '').strip()
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise FileNotFoundError(f'V8_STATE_PLOT_INPUT_DIR does not exist: {p}')
    p = v8_root / 'outputs' / OUTPUT_TAG
    if p.exists():
        return p
    raise FileNotFoundError(
        f'Cannot find state relation output directory: {p}. '\
        'Run state_relation_v8_a first, or set V8_STATE_PLOT_INPUT_DIR.'
    )


def _resolve_output_dir(v8_root: Path) -> Path:
    env = os.environ.get('V8_STATE_PLOT_OUTPUT_DIR', '').strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = v8_root / 'outputs' / OUTPUT_TAG / 'figures_h_jw_state_curves_v8_a'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_required_csvs(state_output_root: Path, window_id: str) -> Dict[str, pd.DataFrame]:
    per_window = state_output_root / 'per_window' / window_id
    obj_path = per_window / f'object_state_curves_{window_id}.csv'
    delta_path = per_window / f'pairwise_delta_state_curves_{window_id}.csv'
    if not obj_path.exists():
        raise FileNotFoundError(f'Missing object state curve file: {obj_path}')
    if not delta_path.exists():
        raise FileNotFoundError(f'Missing pairwise delta curve file: {delta_path}')
    return {
        'object': pd.read_csv(obj_path),
        'delta': pd.read_csv(delta_path),
    }


def _filter_h_jw_curves(df_obj: pd.DataFrame, baseline: str, object_name: str) -> pd.DataFrame:
    out = df_obj[(df_obj['baseline_config'] == baseline) & (df_obj['object'] == object_name)].copy()
    if out.empty:
        raise ValueError(f'No object state rows found for baseline={baseline}, object={object_name}')
    out = out.sort_values('day').reset_index(drop=True)
    return out


def _filter_delta_h_minus_jw(df_delta: pd.DataFrame, baseline: str, branch: str) -> pd.DataFrame:
    out = df_delta[
        (df_delta['baseline_config'] == baseline) &
        (df_delta['branch'] == branch) &
        (df_delta['object_A'] == 'H') &
        (df_delta['object_B'] == 'Jw')
    ].copy()
    if out.empty:
        raise ValueError(f'No H-Jw delta rows found for baseline={baseline}, branch={branch}')
    out = out.sort_values('day').reset_index(drop=True)
    return out


def _save_state_curve_plot(df_h: pd.DataFrame, df_jw: pd.DataFrame, baseline: str, branch: str, ycol: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df_h['day'], df_h[ycol], label='H')
    ax.plot(df_jw['day'], df_jw[ycol], label='Jw')
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel('Day index')
    ax.set_ylabel(ycol)
    ax.set_title(f'W045 H vs Jw state curves | {baseline} | {branch}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_delta_plot(df_delta: pd.DataFrame, baseline: str, branch: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(df_delta['day'], df_delta['delta_S'], label='ΔS(H-Jw)')
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel('Day index')
    ax.set_ylabel('delta_S')
    ax.set_title(f'W045 ΔS(H-Jw) | {baseline} | {branch}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_manifest(rows: List[dict], out_path: Path) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')


def run_plot_h_jw_state_curves_v8_a(v8_root: Path | None = None) -> Path:
    v8_root = Path(v8_root) if v8_root is not None else (_stage_root_from_this_file() / 'V8')
    state_output_root = _resolve_state_output_root(v8_root)
    plot_output_root = _resolve_output_dir(v8_root)
    window_id = os.environ.get('V8_STATE_PLOT_WINDOW', DEFAULT_WINDOW).strip() or DEFAULT_WINDOW

    dfs = _read_required_csvs(state_output_root, window_id)
    df_obj = dfs['object']
    df_delta = dfs['delta']

    window_out = plot_output_root / window_id
    window_out.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []

    for baseline in BASELINE_ORDER:
        df_h = _filter_h_jw_curves(df_obj, baseline, 'H')
        df_jw = _filter_h_jw_curves(df_obj, baseline, 'Jw')
        for branch, ycol in BRANCH_TO_COL.items():
            state_png = window_out / f'H_Jw_state_curves__{baseline}__{branch}.png'
            _save_state_curve_plot(df_h, df_jw, baseline, branch, ycol, state_png)
            manifest_rows.append({
                'window_id': window_id,
                'baseline_config': baseline,
                'plot_type': 'state_curves',
                'branch': branch,
                'file_name': state_png.name,
                'source_csv': f'object_state_curves_{window_id}.csv',
            })

            delta_df = _filter_delta_h_minus_jw(df_delta, baseline, branch)
            delta_png = window_out / f'H_minus_Jw_delta__{baseline}__{branch}.png'
            _save_delta_plot(delta_df, baseline, branch, delta_png)
            manifest_rows.append({
                'window_id': window_id,
                'baseline_config': baseline,
                'plot_type': 'delta_curve',
                'branch': branch,
                'file_name': delta_png.name,
                'source_csv': f'pairwise_delta_state_curves_{window_id}.csv',
            })

    manifest_path = window_out / 'plot_manifest_H_Jw_W045.csv'
    _write_manifest(manifest_rows, manifest_path)
    return window_out


if __name__ == '__main__':
    out = run_plot_h_jw_state_curves_v8_a()
    print(f'Plot output written to: {out}')
