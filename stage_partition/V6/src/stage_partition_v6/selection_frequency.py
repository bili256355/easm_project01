from __future__ import annotations
import numpy as np
import pandas as pd
from .timeline import day_index_to_month_day


def build_selection_frequency_by_day(
    peaks_records_df: pd.DataFrame,
    *,
    max_day_index: int,
    n_total_replicates: int,
) -> pd.DataFrame:
    days = np.arange(int(max_day_index) + 1, dtype=int)
    out = pd.DataFrame({
        'day': days,
        'month_day': [day_index_to_month_day(int(d)) for d in days],
    })
    if peaks_records_df is None or peaks_records_df.empty:
        out['n_selected_replicates'] = 0
        out['selection_frequency_raw'] = 0.0
        return out

    dedup = peaks_records_df[['replicate_id', 'peak_day']].drop_duplicates().copy()
    dedup['peak_day'] = pd.to_numeric(dedup['peak_day'], errors='coerce').astype('Int64')
    counts = dedup.groupby('peak_day').size().rename('n_selected_replicates').reset_index()
    counts['peak_day'] = counts['peak_day'].astype(int)
    out = out.merge(counts, how='left', left_on='day', right_on='peak_day')
    out['n_selected_replicates'] = out['n_selected_replicates'].fillna(0).astype(int)
    out['selection_frequency_raw'] = out['n_selected_replicates'] / float(max(1, int(n_total_replicates)))
    out = out.drop(columns=['peak_day'])
    return out


def smooth_selection_frequency(freq_df: pd.DataFrame, *, window_days: int = 3) -> pd.DataFrame:
    out = freq_df.copy()
    if 'selection_frequency_raw' not in out.columns:
        out['selection_frequency_smooth'] = np.nan
        return out
    win = max(1, int(window_days))
    out['selection_frequency_smooth'] = (
        pd.to_numeric(out['selection_frequency_raw'], errors='coerce')
        .rolling(window=win, center=True, min_periods=1)
        .mean()
    )
    return out


def extract_selection_local_maxima(
    freq_df: pd.DataFrame,
    *,
    min_frequency: float = 0.10,
    use_smoothed: bool = True,
) -> pd.DataFrame:
    value_col = 'selection_frequency_smooth' if use_smoothed and 'selection_frequency_smooth' in freq_df.columns else 'selection_frequency_raw'
    vals = pd.to_numeric(freq_df[value_col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    days = pd.to_numeric(freq_df['day'], errors='coerce').to_numpy(dtype=int)
    rows = []
    peak_id = 1
    for i in range(len(vals)):
        left = vals[i - 1] if i > 0 else -np.inf
        right = vals[i + 1] if i < len(vals) - 1 else -np.inf
        if vals[i] >= left and vals[i] > right and vals[i] >= float(min_frequency):
            rows.append({
                'peak_id': f'SF{i+1:03d}',
                'peak_day': int(days[i]),
                'peak_month_day': day_index_to_month_day(int(days[i])),
                'selection_frequency_raw': float(pd.to_numeric(freq_df.iloc[i]['selection_frequency_raw'], errors='coerce')),
                'selection_frequency_smooth': float(pd.to_numeric(freq_df.iloc[i].get('selection_frequency_smooth', np.nan), errors='coerce')),
                'source_column': value_col,
            })
            peak_id += 1
    return pd.DataFrame(rows, columns=[
        'peak_id', 'peak_day', 'peak_month_day', 'selection_frequency_raw', 'selection_frequency_smooth', 'source_column'
    ])



def build_baseline_peak_context_table(freq_df: pd.DataFrame, baseline_peaks_df: pd.DataFrame) -> pd.DataFrame:
    if baseline_peaks_df is None or baseline_peaks_df.empty:
        return pd.DataFrame(columns=['peak_id','peak_day','peak_month_day','peak_score','peak_prominence','selection_frequency_raw','selection_frequency_smooth'])
    out = baseline_peaks_df.copy()
    if 'peak_day' in out.columns:
        out['peak_day'] = pd.to_numeric(out['peak_day'], errors='coerce').astype('Int64')
    freq_small = freq_df[['day', 'selection_frequency_raw']].copy()
    if 'selection_frequency_smooth' in freq_df.columns:
        freq_small['selection_frequency_smooth'] = pd.to_numeric(freq_df['selection_frequency_smooth'], errors='coerce')
    out = out.merge(freq_small, how='left', left_on='peak_day', right_on='day')
    if 'day' in out.columns:
        out = out.drop(columns=['day'])
    cols = [c for c in ['peak_id','peak_day','peak_month_day','peak_score','peak_prominence','selection_frequency_raw','selection_frequency_smooth'] if c in out.columns]
    return out[cols].copy()


def plot_selection_frequency_curve(
    freq_df: pd.DataFrame,
    baseline_peaks_df: pd.DataFrame,
    output_path,
    *,
    annotate_baseline_peaks: bool = True,
    dpi: int = 200,
) -> dict:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {'status': 'skipped_matplotlib_unavailable', 'error': repr(exc), 'output_path': str(output_path)}
    if freq_df is None or freq_df.empty:
        return {'status': 'skipped_empty_frequency', 'output_path': str(output_path)}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = pd.to_numeric(freq_df['day'], errors='coerce')
    y_raw = pd.to_numeric(freq_df['selection_frequency_raw'], errors='coerce')
    ax.plot(x, y_raw, label='Raw frequency')
    if 'selection_frequency_smooth' in freq_df.columns and pd.to_numeric(freq_df['selection_frequency_smooth'], errors='coerce').notna().any():
        y_smooth = pd.to_numeric(freq_df['selection_frequency_smooth'], errors='coerce')
        ax.plot(x, y_smooth, label='Smoothed frequency')
    if baseline_peaks_df is not None and not baseline_peaks_df.empty:
        bdf = baseline_peaks_df.copy()
        bdf['peak_day'] = pd.to_numeric(bdf['peak_day'], errors='coerce')
        bdf = bdf.dropna(subset=['peak_day'])
        for _, row in bdf.iterrows():
            day = int(row['peak_day'])
            ax.axvline(day, linewidth=0.8, alpha=0.22)
            if annotate_baseline_peaks:
                peak_y = float(pd.to_numeric(freq_df.loc[freq_df['day'] == day, 'selection_frequency_smooth' if 'selection_frequency_smooth' in freq_df.columns else 'selection_frequency_raw'], errors='coerce').fillna(0.0).max())
                ax.text(day, peak_y + 0.01, f'{day}', rotation=90, va='bottom', ha='center', fontsize=8)
    ax.set_title('Bootstrap selection frequency by day')
    ax.set_xlabel('Day index')
    ax.set_ylabel('Selection frequency')
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(max(72, dpi)), bbox_inches='tight')
    plt.close(fig)
    return {'status': 'written', 'output_path': str(output_path)}
