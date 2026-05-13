from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import pandas as pd


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def build_summary(
    peak_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    accepted_points: list[int],
    settings,
) -> dict:
    per_window = []
    if peak_df is not None and not peak_df.empty:
        for window_id, sub in peak_df.groupby("window_id", sort=False):
            ordered = sub.sort_values(["field_peak_day", "field"]).copy()
            per_window.append(
                {
                    "window_id": str(window_id),
                    "window_anchor_day": int(ordered["window_anchor_day"].iloc[0]),
                    "fields_ordered_by_peak_day": [
                        {
                            "field": str(r["field"]),
                            "field_peak_day": int(r["field_peak_day"]) if pd.notna(r["field_peak_day"]) else None,
                            "relative_to_anchor": int(r["relative_to_anchor"]) if pd.notna(r["relative_to_anchor"]) else None,
                            "peak_confidence_label": str(r["peak_confidence_label"]),
                        }
                        for _, r in ordered.iterrows()
                    ],
                }
            )

    return {
        "layer_name": "stage_partition",
        "version_name": "V7",
        "run_scope": "field_specific_transition_timing_for_accepted_windows_only",
        "run_label": settings.output.output_tag,
        "accepted_points": [int(x) for x in accepted_points],
        "excluded_candidate_days": [int(x) for x in settings.accepted_windows.excluded_candidate_days],
        "n_windows_used": int(len(windows_df)) if windows_df is not None else 0,
        "n_field_window_rows": int(len(peak_df)) if peak_df is not None else 0,
        "method": "field_only_ruptures_window_score_profile_peak_within_accepted_v6_1_window",
        "downstream_lead_lag_included": False,
        "pathway_included": False,
        "spatial_earliest_region_included": False,
        "causal_interpretation_included": False,
        "per_window_field_orders": per_window,
    }
