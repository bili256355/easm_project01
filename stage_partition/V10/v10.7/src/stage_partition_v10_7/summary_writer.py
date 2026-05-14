from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import Settings


def _bool_series_all_true(df: pd.DataFrame, col: str) -> bool:
    return (not df.empty) and bool(df[col].astype(bool).all())


def write_summary(cfg: Settings, output_root: Path, reproduction: pd.DataFrame, atlas: pd.DataFrame, stability: pd.DataFrame) -> None:
    ok = _bool_series_all_true(reproduction, "matched_within_tolerance")
    lines: list[str] = []
    lines.append("# V10.7_a H-only main-method event atlas summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("V10.7_a is a H-only main-method baseline export. It reruns the object-native detector semantics for H across multiple detector_width values and exports full-season score curves, candidate catalogs, width-stability tables, and strong-window H event packages.")
    lines.append("")
    lines.append("It is a method-layer baseline for later tests. It is not a physical interpretation layer, not a yearwise validation, not a spatial validation, and not a causal test.")
    lines.append("")
    lines.append("## Baseline reproduction check")
    lines.append("")
    lines.append(f"Expected width=20 H candidates: {list(cfg.reference.expected_h_candidates_width20)}")
    lines.append(f"All expected candidates matched within ±{cfg.reference.expected_match_tolerance_days} days: **{ok}**")
    if not ok:
        lines.append("")
        lines.append("WARNING: baseline reproduction did not fully match the inherited H candidate list. Interpret width sensitivity only after checking implementation/input alignment.")
    lines.append("")
    lines.append("## Strong-window H role overview")
    lines.append("")
    if atlas.empty:
        lines.append("No atlas rows were produced.")
    else:
        for win_id in atlas["window_id"].drop_duplicates().tolist():
            lines.append(f"### {win_id}")
            sub = atlas[atlas["window_id"] == win_id]
            for _, row in sub.iterrows():
                lines.append(
                    f"- width={int(row['detector_width'])}: {row['h_role_class']} | "
                    f"pre={row['h_pre_window_candidates'] or 'none'}; "
                    f"inside={row['h_inside_window_candidates'] or 'none'}; "
                    f"post={row['h_post_window_candidates'] or 'none'}; "
                    f"nearest={row['nearest_candidate_day']} (distance={row['nearest_candidate_distance']})"
                )
            lines.append("")
    lines.append("## Width-stability note")
    lines.append("")
    if stability.empty:
        lines.append("No width-stability rows were produced.")
    else:
        n_stable = int(stability["stable_under_width_flag"].astype(bool).sum())
        lines.append(f"Width-stable baseline candidates: {n_stable} / {len(stability)}")
        lines.append("")
        for _, row in stability.iterrows():
            lines.append(f"- baseline day {int(row['baseline_candidate_day'])}: matched_widths={row['matched_widths']}; max_shift_abs={row['max_shift_abs']}; stable={row['stable_under_width_flag']}")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append("Do not use V10.7_a alone to claim H weak precursor, H condition, or physical mechanism. Use it to choose target H event packages for later yearwise, spatial, or conditional tests.")
    lines.append("")
    (output_root / "summary_h_object_event_atlas_v10_7_a.md").write_text("\n".join(lines), encoding="utf-8")
