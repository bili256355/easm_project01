from __future__ import annotations

from pathlib import Path
import pandas as pd

from .scale_config import ScaleSettings


def _fmt_float(x, nd=3) -> str:
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def write_scale_summary(
    cfg: ScaleSettings,
    output_root: Path,
    ridge_summary: pd.DataFrame,
    target_response: pd.DataFrame,
    decision: pd.DataFrame,
    meta: dict,
) -> None:
    lines: list[str] = []
    lines.append("# V10.7_b H W045 Gaussian derivative scale-space diagnostic")
    lines.append("")
    lines.append("## 1. Purpose and boundary")
    lines.append("")
    lines.append("This is a dedicated scale diagnostic on the H object state matrix around W045.")
    lines.append("It is **not** a `ruptures.Window` rerun and must not be treated as detector-width sensitivity.")
    lines.append("It does not infer causality, physical mechanism, yearwise stability, or spatial continuity.")
    lines.append("Its role is heuristic: identify whether H19/H35/H45/H57 appear as scale-space transition-energy structures and help choose targets for later yearwise/spatial tests.")
    lines.append("")
    lines.append("## 2. Settings")
    lines.append("")
    lines.append(f"- sigmas: {list(cfg.scale.sigmas)}")
    lines.append(f"- focus day range: {cfg.scale.focus_day_min}–{cfg.scale.focus_day_max}")
    lines.append(f"- target days: {cfg.targets.target_days}")
    lines.append(f"- scale backend: {meta.get('scale_backend', 'unknown')}")
    lines.append(f"- used H features: {meta.get('n_used_features', 'NA')} / input features: {meta.get('n_input_features', 'NA')}")
    lines.append("")
    lines.append("## 3. Ridge-family summary")
    lines.append("")
    if ridge_summary is None or ridge_summary.empty:
        lines.append("No ridge families passed the local-maximum thresholds in the focus range.")
    else:
        cols = ["ridge_id", "day_min", "day_max", "day_center_weighted", "sigma_min", "sigma_max", "persistence_fraction", "nearest_target_label", "nearest_target_distance", "role_hint"]
        lines.append(ridge_summary[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 4. Target-day scale identity hints")
    lines.append("")
    if decision is None or decision.empty:
        lines.append("No target-day decision table was generated.")
    else:
        cols = ["target_label", "target_day", "nearest_ridge_id", "scale_identity_hint", "persistence_fraction", "max_energy_norm", "recommended_next_step_target"]
        lines.append(decision[cols].to_markdown(index=False))
        note = str(decision.get("global_h35_note", pd.Series([""])).iloc[0]) if "global_h35_note" in decision.columns else ""
        if note:
            lines.append("")
            lines.append(f"**Global H35 note:** {note}")
    lines.append("")
    lines.append("## 5. Interpretation rules")
    lines.append("")
    lines.append("- If H35 shares a ridge with H19, later tests should use an H19–H35 prewindow package rather than H35 alone.")
    lines.append("- If H35 forms a separate medium-persistence ridge, it can be retained as a heuristic E2 target, but still not as a confirmed weak precursor.")
    lines.append("- If H45 lacks a clear ridge, that supports H absence in the W045 main-cluster at the scale-diagnostic layer only.")
    lines.append("- If H57 forms a ridge, it should be treated as a post-W045 reference candidate, not automatically as W045 response.")
    lines.append("")
    lines.append("## 6. Recommended next step")
    lines.append("")
    if decision is not None and not decision.empty and "target_label" in decision.columns:
        h35_row = decision[decision["target_label"] == "H35"]
        h45_row = decision[decision["target_label"] == "H45"]
        if not h35_row.empty:
            lines.append(f"- H35 next-step target: {h35_row.iloc[0].get('recommended_next_step_target', 'NA')}")
        if not h45_row.empty:
            lines.append(f"- H45 implication: {h45_row.iloc[0].get('recommended_next_step_target', 'NA')}")
    lines.append("- Do not convert these scale hints into physical interpretation until yearwise and spatial-field checks are run.")
    lines.append("")
    (output_root / "summary_h_w045_scale_diagnostic_v10_7_b.md").write_text("\n".join(lines), encoding="utf-8")
