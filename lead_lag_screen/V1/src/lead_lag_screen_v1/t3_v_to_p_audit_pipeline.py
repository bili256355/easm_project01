from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .t3_v_to_p_audit_settings import T3VToPAuditSettings
from .t3_v_to_p_audit_io import (
    read_csv_required,
    read_csv_optional,
    write_csv,
    normalize_v1_keys,
    add_family_columns,
)
from .t3_v_to_p_audit_logic import primary_failure_stage, lag_profile_type, safe_float


def _merge_stability_with_nulls(settings: T3VToPAuditSettings) -> pd.DataFrame:
    stab_path = settings.stability_dir / "tables" / "lead_lag_pair_summary_stability_judged.csv"
    if not stab_path.exists():
        raise FileNotFoundError(
            "Stability judgement table not found. Run "
            "lead_lag_screen\\V1\\scripts\\run_lead_lag_screen_v1_stability_judgement.py first. "
            f"Missing: {stab_path}"
        )
    df = normalize_v1_keys(read_csv_required(stab_path))

    nulls = normalize_v1_keys(read_csv_required(settings.input_dir / "lead_lag_null_summary.csv"))
    audit = normalize_v1_keys(read_csv_required(settings.input_dir / "lead_lag_audit_surrogate_null_summary.csv"))
    phi = read_csv_optional(settings.input_dir / "lead_lag_surrogate_ar1_params.csv")
    pair = normalize_v1_keys(read_csv_required(settings.input_dir / "lead_lag_pair_summary.csv"))

    key = ["window", "source_variable", "target_variable"]

    # Add null-stat columns that may not be in the stability table.
    null_keep = key + [
        c for c in [
            "T_pos_obs", "T_pos_null90", "T_pos_null95", "T_pos_null99",
            "T_neg_obs", "T_neg_null90", "T_neg_null95", "T_neg_null99",
            "T_0_obs", "T_0_null90", "T_0_null95", "T_0_null99",
            "S_pos", "S_neg", "S_0",
        ] if c in nulls.columns
    ]
    df = df.merge(nulls[null_keep], on=key, how="left", suffixes=("", "_null"))

    audit_keep = key + [
        c for c in [
            "T_pos_obs", "T_pos_null90", "T_pos_null95", "T_pos_null99",
            "p_pos_surrogate", "q_pos_within_window", "q_pos_global",
            "S_pos",
        ] if c in audit.columns
    ]
    audit_ren = {c: f"audit_{c}" for c in audit_keep if c not in key}
    df = df.merge(audit[audit_keep].rename(columns=audit_ren), on=key, how="left")

    # Enrich with direction labels / failure reasons if absent.
    pair_keep = key + [
        c for c in ["direction_label", "lead_lag_label", "lead_lag_group", "failure_reason", "risk_note", "sample_status"]
        if c in pair.columns and c not in df.columns
    ]
    if len(pair_keep) > len(key):
        df = df.merge(pair[pair_keep], on=key, how="left")

    # Add AR(1) phi parameters for source and target, if available.
    if phi is not None and {"window", "variable"}.issubset(phi.columns):
        src_phi = phi.rename(columns={
            "variable": "source_variable",
            "raw_phi_before_clip": "source_raw_phi_from_params",
            "phi_after_clip": "source_phi_after_clip_from_params",
            "phi_clip_severity": "source_phi_clip_severity_from_params",
        })
        tgt_phi = phi.rename(columns={
            "variable": "target_variable",
            "raw_phi_before_clip": "target_raw_phi_from_params",
            "phi_after_clip": "target_phi_after_clip_from_params",
            "phi_clip_severity": "target_phi_clip_severity_from_params",
        })
        for side_df, var_col in [(src_phi, "source_variable"), (tgt_phi, "target_variable")]:
            keep = ["window", var_col] + [c for c in side_df.columns if c.endswith("_from_params")]
            df = df.merge(side_df[keep], on=["window", var_col], how="left")

    df = add_family_columns(df)
    return df


def _v_to_p(df: pd.DataFrame, windows: tuple[str, ...] | None = None) -> pd.DataFrame:
    out = df[(df["source_family"].eq("V")) & (df["target_family"].eq("P"))].copy()
    if windows:
        allowed = {w.upper() for w in windows}
        out = out[out["window"].astype(str).str.upper().isin(allowed)].copy()
    return out


def _add_failure_stage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        stage, reason = primary_failure_stage(row)
        rows.append((stage, reason))
    out = df.copy()
    out["failure_stage"] = [r[0] for r in rows]
    out["failure_reason_audit"] = [r[1] for r in rows]
    return out


def _window_attrition(vp: pd.DataFrame) -> pd.DataFrame:
    d = _add_failure_stage(vp)
    records = []
    for window, g in d.groupby("window", dropna=False):
        def count_stage(name: str) -> int:
            return int((g["failure_stage"] == name).sum())

        rec = {
            "window": window,
            "n_all_v_to_p_pairs": int(len(g)),
            "n_stable_lag_dominant": int((g["v1_stability_judgement"] == "stable_lag_dominant").sum()),
            "n_tau0_coupled": int((g["v1_stability_judgement"] == "significant_lagged_but_tau0_coupled").sum()),
            "n_stable_tau0_dominant": int((g["v1_stability_judgement"] == "stable_tau0_dominant_coupling").sum()),
            "n_audit_sensitive": int((g["v1_stability_judgement"] == "audit_sensitive").sum()),
            "n_failed_or_not_candidate": int((~g["v1_stability_judgement"].isin([
                "stable_lag_dominant",
                "significant_lagged_but_tau0_coupled",
                "stable_tau0_dominant_coupling",
                "audit_sensitive",
            ])).sum()),
            "n_positive_surrogate_pass_p05": int((pd.to_numeric(g.get("p_pos_surrogate"), errors="coerce") <= 0.05).sum()),
            "n_positive_fdr_pass_q10": int((pd.to_numeric(g.get("q_pos_within_window"), errors="coerce") <= 0.10).sum()),
            "n_audit_pass_p05": int((pd.to_numeric(g.get("p_pos_audit_surrogate"), errors="coerce") <= 0.05).sum()),
            "n_forward_over_reverse_stable": int((pd.to_numeric(g.get("D_pos_neg_CI90_low"), errors="coerce") > 0).sum()),
            "n_lag_over_tau0_stable": int((pd.to_numeric(g.get("D_pos_0_CI90_low"), errors="coerce") > 0).sum()),
            "n_tau0_competitive_or_close": int((pd.to_numeric(g.get("D_pos_0_CI90_low"), errors="coerce") <= 0).sum()),
        }
        for stage, n in g["failure_stage"].value_counts().to_dict().items():
            rec[f"stage_{stage}"] = int(n)
        records.append(rec)
    out = pd.DataFrame(records)
    order = {w: i for i, w in enumerate(["S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5"])}
    if not out.empty:
        out["_ord"] = out["window"].map(order).fillna(999)
        out = out.sort_values(["_ord", "window"]).drop(columns=["_ord"])
    return out


def _group_matrix(vp: pd.DataFrame, group_col: str) -> pd.DataFrame:
    d = _add_failure_stage(vp)
    recs = []
    for (window, group), g in d.groupby(["window", group_col], dropna=False):
        recs.append({
            "window": window,
            group_col: group,
            "n_pairs": int(len(g)),
            "n_stable_lag_dominant": int((g["v1_stability_judgement"] == "stable_lag_dominant").sum()),
            "n_tau0_coupled": int((g["v1_stability_judgement"] == "significant_lagged_but_tau0_coupled").sum()),
            "n_audit_sensitive": int((g["v1_stability_judgement"] == "audit_sensitive").sum()),
            "n_failed_or_not_candidate": int((~g["v1_stability_judgement"].isin([
                "stable_lag_dominant",
                "significant_lagged_but_tau0_coupled",
                "stable_tau0_dominant_coupling",
                "audit_sensitive",
            ])).sum()),
            "mean_T_pos": float(pd.to_numeric(g.get("T_pos_obs"), errors="coerce").mean()),
            "mean_T_0": float(pd.to_numeric(g.get("T_0_obs"), errors="coerce").mean()),
            "mean_T_neg": float(pd.to_numeric(g.get("T_neg_obs"), errors="coerce").mean()),
            "mean_D_pos_0": float(pd.to_numeric(g.get("D_pos_0"), errors="coerce").mean()),
            "mean_D_pos_neg": float(pd.to_numeric(g.get("D_pos_neg"), errors="coerce").mean()),
            "mean_margin_pos": float((pd.to_numeric(g.get("T_pos_obs"), errors="coerce") - pd.to_numeric(g.get("T_pos_null95"), errors="coerce")).mean()),
        })
    out = pd.DataFrame(recs)
    if not out.empty:
        order = {w: i for i, w in enumerate(["S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5"])}
        out["_ord"] = out["window"].map(order).fillna(999)
        out = out.sort_values(["_ord", group_col]).drop(columns=["_ord"])
    return out


def _lag_profile(settings: T3VToPAuditSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_path = settings.input_dir / "lead_lag_curve_long.csv"
    curve = add_family_columns(normalize_v1_keys(read_csv_required(curve_path)))
    focus = settings.focus_window.upper()
    prof = curve[
        curve["window"].astype(str).str.upper().eq(focus)
        & curve["source_family"].eq("V")
        & curve["target_family"].eq("P")
        & pd.to_numeric(curve["lag"], errors="coerce").between(-5, 5)
    ].copy()
    if prof.empty:
        return prof, pd.DataFrame()

    prof["lag"] = pd.to_numeric(prof["lag"], errors="coerce").astype("Int64")
    prof["abs_r"] = pd.to_numeric(prof["abs_r"], errors="coerce")
    prof["signed_r"] = pd.to_numeric(prof["signed_r"], errors="coerce")
    prof["source_target"] = prof["source_variable"].astype(str) + "→" + prof["target_variable"].astype(str)

    recs = []
    for (src, tgt), g in prof.groupby(["source_variable", "target_variable"]):
        pos = g[g["lag"] > 0].sort_values("abs_r", ascending=False).head(1)
        neg = g[g["lag"] < 0].sort_values("abs_r", ascending=False).head(1)
        zero = g[g["lag"] == 0].head(1)
        rec = {
            "source_variable": src,
            "target_variable": tgt,
            "max_positive_lag": int(pos["lag"].iloc[0]) if not pos.empty else np.nan,
            "max_positive_signed_r": float(pos["signed_r"].iloc[0]) if not pos.empty else np.nan,
            "max_positive_abs_r": float(pos["abs_r"].iloc[0]) if not pos.empty else np.nan,
            "lag0_signed_r": float(zero["signed_r"].iloc[0]) if not zero.empty else np.nan,
            "lag0_abs_r": float(zero["abs_r"].iloc[0]) if not zero.empty else np.nan,
            "max_negative_lag": int(neg["lag"].iloc[0]) if not neg.empty else np.nan,
            "max_negative_signed_r": float(neg["signed_r"].iloc[0]) if not neg.empty else np.nan,
            "max_negative_abs_r": float(neg["abs_r"].iloc[0]) if not neg.empty else np.nan,
        }
        recs.append(rec)
    summary = pd.DataFrame(recs)
    if not summary.empty:
        summary["profile_type"] = [lag_profile_type(row) for _, row in summary.iterrows()]
    return prof, summary


def _null_difficulty(vp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = vp.copy()
    for col in ["T_pos_obs", "T_pos_null95", "audit_T_pos_null95", "T_0_obs", "T_neg_obs"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["margin_pos"] = out.get("T_pos_obs") - out.get("T_pos_null95")
    if "audit_T_pos_null95" in out.columns:
        out["audit_margin_pos"] = out.get("T_pos_obs") - out.get("audit_T_pos_null95")
    else:
        out["audit_margin_pos"] = np.nan

    recs = []
    for window, g in out.groupby("window", dropna=False):
        recs.append({
            "window": window,
            "n_pairs": int(len(g)),
            "mean_T_pos": float(pd.to_numeric(g.get("T_pos_obs"), errors="coerce").mean()),
            "mean_lag0": float(pd.to_numeric(g.get("T_0_obs"), errors="coerce").mean()),
            "mean_T_neg": float(pd.to_numeric(g.get("T_neg_obs"), errors="coerce").mean()),
            "mean_null95_pos": float(pd.to_numeric(g.get("T_pos_null95"), errors="coerce").mean()),
            "mean_margin_pos": float(pd.to_numeric(g.get("margin_pos"), errors="coerce").mean()),
            "median_margin_pos": float(pd.to_numeric(g.get("margin_pos"), errors="coerce").median()),
            "n_margin_positive": int((pd.to_numeric(g.get("margin_pos"), errors="coerce") > 0).sum()),
            "mean_audit_margin_pos": float(pd.to_numeric(g.get("audit_margin_pos"), errors="coerce").mean()),
            "mean_D_pos_0": float(pd.to_numeric(g.get("D_pos_0"), errors="coerce").mean()),
            "mean_D_pos_neg": float(pd.to_numeric(g.get("D_pos_neg"), errors="coerce").mean()),
            "mean_source_phi": float(pd.to_numeric(g.get("source_phi_after_clip"), errors="coerce").mean()) if "source_phi_after_clip" in g.columns else float(pd.to_numeric(g.get("source_phi_after_clip_from_params"), errors="coerce").mean()),
            "mean_target_phi": float(pd.to_numeric(g.get("target_phi_after_clip"), errors="coerce").mean()) if "target_phi_after_clip" in g.columns else float(pd.to_numeric(g.get("target_phi_after_clip_from_params"), errors="coerce").mean()),
        })
    summary = pd.DataFrame(recs)
    if not summary.empty:
        order = {w: i for i, w in enumerate(["S1", "T1", "S2", "T2", "S3", "T3", "S4", "T4", "S5"])}
        summary["_ord"] = summary["window"].map(order).fillna(999)
        summary = summary.sort_values(["_ord", "window"]).drop(columns=["_ord"])
    return out, summary


def _index_validity_context(settings: T3VToPAuditSettings) -> pd.DataFrame:
    if not settings.include_index_validity_context:
        return pd.DataFrame()

    tables_dir = settings.index_validity_tables_dir
    ind = read_csv_optional(tables_dir / "index_window_representativeness.csv")
    joint = read_csv_optional(tables_dir / "window_family_joint_field_coverage.csv")
    if ind is None:
        return pd.DataFrame()

    focus = settings.focus_window.upper()
    ind = ind[ind["window"].astype(str).str.upper().eq(focus)].copy()
    # Keep only V/P indices for this audit context.
    fam_col = "family" if "family" in ind.columns else "object_family"
    if fam_col in ind.columns:
        ind = ind[ind[fam_col].isin(["V", "P"])].copy()

    if joint is not None and "window" in joint.columns:
        joint = joint[joint["window"].astype(str).str.upper().eq(focus)].copy()
        jf = "family" if "family" in joint.columns else "object_family"
        if jf in joint.columns:
            keep_cols = [c for c in [
                "window", jf, "joint_field_R2_year_cv", "joint_eof_coverage_top5_year_cv",
                "coverage_tier", "collapse_risk_update"
            ] if c in joint.columns]
            joint = joint[keep_cols].rename(columns={jf: fam_col})
            ind = ind.merge(joint, on=["window", fam_col], how="left")
    return ind


def _reason_summary(focus_attrition: pd.DataFrame, window_attrition: pd.DataFrame) -> pd.DataFrame:
    counts = focus_attrition["failure_stage"].value_counts().rename_axis("failure_stage").reset_index(name="n_pairs")
    total = max(int(len(focus_attrition)), 1)
    counts["share"] = counts["n_pairs"] / total
    notes = {
        "passed_stable_lag": "T3 V→P survived as stable lag-dominant.",
        "passed_tau0_coupled": "T3 V→P survived only as lagged-but-tau0-coupled.",
        "audit_sensitive_candidate": "Candidate survives only as audit-sensitive.",
        "tau0_competitive": "Likely near-synchronous / lag0-competitive relation.",
        "positive_not_surrogate_significant": "Positive-lag signal did not clear AR(1) surrogate background.",
        "positive_fdr_not_supported": "Positive-lag signal did not clear within-window FDR.",
        "audit_not_stable": "Main support weakens under audit null.",
        "reverse_competitive": "Reverse direction is competitive or stronger.",
        "direction_uncertain": "Forward-vs-reverse interval crosses zero.",
        "not_candidate_other": "Did not enter existing candidate pool; check raw diagnostics.",
    }
    counts["interpretation_hint"] = counts["failure_stage"].map(notes).fillna("Manual review needed.")
    return counts


def run_t3_v_to_p_disappearance_audit(settings: T3VToPAuditSettings) -> dict[str, object]:
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_stability_with_nulls(settings)
    all_vp = _v_to_p(merged, settings.comparison_windows)
    all_vp = _add_failure_stage(all_vp)

    focus = settings.focus_window.upper()
    focus_vp = all_vp[all_vp["window"].astype(str).str.upper().eq(focus)].copy()

    # Core outputs
    write_csv(focus_vp, settings.table_dir / "t3_v_to_p_attrition_waterfall.csv")
    window_attrition = _window_attrition(all_vp)
    write_csv(window_attrition, settings.table_dir / "v_to_p_window_attrition_comparison.csv")

    write_csv(_group_matrix(all_vp, "source_variable"), settings.table_dir / "v_to_p_by_source_v_index.csv")
    write_csv(_group_matrix(all_vp, "target_variable"), settings.table_dir / "v_to_p_by_target_p_index.csv")
    write_csv(_group_matrix(focus_vp, "source_variable"), settings.table_dir / "t3_v_to_p_by_source_v_index.csv")
    write_csv(_group_matrix(focus_vp, "target_variable"), settings.table_dir / "t3_v_to_p_by_target_p_index.csv")

    prof_long, prof_summary = _lag_profile(settings)
    write_csv(prof_long, settings.table_dir / "t3_v_to_p_lag_profile_long.csv")
    write_csv(prof_summary, settings.table_dir / "t3_v_to_p_lag_profile_summary.csv")

    null_detail, null_summary = _null_difficulty(all_vp)
    write_csv(null_detail, settings.table_dir / "v_to_p_window_null_difficulty_detail.csv")
    write_csv(null_summary, settings.table_dir / "v_to_p_window_null_difficulty_summary.csv")

    iv_ctx = _index_validity_context(settings)
    write_csv(iv_ctx, settings.table_dir / "t3_v_to_p_index_validity_context.csv")

    reason_summary = _reason_summary(focus_vp, window_attrition)
    write_csv(reason_summary, settings.table_dir / "t3_v_to_p_disappearance_reason_summary.csv")

    # Lightweight human-readable README
    readme = f"""# T3 V→P disappearance audit

This audit does not rerun V1. It reads existing V1 smooth5 outputs and the V1 stability judgement layer.

Focus question:
Why do V→P candidates shrink in {settings.focus_window} relative to adjacent windows?

Key interpretation:
- `passed_stable_lag` means the pair survives as stable lag-dominant.
- `passed_tau0_coupled` means the pair survives, but lag0 is close/competitive.
- `positive_not_surrogate_significant` means positive-lag signal did not clear the AR(1) surrogate background.
- `audit_not_stable` means main support weakens under the audit null.
- `reverse_competitive` / `direction_uncertain` mean the V→P direction is not stable.
- `tau0_competitive` means the relation is better interpreted as synchronous/rapid adjustment than stable lag.

This is an audit of attrition mechanisms, not a new pathway result.
"""
    (settings.summary_dir / "T3_V_TO_P_AUDIT_README.md").write_text(readme, encoding="utf-8")

    counts = focus_vp["v1_stability_judgement"].value_counts(dropna=False).to_dict() if not focus_vp.empty else {}
    failure_counts = focus_vp["failure_stage"].value_counts(dropna=False).to_dict() if not focus_vp.empty else {}
    summary = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(settings.input_dir),
        "stability_dir": str(settings.stability_dir),
        "output_dir": str(settings.output_dir),
        "focus_window": settings.focus_window,
        "comparison_windows": list(settings.comparison_windows),
        "focus_v_to_p_total_pairs": int(len(focus_vp)),
        "focus_stable_lag_dominant": int((focus_vp["v1_stability_judgement"] == "stable_lag_dominant").sum()) if not focus_vp.empty else 0,
        "focus_tau0_coupled": int((focus_vp["v1_stability_judgement"] == "significant_lagged_but_tau0_coupled").sum()) if not focus_vp.empty else 0,
        "focus_audit_sensitive": int((focus_vp["v1_stability_judgement"] == "audit_sensitive").sum()) if not focus_vp.empty else 0,
        "focus_not_candidate_or_failed": int((~focus_vp["v1_stability_judgement"].isin([
            "stable_lag_dominant",
            "significant_lagged_but_tau0_coupled",
            "stable_tau0_dominant_coupling",
            "audit_sensitive",
        ])).sum()) if not focus_vp.empty else 0,
        "focus_v1_stability_judgement_counts": {str(k): int(v) for k, v in counts.items()},
        "focus_failure_stage_counts": {str(k): int(v) for k, v in failure_counts.items()},
        "outputs": [
            "tables/t3_v_to_p_attrition_waterfall.csv",
            "tables/v_to_p_window_attrition_comparison.csv",
            "tables/t3_v_to_p_by_source_v_index.csv",
            "tables/t3_v_to_p_by_target_p_index.csv",
            "tables/t3_v_to_p_lag_profile_long.csv",
            "tables/t3_v_to_p_lag_profile_summary.csv",
            "tables/v_to_p_window_null_difficulty_detail.csv",
            "tables/v_to_p_window_null_difficulty_summary.csv",
            "tables/t3_v_to_p_index_validity_context.csv",
            "tables/t3_v_to_p_disappearance_reason_summary.csv",
        ],
        "interpretation_guardrail": (
            "This audit traces why T3 V→P shrinks in existing V1/stability outputs. "
            "It does not establish or refute physical pathways."
        ),
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta = {
        "settings": {
            "input_tag": settings.input_tag,
            "stability_tag": settings.stability_tag,
            "output_tag": settings.output_tag,
            "focus_window": settings.focus_window,
            "comparison_windows": list(settings.comparison_windows),
            "include_index_validity_context": settings.include_index_validity_context,
        },
        "required_inputs": [
            str(settings.stability_dir / "tables" / "lead_lag_pair_summary_stability_judged.csv"),
            str(settings.input_dir / "lead_lag_null_summary.csv"),
            str(settings.input_dir / "lead_lag_audit_surrogate_null_summary.csv"),
            str(settings.input_dir / "lead_lag_pair_summary.csv"),
            str(settings.input_dir / "lead_lag_curve_long.csv"),
        ],
        "optional_inputs": [
            str(settings.input_dir / "lead_lag_surrogate_ar1_params.csv"),
            str(settings.index_validity_tables_dir / "index_window_representativeness.csv"),
            str(settings.index_validity_tables_dir / "window_family_joint_field_coverage.csv"),
        ],
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
