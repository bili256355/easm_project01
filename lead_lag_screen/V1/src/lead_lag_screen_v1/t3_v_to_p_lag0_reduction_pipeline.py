from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .t3_v_to_p_lag0_reduction_settings import T3VToPLag0ReductionSettings, WINDOW_DAY_RANGES
from .t3_v_to_p_lag0_reduction_io import (
    read_csv_required,
    read_csv_optional,
    write_csv,
    normalize_v1_keys,
    add_family_columns,
    load_smoothed_fields,
    read_index_values,
)
from .t3_v_to_p_lag0_reduction_logic import (
    V_INDICES,
    P_INDICES,
    lag_profile_from_indices,
    summarize_lag_profile,
    composite_for_index,
    FIELD_BY_FAMILY,
)


def _merge_existing_tables(settings: T3VToPLag0ReductionSettings) -> pd.DataFrame:
    """Build a V1/stability enriched pair table for V->P diagnostics."""
    stab_path = settings.stability_dir / "tables" / "lead_lag_pair_summary_stability_judged.csv"
    if not stab_path.exists():
        raise FileNotFoundError(
            "Missing V1 stability judgement output. Run "
            "lead_lag_screen\\V1\\scripts\\run_lead_lag_screen_v1_stability_judgement.py first.\n"
            f"Missing: {stab_path}"
        )
    df = add_family_columns(normalize_v1_keys(read_csv_required(stab_path)))
    key = ["window", "source_variable", "target_variable"]

    nulls = add_family_columns(normalize_v1_keys(read_csv_required(settings.input_dir / "lead_lag_null_summary.csv")))
    null_keep = key + [
        c for c in [
            "T_pos_obs", "T_pos_null90", "T_pos_null95", "T_pos_null99", "p_pos_surrogate",
            "T_neg_obs", "T_neg_null90", "T_neg_null95", "T_neg_null99", "p_neg_surrogate",
            "T_0_obs", "T_0_null90", "T_0_null95", "T_0_null99", "p_0_surrogate",
            "S_pos", "S_neg", "S_0", "q_pos_within_window", "q_pos_global",
        ] if c in nulls.columns
    ]
    df = df.merge(nulls[null_keep], on=key, how="left", suffixes=("", "_null"))

    audit_path = settings.input_dir / "lead_lag_audit_surrogate_null_summary.csv"
    if audit_path.exists():
        audit = add_family_columns(normalize_v1_keys(read_csv_required(audit_path)))
        audit_keep = key + [
            c for c in ["T_pos_obs", "T_pos_null95", "p_pos_surrogate", "q_pos_within_window", "q_pos_global", "S_pos"]
            if c in audit.columns
        ]
        rename = {c: f"audit_{c}" for c in audit_keep if c not in key}
        df = df.merge(audit[audit_keep].rename(columns=rename), on=key, how="left")

    phi_path = settings.input_dir / "lead_lag_surrogate_ar1_params.csv"
    phi = read_csv_optional(phi_path)
    if phi is not None and {"window", "variable"}.issubset(phi.columns):
        src = phi.rename(columns={
            "variable": "source_variable",
            "phi_after_clip": "source_phi_after_clip",
            "raw_phi_before_clip": "source_raw_phi_before_clip",
            "phi_clip_severity": "source_phi_clip_severity",
        })
        tgt = phi.rename(columns={
            "variable": "target_variable",
            "phi_after_clip": "target_phi_after_clip",
            "raw_phi_before_clip": "target_raw_phi_before_clip",
            "phi_clip_severity": "target_phi_clip_severity",
        })
        # Avoid duplicate merge keys: after renaming, source_variable/target_variable
        # also start with source_/target_, and pandas merge fails when selected twice.
        src_feature_cols = [
            c for c in src.columns
            if c.startswith("source_") and c not in {"source_variable"}
        ]
        tgt_feature_cols = [
            c for c in tgt.columns
            if c.startswith("target_") and c not in {"target_variable"}
        ]
        src_keep = ["window", "source_variable"] + src_feature_cols
        tgt_keep = ["window", "target_variable"] + tgt_feature_cols
        df = df.merge(src.loc[:, src_keep], on=["window", "source_variable"], how="left")
        df = df.merge(tgt.loc[:, tgt_keep], on=["window", "target_variable"], how="left")

    df = add_family_columns(df)
    return df


def _v_to_p(df: pd.DataFrame, windows: tuple[str, ...]) -> pd.DataFrame:
    allowed = {w.upper() for w in windows}
    out = df[(df["source_family"].eq("V")) & (df["target_family"].eq("P"))].copy()
    out = out[out["window"].astype(str).str.upper().isin(allowed)].copy()
    return out


def _lag0_pos_null_pressure(vp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = vp.copy()
    for c in [
        "T_pos_obs", "T_pos_null95", "T_0_obs", "T_0_null95", "T_neg_obs", "T_neg_null95",
        "audit_T_pos_null95", "source_phi_after_clip", "target_phi_after_clip",
        "D_pos_0", "D_pos_0_CI90_low", "D_pos_0_CI90_high",
        "D_pos_neg", "D_pos_neg_CI90_low", "D_pos_neg_CI90_high",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["margin_pos"] = out.get("T_pos_obs") - out.get("T_pos_null95")
    out["margin_0"] = out.get("T_0_obs") - out.get("T_0_null95")
    out["audit_margin_pos"] = out.get("T_pos_obs") - out.get("audit_T_pos_null95") if "audit_T_pos_null95" in out.columns else np.nan
    out["pos_above_null95"] = out["margin_pos"] > 0
    out["lag0_above_null95"] = out["margin_0"] > 0

    recs = []
    for window, g in out.groupby("window", dropna=False):
        recs.append({
            "window": window,
            "n_pairs": int(len(g)),
            "mean_T_pos": float(g["T_pos_obs"].mean()),
            "mean_T_0": float(g["T_0_obs"].mean()),
            "mean_T_neg": float(g["T_neg_obs"].mean()),
            "mean_null95_pos": float(g["T_pos_null95"].mean()),
            "mean_null95_0": float(g["T_0_null95"].mean()),
            "mean_margin_pos": float(g["margin_pos"].mean()),
            "median_margin_pos": float(g["margin_pos"].median()),
            "mean_margin_0": float(g["margin_0"].mean()),
            "median_margin_0": float(g["margin_0"].median()),
            "n_pos_above_null95": int(g["pos_above_null95"].sum()),
            "n_lag0_above_null95": int(g["lag0_above_null95"].sum()),
            "mean_source_phi": float(pd.to_numeric(g.get("source_phi_after_clip"), errors="coerce").mean()),
            "mean_target_phi": float(pd.to_numeric(g.get("target_phi_after_clip"), errors="coerce").mean()),
            "n_stable_lag_dominant": int((g.get("v1_stability_judgement") == "stable_lag_dominant").sum()),
            "n_tau0_coupled": int((g.get("v1_stability_judgement") == "significant_lagged_but_tau0_coupled").sum()),
        })
    summary = pd.DataFrame(recs)
    order = {w: i for i, w in enumerate(WINDOW_DAY_RANGES)}
    if not summary.empty:
        summary["_ord"] = summary["window"].map(order).fillna(999)
        summary = summary.sort_values(["_ord", "window"]).drop(columns="_ord")
    return out, summary


def _subwindow_profiles(index_df: pd.DataFrame, settings: T3VToPLag0ReductionSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_parts = []
    summary_rows = []
    for sub_name, day_range in settings.focus_subwindows().items():
        for src in V_INDICES:
            for tgt in P_INDICES:
                prof = lag_profile_from_indices(index_df, src, tgt, day_range, settings.max_lag)
                prof.insert(0, "subwindow", sub_name)
                prof.insert(1, "day_start", day_range[0])
                prof.insert(2, "day_end", day_range[1])
                long_parts.append(prof)
                rec = summarize_lag_profile(prof)
                rec.update({
                    "subwindow": sub_name,
                    "day_start": day_range[0],
                    "day_end": day_range[1],
                    "source_variable": src,
                    "target_variable": tgt,
                })
                summary_rows.append(rec)
    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return long_df, summary_df


def _subwindow_shift_summary(sub_summary: pd.DataFrame) -> pd.DataFrame:
    if sub_summary.empty:
        return pd.DataFrame()
    pivot_cols = ["positive_peak_abs_r", "lag0_abs_r", "negative_peak_abs_r", "profile_type"]
    wide = sub_summary.pivot_table(
        index=["source_variable", "target_variable"],
        columns="subwindow",
        values=[c for c in pivot_cols if c != "profile_type"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    # profile_type cannot be pivoted numerically in pivot_table above; handle separately.
    prof_wide = sub_summary.pivot_table(index=["source_variable", "target_variable"], columns="subwindow", values="profile_type", aggfunc="first")
    prof_wide.columns = [f"profile_type_{c}" for c in prof_wide.columns]
    prof_wide = prof_wide.reset_index()
    out = wide.merge(prof_wide, on=["source_variable", "target_variable"], how="left")
    subs = list(sorted(sub_summary["subwindow"].unique()))
    if len(subs) >= 2:
        a, b = subs[0], subs[1]
        for metric in ["positive_peak_abs_r", "lag0_abs_r", "negative_peak_abs_r"]:
            ca = f"{metric}_{a}"
            cb = f"{metric}_{b}"
            if ca in out.columns and cb in out.columns:
                out[f"delta_{metric}_{b}_minus_{a}"] = pd.to_numeric(out[cb], errors="coerce") - pd.to_numeric(out[ca], errors="coerce")
        out["profile_type_changed"] = out.get(f"profile_type_{a}").astype(str) != out.get(f"profile_type_{b}").astype(str)
    return out


def _metric_rows_from_composites(comps: list[dict[str, object]], relation: str) -> pd.DataFrame:
    rows = []
    for c in comps:
        rows.append({
            "relation": relation,
            "index_name": c["index_name"],
            "field_key": c["field_key"],
            "n_total_samples": c["n_total_samples"],
            "n_high_samples": c["n_high_samples"],
            "n_low_samples": c["n_low_samples"],
            "high_threshold": c["high_threshold"],
            "low_threshold": c["low_threshold"],
            "diff_mean": c["diff_mean"],
            "diff_std": c["diff_std"],
            "diff_max_abs": c["diff_max_abs"],
            "zonal_profile_max_abs": c["zonal_profile_max_abs"],
            "meridional_profile_max_abs": c["meridional_profile_max_abs"],
        })
    return pd.DataFrame(rows)


def _plot_composite_png(comp: dict[str, object], path: Path, title: str, use_cartopy: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = np.asarray(comp["diff"], dtype=float)
        lat = np.asarray(comp["lat"], dtype=float)
        lon = np.asarray(comp["lon"], dtype=float)
        vmax = np.nanpercentile(np.abs(data), 98) if np.isfinite(data).any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        fig = plt.figure(figsize=(8, 5))
        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                ax = plt.axes(projection=ccrs.PlateCarree())
                im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
                ax.coastlines(linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))], crs=ccrs.PlateCarree())
                ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
            except Exception:
                ax = plt.axes()
                im = ax.pcolormesh(lon, lat, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
        else:
            ax = plt.axes()
            im = ax.pcolormesh(lon, lat, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="high-low diff")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return "ok"
    except Exception as exc:
        return f"plot_failed: {type(exc).__name__}: {exc}"


def _build_composites(index_df: pd.DataFrame, fields: dict[str, np.ndarray], settings: T3VToPLag0ReductionSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_range = settings.focus_days()
    comps: list[dict[str, object]] = []
    manifest_rows = []
    use_cartopy = (not settings.no_cartopy)

    # V index high/low on V field and P field.
    for idx in V_INDICES:
        for field_key, relation in [("v850", "V_index_to_V_field"), ("precip", "V_index_to_P_field")]:
            c = composite_for_index(index_df, fields, idx, field_key, focus_range, settings.high_quantile, settings.low_quantile, settings.map_extent)
            comps.append({**c, "relation": relation})
            if settings.make_figures:
                fig_name = f"{settings.focus_window}_{idx}_{relation}_composite.png".replace("/", "_")
                fig_path = settings.figure_dir / fig_name
                status = _plot_composite_png(c, fig_path, f"{settings.focus_window} {idx} high-low on {field_key}", use_cartopy)
                manifest_rows.append({"figure": str(fig_path.relative_to(settings.output_dir)), "index_name": idx, "field_key": field_key, "relation": relation, "status": status})

    # P component high/low on P field.
    for idx in P_INDICES:
        c = composite_for_index(index_df, fields, idx, "precip", focus_range, settings.high_quantile, settings.low_quantile, settings.map_extent)
        comps.append({**c, "relation": "P_index_to_P_field"})
        if settings.make_figures:
            fig_name = f"{settings.focus_window}_{idx}_P_index_to_P_field_composite.png".replace("/", "_")
            fig_path = settings.figure_dir / fig_name
            status = _plot_composite_png(c, fig_path, f"{settings.focus_window} {idx} high-low on precip", use_cartopy)
            manifest_rows.append({"figure": str(fig_path.relative_to(settings.output_dir)), "index_name": idx, "field_key": "precip", "relation": "P_index_to_P_field", "status": status})

    metric_frames = []
    for relation in sorted(set(c.get("relation") for c in comps)):
        metric_frames.append(_metric_rows_from_composites([c for c in comps if c.get("relation") == relation], str(relation)))
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    return metrics, manifest


def _index_validity_context(settings: T3VToPLag0ReductionSettings) -> pd.DataFrame:
    tables = settings.index_validity_tables_dir
    ind = read_csv_optional(tables / "index_window_representativeness.csv")
    joint = read_csv_optional(tables / "window_family_joint_field_coverage.csv")
    if ind is None:
        return pd.DataFrame()
    focus = settings.focus_window.upper()
    ind = ind[ind["window"].astype(str).str.upper().eq(focus)].copy()
    fam_col = "family" if "family" in ind.columns else "object_family"
    if fam_col in ind.columns:
        ind = ind[ind[fam_col].isin(["V", "P"])].copy()
    if joint is not None and "window" in joint.columns:
        joint = joint[joint["window"].astype(str).str.upper().eq(focus)].copy()
        jf = "family" if "family" in joint.columns else "object_family"
        keep = [c for c in ["window", jf, "joint_field_R2_year_cv", "joint_eof_coverage_top5_year_cv", "coverage_tier", "collapse_risk_update"] if c in joint.columns]
        if jf in joint.columns and keep:
            joint = joint[keep].rename(columns={jf: fam_col})
            ind = ind.merge(joint, on=["window", fam_col], how="left")
    return ind


def _reason_summary(vp_focus: pd.DataFrame, null_summary: pd.DataFrame, sub_shift: pd.DataFrame, composite_metrics: pd.DataFrame, iv_ctx: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    t3_row = null_summary[null_summary["window"].astype(str).str.upper().eq("T3")].head(1)
    if not t3_row.empty:
        r = t3_row.iloc[0]
        rows.append({
            "reason_candidate": "positive_lag_signal_near_ar1_null",
            "evidence_strength": "strong" if float(r.get("mean_margin_pos", np.nan)) < 0.05 else "moderate",
            "supporting_metrics": f"mean_T_pos={r.get('mean_T_pos'):.3f}; mean_null95_pos={r.get('mean_null95_pos'):.3f}; mean_margin_pos={r.get('mean_margin_pos'):.3f}; n_pos_above_null95={int(r.get('n_pos_above_null95'))}",
            "interpretation": "T3 positive-lag V→P index-pair signal is close to the AR(1) null threshold.",
        })
        rows.append({
            "reason_candidate": "lag0_signal_also_weakened",
            "evidence_strength": "strong" if int(r.get("n_lag0_above_null95", 999)) <= 10 else "moderate",
            "supporting_metrics": f"mean_T_0={r.get('mean_T_0'):.3f}; mean_null95_0={r.get('mean_null95_0'):.3f}; mean_margin_0={r.get('mean_margin_0'):.3f}; n_lag0_above_null95={int(r.get('n_lag0_above_null95'))}",
            "interpretation": "Lag0 does not systematically compensate for lost positive-lag relations in fixed V/P index pairs.",
        })

    if not vp_focus.empty:
        vstrength = vp_focus[vp_focus["source_variable"].eq("V_strength")]
        if not vstrength.empty:
            all_fail = int((~vstrength["v1_stability_judgement"].isin(["stable_lag_dominant", "significant_lagged_but_tau0_coupled"])).sum()) == len(vstrength)
            rows.append({
                "reason_candidate": "v_strength_decoupling_from_p",
                "evidence_strength": "strong" if all_fail else "moderate",
                "supporting_metrics": f"V_strength retained_pairs={int((vstrength['v1_stability_judgement'].isin(['stable_lag_dominant','significant_lagged_but_tau0_coupled'])).sum())}/{len(vstrength)}",
                "interpretation": "V_strength contributes little to retained T3 V→P fixed-index relations.",
            })
        vns = vp_focus[vp_focus["source_variable"].eq("V_NS_diff")]
        if not vns.empty:
            retained = int((vns["v1_stability_judgement"].isin(["stable_lag_dominant", "significant_lagged_but_tau0_coupled"])).sum())
            rows.append({
                "reason_candidate": "v_ns_diff_partial_retention",
                "evidence_strength": "strong" if retained >= 2 else "moderate" if retained else "weak",
                "supporting_metrics": f"V_NS_diff retained_pairs={retained}/{len(vns)}",
                "interpretation": "T3 V→P retention is concentrated more in V_NS_diff than in V_strength.",
            })

    if not sub_shift.empty:
        changed = int(pd.to_numeric(sub_shift.get("profile_type_changed"), errors="coerce").fillna(False).astype(bool).sum()) if "profile_type_changed" in sub_shift.columns else 0
        rows.append({
            "reason_candidate": "subwindow_state_mixing",
            "evidence_strength": "moderate" if changed >= 5 else "weak",
            "supporting_metrics": f"profile_type_changed_pairs={changed}/21",
            "interpretation": "Early/late T3 lag-profile changes may dilute full-window lag0 and positive-lag signals.",
        })

    if not composite_metrics.empty:
        rows.append({
            "reason_candidate": "composite_structure_check_available",
            "evidence_strength": "diagnostic",
            "supporting_metrics": f"n_composite_metrics={len(composite_metrics)}",
            "interpretation": "Composite maps/metrics were generated for V-index→V/P fields and P-index→P field checks.",
        })

    if not iv_ctx.empty:
        bad = iv_ctx[iv_ctx.astype(str).apply(lambda s: s.str.contains("high_risk|not_supported|collapse", case=False, na=False)).any(axis=1)]
        rows.append({
            "reason_candidate": "index_validity_not_failure",
            "evidence_strength": "strong" if bad.empty else "mixed",
            "supporting_metrics": f"index_validity_rows={len(iv_ctx)}; suspicious_rows={len(bad)}",
            "interpretation": "Existing index_validity context does not support V/P index-family collapse as the cause.",
        })
    return pd.DataFrame(rows)


def run_t3_v_to_p_lag0_reduction_audit(settings: T3VToPLag0ReductionSettings) -> dict[str, object]:
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    if settings.make_figures:
        settings.figure_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_existing_tables(settings)
    vp = _v_to_p(merged, settings.comparison_windows)
    focus = vp[vp["window"].astype(str).str.upper().eq(settings.focus_window.upper())].copy()

    null_detail, null_summary = _lag0_pos_null_pressure(vp)
    write_csv(null_detail, settings.table_dir / "v_to_p_window_null_difficulty_lag0_pos_detail.csv")
    write_csv(null_summary, settings.table_dir / "v_to_p_window_null_difficulty_lag0_pos_summary.csv")

    index_df = read_index_values(settings.index_values_path)
    sub_long, sub_summary = _subwindow_profiles(index_df, settings)
    sub_shift = _subwindow_shift_summary(sub_summary)
    write_csv(sub_long, settings.table_dir / "t3_v_to_p_subwindow_lag_profile_long.csv")
    write_csv(sub_summary, settings.table_dir / "t3_v_to_p_subwindow_lag_profile_summary.csv")
    write_csv(sub_shift, settings.table_dir / "t3_v_to_p_subwindow_shift_summary.csv")

    fields = load_smoothed_fields(settings.smoothed_fields_path)
    comp_metrics, fig_manifest = _build_composites(index_df, fields, settings)
    write_csv(comp_metrics, settings.table_dir / "t3_v_p_composite_metrics.csv")
    write_csv(fig_manifest, settings.table_dir / "figure_manifest.csv")

    iv_ctx = _index_validity_context(settings)
    write_csv(iv_ctx, settings.table_dir / "t3_v_to_p_index_validity_context_expanded.csv")

    reason_summary = _reason_summary(focus, null_summary, sub_shift, comp_metrics, iv_ctx)
    write_csv(reason_summary, settings.table_dir / "t3_v_to_p_lag0_and_lagged_reduction_reason_summary.csv")

    readme = f"""# T3 V→P lag0 and lagged reduction audit

This audit extends the earlier T3 V→P attrition audit. It asks why T3 loses many V→P fixed-index relations in both positive-lag and lag0 diagnostics.

It does not rerun the V1 main lead-lag screen. It adds:

1. T3 early/late subwindow raw lag profiles for all 21 V→P index pairs.
2. V-index high/low composites on V field and P field.
3. P-index high/low composites on P field.
4. lag0 and positive-lag null-pressure tables.
5. index_validity context and a reason summary.

Interpretation guardrail:
This is a diagnostic audit of why fixed-index V→P relations shrink in T3. It is not a new pathway result and does not prove physical mechanisms.
"""
    (settings.summary_dir / "T3_V_TO_P_LAG0_REDUCTION_AUDIT_README.md").write_text(readme, encoding="utf-8")

    t3_null = null_summary[null_summary["window"].astype(str).str.upper().eq(settings.focus_window.upper())].to_dict("records")
    summary = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(settings.input_dir),
        "stability_dir": str(settings.stability_dir),
        "previous_audit_dir": str(settings.previous_audit_dir),
        "foundation_dir": str(settings.foundation_dir),
        "output_dir": str(settings.output_dir),
        "focus_window": settings.focus_window,
        "focus_days": list(settings.focus_days()),
        "subwindows": settings.focus_subwindows(),
        "n_focus_v_to_p_pairs": int(len(focus)),
        "n_subwindow_lag_profile_rows": int(len(sub_long)),
        "n_composite_metric_rows": int(len(comp_metrics)),
        "n_figures": int(len(fig_manifest)),
        "t3_null_pressure_summary": t3_null[0] if t3_null else {},
        "reason_summary": reason_summary.to_dict("records"),
        "outputs": [
            "tables/v_to_p_window_null_difficulty_lag0_pos_detail.csv",
            "tables/v_to_p_window_null_difficulty_lag0_pos_summary.csv",
            "tables/t3_v_to_p_subwindow_lag_profile_long.csv",
            "tables/t3_v_to_p_subwindow_lag_profile_summary.csv",
            "tables/t3_v_to_p_subwindow_shift_summary.csv",
            "tables/t3_v_p_composite_metrics.csv",
            "tables/figure_manifest.csv",
            "tables/t3_v_to_p_index_validity_context_expanded.csv",
            "tables/t3_v_to_p_lag0_and_lagged_reduction_reason_summary.csv",
        ],
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta = {
        "settings": {
            "input_tag": settings.input_tag,
            "stability_tag": settings.stability_tag,
            "previous_audit_tag": settings.previous_audit_tag,
            "output_tag": settings.output_tag,
            "foundation_tag": settings.foundation_tag,
            "focus_window": settings.focus_window,
            "high_quantile": settings.high_quantile,
            "low_quantile": settings.low_quantile,
            "max_lag": settings.max_lag,
            "make_figures": settings.make_figures,
            "no_cartopy": settings.no_cartopy,
            "map_extent": list(settings.map_extent),
        },
        "required_inputs": [
            str(settings.stability_dir / "tables" / "lead_lag_pair_summary_stability_judged.csv"),
            str(settings.input_dir / "lead_lag_null_summary.csv"),
            str(settings.index_values_path),
            str(settings.smoothed_fields_path),
        ],
        "optional_inputs": [
            str(settings.input_dir / "lead_lag_audit_surrogate_null_summary.csv"),
            str(settings.input_dir / "lead_lag_surrogate_ar1_params.csv"),
            str(settings.index_validity_tables_dir / "index_window_representativeness.csv"),
            str(settings.index_validity_tables_dir / "window_family_joint_field_coverage.csv"),
        ],
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
