from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .core import run_screen
from .logging_utils import setup_logger
from .reporting import build_v1_1_reports
from .settings import LeadLagScreenSettings
from .structural_indices import compute_v1_1_structural_indices


ROOT_OUTPUT_TAG = "lead_lag_screen_v1_1_t3_window_length_sensitivity_a"


def _variant_registry() -> List[dict]:
    """
    Window-length sensitivity variants.

    Every variant is a single target-side window run under the same V1_1 V→P
    test framework.  The equal-length controls test whether an 11-day window
    alone can explain the low T3 pair count.  The T3 expansions test whether
    relationships recover when the transition window is widened forward,
    backward, or symmetrically.
    """
    return [
        {
            "variant": "T3_current_107_117_len11",
            "family": "T3_current",
            "comparison_role": "baseline_T3_current",
            "window_name": "T3_current",
            "start_day": 107,
            "end_day": 117,
            "reference": "current V1_1 T3 window",
        },
        {
            "variant": "S3_center_equal11_092_102",
            "family": "equal_length_control",
            "comparison_role": "S3_11day_control",
            "window_name": "S3_equal11",
            "start_day": 92,
            "end_day": 102,
            "reference": "11-day control centered within S3",
        },
        {
            "variant": "S4_center_equal11_131_141",
            "family": "equal_length_control",
            "comparison_role": "S4_11day_control",
            "window_name": "S4_equal11",
            "start_day": 131,
            "end_day": 141,
            "reference": "11-day control centered within S4",
        },
        {
            "variant": "T3_expand_symmetric17_104_120",
            "family": "T3_expansion",
            "comparison_role": "T3_symmetric_expansion_17day",
            "window_name": "T3_expand17",
            "start_day": 104,
            "end_day": 120,
            "reference": "T3 expanded 3 days backward and forward",
        },
        {
            "variant": "T3_expand_symmetric23_101_123",
            "family": "T3_expansion",
            "comparison_role": "T3_symmetric_expansion_23day",
            "window_name": "T3_expand23",
            "start_day": 101,
            "end_day": 123,
            "reference": "T3 expanded 6 days backward and forward",
        },
        {
            "variant": "T3_expand_backward17_101_117",
            "family": "T3_expansion",
            "comparison_role": "T3_backward_expansion_17day",
            "window_name": "T3_back17",
            "start_day": 101,
            "end_day": 117,
            "reference": "T3 expanded only backward toward S3",
        },
        {
            "variant": "T3_expand_forward17_107_123",
            "family": "T3_expansion",
            "comparison_role": "T3_forward_expansion_17day",
            "window_name": "T3_forw17",
            "start_day": 107,
            "end_day": 123,
            "reference": "T3 expanded only forward toward S4",
        },
    ]


def _prepare_base_indices(settings: LeadLagScreenSettings, logger) -> dict:
    """Ensure the main V1_1 structural index files exist and return metadata."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    if settings.generated_index_anomalies.exists() and settings.generated_index_raw.exists():
        logger.info("Reusing existing V1_1 structural index files from %s", settings.index_dir)
        return {
            "status": "reused_existing",
            "index_anomalies": str(settings.generated_index_anomalies),
            "index_raw": str(settings.generated_index_raw),
        }
    logger.info("Base V1_1 structural index files are missing; computing them once")
    return compute_v1_1_structural_indices(settings, logger=logger)


def _copy_index_inputs(src_settings: LeadLagScreenSettings, dst_settings: LeadLagScreenSettings) -> None:
    """Copy V1_1 generated index files into a variant run directory.

    The inherited V1_1 settings expose input_index_anomalies as a property tied
    to each run's output dir.  We copy the same generated index table into each
    variant so run_screen can be reused without changing V1 or the base V1_1
    pipeline.
    """
    dst_settings.index_dir.mkdir(parents=True, exist_ok=True)
    copies = [
        (src_settings.generated_index_anomalies, dst_settings.generated_index_anomalies),
        (src_settings.generated_index_raw, dst_settings.generated_index_raw),
        (src_settings.index_dir / "v1_1_index_registry.csv", dst_settings.index_dir / "v1_1_index_registry.csv"),
        (src_settings.index_dir / "v1_1_index_quality_flags.csv", dst_settings.index_dir / "v1_1_index_quality_flags.csv"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def _pair_type(row: pd.Series) -> str:
    src = str(row.get("source_index_type", ""))
    tgt = str(row.get("target_index_type", ""))
    if src == "old_v1" and tgt == "old_v1":
        return "oldV_oldP"
    if src == "new_v1_1" and tgt == "old_v1":
        return "newV_oldP"
    if src == "old_v1" and tgt == "new_v1_1":
        return "oldV_newP"
    if src == "new_v1_1" and tgt == "new_v1_1":
        return "newV_newP"
    return "unknown"


def _is_stable(df: pd.DataFrame) -> pd.Series:
    if "v1_stability_judgement" in df.columns:
        return df["v1_stability_judgement"].astype(str).eq("stable_lag_dominant")
    return pd.Series(False, index=df.index)


def _highlat_structural_mask(df: pd.DataFrame) -> pd.Series:
    src = df.get("source_variable", pd.Series("", index=df.index)).astype(str)
    tgt = df.get("target_variable", pd.Series("", index=df.index)).astype(str)
    src_hit = src.str.contains("north_edge|band_width|highlat_35_55|high_minus_low|centroid_lat_recomputed", regex=True)
    tgt_hit = tgt.str.contains("P_highlat_40_60|P_highlat_35_60", regex=True)
    return src_hit & tgt_hit


def _summarize_classified(variant_meta: dict, classified: pd.DataFrame) -> dict:
    stable = _is_stable(classified)
    c = classified.copy()
    c["pair_type"] = c.apply(_pair_type, axis=1)
    duration = int(variant_meta["end_day"] - variant_meta["start_day"] + 1)

    out = {
        "variant": variant_meta["variant"],
        "family": variant_meta["family"],
        "comparison_role": variant_meta["comparison_role"],
        "window_name": variant_meta["window_name"],
        "start_day": int(variant_meta["start_day"]),
        "end_day": int(variant_meta["end_day"]),
        "duration_days": duration,
        "reference": variant_meta["reference"],
        "n_pairs_total": int(len(c)),
        "n_stable_total": int(stable.sum()),
        "n_highlat_structural_stable": int((stable & _highlat_structural_mask(c)).sum()),
    }
    for pt in ["oldV_oldP", "newV_oldP", "oldV_newP", "newV_newP", "unknown"]:
        m = c["pair_type"].eq(pt)
        out[f"{pt}_n_total"] = int(m.sum())
        out[f"{pt}_n_stable"] = int((m & stable).sum())
    out["any_new_index_n_stable"] = int((stable & ~c["pair_type"].eq("oldV_oldP")).sum())
    if "v1_stability_judgement" in c.columns:
        for label, n in c["v1_stability_judgement"].astype(str).value_counts(dropna=False).items():
            out[f"judgement_count__{label}"] = int(n)
    return out


def _effect_size_summary(variant_meta: dict, classified: pd.DataFrame) -> pd.DataFrame:
    c = classified.copy()
    c["pair_type"] = c.apply(_pair_type, axis=1)
    stable = _is_stable(c)
    rows = []
    groups = ["all", "oldV_oldP", "newV_oldP", "oldV_newP", "newV_newP", "any_new"]
    for group in groups:
        if group == "all":
            g = c
        elif group == "any_new":
            g = c[~c["pair_type"].eq("oldV_oldP")]
        else:
            g = c[c["pair_type"].eq(group)]
        if g.empty:
            vals = pd.Series(dtype=float)
            lag0 = pd.Series(dtype=float)
        else:
            vals = pd.to_numeric(g.get("positive_peak_abs_r", np.nan), errors="coerce")
            lag0 = pd.to_numeric(g.get("lag0_abs_r", np.nan), errors="coerce")
        rows.append({
            "variant": variant_meta["variant"],
            "family": variant_meta["family"],
            "comparison_role": variant_meta["comparison_role"],
            "window_name": variant_meta["window_name"],
            "start_day": int(variant_meta["start_day"]),
            "end_day": int(variant_meta["end_day"]),
            "duration_days": int(variant_meta["end_day"] - variant_meta["start_day"] + 1),
            "pair_group": group,
            "n_pairs": int(len(g)),
            "n_stable": int((_is_stable(g)).sum()) if not g.empty else 0,
            "positive_abs_r_mean": float(vals.mean()) if len(vals) else np.nan,
            "positive_abs_r_median": float(vals.median()) if len(vals) else np.nan,
            "positive_abs_r_q75": float(vals.quantile(0.75)) if len(vals) else np.nan,
            "positive_abs_r_q90": float(vals.quantile(0.90)) if len(vals) else np.nan,
            "positive_abs_r_max": float(vals.max()) if len(vals) else np.nan,
            "lag0_abs_r_median": float(lag0.median()) if len(lag0) else np.nan,
            "lag0_abs_r_q90": float(lag0.quantile(0.90)) if len(lag0) else np.nan,
        })
    return pd.DataFrame(rows)


def _read_classified(settings: LeadLagScreenSettings) -> pd.DataFrame:
    path = settings.table_dir / "v1_1_v_to_p_classified_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_window_registry(root_dir: Path, variants: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(variants).copy()
    df["duration_days"] = df["end_day"] - df["start_day"] + 1
    out = root_dir / "summary" / "window_length_sensitivity_registry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def _build_interpretation_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Small decision table; it is diagnostic, not a physical conclusion."""
    rows = []
    base = counts[counts["variant"].eq("T3_current_107_117_len11")]
    if base.empty:
        return pd.DataFrame()
    base = base.iloc[0]
    s3 = counts[counts["variant"].eq("S3_center_equal11_092_102")]
    s4 = counts[counts["variant"].eq("S4_center_equal11_131_141")]
    exp = counts[counts["family"].eq("T3_expansion")]

    base_stable = int(base.get("n_stable_total", 0))
    base_newnew = int(base.get("newV_newP_n_stable", 0))
    base_oldold = int(base.get("oldV_oldP_n_stable", 0))

    if not s3.empty and not s4.empty:
        s3_stable = int(s3.iloc[0].get("n_stable_total", 0))
        s4_stable = int(s4.iloc[0].get("n_stable_total", 0))
        if base_stable < min(s3_stable, s4_stable):
            support = "supports_T3_specific_low_count_beyond_equal_length_alone"
            allowed = "The current 11-day T3 has fewer stable V→P pairs than equal-length S3/S4 controls, so short length alone is unlikely to fully explain the T3 drop."
        else:
            support = "compatible_with_short_window_power_loss"
            allowed = "Equal-length controls also lose many pairs; window length may account for a large part of the T3 drop."
        rows.append({
            "diagnosis_id": "equal_length_control",
            "support_level": support,
            "primary_evidence": f"T3_current stable={base_stable}; S3_equal11 stable={s3_stable}; S4_equal11 stable={s4_stable}",
            "allowed_statement": allowed,
            "forbidden_statement": "Do not claim T3 pair loss is only due to missing indices before checking equal-length controls.",
        })

    if not exp.empty:
        max_exp = int(exp["n_stable_total"].max())
        best = exp.loc[exp["n_stable_total"].idxmax()]
        gain = max_exp - base_stable
        if gain > 0:
            support = "expansion_recovers_additional_pairs"
            allowed = f"Expanding T3 recovers additional stable pairs; best variant is {best['variant']} with stable={max_exp}, gain={gain}."
        else:
            support = "expansion_does_not_recover_pairs"
            allowed = "Expanding T3 does not recover additional stable pairs; window length/edge placement is not the main reason for low pair count under this audit."
        rows.append({
            "diagnosis_id": "t3_expansion_recovery",
            "support_level": support,
            "primary_evidence": f"T3_current stable={base_stable}; max expanded stable={max_exp}; best={best['variant']}",
            "allowed_statement": allowed,
            "forbidden_statement": "Do not treat expansion recovery as proof of physical causality; it is a window-sensitivity result.",
        })

        exp_oldold = int(exp["oldV_oldP_n_stable"].max())
        exp_newnew = int(exp["newV_newP_n_stable"].max())
        if exp_newnew >= exp_oldold and base_newnew > base_oldold:
            support = "index_projection_mismatch_remains_visible"
            allowed = "Recovered T3 relationships remain concentrated in newV-newP structural pairs, so index projection mismatch remains a supported component."
        else:
            support = "old_index_recovery_with_expansion_possible"
            allowed = "Expanded windows recover old-index pairs; part of the T3 drop may be due to short/edge placement rather than only index projection mismatch."
        rows.append({
            "diagnosis_id": "old_vs_new_recovery_pattern",
            "support_level": support,
            "primary_evidence": f"base old-old={base_oldold}, base new-new={base_newnew}; expanded max old-old={exp_oldold}, expanded max new-new={exp_newnew}",
            "allowed_statement": allowed,
            "forbidden_statement": "Do not summarize this audit only as 'T3 is too short' without checking whether recovered pairs are old-index or new-index pairs.",
        })
    return pd.DataFrame(rows)


def run_v1_1_t3_window_length_sensitivity_a(settings: LeadLagScreenSettings | None = None) -> dict:
    settings = settings or LeadLagScreenSettings()
    root_settings = replace(settings, output_tag=ROOT_OUTPUT_TAG)
    root_dir = root_settings.output_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "tables").mkdir(parents=True, exist_ok=True)
    (root_dir / "summary").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(root_settings.log_dir)
    logger.info("Starting V1_1 T3 window-length sensitivity audit")
    logger.info("Root output dir: %s", root_dir)

    base_index_meta = _prepare_base_indices(settings, logger)
    variants = _variant_registry()
    registry_df = _write_window_registry(root_dir, variants)

    count_rows: List[dict] = []
    effect_tables: List[pd.DataFrame] = []
    subrun_summaries: Dict[str, dict] = {}

    for i, meta in enumerate(variants, start=1):
        variant = meta["variant"]
        logger.info("[%d/%d] Running sensitivity variant %s (%d-%d)", i, len(variants), variant, meta["start_day"], meta["end_day"])
        v_settings = replace(
            settings,
            output_tag=f"{ROOT_OUTPUT_TAG}/{variant}",
            windows={meta["window_name"]: (int(meta["start_day"]), int(meta["end_day"]))},
            random_seed=int(settings.random_seed + i * 101),
        )
        v_settings.output_dir.mkdir(parents=True, exist_ok=True)
        v_settings.index_dir.mkdir(parents=True, exist_ok=True)
        v_settings.table_dir.mkdir(parents=True, exist_ok=True)
        v_settings.summary_dir.mkdir(parents=True, exist_ok=True)
        _copy_index_inputs(settings, v_settings)

        main_summary = run_screen(v_settings, logger)
        report_summary = build_v1_1_reports(v_settings)
        classified = _read_classified(v_settings)

        row = _summarize_classified(meta, classified)
        count_rows.append(row)
        effect_tables.append(_effect_size_summary(meta, classified))
        subrun_summaries[variant] = {
            "output_dir": str(v_settings.output_dir),
            "main_summary": main_summary,
            "report_summary": report_summary,
            "n_classified_pairs": int(len(classified)),
        }

    counts = pd.DataFrame(count_rows)
    effects = pd.concat(effect_tables, ignore_index=True) if effect_tables else pd.DataFrame()

    counts_path = root_dir / "tables" / "window_length_sensitivity_pair_counts.csv"
    effects_path = root_dir / "tables" / "window_length_sensitivity_effect_size.csv"
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    effects.to_csv(effects_path, index=False, encoding="utf-8-sig")

    t3_exp = counts[counts["variant"].str.startswith("T3_")].copy()
    t3_path = root_dir / "tables" / "t3_expansion_recovery_summary.csv"
    t3_exp.to_csv(t3_path, index=False, encoding="utf-8-sig")

    diagnosis = _build_interpretation_table(counts)
    diagnosis_path = root_dir / "tables" / "window_length_sensitivity_diagnosis_table.csv"
    diagnosis.to_csv(diagnosis_path, index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "audit": "v1_1_t3_window_length_sensitivity_a",
        "root_output_dir": str(root_dir),
        "base_index_meta": base_index_meta,
        "n_variants": int(len(variants)),
        "registry": str(root_dir / "summary" / "window_length_sensitivity_registry.csv"),
        "pair_counts": str(counts_path),
        "effect_size": str(effects_path),
        "t3_expansion_recovery": str(t3_path),
        "diagnosis_table": str(diagnosis_path),
        "subruns": subrun_summaries,
        "primary_question": "Whether the low T3 V→P stable-pair count is mainly a short-window artifact or remains concentrated in structural newV-newP recovery.",
        "guardrails": [
            "This audit does not change V1 or V1_1 main outputs.",
            "Each variant is a one-window V→P rerun under the same V1_1 test framework.",
            "Interpretation is limited to window-length and edge-placement sensitivity, not physical causality.",
        ],
    }
    (root_dir / "summary" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "settings": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items() if k not in {"windows"}},
            "base_windows": settings.windows,
            "sensitivity_variants": variants,
        },
        "method_contract": {
            "v1_readonly": True,
            "v1_1_main_outputs_not_modified": True,
            "index_inputs": "reuse/copy V1_1 generated structural anomaly indices",
            "screen_core": "same V1_1 run_screen + build_v1_1_reports",
            "main_direction": "V→P only",
            "target_side_windows": True,
            "year_pairing": "same-year only",
        },
    }
    (root_dir / "summary" / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished V1_1 T3 window-length sensitivity audit")
    return summary
