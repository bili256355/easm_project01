from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from .settings import LeadLagScreenSettings
from .stability_judgement_classifier import StabilityThresholds, classify_stability


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        lc = c.lower()
        if lc == "source":
            rename[c] = "source_variable"
        elif lc == "target":
            rename[c] = "target_variable"
    return out.rename(columns=rename)


def _index_kind(name: str, settings: LeadLagScreenSettings) -> str:
    return settings.index_type_map.get(str(name), "unknown")


def _stable_count(df: pd.DataFrame) -> int:
    if df.empty or "v1_stability_judgement" not in df.columns:
        return 0
    return int((df["v1_stability_judgement"] == "stable_lag_dominant").sum())


def _stability_postprocess(settings: LeadLagScreenSettings) -> pd.DataFrame:
    evidence = _normalize_keys(_read_csv(settings.output_dir / "lead_lag_evidence_tier_summary.csv"))
    robust = _normalize_keys(_read_csv(settings.output_dir / "lead_lag_directional_robustness.csv"))
    pair = _normalize_keys(_read_csv(settings.output_dir / "lead_lag_pair_summary.csv", required=False))
    key = ["window", "source_variable", "target_variable"]
    merged = evidence.merge(robust, on=key, how="left", suffixes=("", "_robustness"))
    if not pair.empty:
        keep = key + [c for c in pair.columns if c not in set(merged.columns) and c not in key]
        if len(keep) > len(key):
            merged = merged.merge(pair[keep], on=key, how="left")
    th = StabilityThresholds()
    bits = [classify_stability(row, th) for _, row in merged.iterrows()]
    classified = pd.concat([merged.reset_index(drop=True), pd.DataFrame(bits)], axis=1)
    classified["source_family"] = "V"
    classified["target_family"] = "P"
    classified["source_index_type"] = classified["source_variable"].map(lambda x: _index_kind(x, settings))
    classified["target_index_type"] = classified["target_variable"].map(lambda x: _index_kind(x, settings))
    classified.to_csv(settings.table_dir / "v1_1_lag_tau0_stability.csv", index=False, encoding="utf-8-sig")
    return classified


def _write_alias_tables(settings: LeadLagScreenSettings, classified: pd.DataFrame) -> None:
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    alias = {
        "lead_lag_curve_long.csv": "v1_1_v_to_p_lead_lag_long.csv",
        "lead_lag_pair_summary.csv": "v1_1_v_to_p_best_positive_lag.csv",
        "lead_lag_null_summary.csv": "v1_1_v_to_p_null_summary.csv",
        "lead_lag_directional_robustness.csv": "v1_1_v_to_p_directional_robustness.csv",
        "lead_lag_evidence_tier_summary.csv": "v1_1_v_to_p_evidence_tier_summary.csv",
        "lead_lag_audit_surrogate_null_summary.csv": "v1_1_v_to_p_audit_surrogate_null_summary.csv",
        "lead_lag_surrogate_ar1_params.csv": "v1_1_v_to_p_surrogate_ar1_params.csv",
        "lead_lag_surrogate_yearwise_scale.csv": "v1_1_v_to_p_surrogate_yearwise_scale.csv",
    }
    for src, dst in alias.items():
        p = settings.output_dir / src
        if p.exists():
            shutil.copy2(p, settings.table_dir / dst)
    preferred = [
        "window", "source_variable", "target_variable", "source_family", "target_family",
        "source_index_type", "target_index_type", "positive_peak_lag", "positive_peak_signed_r",
        "positive_peak_abs_r", "lag0_signed_r", "lag0_abs_r", "p_pos_surrogate",
        "q_pos_within_window", "q_pos_global", "evidence_tier", "lead_lag_label",
        "lead_lag_group", "lag_vs_tau0_label", "forward_vs_reverse_label",
        "v1_stability_judgement", "v1_stability_use_class",
        "v1_stability_interpretation_guardrail", "D_pos_0_used", "D_pos_neg_used",
        "same_day_coupling_detected", "failure_reason", "risk_note",
    ]
    classified[[c for c in preferred if c in classified.columns]].to_csv(
        settings.table_dir / "v1_1_v_to_p_classified_pairs.csv", index=False, encoding="utf-8-sig"
    )


def _recovery_tables(settings: LeadLagScreenSettings, classified: pd.DataFrame) -> None:
    prev = pd.DataFrame()
    prev_path = settings.previous_v1_stability_dir / "lead_lag_pair_summary_stability_judged.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, encoding="utf-8-sig")
        if "source_variable" not in prev.columns and "source" in prev.columns:
            prev = prev.rename(columns={"source": "source_variable", "target": "target_variable"})
    rows = []
    for w in settings.windows.keys():
        g = classified[classified["window"].astype(str).eq(w)]
        old_old = g[(g["source_index_type"] == "old_v1") & (g["target_index_type"] == "old_v1")]
        new_v_old_p = g[(g["source_index_type"] == "new_v1_1") & (g["target_index_type"] == "old_v1")]
        old_v_new_p = g[(g["source_index_type"] == "old_v1") & (g["target_index_type"] == "new_v1_1")]
        new_new = g[(g["source_index_type"] == "new_v1_1") & (g["target_index_type"] == "new_v1_1")]
        old_v1_total = np.nan; old_v1_stable = np.nan
        if not prev.empty:
            p = prev[prev["window"].astype(str).eq(w)].copy()
            if "source_family" in p and "target_family" in p:
                p = p[(p["source_family"].astype(str).eq("V")) & (p["target_family"].astype(str).eq("P"))]
            old_v1_total = int(len(p))
            if "v1_stability_judgement" in p.columns:
                old_v1_stable = int((p["v1_stability_judgement"] == "stable_lag_dominant").sum())
        rows.append({
            "window": w,
            "old_v1_vp_n_total": old_v1_total,
            "old_v1_vp_n_stable_lag": old_v1_stable,
            "v1_1_old_indices_n_total": int(len(old_old)),
            "v1_1_old_indices_n_stable_lag": _stable_count(old_old),
            "v1_1_newV_oldP_n_total": int(len(new_v_old_p)),
            "v1_1_newV_oldP_n_stable_lag": _stable_count(new_v_old_p),
            "v1_1_oldV_newP_n_total": int(len(old_v_new_p)),
            "v1_1_oldV_newP_n_stable_lag": _stable_count(old_v_new_p),
            "v1_1_newV_newP_n_total": int(len(new_new)),
            "v1_1_newV_newP_n_stable_lag": _stable_count(new_new),
            "v1_1_any_new_index_n_stable_lag": _stable_count(pd.concat([new_v_old_p, old_v_new_p, new_new], ignore_index=True)),
        })
    rec = pd.DataFrame(rows)
    rec["recovery_gain_from_new_indices_vs_v1_1_old_only"] = rec["v1_1_any_new_index_n_stable_lag"] - rec["v1_1_old_indices_n_stable_lag"]
    rec.to_csv(settings.table_dir / "v1_vs_v1_1_pair_recovery_summary.csv", index=False, encoding="utf-8-sig")
    contrib = classified[classified["v1_stability_judgement"].astype(str).eq("stable_lag_dominant")].copy()
    contrib["is_new_source"] = contrib["source_index_type"].eq("new_v1_1")
    contrib["is_new_target"] = contrib["target_index_type"].eq("new_v1_1")
    def tag(row):
        s = str(row.get("source_variable", "")); t = str(row.get("target_variable", "")); tags = []
        if "north_edge" in s or "band_width" in s or "high_minus_low" in s:
            tags.append("boundary_or_latband_v")
        if "highlat" in t: tags.append("highlat_p_target")
        if "south" in t or "scs" in t: tags.append("south_p_target")
        if "main" in t: tags.append("main_p_target")
        return ";".join(tags) if tags else "old_or_unlabeled"
    if not contrib.empty:
        contrib["interpretation_tag_derived_not_claim"] = contrib.apply(tag, axis=1)
    keep = ["window", "source_variable", "target_variable", "source_index_type", "target_index_type",
            "positive_peak_lag", "positive_peak_signed_r", "positive_peak_abs_r",
            "p_pos_surrogate", "q_pos_within_window", "v1_stability_judgement",
            "is_new_source", "is_new_target", "interpretation_tag_derived_not_claim"]
    contrib[[c for c in keep if c in contrib.columns]].to_csv(settings.table_dir / "v1_1_new_index_contribution_by_window.csv", index=False, encoding="utf-8-sig")


def build_v1_1_reports(settings: LeadLagScreenSettings) -> dict[str, object]:
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    classified = _stability_postprocess(settings)
    _write_alias_tables(settings, classified)
    _recovery_tables(settings, classified)
    counts = classified["v1_stability_judgement"].value_counts(dropna=False).to_dict()
    return {
        "status": "success",
        "n_classified_pairs": int(len(classified)),
        "stability_judgement_counts": {str(k): int(v) for k, v in counts.items()},
        "n_stable_lag_dominant": int((classified["v1_stability_judgement"] == "stable_lag_dominant").sum()),
        "classified_pairs": str(settings.table_dir / "v1_1_v_to_p_classified_pairs.csv"),
        "recovery_summary": str(settings.table_dir / "v1_vs_v1_1_pair_recovery_summary.csv"),
    }
