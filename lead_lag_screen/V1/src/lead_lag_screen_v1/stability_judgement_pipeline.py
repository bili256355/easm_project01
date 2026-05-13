from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pandas as pd

from .stability_judgement_settings import V1StabilityJudgementSettings
from .stability_judgement_io import read_first_existing, normalize_keys, write_csv, family_from_variable
from .stability_judgement_classifier import StabilityThresholds, classify_stability


def _load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    evidence = read_first_existing(input_dir, [
        "lead_lag_evidence_tier_summary.csv",
        "lead_lag_pair_summary.csv",
    ], required=True)
    robustness = read_first_existing(input_dir, [
        "lead_lag_directional_robustness.csv",
        "directional_robustness.csv",
    ], required=True)
    pair = read_first_existing(input_dir, [
        "lead_lag_pair_summary.csv",
    ], required=False)
    return evidence, robustness, pair


def _merge_inputs(evidence: pd.DataFrame, robustness: pd.DataFrame, pair: pd.DataFrame | None) -> pd.DataFrame:
    ev = normalize_keys(evidence)
    rb = normalize_keys(robustness)
    key = ["window", "source_variable", "target_variable"]

    merged = ev.merge(rb, on=key, how="left", suffixes=("", "_robustness"))

    if pair is not None:
        pr = normalize_keys(pair)
        keep = key + [c for c in pr.columns if c not in set(merged.columns) and c not in key]
        if len(keep) > len(key):
            merged = merged.merge(pr[keep], on=key, how="left", suffixes=("", "_pair"))

    return merged


def _add_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_family"] = out["source_variable"].map(family_from_variable)
    out["target_family"] = out["target_variable"].map(family_from_variable)
    out["family_direction"] = out["source_family"] + "→" + out["target_family"]
    return out


def _rollups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = _add_family_columns(df)
    tables: dict[str, pd.DataFrame] = {}

    tables["window_stability_rollup"] = (
        d.groupby(["window", "v1_stability_judgement"], dropna=False)
        .size().reset_index(name="n_pairs")
        .sort_values(["window", "v1_stability_judgement"])
    )

    tables["window_family_stability_rollup"] = (
        d.groupby(["window", "family_direction", "v1_stability_judgement"], dropna=False)
        .size().reset_index(name="n_pairs")
        .sort_values(["window", "family_direction", "v1_stability_judgement"])
    )

    tier_col = "evidence_tier" if "evidence_tier" in d.columns else "evidence_tier_prefix"
    tables["evidence_tier_vs_stability_rollup"] = (
        d.groupby([tier_col, "v1_stability_judgement"], dropna=False)
        .size().reset_index(name="n_pairs")
        .sort_values([tier_col, "v1_stability_judgement"])
    )

    tables["lag_vs_tau0_stability_rollup"] = (
        d.groupby(["window", "lag_vs_tau0_label"], dropna=False)
        .size().reset_index(name="n_pairs")
        .sort_values(["window", "lag_vs_tau0_label"])
    )

    tables["t3_stability_audit"] = d[d["window"].astype(str).str.upper().eq("T3")].copy()

    tables["v1_core_candidate_pool_stability_judged"] = d[d["v1_stability_judgement"].isin([
        "stable_lag_dominant",
        "significant_lagged_but_tau0_coupled",
        "stable_tau0_dominant_coupling",
    ])].copy()

    return tables


def run_v1_stability_judgement(settings: V1StabilityJudgementSettings) -> dict[str, object]:
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)

    evidence, robustness, pair = _load_inputs(settings.input_dir)
    merged = _merge_inputs(evidence, robustness, pair)

    thresholds = StabilityThresholds(
        ci_level=settings.ci_level,
        p_lag_gt_tau0_threshold=settings.p_lag_gt_tau0_threshold,
        p_forward_gt_reverse_threshold=settings.p_forward_gt_reverse_threshold,
        core_evidence_prefixes=settings.core_evidence_prefixes,
        sensitive_evidence_prefixes=settings.sensitive_evidence_prefixes,
    )
    classified_bits = [classify_stability(row, thresholds) for _, row in merged.iterrows()]
    classified = pd.concat([merged.reset_index(drop=True), pd.DataFrame(classified_bits)], axis=1)
    classified = _add_family_columns(classified)

    write_csv(classified, settings.table_dir / "lead_lag_pair_summary_stability_judged.csv")
    for name, table in _rollups(classified).items():
        write_csv(table, settings.table_dir / f"{name}.csv")

    counts = classified["v1_stability_judgement"].value_counts(dropna=False).to_dict()
    use_counts = classified["v1_stability_use_class"].value_counts(dropna=False).to_dict()
    lag_counts = classified["lag_vs_tau0_label"].value_counts(dropna=False).to_dict()

    summary = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(settings.input_dir),
        "output_dir": str(settings.output_dir),
        "n_rows": int(len(classified)),
        "stability_judgement_counts": {str(k): int(v) for k, v in counts.items()},
        "use_class_counts": {str(k): int(v) for k, v in use_counts.items()},
        "lag_vs_tau0_label_counts": {str(k): int(v) for k, v in lag_counts.items()},
        "interpretation": (
            "V1 stability judgement is a post-processing layer. It formalizes lag-vs-tau0 and "
            "forward-vs-reverse stability using existing V1 diagnostics. It does not rerun correlations, "
            "surrogates, or establish causality/pathways."
        ),
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    run_meta = {
        "settings": {
            "input_tag": settings.input_tag,
            "output_tag": settings.output_tag,
            "ci_level": settings.ci_level,
            "p_lag_gt_tau0_threshold": settings.p_lag_gt_tau0_threshold,
            "p_forward_gt_reverse_threshold": settings.p_forward_gt_reverse_threshold,
            "core_evidence_prefixes": list(settings.core_evidence_prefixes),
            "sensitive_evidence_prefixes": list(settings.sensitive_evidence_prefixes),
        },
        "required_inputs": [
            "lead_lag_evidence_tier_summary.csv",
            "lead_lag_directional_robustness.csv",
            "lead_lag_pair_summary.csv (optional enrichment)",
        ],
        "outputs": [
            "tables/lead_lag_pair_summary_stability_judged.csv",
            "tables/window_stability_rollup.csv",
            "tables/window_family_stability_rollup.csv",
            "tables/evidence_tier_vs_stability_rollup.csv",
            "tables/lag_vs_tau0_stability_rollup.csv",
            "tables/t3_stability_audit.csv",
            "tables/v1_core_candidate_pool_stability_judged.csv",
        ],
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
