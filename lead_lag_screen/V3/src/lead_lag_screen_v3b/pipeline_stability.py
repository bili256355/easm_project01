from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
from typing import Dict

import numpy as np

from lead_lag_screen_v3.comparison import build_v1_v3_comparisons
from lead_lag_screen_v3.data_io import ensure_dirs, load_field_anomalies, load_index_anomalies, save_json
from lead_lag_screen_v3.eof_pc1 import build_eof_pc1_panels
from lead_lag_screen_v3.lead_lag_core import run_pc1_lead_lag
from lead_lag_screen_v3.logging_utils import setup_logger
from lead_lag_screen_v3.settings import LeadLagScreenV3Settings

from .settings_b import StabilitySettings, make_base_settings
from .stability_core import (
    attach_formal_stability_judgement,
    build_v1_v3b_comparison,
    compute_pc1_mode_stability,
    compute_relation_stability,
)


def _write_tables(output_dir: Path, tables: Dict[str, object]) -> None:
    for name, obj in tables.items():
        if hasattr(obj, "to_csv"):
            obj.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def run_lead_lag_screen_v3_b_stability(
    base_settings: LeadLagScreenV3Settings | None = None,
    stability_settings: StabilitySettings | None = None,
) -> Dict[str, object]:
    stability = stability_settings or StabilitySettings()
    settings = make_base_settings(base_settings, output_tag=stability.output_tag)
    ensure_dirs(settings.output_dir, settings.log_dir)
    logger = setup_logger(settings.log_dir, "lead_lag_screen_v3_b_stability")
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    logger.info("Starting lead_lag_screen/V3_b EOF-PC1 stability-judged audit")
    logger.info("Output: %s", settings.output_dir)
    logger.info("Preprocess: %s", settings.preprocess_dir)
    save_json(settings.output_dir / "settings_summary.json", {
        "base_v3_settings": settings.to_jsonable(),
        "stability_settings": stability.to_jsonable(),
    })

    rng = np.random.default_rng(int(settings.random_seed + stability.random_seed_offset))

    fields, lat, lon, years, field_meta = load_field_anomalies(settings)
    index_df = load_index_anomalies(settings.input_index_anomalies)
    logger.info("Loaded fields: %s", {k: v.shape for k, v in fields.items()})
    logger.info("Loaded years=%d lat=%d lon=%d index_rows=%d", len(years), len(lat), len(lon), len(index_df))

    logger.info("Fitting window-wise EOF-PC1 panels")
    score_df, quality_df, loadings_long, aux = build_eof_pc1_panels(
        fields=fields,
        lat=lat,
        lon=lon,
        years=years,
        index_df=index_df,
        settings=settings,
        logger=logger,
    )
    panels = aux["panels"]
    loadings_npz = aux["loadings_npz"]

    logger.info("Running inherited V3_a PC1 lead-lag screen as statistical base")
    leadlag_outputs = run_pc1_lead_lag(panels=panels, years=years, settings=settings, logger=logger)

    logger.info("Computing formal PC1 mode stability")
    mode_stability = compute_pc1_mode_stability(
        fields=fields,
        lat=lat,
        lon=lon,
        years=years,
        score_df=score_df,
        quality_df=quality_df,
        loadings_long=loadings_long,
        base_settings=settings,
        stability=stability,
        rng=rng,
        logger=logger,
    )

    logger.info("Computing formal relation stability: lag-vs-tau0, forward-vs-reverse, peak-lag")
    rel_stab = compute_relation_stability(
        panels=panels,
        years=years,
        loadings_npz=loadings_npz,
        base_settings=settings,
        stability=stability,
        rng=rng,
        logger=logger,
    )

    logger.info("Attaching final stability judgement")
    judged = attach_formal_stability_judgement(
        pair_summary=leadlag_outputs["eof_pc1_pair_summary"],
        mode_stability=mode_stability,
        lag_tau0=rel_stab["eof_pc1_lag_vs_tau0_stability"],
        fwd_rev=rel_stab["eof_pc1_forward_reverse_stability"],
        peak=rel_stab["eof_pc1_peak_lag_stability"],
        stability=stability,
    )

    logger.info("Building V1-vs-V3_b comparison table")
    v1_v3b = build_v1_v3b_comparison(settings.v1_output_dir, judged)
    t3_audit = judged[judged["window"] == "T3"].copy()

    logger.info("Writing outputs")
    score_df.to_csv(settings.output_dir / "eof_pc1_scores_long.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(settings.output_dir / "eof_pc1_quality.csv", index=False, encoding="utf-8-sig")
    loadings_long.to_csv(settings.output_dir / "eof_pc1_loadings_long.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(settings.output_dir / "eof_pc1_loadings.npz", **loadings_npz)

    _write_tables(settings.output_dir, leadlag_outputs)
    mode_stability.to_csv(settings.output_dir / "eof_pc1_mode_stability.csv", index=False, encoding="utf-8-sig")
    _write_tables(settings.output_dir, rel_stab)
    judged.to_csv(settings.output_dir / "eof_pc1_pair_summary_stability_judged.csv", index=False, encoding="utf-8-sig")
    if not v1_v3b.empty:
        v1_v3b.to_csv(settings.output_dir / "v1_index_vs_v3b_pc1_stability_comparison.csv", index=False, encoding="utf-8-sig")
    t3_audit.to_csv(settings.output_dir / "t3_meiyu_end_pc1_stability_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "output_tag": settings.output_tag,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 3),
        "field_input_meta": field_meta,
        "n_years": int(len(years)),
        "n_windows": int(len(settings.windows)),
        "n_objects": int(len(settings.objects)),
        "n_pc1_pair_rows": int(len(judged)),
        "mode_stability_counts": mode_stability["pc1_mode_stability_label"].value_counts(dropna=False).to_dict(),
        "lag_vs_tau0_counts": rel_stab["eof_pc1_lag_vs_tau0_stability"]["lag_vs_tau0_label"].value_counts(dropna=False).to_dict(),
        "forward_reverse_counts": rel_stab["eof_pc1_forward_reverse_stability"]["direction_vs_reverse_label"].value_counts(dropna=False).to_dict(),
        "peak_lag_stability_counts": rel_stab["eof_pc1_peak_lag_stability"]["peak_lag_stability_label"].value_counts(dropna=False).to_dict(),
        "stability_judgement_counts": judged["stability_judgement"].value_counts(dropna=False).to_dict(),
        "interpretation_scope": "formal EOF-PC1 lead-lag stability judgement; V3_a exploratory labels are not sufficient for formal lead-lag claims",
        "primary_new_gates": [
            "pc1_mode_stability",
            "lag_vs_tau0_stability",
            "forward_vs_reverse_stability",
            "peak_lag_stability",
            "final_stability_judgement",
        ],
        "hard_boundaries": {
            "does_pathway_inference": False,
            "does_causal_inference": False,
            "uses_only_PC1": True,
            "does_not_treat_positive_lag_significance_as_stable_lag": True,
        },
    }
    save_json(settings.output_dir / "summary.json", summary)
    save_json(settings.output_dir / "run_meta.json", {
        "layer": "lead_lag_screen",
        "version": "V3_b",
        "method": "window_wise_field_EOF_PC1_lead_lag_with_formal_stability_judgement",
        "base_v3a_method_reused": True,
        "settings": {
            "base_v3_settings": settings.to_jsonable(),
            "stability_settings": stability.to_jsonable(),
        },
        "summary": summary,
    })
    logger.info("Finished lead_lag_screen/V3_b: %s", summary)
    return summary
