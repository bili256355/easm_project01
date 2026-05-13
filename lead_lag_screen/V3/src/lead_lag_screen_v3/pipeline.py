from __future__ import annotations

from datetime import datetime
import time
from typing import Dict

import numpy as np

from .comparison import build_v1_v3_comparisons
from .data_io import ensure_dirs, load_field_anomalies, load_index_anomalies, save_json
from .eof_pc1 import build_eof_pc1_panels
from .lead_lag_core import run_pc1_lead_lag
from .logging_utils import setup_logger
from .settings import LeadLagScreenV3Settings


def run_lead_lag_screen_v3(settings: LeadLagScreenV3Settings | None = None) -> Dict[str, object]:
    settings = settings or LeadLagScreenV3Settings()
    ensure_dirs(settings.output_dir, settings.log_dir)
    logger = setup_logger(settings.log_dir)
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    logger.info("Starting lead_lag_screen/V3 EOF-PC1 audit")
    logger.info("Output: %s", settings.output_dir)
    logger.info("Preprocess: %s", settings.preprocess_dir)
    save_json(settings.output_dir / "settings_summary.json", settings.to_jsonable())

    fields, lat, lon, years, field_meta = load_field_anomalies(settings)
    index_df = load_index_anomalies(settings.input_index_anomalies)
    logger.info("Loaded fields: %s", {k: v.shape for k, v in fields.items()})
    logger.info("Loaded years=%d lat=%d lon=%d index_rows=%d", len(years), len(lat), len(lon), len(index_df))

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

    logger.info("Running PC1 lead-lag screen")
    leadlag_outputs = run_pc1_lead_lag(panels=panels, years=years, settings=settings, logger=logger)

    logger.info("Building V1-vs-V3 comparison tables")
    comparisons = build_v1_v3_comparisons(
        v1_output_dir=settings.v1_output_dir,
        pc1_pair_summary=leadlag_outputs["eof_pc1_pair_summary"],
        pc1_quality=quality_df,
    )

    logger.info("Writing outputs")
    score_df.to_csv(settings.output_dir / "eof_pc1_scores_long.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(settings.output_dir / "eof_pc1_quality.csv", index=False, encoding="utf-8-sig")
    loadings_long.to_csv(settings.output_dir / "eof_pc1_loadings_long.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(settings.output_dir / "eof_pc1_loadings.npz", **loadings_npz)

    for name, df in leadlag_outputs.items():
        df.to_csv(settings.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, df in comparisons.items():
        df.to_csv(settings.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "output_tag": settings.output_tag,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 3),
        "field_input_meta": field_meta,
        "n_years": int(len(years)),
        "n_objects": int(len(settings.objects)),
        "n_windows": int(len(settings.windows)),
        "n_pc1_score_rows": int(len(score_df)),
        "n_pc1_quality_rows": int(len(quality_df)),
        "pc1_quality_flag_counts": quality_df["quality_flag"].value_counts(dropna=False).to_dict(),
        "n_pc1_pair_rows": int(len(leadlag_outputs["eof_pc1_pair_summary"])),
        "pc1_label_counts": leadlag_outputs["eof_pc1_pair_summary"]["pc1_lead_lag_label"].value_counts(dropna=False).to_dict(),
        "pc1_group_counts": leadlag_outputs["eof_pc1_pair_summary"]["pc1_lead_lag_group"].value_counts(dropna=False).to_dict(),
        "pc1_evidence_tier_counts": leadlag_outputs["eof_pc1_pair_summary"]["pc1_evidence_tier"].value_counts(dropna=False).to_dict(),
        "interpretation_scope": "field-mode EOF-PC1 audit of V1 index applicability; not pathway proof; not PCMCI; PC1 only",
        "primary_audit_focus": "T3/meiyu-ending index applicability; inspect v1_index_vs_v3_pc1_family_comparison.csv and t3_meiyu_end_pc1_audit.csv",
    }
    save_json(settings.output_dir / "summary.json", summary)
    save_json(settings.output_dir / "run_meta.json", {
        "layer": "lead_lag_screen",
        "version": "V3",
        "method": "window_wise_field_EOF_PC1_lead_lag_audit",
        "hard_boundaries": {
            "uses_indices_as_main_variables": False,
            "uses_only_PC1": True,
            "uses_pcmci": False,
            "does_pathway_inference": False,
            "changes_window_definition": False,
        },
        "settings": settings.to_jsonable(),
        "summary": summary,
    })
    logger.info("Finished lead_lag_screen/V3: %s", summary)
    return summary
