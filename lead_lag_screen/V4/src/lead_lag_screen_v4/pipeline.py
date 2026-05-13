from __future__ import annotations

from datetime import datetime
import time
from typing import Dict

from .cca_core import build_score_panels, run_cca_lags, summarize_cca
from .comparison import build_t3_cca_audit, build_v1_v3_v4_comparison
from .data_io import ensure_dirs, load_field_anomalies, save_json
from .eof_reduce import build_window_eof_models
from .logging_utils import setup_logger
from .settings import LeadLagScreenV4Settings


def run_lead_lag_screen_v4(settings: LeadLagScreenV4Settings | None = None) -> Dict[str, object]:
    settings = settings or LeadLagScreenV4Settings()
    ensure_dirs(settings.output_dir, settings.log_dir)
    logger = setup_logger(settings.log_dir)
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    logger.info("Starting lead_lag_screen/V4 field lagged-CCA audit")
    logger.info("Output: %s", settings.output_dir)
    logger.info("Preprocess: %s", settings.preprocess_dir)
    save_json(settings.output_dir / "settings_summary.json", settings.to_jsonable())

    fields, lat, lon, years, field_meta = load_field_anomalies(settings)
    logger.info("Loaded fields: %s", {k: v.shape for k, v in fields.items()})
    logger.info("Loaded years=%d lat=%d lon=%d", len(years), len(lat), len(lon))

    eof_models, eof_score_df, eof_quality_df = build_window_eof_models(fields, lat, lon, years, settings, logger)
    eof_score_df.to_csv(settings.output_dir / "cca_eof_scores_long.csv", index=False, encoding="utf-8-sig")
    eof_quality_df.to_csv(settings.output_dir / "cca_eof_mode_quality.csv", index=False, encoding="utf-8-sig")

    panels = build_score_panels(eof_score_df, years, settings.eof_max_modes)
    lag_long, perm_df, boot_df, pattern_df = run_cca_lags(panels, years, settings, logger)
    cca_summary, cca_summary_main = summarize_cca(lag_long, perm_df, boot_df, settings)

    logger.info("Building V1/V3/V4 comparison")
    comparison = build_v1_v3_v4_comparison(settings.v1_output_dir, settings.v3_output_dir, cca_summary)
    t3_audit = build_t3_cca_audit(comparison)

    lag_long.to_csv(settings.output_dir / "cca_lag_long.csv", index=False, encoding="utf-8-sig")
    perm_df.to_csv(settings.output_dir / "cca_permutation_summary.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(settings.output_dir / "cca_bootstrap_stability.csv", index=False, encoding="utf-8-sig")
    cca_summary.to_csv(settings.output_dir / "cca_pair_summary.csv", index=False, encoding="utf-8-sig")
    cca_summary_main.to_csv(settings.output_dir / "cca_pair_summary_main_k5.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(settings.output_dir / "v1_v3_v4_cca_comparison.csv", index=False, encoding="utf-8-sig")
    t3_audit.to_csv(settings.output_dir / "t3_meiyu_end_cca_audit.csv", index=False, encoding="utf-8-sig")
    if not pattern_df.empty:
        pattern_df.to_csv(settings.output_dir / "cca_canonical_patterns_long.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "output_tag": settings.output_tag,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 3),
        "field_input_meta": field_meta,
        "n_years": int(len(years)),
        "n_windows": int(len(settings.windows)),
        "n_core_pair_directions": int(len(settings.core_pairs)),
        "eof_k_values": list(settings.eof_k_values),
        "n_cca_lag_rows": int(len(lag_long)),
        "n_cca_summary_rows": int(len(cca_summary)),
        "cca_time_structure_counts_main_k5": cca_summary_main["cca_time_structure_label"].value_counts(dropna=False).to_dict() if not cca_summary_main.empty else {},
        "cca_evidence_tier_counts_main_k5": cca_summary_main["cca_evidence_tier"].value_counts(dropna=False).to_dict() if not cca_summary_main.empty else {},
        "interpretation_scope": "field-to-field lagged CCA coupling-mode audit; not pathway proof; not PCMCI; not index replacement",
        "primary_audit_focus": "Does T3/meiyu-ending field-to-field coupling remain when using CCA coupling modes instead of manual indices or EOF-PC1 only?",
        "hard_boundary": "CCA can expose coupling modes but cannot by itself establish causal pathway or mediator chains.",
    }
    save_json(settings.output_dir / "summary.json", summary)
    save_json(settings.output_dir / "run_meta.json", {
        "layer": "lead_lag_screen",
        "version": "V4",
        "method": "smooth5_field_lagged_CCA_audit",
        "hard_boundaries": {
            "uses_indices_as_main_variables": False,
            "uses_cca": True,
            "uses_pcmci": False,
            "does_pathway_inference": False,
            "changes_window_definition": False,
            "core_pairs_only": True,
        },
        "settings": settings.to_jsonable(),
        "summary": summary,
    })
    logger.info("Finished lead_lag_screen/V4: %s", summary)
    return summary
