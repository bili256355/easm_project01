from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .core import run_screen
from .logging_utils import setup_logger
from .settings import LeadLagScreenSettings
from .structural_indices import compute_v1_1_structural_indices
from .reporting import build_v1_1_reports


def run_lead_lag_screen_v1_1_structural_vp_a(settings: LeadLagScreenSettings | None = None):
    settings = settings or LeadLagScreenSettings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(settings.log_dir)
    logger.info("Starting V1_1 structural V→P screen")
    logger.info("Project root: %s", settings.project_root)
    logger.info("Output dir: %s", settings.output_dir)
    logger.info("V1 read-only index anomalies: %s", settings.input_v1_index_anomalies)
    logger.info("Smooth5 fields: %s", settings.input_smoothed_fields)

    index_meta = compute_v1_1_structural_indices(settings, logger=logger)
    logger.info("Structural indices ready: %s", settings.generated_index_anomalies)

    main_summary = run_screen(settings, logger)
    report_summary = build_v1_1_reports(settings)

    combined_summary = {
        "status": "success",
        "audit": "lead_lag_screen_v1_1_structural_vp_a",
        "v1_readonly": True,
        "index_meta": index_meta,
        "main_screen_summary": main_summary,
        "report_summary": report_summary,
        "primary_question": (
            "Whether adding structural V/P indices restores V→P lead-lag support under the same V1-style test framework."
        ),
        "guardrails": [
            "V1 is not modified.",
            "Only V→P main direction is run.",
            "Other fields H/Je/Jw are not included.",
            "Derived interpretation tags are not physical conclusions.",
        ],
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_meta = {
        "settings": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items() if k not in {"windows"}},
            "windows": settings.windows,
            "variable_families": settings.variable_families,
            "variables": settings.variables,
        },
        "method_contract": {
            "v1_readonly": True,
            "source_direction": "V",
            "target_direction": "P",
            "window_assignment": "target-side Y(t) belongs to W",
            "year_pairing": "same-year only; no cross-year concatenation",
            "index_value_mode": "day-of-season anomalies; old V1 anomalies + new V1_1 structural anomalies",
            "main_surrogate_mode": settings.surrogate_mode,
            "audit_surrogate_mode": settings.audit_surrogate_mode,
            "lag_tau0_stability": "post-processed with V1 stability judgement classifier",
        },
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished V1_1 structural V→P screen")
    return combined_summary
