from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .eof_pc1_v1_adjudication_core import (
    build_adjudication_diagnosis,
    build_eof_pc1_v1_style_classification,
    build_old_index_pc1_leadlag,
    build_pc1_seasonal_progression_control,
    compute_pc1_old_index_alignment,
    load_v1_1_old_pair_counts,
)
from .eof_pc1_v1_adjudication_settings import EOFPC1V1AdjudicationSettings
from .logging_utils import setup_logger


def _jsonable(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _read_csv_required(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {desc} not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    if "day" in df.columns:
        df["day"] = df["day"].astype(int)
    return df


def run_eof_pc1_v1_adjudication_a(settings: EOFPC1V1AdjudicationSettings | None = None) -> dict:
    settings = settings or EOFPC1V1AdjudicationSettings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.table_dir.mkdir(parents=True, exist_ok=True)
    settings.summary_dir.mkdir(parents=True, exist_ok=True)
    settings.figure_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(settings.log_dir)
    logger.info("Starting EOF-PC1 V1 adjudication audit")
    logger.info("Output dir: %s", settings.output_dir)

    p_pc = _read_csv_required(settings.p_pc_scores_path, "P EOF PC scores")
    v_pc = _read_csv_required(settings.v_pc_scores_path, "V EOF PC scores")
    idx = _read_csv_required(settings.v1_1_index_anomalies, "V1_1 index anomaly table")
    if "P_PC1" not in p_pc.columns or "V_PC1" not in v_pc.columns:
        raise ValueError("Expected P_PC1 and V_PC1 columns in EOF PC score tables.")

    logger.info("Computing PC1-old-index alignment")
    alignment = compute_pc1_old_index_alignment(p_pc, v_pc, idx, settings)
    alignment.to_csv(settings.table_dir / "pc1_old_index_alignment_by_window.csv", index=False, encoding="utf-8-sig")

    logger.info("Computing old-index aggregate PC1 lead-lag")
    old_pc_ll, old_pc_meta = build_old_index_pc1_leadlag(idx, settings)
    old_pc_ll.to_csv(settings.table_dir / "old_index_pc1_leadlag_by_window.csv", index=False, encoding="utf-8-sig")
    old_pc_meta.to_csv(settings.table_dir / "old_index_pc1_metadata.csv", index=False, encoding="utf-8-sig")

    logger.info("Computing seasonal/background controls for EOF-PC1")
    seasonal = build_pc1_seasonal_progression_control(p_pc, v_pc, settings)
    seasonal.to_csv(settings.table_dir / "pc1_seasonal_progression_control.csv", index=False, encoding="utf-8-sig")

    logger.info("Computing EOF-PC1 V1-style classification")
    eof_v1_style = build_eof_pc1_v1_style_classification(p_pc, v_pc, settings)
    eof_v1_style.to_csv(settings.table_dir / "eof_pc1_v1_style_classification.csv", index=False, encoding="utf-8-sig")

    old_pair_counts = load_v1_1_old_pair_counts(settings)
    old_pair_counts.to_csv(settings.table_dir / "v1_1_old_pair_count_reference.csv", index=False, encoding="utf-8-sig")

    logger.info("Building adjudication diagnosis")
    diagnosis = build_adjudication_diagnosis(
        alignment=alignment,
        old_pc_ll=old_pc_ll,
        seasonal=seasonal,
        eof_v1_style=eof_v1_style,
        old_pair_counts=old_pair_counts,
        settings=settings,
    )
    diagnosis.to_csv(settings.table_dir / "eof_pc1_v1_adjudication_diagnosis.csv", index=False, encoding="utf-8-sig")

    # Compact T3 summary panel table for quick reading.
    t3_rows = []
    for name, df in [
        ("pc1_old_index_alignment", alignment),
        ("old_index_pc1_leadlag", old_pc_ll),
        ("pc1_seasonal_control", seasonal),
        ("eof_pc1_v1_style", eof_v1_style),
        ("diagnosis", diagnosis),
    ]:
        if "window" in df.columns:
            sub = df[df["window"] == "T3"].copy()
        else:
            sub = df.copy()
        t3_rows.append({"section": name, "n_rows": int(len(sub))})
    pd.DataFrame(t3_rows).to_csv(settings.summary_dir / "t3_adjudication_row_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "status": "success",
        "audit": "v1_1_eof_pc1_v1_adjudication_a",
        "primary_question": "Whether EOF-PC1 is eligible to adjudicate V1 old-index T3 V→P weakening.",
        "output_dir": str(settings.output_dir),
        "inputs": {
            "p_pc_scores": str(settings.p_pc_scores_path),
            "v_pc_scores": str(settings.v_pc_scores_path),
            "v1_1_index_anomalies": str(settings.v1_1_index_anomalies),
            "v1_1_classified_pairs": str(settings.v1_1_classified_pairs_path),
        },
        "diagnosis_counts": diagnosis["support_level"].value_counts(dropna=False).to_dict() if not diagnosis.empty else {},
        "t3_diagnosis": diagnosis.to_dict(orient="records"),
        "guardrails": [
            "This audit does not use highlat branch as the adjudication criterion.",
            "EOF-PC1 can adjudicate V1 only if it aligns with old-index spaces, survives seasonal controls, and is V1-style stable-lag in T3.",
            "Old-index aggregate PC1 distinguishes pair-level weakening from aggregate-space decoupling.",
        ],
    }
    (settings.summary_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "settings": _jsonable(asdict(settings)),
        "outputs": {"tables": [p.name for p in settings.table_dir.glob("*.csv")]},
        "method_notes": {
            "alignment": "linear R2 of EOF-PC1 explained by V1 old-index sets within each window",
            "old_index_pc1": "global PCA of old V indices and old P indices; lead-lag uses same target-side windows",
            "seasonal_control": "raw PC1, day-of-year residual PC1, and window-centered PC1 are compared",
            "v1_style_classification": "lag-vs-tau0 diagnostic classification only; this is an adjudication audit, not a new main screen",
        },
    }
    (settings.summary_dir / "run_meta.json").write_text(json.dumps(_jsonable(run_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished EOF-PC1 V1 adjudication audit")
    return summary
