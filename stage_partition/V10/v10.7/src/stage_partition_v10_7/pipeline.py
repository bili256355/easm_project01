from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .io_utils import load_smoothed_fields
from .h_profile_builder import build_h_profile, build_h_state_matrix, summarize_h_profile_validity
from .main_method_detector import run_point_detector_for_width
from .atlas_builder import (
    build_baseline_reproduction_audit,
    build_candidate_catalog,
    build_full_curve_table,
    build_h_event_atlas,
    build_width_stability_summary,
)
from .plotting import plot_h_candidate_raster, plot_h_fullseason_by_width, plot_h_window_panels
from .summary_writer import write_summary
from .utils import clean_output_root, now_utc, safe_read_csv, write_dataframe, write_json


def run_h_object_event_atlas_v10_7_a(project_root: str | Path | None = None) -> dict[str, Any]:
    cfg = Settings()
    if project_root is not None:
        cfg.with_project_root(Path(project_root))
    output_root = cfg.output.output_root(cfg.project_root)
    clean_output_root(output_root)

    print("[V10.7_a] Loading smoothed fields...")
    smoothed_path = cfg.foundation.smoothed_fields_path()
    smoothed = load_smoothed_fields(smoothed_path)

    print("[V10.7_a] Building H profile and state matrix...")
    h_profile = build_h_profile(smoothed, cfg.h_profile)
    validity = summarize_h_profile_validity(h_profile)
    state = build_h_state_matrix(h_profile, cfg.state)

    width_outputs: dict[int, dict[str, Any]] = {}
    for width in cfg.detector.widths:
        print(f"[V10.7_a] Running H main-method detector: detector_width={width}...")
        width_outputs[int(width)] = run_point_detector_for_width(state["state_matrix"], state["valid_day_index"], cfg.detector, int(width))

    print("[V10.7_a] Building full curve/candidate/atlas outputs...")
    curve = build_full_curve_table(width_outputs)
    candidates = build_candidate_catalog(width_outputs)
    reproduction = build_baseline_reproduction_audit(cfg, candidates)
    atlas = build_h_event_atlas(cfg, curve, candidates)
    stability = build_width_stability_summary(cfg, candidates)

    old_ref_path = cfg.project_root / cfg.reference.v10_2_candidate_catalog_relpath
    old_ref = safe_read_csv(old_ref_path)
    if old_ref is not None and not old_ref.empty:
        old_ref_h = old_ref[old_ref.astype(str).apply(lambda r: r.str.contains("H", case=False, regex=False).any(), axis=1)].copy()
    else:
        old_ref_h = None

    tables = output_root / "tables"
    write_dataframe(validity, tables / "h_profile_validity_v10_7_a.csv")
    write_dataframe(state["feature_table"], tables / "h_state_feature_table_v10_7_a.csv")
    write_dataframe(state["state_empty_feature_audit"], tables / "h_state_empty_feature_audit_v10_7_a.csv")
    write_dataframe(curve, tables / "h_detector_score_curves_by_width_v10_7_a.csv")
    write_dataframe(candidates, tables / "h_candidate_catalog_by_width_v10_7_a.csv")
    write_dataframe(reproduction, tables / "h_width20_baseline_reproduction_audit_v10_7_a.csv")
    write_dataframe(atlas, tables / "h_event_atlas_by_window_width_v10_7_a.csv")
    write_dataframe(stability, tables / "h_width_stability_summary_v10_7_a.csv")
    if old_ref_h is not None:
        write_dataframe(old_ref_h, tables / "h_v10_2_reference_rows_detected_v10_7_a.csv")

    print("[V10.7_a] Creating figures...")
    figures = output_root / "figures"
    try:
        plot_h_fullseason_by_width(cfg, curve, candidates, figures / "h_detector_score_fullseason_by_width_v10_7_a.png")
        plot_h_candidate_raster(cfg, candidates, figures / "h_candidate_days_by_width_raster_v10_7_a.png")
        plot_h_window_panels(cfg, curve, candidates, figures / "h_window_event_panels_v10_7_a.png")
    except Exception as e:
        # Keep tables usable even when matplotlib or backend fails.
        (output_root / "figures" / "FIGURE_ERROR.txt").write_text(str(e), encoding="utf-8")

    write_summary(cfg, output_root, reproduction, atlas, stability)

    run_meta = {
        "version": "v10.7_a",
        "task": "H-only main-method event atlas with detector_width sensitivity",
        "created_at_utc": now_utc(),
        "project_root": str(cfg.project_root),
        "input_smoothed_fields": str(smoothed_path),
        "settings": cfg.to_dict(),
        "state_vector_meta": state["state_vector_meta"],
        "outputs": {
            "output_root": str(output_root),
            "tables": str(tables),
            "figures": str(figures),
            "summary": str(output_root / "summary_h_object_event_atlas_v10_7_a.md"),
        },
        "not_implemented": [
            "bootstrap recurrence support",
            "yearwise validation",
            "cartopy spatial continuity validation",
            "physical mechanism interpretation",
            "causal or quasi-causal inference",
            "multi-object expansion beyond H",
        ],
        "interpretation_boundary": "method-layer H-only baseline export; detector_width is a detection temporal scale, not physical process duration",
    }
    write_json(run_meta, output_root / "run_meta" / "run_meta_v10_7_a.json")
    print(f"[V10.7_a] Done. Output root: {output_root}")
    return run_meta
