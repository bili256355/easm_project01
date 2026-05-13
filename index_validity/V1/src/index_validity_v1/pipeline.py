\
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .anomaly_basic_check import anomaly_daily_mean_check, anomaly_reconstruction_check
from .data_io import ensure_dirs, load_smoothed_bundle, read_index_tables, save_csv
from .index_metadata import VARIABLE_ORDER
from .logging_utils import setup_logger
from .physical_composites import build_physical_composites, build_physical_summary
from .plotting_physical import plot_physical_figures
from .plotting_yearwise import plot_yearwise_figures
from .settings import IndexValiditySettings
from .yearwise_diagnostics import build_yearwise_shape_audit


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_audit(settings: IndexValiditySettings) -> pd.DataFrame:
    paths = {
        "index_values_smoothed": settings.index_values_path,
        "index_daily_climatology": settings.index_climatology_path,
        "index_anomalies": settings.index_anomalies_path,
        "smoothed_fields_bundle": settings.smoothed_fields_bundle,
    }
    rows = []
    for name, path in paths.items():
        rows.append({
            "input_name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        })
    return pd.DataFrame(rows)


def run_index_validity_smooth5_v1(settings: IndexValiditySettings | None = None) -> dict:
    settings = settings or IndexValiditySettings()
    ensure_dirs(
        settings.output_dir,
        settings.log_dir,
        settings.tables_dir,
        settings.yearwise_figures_dir,
        settings.physical_figures_dir,
        settings.summary_dir,
    )
    logger = setup_logger(settings.log_dir)
    logger.info("Starting index_validity_smooth5_v1")
    logger.info("Output dir: %s", settings.output_dir)

    file_audit = _file_audit(settings)
    save_csv(settings.tables_dir / "input_file_audit.csv", file_audit)
    if not bool(file_audit["exists"].all()):
        missing = file_audit[~file_audit["exists"]]["path"].tolist()
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    logger.info("Loading index tables")
    values, clim, anom = read_index_tables(
        settings.index_values_path,
        settings.index_climatology_path,
        settings.index_anomalies_path,
    )

    logger.info("Running yearwise index shape diagnostics")
    yearwise = build_yearwise_shape_audit(values)
    save_csv(settings.tables_dir / "yearwise_index_shape_audit.csv", yearwise)

    logger.info("Plotting yearwise index figures")
    n_yearwise_figures = plot_yearwise_figures(
        index_values=values,
        yearwise_audit=yearwise,
        output_dir=settings.yearwise_figures_dir,
        dpi=settings.figure_dpi,
    )

    logger.info("Running basic anomaly checks")
    recon = anomaly_reconstruction_check(values, clim, anom, tolerance=settings.tolerance)
    daily = anomaly_daily_mean_check(anom, tolerance=settings.tolerance)
    save_csv(settings.tables_dir / "anomaly_reconstruction_basic_check.csv", recon)
    save_csv(settings.tables_dir / "anomaly_daily_mean_basic_check.csv", daily)

    logger.info("Loading 5-day smoothed physical fields")
    fields = load_smoothed_bundle(settings.smoothed_fields_bundle)

    logger.info("Computing high/low physical composites")
    composites, sample_info = build_physical_composites(
        index_values=values,
        fields=fields,
        high_q=settings.high_quantile,
        low_q=settings.low_quantile,
    )
    physical_summary = build_physical_summary(composites)
    save_csv(settings.tables_dir / "physical_composite_sample_info.csv", sample_info)
    save_csv(settings.tables_dir / "physical_representativeness_summary.csv", physical_summary)

    logger.info("Plotting physical representativeness figures")
    n_physical_figures, cartopy_status = plot_physical_figures(
        composites=composites,
        output_dir=settings.physical_figures_dir,
        dpi=settings.figure_dpi,
        use_cartopy_if_available=settings.use_cartopy_if_available,
    )

    anomaly_basic_status = "pass" if (recon["status"].eq("pass").all() and daily["status"].eq("pass").all()) else "warning"
    n_offset_flags = int(yearwise["flag_large_offset"].sum())
    n_rough_flags = int(yearwise["flag_large_roughness"].sum())
    n_curv_flags = int(yearwise["flag_large_curvature"].sum())

    summary = {
        "status": "success",
        "runtime_tag": settings.runtime_tag,
        "task_scope": "index_validity_only_no_leadlag_no_pathway",
        "n_indices": len(VARIABLE_ORDER),
        "n_years": int(values["year"].nunique()),
        "n_days": int(values["day"].nunique()),
        "n_yearwise_rows": int(len(yearwise)),
        "n_yearwise_figures": int(n_yearwise_figures),
        "n_physical_figures": int(n_physical_figures),
        "cartopy_status": cartopy_status,
        "anomaly_basic_status": anomaly_basic_status,
        "n_large_offset_flags": n_offset_flags,
        "n_large_roughness_flags": n_rough_flags,
        "n_large_curvature_flags": n_curv_flags,
        "outputs": {
            "tables_dir": str(settings.tables_dir),
            "yearwise_figures_dir": str(settings.yearwise_figures_dir),
            "physical_figures_dir": str(settings.physical_figures_dir),
        },
        "notes": [
            "This diagnostic checks 5-day smoothed index validity only.",
            "It does not evaluate lead-lag, autocorrelation, pathway, or downstream effects.",
            "Physical representativeness grades are initialized as to_review and require figure review.",
            "Cartopy is used when available; otherwise plain matplotlib maps are explicitly recorded.",
        ],
    }

    run_meta = {
        "settings": {
            "project_root": str(settings.project_root),
            "runtime_tag": settings.runtime_tag,
            "high_quantile": settings.high_quantile,
            "low_quantile": settings.low_quantile,
            "yearwise_flag_quantile": settings.yearwise_flag_quantile,
            "figure_dpi": settings.figure_dpi,
            "tolerance": settings.tolerance,
            "index_values_path": str(settings.index_values_path),
            "index_climatology_path": str(settings.index_climatology_path),
            "index_anomalies_path": str(settings.index_anomalies_path),
            "smoothed_fields_bundle": str(settings.smoothed_fields_bundle),
            "output_dir": str(settings.output_dir),
        },
        "scope_exclusions": [
            "lead_lag",
            "pathway",
            "autocorrelation_as_downstream_risk",
            "windowwise_stability",
            "raw_vs_anomaly_leadlag",
        ],
    }

    _write_json(settings.summary_dir / "summary.json", summary)
    _write_json(settings.summary_dir / "run_meta.json", run_meta)
    logger.info("Finished index_validity_smooth5_v1")
    return summary
