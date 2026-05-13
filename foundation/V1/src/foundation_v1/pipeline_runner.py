from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data_contract import summarize_finite_status, validate_input_contract
from .field_preprocessing import (
    build_field_stats,
    build_nan_report,
    compute_all_anomalies,
    compute_all_daily_climatology,
    smooth_all_fields,
)
from .io_utils import ensure_dirs, load_input_arrays, load_npz_bundle, save_csv, save_json, save_npz
from .logging_utils import build_logger, log_step
from .object_index_builder import (
    build_daily_climatology_table,
    build_index_summary_table,
    build_index_value_table,
    build_region_table,
    build_variable_definition_table,
    compute_index_anomalies,
    compute_index_daily_climatology,
    compute_indices,
)
from .settings import (
    CORE_FIELDS,
    FullConfig,
    LAYER_NAME,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PROJECT_NAME,
    VERSION_NAME,
)

REQUIRED_SMOOTHED_KEYS = (
    "precip_smoothed",
    "u200_smoothed",
    "z500_smoothed",
    "v850_smoothed",
    "lat",
    "lon",
    "years",
)


def _build_raw_bundle(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "precip_raw": arrays["precip"],
        "u200_raw": arrays["u200"],
        "z500_raw": arrays["z500"],
        "v850_raw": arrays["v850"],
        "lat": arrays["lat"],
        "lon": arrays["lon"],
        "years": arrays["years"],
    }


def _rename_with_suffix(arrays: Dict[str, np.ndarray], suffix: str) -> Dict[str, np.ndarray]:
    return {f"{name}_{suffix}": arr for name, arr in arrays.items()}


def _attach_coords(bundle: Dict[str, np.ndarray], arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(bundle)
    out["lat"] = np.asarray(arrays["lat"], dtype=np.float64)
    out["lon"] = np.asarray(arrays["lon"], dtype=np.float64)
    out["years"] = np.asarray(arrays["years"])
    return out


def _extract_smoothed_inputs(bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_SMOOTHED_KEYS if key not in bundle]
    if missing:
        raise KeyError("smoothed_fields.npz 缺少以下键：" + ", ".join(missing))
    return {
        "precip": np.asarray(bundle["precip_smoothed"], dtype=np.float64),
        "u200": np.asarray(bundle["u200_smoothed"], dtype=np.float64),
        "z500": np.asarray(bundle["z500_smoothed"], dtype=np.float64),
        "v850": np.asarray(bundle["v850_smoothed"], dtype=np.float64),
        "lat": np.asarray(bundle["lat"], dtype=np.float64),
        "lon": np.asarray(bundle["lon"], dtype=np.float64),
        "years": np.asarray(bundle["years"]),
    }


def run_foundation_v1(config: FullConfig | None = None) -> Dict[str, Any]:
    config = config or FullConfig()
    paths = config.paths
    ensure_dirs(paths.outputs_root, paths.logs_root, paths.preprocess_root, paths.indices_root)
    logger = build_logger(paths.logs_root / "mainline.log", name="foundation_v1.mainline")

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    total_steps = 10

    base_status = {
        "project_name": PROJECT_NAME,
        "layer_name": LAYER_NAME,
        "version_name": VERSION_NAME,
        "project_root": str(paths.project_root),
        "layer_root": str(paths.layer_root),
        "data_root": str(paths.data_root),
        "outputs_root": str(paths.outputs_root),
        "logs_root": str(paths.logs_root),
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
    }

    save_json(paths.outputs_root / "run_status.json", {"status": "running", "started_at": started_at, **base_status})
    save_json(paths.outputs_root / "run_config.json", config.to_jsonable())

    try:
        log_step(logger, 1, total_steps, "读取输入数据")
        arrays, input_manifest = load_input_arrays(paths.data_root)

        log_step(logger, 2, total_steps, "检查输入契约与 finite 摘要")
        shape_report = validate_input_contract(arrays)
        finite_summary = summarize_finite_status(arrays)

        log_step(logger, 3, total_steps, "执行场级 smoothing")
        raw_fields = {name: np.asarray(arrays[name], dtype=np.float64) for name in CORE_FIELDS}
        smoothed_fields = smooth_all_fields(raw_fields, window=config.preprocess.smooth_window)

        log_step(logger, 4, total_steps, "计算场级 daily climatology")
        daily_climatology = compute_all_daily_climatology(smoothed_fields)

        log_step(logger, 5, total_steps, "计算场级 anomaly")
        anomaly_fields = compute_all_anomalies(smoothed_fields, daily_climatology)

        log_step(logger, 6, total_steps, "生成 preprocess 质控表")
        nan_report = build_nan_report(raw_fields, smoothed_fields, daily_climatology, anomaly_fields)
        field_stats = build_field_stats(
            {
                "raw": raw_fields,
                "smoothed": smoothed_fields,
                "climatology": daily_climatology,
                "anomaly": anomaly_fields,
            }
        )

        preprocess_meta = {
            "task_scope": "field_level_preprocess_only",
            "project_name": PROJECT_NAME,
            "layer_name": LAYER_NAME,
            "version_name": VERSION_NAME,
            "hard_constraints": {
                "no_stage_generation": True,
                "no_window_detection": True,
                "no_pathway_inference": True,
            },
            "input_manifest": input_manifest,
            "shape_report": shape_report,
            "finite_summary": finite_summary,
            "smooth_window": config.preprocess.smooth_window,
            "anomaly_definition": config.preprocess.anomaly_definition,
        }

        log_step(logger, 7, total_steps, "保存 preprocess 输出")
        save_npz(paths.preprocess_root / "raw_fields.npz", _build_raw_bundle(arrays))
        save_npz(paths.preprocess_root / "smoothed_fields.npz", _attach_coords(_rename_with_suffix(smoothed_fields, "smoothed"), arrays))
        save_npz(paths.preprocess_root / "daily_climatology.npz", _attach_coords(_rename_with_suffix(daily_climatology, "clim"), arrays))
        save_npz(paths.preprocess_root / "anomaly_fields.npz", _attach_coords(_rename_with_suffix(anomaly_fields, "anom"), arrays))
        save_csv(paths.preprocess_root / "nan_report.csv", nan_report)
        save_csv(paths.preprocess_root / "field_stats.csv", field_stats)
        save_json(paths.preprocess_root / "preprocess_meta.json", preprocess_meta)

        log_step(logger, 8, total_steps, "从 preprocess 输出读取 smoothed fields")
        smoothed_bundle, smoothed_manifest = load_npz_bundle(paths.preprocess_root / config.indices.source_bundle_name)
        smoothed_inputs = _extract_smoothed_inputs(smoothed_bundle)

        log_step(logger, 9, total_steps, "计算对象指数")
        index_arrays, index_meta = compute_indices(
            smoothed_fields={
                "precip": smoothed_inputs["precip"],
                "u200": smoothed_inputs["u200"],
                "z500": smoothed_inputs["z500"],
                "v850": smoothed_inputs["v850"],
            },
            lat=smoothed_inputs["lat"],
            lon=smoothed_inputs["lon"],
        )
        index_daily_climatology = compute_index_daily_climatology(index_arrays)
        index_anomalies = compute_index_anomalies(index_arrays, index_daily_climatology)

        value_table = build_index_value_table(index_arrays, smoothed_inputs["years"])
        clim_table = build_daily_climatology_table(index_daily_climatology)
        anom_table = build_index_value_table(index_anomalies, smoothed_inputs["years"])
        summary_table = build_index_summary_table(index_arrays)
        region_table = pd.DataFrame(build_region_table())
        variable_table = pd.DataFrame(build_variable_definition_table())

        finished_at = datetime.now().isoformat(timespec="seconds")
        elapsed = round(time.time() - t0, 3)
        indices_meta = {
            "task_scope": "object_index_extraction_from_smoothed_fields",
            "project_name": PROJECT_NAME,
            "layer_name": LAYER_NAME,
            "version_name": VERSION_NAME,
            "hard_constraints": {
                "compute_on_smoothed_fields_only": True,
                "index_anomaly_after_index_build": True,
                "no_stage_generation": True,
                "no_window_detection": True,
                "no_pathway_inference": True,
            },
            "source_smoothed_bundle_manifest": smoothed_manifest,
            "implementation_meta": index_meta,
            "elapsed_seconds": elapsed,
        }

        log_step(logger, 10, total_steps, "保存 indices 输出并结束")
        save_npz(paths.indices_root / "index_values_smoothed.npz", _rename_with_suffix(index_arrays, "smoothed_index"))
        save_npz(paths.indices_root / "index_daily_climatology.npz", _rename_with_suffix(index_daily_climatology, "index_clim"))
        save_npz(paths.indices_root / "index_anomalies.npz", _rename_with_suffix(index_anomalies, "index_anom"))
        save_csv(paths.indices_root / "index_values_smoothed.csv", value_table)
        save_csv(paths.indices_root / "index_daily_climatology.csv", clim_table)
        save_csv(paths.indices_root / "index_anomalies.csv", anom_table)
        save_csv(paths.indices_root / "index_build_summary.csv", summary_table)
        save_csv(paths.indices_root / "index_region_table.csv", region_table)
        save_csv(paths.indices_root / "index_variable_definition_table.csv", variable_table)
        save_json(paths.indices_root / "index_meta.json", indices_meta)

        save_json(
            paths.outputs_root / "run_status.json",
            {
                "status": "success",
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed,
                **base_status,
            },
        )
        logger.info("foundation_v1 完成。输出根目录：%s", paths.outputs_root)
        return {
            "status": "success",
            "layer_name": LAYER_NAME,
            "version_name": VERSION_NAME,
            "outputs_root": str(paths.outputs_root),
            "preprocess_root": str(paths.preprocess_root),
            "indices_root": str(paths.indices_root),
            "logs_root": str(paths.logs_root),
        }
    except Exception as exc:
        save_json(
            paths.outputs_root / "run_status.json",
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **base_status,
            },
        )
        logger.exception("foundation_v1 失败：%s", exc)
        raise
