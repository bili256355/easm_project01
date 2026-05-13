
from __future__ import annotations

import time
from datetime import datetime
import logging

from .block_audit import run_block_ablation_audit
from .candidate import build_general_candidates
from .diagnostics_h_dominance import build_block_score_contribution, summarize_block_energy, summarize_block_weight_effect
from .config import StagePartitionV1Settings
from .decision import decide_general_status
from .io import load_daily_climatology, load_smoothed_fields, prepare_output_dirs, resolve_foundation_inputs, save_csv, save_json
from .profiles import build_all_profiles, summarize_profile_validity
from .report import plot_candidate_overview, plot_general_score_curve, plot_yearwise_anchor_density, write_general_layer_summary
from .score import build_general_score_curve, build_yearwise_score_cube
from .state_vector import assemble_state_matrix
from .tests_general import assemble_general_test_table, run_continuity_test, run_struct_difference_test, run_yearwise_occurrence_test


def _build_logger(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('stage_partition_v1.mainline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def run_stage_partition_v1(settings: StagePartitionV1Settings | None = None):
    settings = settings or StagePartitionV1Settings()
    io_dirs = prepare_output_dirs(settings)
    logger = _build_logger(io_dirs['logs_root'] / 'mainline.log')
    started_at = datetime.now().isoformat(timespec='seconds')
    t0 = time.time()
    save_json(io_dirs['outputs_root'] / 'run_meta.json', {'status': 'running', 'started_at': started_at, 'layer_name': 'stage_partition', 'version_name': 'V1', 'foundation_root': str(settings.foundation.foundation_root)})
    try:
        logger.info('[1/10] 解析 foundation 输入')
        input_paths = resolve_foundation_inputs(settings)
        logger.info('[2/10] 读取 smoothed/climatology')
        smoothed_bundle = load_smoothed_fields(input_paths['smoothed_fields'])
        clim_bundle = load_daily_climatology(input_paths['daily_climatology'])
        logger.info('[3/10] 构造对象 profiles')
        profile_dict = build_all_profiles(smoothed_bundle, clim_bundle, settings)
        validity_df = summarize_profile_validity(profile_dict)
        save_csv(io_dirs['outputs_root'] / 'profile_validity_summary.csv', validity_df)
        logger.info('[4/10] 构造联合状态向量')
        state_payload = assemble_state_matrix(profile_dict, settings)
        save_csv(io_dirs['outputs_root'] / 'feature_table.csv', state_payload['feature_table'])
        save_json(io_dirs['outputs_root'] / 'state_vector_meta.json', state_payload['meta'])
        logger.info('[4.5/10] 产出 H 主导来源诊断表')
        raw_energy_df = summarize_block_energy(state_payload['raw_blocks'], stage_label='raw_anomaly')
        std_energy_df = summarize_block_energy(state_payload['standardized_blocks'], stage_label='standardized')
        weight_effect_df = summarize_block_weight_effect(
            state_payload['standardized_blocks'],
            state_payload['weighted_blocks'],
            state_payload['block_weights'],
        )
        save_csv(io_dirs['outputs_root'] / 'diag_block_energy_raw.csv', raw_energy_df)
        save_csv(io_dirs['outputs_root'] / 'diag_block_energy_standardized.csv', std_energy_df)
        save_csv(io_dirs['outputs_root'] / 'diag_block_weight_effect.csv', weight_effect_df)
        logger.info('[5/10] 计算一般层 score 曲线')
        score_df = build_general_score_curve(state_payload['state_mean'], settings)
        save_csv(io_dirs['outputs_root'] / 'general_score_curve.csv', score_df)
        yearwise_score_cube = build_yearwise_score_cube(state_payload['state_cube'], settings)
        logger.info('[6/10] 生成一般层候选窗口')
        candidate_df = build_general_candidates(score_df, settings)
        save_csv(io_dirs['outputs_root'] / 'general_candidate_windows.csv', candidate_df)
        logger.info('[7/10] 执行一般层检验')
        struct_df = run_struct_difference_test(state_payload['state_cube'], candidate_df, settings)
        continuity_df = run_continuity_test(score_df, candidate_df, settings)
        yearwise_df, yearwise_summary_df = run_yearwise_occurrence_test(yearwise_score_cube, candidate_df, settings)
        test_df = assemble_general_test_table(struct_df, continuity_df, yearwise_summary_df)
        save_csv(io_dirs['outputs_root'] / 'general_test_table.csv', test_df)
        save_csv(io_dirs['outputs_root'] / 'general_yearwise_summary.csv', yearwise_summary_df)
        save_csv(io_dirs['outputs_root'] / 'general_yearwise_detail.csv', yearwise_df)
        logger.info('[8/10] 执行单对象依赖审计')
        block_df = run_block_ablation_audit(state_payload['state_mean'], state_payload['block_slices'], candidate_df, settings)
        save_csv(io_dirs['outputs_root'] / 'general_block_ablation.csv', block_df)
        block_score_df = build_block_score_contribution(state_payload['state_mean'], state_payload['block_slices'], candidate_df, settings)
        save_csv(io_dirs['outputs_root'] / 'diag_score_block_contribution.csv', block_score_df)
        logger.info('[9/10] 形成一般层判定表')
        decision_df = decide_general_status(candidate_df, test_df, block_df, settings)
        save_csv(io_dirs['outputs_root'] / 'general_decision_table.csv', decision_df)
        logger.info('[10/10] 写出图件与元数据')
        plot_general_score_curve(score_df, candidate_df, io_dirs['outputs_root'])
        plot_candidate_overview(candidate_df, io_dirs['outputs_root'])
        plot_yearwise_anchor_density(yearwise_df, io_dirs['outputs_root'])
        write_general_layer_summary(candidate_df, decision_df, io_dirs['outputs_root'])
        save_json(io_dirs['outputs_root'] / 'config_used.json', settings.to_jsonable())
        save_json(io_dirs['outputs_root'] / 'run_meta.json', {
            'status': 'success',
            'started_at': started_at,
            'finished_at': datetime.now().isoformat(timespec='seconds'),
            'elapsed_seconds': round(time.time() - t0, 3),
            'layer_name': 'stage_partition',
            'version_name': 'V1',
            'foundation_root': str(settings.foundation.foundation_root),
            'output_root': str(io_dirs['outputs_root']),
        })
        return {'status': 'success', 'output_root': str(io_dirs['outputs_root'])}
    except Exception as exc:
        save_json(io_dirs['outputs_root'] / 'run_meta.json', {
            'status': 'failed', 'started_at': started_at, 'finished_at': datetime.now().isoformat(timespec='seconds'), 'error_type': type(exc).__name__, 'error_message': str(exc), 'layer_name': 'stage_partition', 'version_name': 'V1'
        })
        logger.exception('stage_partition/V1 失败：%s', exc)
        raise
