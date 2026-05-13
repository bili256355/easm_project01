from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .utils import nanmean_no_warn, safe_zscore_cube


def assemble_state_matrix(profile_dict: Dict[str, Dict[str, np.ndarray]], settings: StagePartitionV1Settings) -> Dict[str, object]:
    weighted_blocks: List[np.ndarray] = []
    feature_rows: List[dict] = []
    block_slices: Dict[str, Tuple[int, int]] = {}
    raw_blocks: Dict[str, np.ndarray] = {}
    standardized_blocks: Dict[str, np.ndarray] = {}
    weighted_block_map: Dict[str, np.ndarray] = {}
    block_weights: Dict[str, float] = {}
    start = 0

    for obj, payload in profile_dict.items():
        raw_block = np.asarray(payload['anomaly'], dtype=np.float64)
        raw_blocks[obj] = raw_block.copy()

        block = raw_block.copy()
        if settings.state_vector.standardize:
            block = safe_zscore_cube(block, eps=settings.state_vector.standardize_eps)
        standardized_blocks[obj] = block.copy()

        block_size = block.shape[-1]
        weight = 1.0
        if settings.state_vector.block_equal_weight and block_size > 0:
            weight = float(1.0 / np.sqrt(block_size))
            block = block * weight
        block_weights[obj] = weight
        weighted_block_map[obj] = block.copy()

        end = start + block_size
        block_slices[obj] = (start, end)
        for lat_value in payload['lat_grid']:
            feature_rows.append({
                'object_name': obj,
                'feature_name': f'{obj}_lat_{lat_value:.1f}',
                'lat_value': float(lat_value),
            })
        weighted_blocks.append(block)
        start = end

    state_cube = np.concatenate(weighted_blocks, axis=2)
    state_mean = nanmean_no_warn(state_cube, axis=0)
    feature_table = pd.DataFrame(feature_rows)
    meta = {
        'total_feature_count': int(state_cube.shape[-1]),
        'block_slices': {k: [int(v[0]), int(v[1])] for k, v in block_slices.items()},
        'block_feature_counts': {k: int(v[1] - v[0]) for k, v in block_slices.items()},
        'block_weights': {k: float(v) for k, v in block_weights.items()},
        'standardize': settings.state_vector.standardize,
        'block_equal_weight': settings.state_vector.block_equal_weight,
    }
    return {
        'state_cube': state_cube,
        'state_mean': state_mean,
        'feature_table': feature_table,
        'block_slices': block_slices,
        'block_weights': block_weights,
        'raw_blocks': raw_blocks,
        'standardized_blocks': standardized_blocks,
        'weighted_blocks': weighted_block_map,
        'meta': meta,
    }
