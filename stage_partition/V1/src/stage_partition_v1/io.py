from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import StagePartitionV1Settings
from .utils import file_sha256


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def save_npz(path: Path, arrays: Dict[str, np.ndarray], compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def load_npz_bundle(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'找不到 npz 文件：{path}')
    with np.load(path, allow_pickle=False) as data:
        arrays = {str(k): np.asarray(data[k]) for k in data.files}
    manifest = {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'sha256': file_sha256(path),
        'npz_keys': sorted(arrays.keys()),
    }
    return arrays, manifest


def resolve_foundation_inputs(settings: StagePartitionV1Settings) -> Dict[str, Path]:
    preprocess_root = settings.foundation.preprocess_root
    paths = {
        'smoothed_fields': preprocess_root / 'smoothed_fields.npz',
        'daily_climatology': preprocess_root / 'daily_climatology.npz',
        'foundation_run_status': settings.foundation.foundation_root / 'outputs' / 'baseline_a' / 'run_status.json',
        'foundation_run_config': settings.foundation.foundation_root / 'outputs' / 'baseline_a' / 'run_config.json',
    }
    if settings.foundation.strict_input_check:
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError('foundation/V1 依赖缺失：\n' + '\n'.join(missing))
    return paths


def load_smoothed_fields(path: Path) -> Dict[str, np.ndarray]:
    bundle, _ = load_npz_bundle(path)
    required = ['precip_smoothed', 'u200_smoothed', 'z500_smoothed', 'v850_smoothed', 'lat', 'lon', 'years']
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError('smoothed_fields 缺少键：' + ', '.join(missing))
    return bundle


def load_daily_climatology(path: Path) -> Dict[str, np.ndarray]:
    bundle, _ = load_npz_bundle(path)
    required = ['precip_clim', 'u200_clim', 'z500_clim', 'v850_clim', 'lat', 'lon', 'years']
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError('daily_climatology 缺少键：' + ', '.join(missing))
    return bundle


def prepare_output_dirs(settings: StagePartitionV1Settings) -> Dict[str, Path]:
    ensure_dirs(settings.paths.outputs_root, settings.paths.logs_root)
    return {'outputs_root': settings.paths.outputs_root, 'logs_root': settings.paths.logs_root}
