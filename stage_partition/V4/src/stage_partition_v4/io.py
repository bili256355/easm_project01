from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
from .config import StagePartitionV4Settings

REQUIRED_SMOOTHED_KEYS = ['precip_smoothed','u200_smoothed','z500_smoothed','v850_smoothed','lat','lon','years']

def prepare_output_dirs(settings: StagePartitionV4Settings) -> dict[str, Path]:
    out = settings.output_root(); log = settings.log_root()
    out.mkdir(parents=True, exist_ok=True); log.mkdir(parents=True, exist_ok=True)
    return {'output_root': out, 'log_root': log}

def resolve_smoothed_fields_path(settings: StagePartitionV4Settings) -> Path:
    path = settings.foundation.smoothed_fields_path()
    if not path.exists():
        raise FileNotFoundError(f'Missing smoothed_fields.npz at {path}')
    return path

def load_smoothed_fields(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED_SMOOTHED_KEYS if k not in data]
    if missing:
        raise KeyError(f'smoothed_fields is missing keys: {missing}')
    return {k: data[k] for k in REQUIRED_SMOOTHED_KEYS}

def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)

def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
