from __future__ import annotations

from pathlib import Path
import numpy as np


def load_smoothed_fields(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing smoothed_fields.npz: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
