from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
_BASE_DATE = date(2001, 4, 1)

def day_index_to_month_day(day_index: int | float | None) -> str | None:
    if day_index is None:
        return None
    try:
        if not np.isfinite(day_index):
            return None
    except Exception:
        return None
    return (_BASE_DATE + timedelta(days=int(day_index))).strftime('%m-%d')

def month_day_to_day_index(month_day: str) -> int:
    month, day = [int(x) for x in month_day.split('-')]
    return int((date(2001, month, day) - _BASE_DATE).days)

def build_valid_day_metadata(valid_day_mask: np.ndarray) -> pd.DataFrame:
    idx = np.arange(len(valid_day_mask), dtype=int)
    return pd.DataFrame({'day_index': idx, 'month_day': [day_index_to_month_day(i) for i in idx], 'is_valid_day': valid_day_mask.astype(bool)})
