from __future__ import annotations

import pandas as pd

from .config import W045PreclusterConfig


def build_cluster_definition_table(cfg: W045PreclusterConfig) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cluster_id": c.cluster_id,
            "center_day": c.center_day,
            "day_min": c.day_min,
            "day_max": c.day_max,
            "role_seed": c.role_seed,
            "included_in_order_test": c.included_in_order_test,
        }
        for c in cfg.clusters
    ])
