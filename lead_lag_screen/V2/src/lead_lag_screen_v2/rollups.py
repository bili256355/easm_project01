from __future__ import annotations

import pandas as pd


def build_family_rollup(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    df = edges.copy()
    df["family_edge"] = df["source_family"] + "->" + df["target_family"]
    roll = (
        df.groupby(["window", "source_family", "target_family", "family_edge"], dropna=False)
        .agg(
            n_candidate_lagged_tests=("p_matrix", "size"),
            n_graph_selected=("pcmci_graph_selected", "sum"),
            n_pcmci_plus_supported=("pcmci_plus_supported", "sum"),
            min_q=("q_within_window", "min"),
            max_abs_val=("val_matrix", lambda x: x.abs().max()),
        )
        .reset_index()
        .sort_values(["window", "n_pcmci_plus_supported", "min_q"], ascending=[True, False, True])
    )
    return roll


def build_window_summary(edges: pd.DataFrame, tau0: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(set(edges["window"].unique()) | set(tau0["window"].unique())) if (not edges.empty or not tau0.empty) else []
    rows = []
    for w in windows:
        e = edges[edges["window"] == w] if not edges.empty else pd.DataFrame()
        z = tau0[tau0["window"] == w] if not tau0.empty else pd.DataFrame()
        rows.append({
            "window": w,
            "n_lagged_candidate_tests": int(len(e)),
            "n_lagged_graph_selected": int(e["pcmci_graph_selected"].sum()) if not e.empty else 0,
            "n_lagged_pcmci_plus_supported": int(e["pcmci_plus_supported"].sum()) if not e.empty else 0,
            "n_tau0_candidate_tests": int(len(z)),
            "n_tau0_graph_selected": int(z["pcmci_graph_selected"].sum()) if not z.empty else 0,
            "n_tau0_supported": int(z["tau0_supported"].sum()) if (not z.empty and "tau0_supported" in z.columns) else 0,
        })
    return pd.DataFrame(rows)
