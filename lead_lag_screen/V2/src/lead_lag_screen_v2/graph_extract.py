from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .stats_utils import fdr_bh


def _safe_get_matrix(results: dict, key: str, n_var: int, tau_max: int) -> np.ndarray:
    arr = results.get(key)
    if arr is None:
        return np.full((n_var, n_var, tau_max + 1), np.nan, dtype=float)
    arr = np.asarray(arr)
    if arr.shape[0] != n_var or arr.shape[1] != n_var:
        raise ValueError(f"Unexpected {key} shape: {arr.shape}")
    return arr


def _graph_entry_selected(entry: object, tau: int) -> bool:
    """
    Interpret tigramite graph entries without assuming a single version-specific
    string vocabulary.

    For lagged links tau>=1, any non-empty entry containing an arrow/edge marker is
    treated as selected. Directional export itself is fixed as source(t-tau)->target(t)
    because tigramite stores lagged graph entries as graph[source, target, lag].
    """
    if entry is None:
        return False
    s = str(entry).strip()
    if s == "" or s.lower() in {"nan", "none", "false", "0"}:
        return False
    if tau >= 1:
        return any(marker in s for marker in ["-->", "<--", "o-o", "x-x", "-?>", "<?-"])
    return any(marker in s for marker in ["-->", "<--", "o-o", "x-x"])


def extract_pcmci_tables(
    results: dict,
    variables: List[str],
    variable_families: Dict[str, str],
    window: str,
    tau_min: int,
    tau_max: int,
    main_tau_min: int,
    main_tau_max: int,
    fdr_alpha: float,
    include_same_family_reported_edges: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = results.get("graph")
    if graph is None:
        raise ValueError("PCMCI+ results do not contain a graph matrix.")
    graph = np.asarray(graph, dtype=object)
    n_var = len(variables)
    if graph.shape[0] != n_var or graph.shape[1] != n_var:
        raise ValueError(f"Unexpected graph shape: {graph.shape}")

    val = _safe_get_matrix(results, "val_matrix", n_var, tau_max)
    pval = _safe_get_matrix(results, "p_matrix", n_var, tau_max)

    main_rows = []
    tau0_rows = []

    for si, source in enumerate(variables):
        for ti, target in enumerate(variables):
            if si == ti:
                continue
            sf = variable_families[source]
            tf = variable_families[target]
            same_family = sf == tf
            if same_family and not include_same_family_reported_edges:
                continue

            for tau in range(tau_min, tau_max + 1):
                try:
                    entry = graph[si, ti, tau]
                except IndexError:
                    entry = ""
                row = {
                    "window": window,
                    "source": source,
                    "target": target,
                    "source_family": sf,
                    "target_family": tf,
                    "same_family_edge": bool(same_family),
                    "tau": int(tau),
                    "graph_entry": str(entry),
                    "pcmci_graph_selected": bool(_graph_entry_selected(entry, tau)),
                    "val_matrix": float(val[si, ti, tau]) if np.isfinite(val[si, ti, tau]) else np.nan,
                    "p_matrix": float(pval[si, ti, tau]) if np.isfinite(pval[si, ti, tau]) else np.nan,
                    "edge_interpretation": "contemporaneous_diagnostic" if tau == 0 else "lagged_direct_edge_control",
                    "direction_semantics": (
                        "source(t)->target(t), contemporaneous orientation diagnostic only"
                        if tau == 0 else "source(t-tau)->target(t)"
                    ),
                }
                if tau == 0:
                    tau0_rows.append(row)
                elif main_tau_min <= tau <= main_tau_max:
                    main_rows.append(row)

    main_df = pd.DataFrame(main_rows)
    tau0_df = pd.DataFrame(tau0_rows)

    if not main_df.empty:
        main_df["q_within_window"] = fdr_bh(main_df["p_matrix"].to_numpy())
        main_df["pcmci_plus_supported"] = (
            main_df["pcmci_graph_selected"].astype(bool)
            & main_df["q_within_window"].le(float(fdr_alpha)).fillna(False)
        )
        main_df["support_rule"] = "graph_selected_and_window_BH_q_le_alpha"
    else:
        main_df["q_within_window"] = []
        main_df["pcmci_plus_supported"] = []
        main_df["support_rule"] = []

    if not tau0_df.empty:
        tau0_df["q_within_window_tau0"] = fdr_bh(tau0_df["p_matrix"].to_numpy())
        tau0_df["tau0_supported"] = (
            tau0_df["pcmci_graph_selected"].astype(bool)
            & tau0_df["q_within_window_tau0"].le(float(fdr_alpha)).fillna(False)
        )
        tau0_df["support_rule"] = "tau0_graph_selected_and_window_BH_q_le_alpha"

    return main_df, tau0_df
