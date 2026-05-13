from __future__ import annotations

import inspect
from typing import Dict, List

import numpy as np


class TigramiteUnavailableError(RuntimeError):
    pass


def _import_tigramite():
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except Exception as exc:  # pragma: no cover - depends on local env
        raise TigramiteUnavailableError(
            "tigramite is required for lead_lag_screen/V2_c PCMCI+ sensitivity reruns. "
            "No fallback implementation is allowed. Original import error: "
            f"{exc!r}"
        ) from exc
    return pp, PCMCI, ParCorr


def _make_parcorr(ParCorr, significance: str):
    kwargs = {"significance": significance}
    sig = inspect.signature(ParCorr)
    if "mask_type" in sig.parameters:
        kwargs["mask_type"] = "y"
    return ParCorr(**kwargs)


def _make_dataframe(pp, data_dict: Dict[int, np.ndarray], mask_dict: Dict[int, np.ndarray], variables: List[str]):
    datatime = {k: np.arange(v.shape[0]) for k, v in data_dict.items()}
    try:
        return pp.DataFrame(
            data=data_dict,
            mask=mask_dict,
            datatime=datatime,
            var_names=variables,
            analysis_mode="multiple",
        )
    except TypeError as exc:
        raise RuntimeError(
            "The installed tigramite does not appear to support the required multiple-dataset DataFrame API. "
            "V2_c intentionally does not fall back to cross-year concatenation. Original error: "
            f"{exc!r}"
        ) from exc


def run_pcmci_plus(
    data_dict: Dict[int, np.ndarray],
    mask_dict: Dict[int, np.ndarray],
    variables: List[str],
    tau_min: int,
    tau_max: int,
    pc_alpha: float,
    parcorr_significance: str,
    verbosity: int,
) -> dict:
    pp, PCMCI, ParCorr = _import_tigramite()
    dataframe = _make_dataframe(pp, data_dict, mask_dict, variables)
    cond_ind_test = _make_parcorr(ParCorr, parcorr_significance)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)
    kwargs = {
        "tau_min": int(tau_min),
        "tau_max": int(tau_max),
        "pc_alpha": float(pc_alpha),
    }
    sig = inspect.signature(pcmci.run_pcmciplus)
    if "fdr_method" in sig.parameters:
        kwargs["fdr_method"] = "none"
    return pcmci.run_pcmciplus(**kwargs)
