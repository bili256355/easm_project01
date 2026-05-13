from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_1_ROOT = THIS_FILE.parents[1]
SRC = V1_1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1_1.eof_pc1_interpretability_settings import EOFPC1InterpretabilitySettings
from lead_lag_screen_v1_1.eof_pc1_interpretability_pipeline import run_eof_pc1_interpretability_audit_v1_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V1_1 EOF-PC1 interpretability audit.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--spatial-stride", type=int, default=None)
    parser.add_argument("--eof-value-mode", choices=["doy_anomaly", "raw_centered"], default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-cartopy", action="store_true")
    args = parser.parse_args()

    settings = EOFPC1InterpretabilitySettings()
    updates = {}
    if args.project_root:
        updates["project_root"] = Path(args.project_root)
    if args.n_modes is not None:
        updates["n_modes"] = args.n_modes
    if args.n_iter is not None:
        updates["n_iter"] = args.n_iter
    if args.spatial_stride is not None:
        updates["spatial_stride"] = args.spatial_stride
    if args.eof_value_mode is not None:
        updates["eof_value_mode"] = args.eof_value_mode
    if updates:
        settings = replace(settings, **updates)

    run_eof_pc1_interpretability_audit_v1_a(settings, make_figures=not args.no_figures, use_cartopy=not args.no_cartopy)


if __name__ == "__main__":
    main()
