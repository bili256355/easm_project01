from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parents[1]
SRC = V1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1.t3_v_to_p_lag0_reduction_settings import T3VToPLag0ReductionSettings
from lead_lag_screen_v1.t3_v_to_p_lag0_reduction_pipeline import run_t3_v_to_p_lag0_reduction_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit why T3 V->P has fewer fixed-index relations in both lagged and lag0 diagnostics. "
            "This reads existing V1/stability results plus smooth5 fields/indices and does not rerun V1."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\easm_project01"))
    parser.add_argument("--input-tag", default="lead_lag_screen_v1_smooth5_a")
    parser.add_argument("--stability-tag", default="lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b")
    parser.add_argument("--previous-audit-tag", default="lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit")
    parser.add_argument("--output-tag", default="lead_lag_screen_v1_smooth5_a_t3_v_to_p_lag0_reduction_audit")
    parser.add_argument("--foundation-tag", default="baseline_smooth5_a")
    parser.add_argument("--focus-window", default="T3")
    parser.add_argument("--comparison-windows", default="S1,T1,S2,T2,S3,T3,S4,T4,S5")
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--no-figures", action="store_true", help="Skip composite PNG generation; numeric tables are still written.")
    parser.add_argument("--no-cartopy", action="store_true", help="Do not use cartopy even if installed; use plain lon/lat plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = T3VToPLag0ReductionSettings(
        project_root=args.project_root,
        input_tag=args.input_tag,
        stability_tag=args.stability_tag,
        previous_audit_tag=args.previous_audit_tag,
        output_tag=args.output_tag,
        foundation_tag=args.foundation_tag,
        focus_window=args.focus_window,
        comparison_windows=tuple(w.strip() for w in args.comparison_windows.split(",") if w.strip()),
        high_quantile=args.high_quantile,
        low_quantile=args.low_quantile,
        max_lag=args.max_lag,
        make_figures=not args.no_figures,
        no_cartopy=args.no_cartopy,
    )
    summary = run_t3_v_to_p_lag0_reduction_audit(settings)
    print("T3 V->P lag0/lagged reduction audit completed.")
    print(f"Output: {summary['output_dir']}")
    print(f"Focus window: {summary['focus_window']} days={summary['focus_days']}")
    print(f"V->P pairs: {summary['n_focus_v_to_p_pairs']}")
    print(f"Composite metric rows: {summary['n_composite_metric_rows']}")
    print(f"Figures: {summary['n_figures']}")


if __name__ == "__main__":
    main()
