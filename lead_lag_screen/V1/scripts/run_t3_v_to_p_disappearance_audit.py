from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parents[1]
SRC = V1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1.t3_v_to_p_audit_settings import T3VToPAuditSettings
from lead_lag_screen_v1.t3_v_to_p_audit_pipeline import run_t3_v_to_p_disappearance_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit why V->P candidates shrink in T3 by tracing existing V1 outputs through "
            "positive-lag significance, audit null, forward-vs-reverse stability, and lag-vs-tau0 stability."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\easm_project01"))
    parser.add_argument("--input-tag", default="lead_lag_screen_v1_smooth5_a")
    parser.add_argument("--stability-tag", default="lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b")
    parser.add_argument("--output-tag", default="lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit")
    parser.add_argument("--focus-window", default="T3")
    parser.add_argument(
        "--comparison-windows",
        default="S1,T1,S2,T2,S3,T3,S4,T4,S5",
        help="Comma-separated windows to include in window-level V->P comparison tables.",
    )
    parser.add_argument(
        "--skip-index-validity-context",
        action="store_true",
        help="Do not try to attach index_validity V1_b context tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = T3VToPAuditSettings(
        project_root=args.project_root,
        input_tag=args.input_tag,
        stability_tag=args.stability_tag,
        output_tag=args.output_tag,
        focus_window=args.focus_window,
        comparison_windows=tuple(w.strip() for w in args.comparison_windows.split(",") if w.strip()),
        include_index_validity_context=not args.skip_index_validity_context,
    )
    summary = run_t3_v_to_p_disappearance_audit(settings)
    print("T3 V->P disappearance audit completed.")
    print(f"Input V1:     {summary['input_dir']}")
    print(f"Stability:    {summary['stability_dir']}")
    print(f"Output:       {summary['output_dir']}")
    print(f"Focus window: {summary['focus_window']}")
    print("Key counts:")
    for key in [
        "focus_v_to_p_total_pairs",
        "focus_stable_lag_dominant",
        "focus_tau0_coupled",
        "focus_audit_sensitive",
        "focus_not_candidate_or_failed",
    ]:
        print(f"  {key}: {summary.get(key)}")


if __name__ == "__main__":
    main()
