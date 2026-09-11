"""Run the complete dissertation analysis in its documented order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-boundary-download",
        action="store_true",
        help="Use an existing data/spatial/gm_lsoa_2021_bgc_v5.gpkg file.",
    )
    return parser.parse_args()


def run(label: str, *arguments: str) -> None:
    print(f"\n== {label} ==", flush=True)
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    run("Check source files", "check_setup.py", "--require-inputs")
    run("Build LSOA analytical dataset", "src/build_lsoa_dataset.py")

    if not args.skip_boundary_download:
        run("Download Greater Manchester LSOA boundaries", "src/download_gm_lsoa_boundaries.py")

    run(
        "Run spatial analysis",
        "src/run_spatial_analysis.py",
        "--permutations",
        "999",
        "--seed",
        "42",
    )
    run(
        "Evaluate models and fit the explanatory model",
        "src/run_explainable_models.py",
        "--n-iter",
        "15",
        "--inner-splits",
        "4",
        "--seed",
        "42",
        "--permutations",
        "999",
    )
    run("Render explainability figures", "src/render_explainability_figures.py")
    run("Run sensitivity analysis", "src/run_sensitivity_analysis.py")
    run("Audit candidate variables", "src/run_candidate_variable_audit.py")
    run("Render methodology workflow", "src/build_methodology_workflow.py")
    print("\nPipeline complete. Compare outputs/ with verification_outputs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

