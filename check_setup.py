"""Check the local environment and required input paths without reading the data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_INPUTS = {
    "Domestic EPC extract": ROOT / "EPC" / "epc_domestic_certificates_202607091646.csv",
    "ONS Postcode Directory": ROOT
    / "postcode-LSOA look-up"
    / "Data"
    / "ONSPD_MAY_2025_UK.csv",
    "English Indices of Deprivation 2025": ROOT
    / "IMD"
    / "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv",
    "Census 2021 tenure table": ROOT / "census" / "4169201263056497.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-inputs",
        action="store_true",
        help="Return a non-zero exit code when any expected source file is absent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        name: {"path": str(path.relative_to(ROOT)), "present": path.is_file()}
        for name, path in REQUIRED_INPUTS.items()
    }
    report = {
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 12),
        "repository_root": str(ROOT),
        "inputs": inputs,
        "all_inputs_present": all(item["present"] for item in inputs.values()),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {report['python']} (3.12+ recommended)")
        for name, item in inputs.items():
            status = "FOUND" if item["present"] else "MISSING"
            print(f"[{status}] {name}: {item['path']}")

    if args.require_inputs and not report["all_inputs_present"]:
        print("\nSee docs/data_access.md for access and placement instructions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

