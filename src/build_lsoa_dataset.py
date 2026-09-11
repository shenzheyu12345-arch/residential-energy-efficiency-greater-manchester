"""Build the Greater Manchester LSOA-level analytical dataset.

This script implements the agreed first stage of the dissertation methodology:

1. identify dwellings using UPRN, falling back to building reference number;
2. retain the latest EPC for each dwelling;
3. apply the active ten-year EPC window;
4. match normalized postcodes to 2021 LSOAs through ONSPD;
5. restrict records to the ten Greater Manchester local authorities;
6. aggregate EPC outcomes and dwelling-stock characteristics to LSOA;
7. join Census 2021 tenure and IoD 2025 deprivation variables; and
8. write an analytical CSV plus a machine-readable quality summary.

The script deliberately does not apply the provisional minimum-EPC or coverage
thresholds. It creates the diagnostic fields needed to make those decisions
from the observed LSOA distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GM_LADS = {
    "Bolton",
    "Bury",
    "Manchester",
    "Oldham",
    "Rochdale",
    "Salford",
    "Stockport",
    "Tameside",
    "Trafford",
    "Wigan",
}

IMD_LSOA = "LSOA code (2021)"
IMD_LSOA_NAME = "LSOA name (2021)"
IMD_LAD_CODE = "Local Authority District code (2024)"
IMD_LAD_NAME = "Local Authority District name (2024)"
IMD_SCORE = "Index of Multiple Deprivation (IMD) Score"
IMD_RANK = "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)"
IMD_DECILE = "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)"

EPC_USECOLS = [
    "lmk_key",
    "uprn",
    "building_reference_number",
    "postcode",
    "lodgement_date",
    "current_energy_rating",
    "current_energy_efficiency",
    "property_type",
    "built_form",
    "construction_age_band",
    "total_floor_area",
    "main_fuel",
    "tenure",
]

EPC_STRING_COLUMNS = {
    column: "string"
    for column in EPC_USECOLS
    if column not in {"current_energy_efficiency", "total_floor_area"}
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epc",
        type=Path,
        default=root / "EPC" / "epc_domestic_certificates_202607091646.csv",
    )
    parser.add_argument(
        "--onspd",
        type=Path,
        default=root / "postcode-LSOA look-up" / "Data" / "ONSPD_MAY_2025_UK.csv",
    )
    parser.add_argument(
        "--imd",
        type=Path,
        default=root
        / "IMD"
        / "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv",
    )
    parser.add_argument(
        "--census",
        type=Path,
        default=root / "census" / "4169201263056497.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "lsoa_analytical_dataset.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "outputs" / "data_quality" / "data_build_summary.json",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=root / "outputs" / "data_quality" / "lsoa_diagnostics.csv",
    )
    parser.add_argument("--window-start", default="2015-11-01")
    parser.add_argument("--window-end", default="2025-10-31")
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input files not found: " + ", ".join(missing))


def normalize_postcode(values: pd.Series) -> pd.Series:
    return values.astype("string").str.upper().str.replace(" ", "", regex=False).str.strip()


def load_gm_imd(path: Path) -> pd.DataFrame:
    columns = [
        IMD_LSOA,
        IMD_LSOA_NAME,
        IMD_LAD_CODE,
        IMD_LAD_NAME,
        IMD_SCORE,
        IMD_RANK,
        IMD_DECILE,
    ]
    imd = pd.read_csv(path, usecols=columns, low_memory=False)
    imd = imd.loc[imd[IMD_LAD_NAME].isin(GM_LADS)].copy()
    imd = imd.rename(
        columns={
            IMD_LSOA: "lsoa21",
            IMD_LSOA_NAME: "lsoa21_name",
            IMD_LAD_CODE: "lad24_code",
            IMD_LAD_NAME: "lad24_name",
            IMD_SCORE: "imd_score",
            IMD_RANK: "imd_rank",
            IMD_DECILE: "imd_decile",
        }
    )
    if imd["lsoa21"].duplicated().any():
        raise ValueError("IoD file contains duplicate Greater Manchester LSOA codes")
    if len(imd) != 1702:
        raise ValueError(f"Expected 1,702 Greater Manchester LSOAs in IoD; found {len(imd):,}")
    return imd


def load_epc_latest(
    path: Path, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    epc = pd.read_csv(
        path,
        usecols=EPC_USECOLS,
        dtype=EPC_STRING_COLUMNS,
        low_memory=False,
    )
    raw_rows = len(epc)

    epc["lodgement_date"] = pd.to_datetime(epc["lodgement_date"], errors="coerce")
    epc["current_energy_efficiency"] = pd.to_numeric(
        epc["current_energy_efficiency"], errors="coerce"
    )
    epc["total_floor_area"] = pd.to_numeric(epc["total_floor_area"], errors="coerce")

    epc["dwelling_key"] = epc["uprn"].where(
        epc["uprn"].notna(), "BRN:" + epc["building_reference_number"]
    )
    if epc["dwelling_key"].isna().any():
        raise ValueError("Some EPC rows have neither UPRN nor building reference number")

    epc = epc.sort_values(
        ["dwelling_key", "lodgement_date", "lmk_key"], kind="mergesort"
    ).drop_duplicates("dwelling_key", keep="last")
    latest_dwellings = len(epc)

    valid_date = epc["lodgement_date"].between(window_start, window_end, inclusive="both")
    epc = epc.loc[valid_date].copy()
    epc["postcode_key"] = normalize_postcode(epc["postcode"])

    summary = {
        "raw_epc_certificates": raw_rows,
        "latest_unique_dwellings": latest_dwellings,
        "duplicate_certificate_rows_removed": raw_rows - latest_dwellings,
        "window_start": window_start.date().isoformat(),
        "window_end": window_end.date().isoformat(),
        "latest_dwellings_in_window": len(epc),
        "share_latest_dwellings_in_window": len(epc) / latest_dwellings,
    }
    return epc, summary


def load_relevant_onspd(path: Path, postcode_keys: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["pcds", "lsoa21"],
        dtype="string",
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["postcode_key"] = normalize_postcode(chunk["pcds"])
        keep = chunk["postcode_key"].isin(postcode_keys)
        if keep.any():
            parts.append(chunk.loc[keep, ["postcode_key", "lsoa21"]])

    if not parts:
        raise ValueError("No EPC postcodes matched ONSPD")

    lookup = pd.concat(parts, ignore_index=True)
    conflicting = lookup.groupby("postcode_key")["lsoa21"].nunique(dropna=True)
    if (conflicting > 1).any():
        raise ValueError("ONSPD contains postcode keys mapped to more than one non-null LSOA")
    return lookup.drop_duplicates("postcode_key", keep="last")


def add_epc_features(epc: pd.DataFrame) -> pd.DataFrame:
    rating = epc["current_energy_rating"].str.upper().str.strip()
    epc["rating_valid"] = rating.isin(list("ABCDEFG"))
    epc["below_c"] = rating.isin(list("DEFG"))

    property_type = epc["property_type"].str.lower().str.strip()
    epc["property_type_valid"] = property_type.notna() & ~property_type.isin(
        ["not recorded", "unknown"]
    )
    epc["house"] = property_type.eq("house")
    epc["bungalow"] = property_type.eq("bungalow")
    epc["flat_maisonette"] = property_type.isin(["flat", "maisonette"])
    epc["park_home"] = property_type.eq("park home")

    built_form = epc["built_form"].str.lower().str.strip()
    epc["built_form_valid"] = built_form.notna() & ~built_form.isin(
        ["not recorded", "unknown"]
    )
    epc["detached"] = built_form.eq("detached")
    epc["semi_detached"] = built_form.eq("semi-detached")
    epc["terraced"] = built_form.str.contains("terrace", na=False)

    age = epc["construction_age_band"].str.lower().str.strip()
    exact_year = pd.to_numeric(age.where(age.str.fullmatch(r"\d{4}", na=False)), errors="coerce")
    invalid_exact_year = exact_year.notna() & ~exact_year.between(1700, 2025)
    epc["age_valid"] = age.notna() & ~invalid_exact_year
    epc["pre_1930"] = (
        age.str.contains(r"before 1900|1900-1929", regex=True, na=False)
        | exact_year.between(1700, 1929, inclusive="both")
    ).fillna(False) & epc["age_valid"]
    epc["post_1990"] = (
        age.str.contains(
            r"1991-1995|1996-2002|2003-2006|2007|2012|2022 onwards",
            regex=True,
            na=False,
        )
        | exact_year.between(1991, 2025, inclusive="both")
    ).fillna(False) & epc["age_valid"]
    epc["mid_1930_1990"] = (
        epc["age_valid"] & ~epc["pre_1930"] & ~epc["post_1990"]
    )

    fuel = epc["main_fuel"].str.lower().str.strip()
    epc["fuel_valid"] = fuel.notna()
    epc["mains_gas"] = fuel.str.contains("mains gas", na=False)
    epc["non_mains_gas"] = epc["fuel_valid"] & ~epc["mains_gas"]
    epc["electric_fuel"] = fuel.str.contains("electricity", na=False)
    epc["other_fuel"] = (
        epc["fuel_valid"] & ~epc["mains_gas"] & ~epc["electric_fuel"]
    )

    epc["floor_area_valid"] = epc["total_floor_area"].gt(0)
    epc["floor_area_for_aggregation"] = epc["total_floor_area"].where(
        epc["floor_area_valid"]
    )
    epc["lodgement_year"] = epc["lodgement_date"].dt.year
    return epc


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.gt(0)))


def aggregate_epc(epc: pd.DataFrame) -> pd.DataFrame:
    grouped = epc.groupby("lsoa21", observed=True)
    lsoa = grouped.agg(
        n_epc=("dwelling_key", "size"),
        n_valid_rating=("rating_valid", "sum"),
        n_below_c=("below_c", "sum"),
        mean_current_energy_efficiency=("current_energy_efficiency", "mean"),
        median_current_energy_efficiency=("current_energy_efficiency", "median"),
        n_property_type_valid=("property_type_valid", "sum"),
        n_house=("house", "sum"),
        n_bungalow=("bungalow", "sum"),
        n_flat_maisonette=("flat_maisonette", "sum"),
        n_park_home=("park_home", "sum"),
        n_built_form_valid=("built_form_valid", "sum"),
        n_detached=("detached", "sum"),
        n_semi_detached=("semi_detached", "sum"),
        n_terraced=("terraced", "sum"),
        n_age_valid=("age_valid", "sum"),
        n_pre_1930=("pre_1930", "sum"),
        n_mid_1930_1990=("mid_1930_1990", "sum"),
        n_post_1990=("post_1990", "sum"),
        n_fuel_valid=("fuel_valid", "sum"),
        n_mains_gas=("mains_gas", "sum"),
        n_non_mains_gas=("non_mains_gas", "sum"),
        n_electric_fuel=("electric_fuel", "sum"),
        n_other_fuel=("other_fuel", "sum"),
        n_floor_area_valid=("floor_area_valid", "sum"),
        median_floor_area=("floor_area_for_aggregation", "median"),
        floor_area_iqr=(
            "floor_area_for_aggregation",
            lambda x: x.quantile(0.75) - x.quantile(0.25),
        ),
        median_lodgement_year=("lodgement_year", "median"),
    ).reset_index()

    lsoa["below_c_rate"] = safe_rate(lsoa["n_below_c"], lsoa["n_valid_rating"])
    lsoa["house_pct"] = safe_rate(lsoa["n_house"], lsoa["n_property_type_valid"])
    lsoa["bungalow_pct"] = safe_rate(
        lsoa["n_bungalow"], lsoa["n_property_type_valid"]
    )
    lsoa["flat_maisonette_pct"] = safe_rate(
        lsoa["n_flat_maisonette"], lsoa["n_property_type_valid"]
    )
    lsoa["park_home_pct"] = safe_rate(
        lsoa["n_park_home"], lsoa["n_property_type_valid"]
    )
    lsoa["detached_pct"] = safe_rate(
        lsoa["n_detached"], lsoa["n_built_form_valid"]
    )
    lsoa["semi_detached_pct"] = safe_rate(
        lsoa["n_semi_detached"], lsoa["n_built_form_valid"]
    )
    lsoa["terraced_pct"] = safe_rate(lsoa["n_terraced"], lsoa["n_built_form_valid"])
    lsoa["pre_1930_pct"] = safe_rate(lsoa["n_pre_1930"], lsoa["n_age_valid"])
    lsoa["mid_1930_1990_pct"] = safe_rate(
        lsoa["n_mid_1930_1990"], lsoa["n_age_valid"]
    )
    lsoa["post_1990_pct"] = safe_rate(lsoa["n_post_1990"], lsoa["n_age_valid"])
    lsoa["age_missing_rate"] = 1 - safe_rate(lsoa["n_age_valid"], lsoa["n_epc"])
    lsoa["non_mains_gas_pct"] = safe_rate(
        lsoa["n_non_mains_gas"], lsoa["n_fuel_valid"]
    )
    lsoa["electric_fuel_pct"] = safe_rate(
        lsoa["n_electric_fuel"], lsoa["n_fuel_valid"]
    )
    lsoa["mains_gas_pct"] = safe_rate(lsoa["n_mains_gas"], lsoa["n_fuel_valid"])
    lsoa["other_fuel_pct"] = safe_rate(lsoa["n_other_fuel"], lsoa["n_fuel_valid"])
    lsoa["fuel_missing_rate"] = 1 - safe_rate(lsoa["n_fuel_valid"], lsoa["n_epc"])
    lsoa["floor_area_missing_rate"] = 1 - safe_rate(
        lsoa["n_floor_area_valid"], lsoa["n_epc"]
    )
    return lsoa


def load_census_tenure(path: Path) -> pd.DataFrame:
    census = pd.read_csv(path, skiprows=7, low_memory=False)
    required = [
        "mnemonic",
        "Total: All households",
        "Owned",
        "Shared ownership",
        "Social rented",
        "Private rented",
        "Lives rent free",
    ]
    missing = [column for column in required if column not in census.columns]
    if missing:
        raise ValueError(f"Census file missing required columns: {missing}")

    census = census.loc[census["mnemonic"].astype("string").str.match(r"^E010", na=False)].copy()
    census = census.rename(
        columns={
            "mnemonic": "lsoa21",
            "Total: All households": "census_total_households",
            "Owned": "census_owned_households",
            "Shared ownership": "census_shared_ownership_households",
            "Social rented": "census_social_rented_households",
            "Private rented": "census_private_rented_households",
            "Lives rent free": "census_rent_free_households",
        }
    )
    for column in [
        "census_total_households",
        "census_owned_households",
        "census_shared_ownership_households",
        "census_social_rented_households",
        "census_private_rented_households",
        "census_rent_free_households",
    ]:
        census[column] = pd.to_numeric(census[column], errors="coerce")

    if census["lsoa21"].duplicated().any():
        raise ValueError("Census file contains duplicate LSOA codes")

    census["owner_occupied_pct"] = safe_rate(
        census["census_owned_households"], census["census_total_households"]
    )
    census["shared_ownership_pct"] = safe_rate(
        census["census_shared_ownership_households"],
        census["census_total_households"],
    )
    census["social_rented_pct"] = safe_rate(
        census["census_social_rented_households"], census["census_total_households"]
    )
    census["private_rented_pct"] = safe_rate(
        census["census_private_rented_households"], census["census_total_households"]
    )
    census["rent_free_pct"] = safe_rate(
        census["census_rent_free_households"], census["census_total_households"]
    )
    return census[
        [
            "lsoa21",
            "census_total_households",
            "census_owned_households",
            "census_shared_ownership_households",
            "census_social_rented_households",
            "census_private_rented_households",
            "census_rent_free_households",
            "owner_occupied_pct",
            "shared_ownership_pct",
            "social_rented_pct",
            "private_rented_pct",
            "rent_free_pct",
        ]
    ]


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def quantile_summary(series: pd.Series) -> dict[str, Any]:
    quantiles = series.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    return {str(index): scalar(value) for index, value in quantiles.items()}


def build_summary(
    analytical: pd.DataFrame,
    epc_summary: dict[str, Any],
    geography_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epc": epc_summary,
        "geography": geography_summary,
        "analytical_dataset": {
            "rows": len(analytical),
            "rows_with_epc": int(analytical["n_epc"].notna().sum()),
            "rows_without_epc": int(analytical["n_epc"].isna().sum()),
            "rows_n_epc_below_10": int(analytical["n_epc"].fillna(0).lt(10).sum()),
            "rows_n_epc_below_30": int(analytical["n_epc"].fillna(0).lt(30).sum()),
            "rows_coverage_above_1": int(analytical["epc_coverage_rate"].gt(1).sum()),
            "rows_coverage_below_0_25": int(analytical["epc_coverage_rate"].lt(0.25).sum()),
            "n_epc_quantiles": quantile_summary(analytical["n_epc"].dropna()),
            "coverage_quantiles": quantile_summary(
                analytical["epc_coverage_rate"].dropna()
            ),
            "below_c_rate_quantiles": quantile_summary(
                analytical["below_c_rate"].dropna()
            ),
            "median_floor_area_quantiles": quantile_summary(
                analytical["median_floor_area"].dropna()
            ),
            "missing_values_by_column": {
                column: int(count)
                for column, count in analytical.isna().sum().items()
                if count > 0
            },
        },
    }


def main() -> None:
    args = parse_args()
    require_files([args.epc, args.onspd, args.imd, args.census])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)

    window_start = pd.Timestamp(args.window_start)
    window_end = pd.Timestamp(args.window_end)
    if window_start > window_end:
        raise ValueError("window-start must be on or before window-end")

    imd = load_gm_imd(args.imd)
    gm_lsoas = set(imd["lsoa21"])

    epc, epc_summary = load_epc_latest(args.epc, window_start, window_end)
    postcode_keys = set(epc["postcode_key"].dropna())
    onspd = load_relevant_onspd(args.onspd, postcode_keys)

    epc = epc.merge(onspd, on="postcode_key", how="left", validate="many_to_one")
    matched_lsoa = epc["lsoa21"].astype("string").str.match(r"^E010", na=False)
    in_gm = epc["lsoa21"].isin(gm_lsoas)
    geography_summary = {
        "active_latest_dwellings_before_geography_filter": len(epc),
        "matched_to_english_lsoa21": int(matched_lsoa.sum()),
        "matched_to_english_lsoa21_share": float(matched_lsoa.mean()),
        "matched_to_greater_manchester_lsoa": int(in_gm.sum()),
        "matched_to_greater_manchester_lsoa_share": float(in_gm.mean()),
        "outside_gm_or_unmatched": int((~in_gm).sum()),
    }
    epc = epc.loc[in_gm].copy()
    epc = add_epc_features(epc)
    epc_lsoa = aggregate_epc(epc)

    census = load_census_tenure(args.census)
    analytical = imd.merge(census, on="lsoa21", how="left", validate="one_to_one")
    analytical = analytical.merge(epc_lsoa, on="lsoa21", how="left", validate="one_to_one")
    analytical["epc_coverage_rate"] = safe_rate(
        analytical["n_epc"], analytical["census_total_households"]
    )
    analytical = analytical.sort_values(["lad24_name", "lsoa21"]).reset_index(drop=True)

    summary = build_summary(analytical, epc_summary, geography_summary)
    analytical.to_csv(args.output, index=False, encoding="utf-8")

    diagnostic_columns = [
        "lsoa21",
        "lsoa21_name",
        "lad24_name",
        "n_epc",
        "n_valid_rating",
        "epc_coverage_rate",
        "age_missing_rate",
        "fuel_missing_rate",
        "floor_area_missing_rate",
        "below_c_rate",
    ]
    analytical[diagnostic_columns].to_csv(
        args.diagnostics, index=False, encoding="utf-8"
    )
    with args.summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=scalar)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=scalar))
    print(f"Wrote analytical dataset: {args.output}")
    print(f"Wrote data quality summary: {args.summary}")
    print(f"Wrote LSOA diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
