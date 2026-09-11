"""Audit the complete set of dwelling, fuel and tenure candidate variables.

This analysis separates descriptive completeness from model parsimony.  It reports
all broad category shares available in the EPC and Census inputs, their bivariate
relationships with the LSOA Below-C rate, and correlations between candidates that
may make simultaneous SHAP attribution unstable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lsoa_analytical_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "candidate_audit"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

GROUPS = {
    "Dwelling age": ["pre_1930_pct", "mid_1930_1990_pct", "post_1990_pct"],
    "EPC property category": [
        "house_pct",
        "bungalow_pct",
        "flat_maisonette_pct",
        "park_home_pct",
    ],
    "EPC built form": ["detached_pct", "semi_detached_pct", "terraced_pct"],
    "Dwelling size": ["median_floor_area"],
    "Main fuel": ["mains_gas_pct", "electric_fuel_pct", "other_fuel_pct"],
    "Census tenure": [
        "owner_occupied_pct",
        "shared_ownership_pct",
        "social_rented_pct",
        "private_rented_pct",
        "rent_free_pct",
    ],
    "Deprivation": ["imd_score"],
}

DISPLAY_LABELS = {
    "below_c_rate": "Below-C rate",
    "pre_1930_pct": "Pre-1930",
    "mid_1930_1990_pct": "1930-1990",
    "post_1990_pct": "Post-1990",
    "house_pct": "House",
    "bungalow_pct": "Bungalow",
    "flat_maisonette_pct": "Flat/maisonette",
    "park_home_pct": "Park home",
    "detached_pct": "Detached",
    "semi_detached_pct": "Semi-detached",
    "terraced_pct": "Terraced",
    "median_floor_area": "Median floor area",
    "mains_gas_pct": "Mains gas",
    "electric_fuel_pct": "Electric fuel",
    "other_fuel_pct": "Other fuel",
    "owner_occupied_pct": "Owner occupied",
    "shared_ownership_pct": "Shared ownership",
    "social_rented_pct": "Social rented",
    "private_rented_pct": "Private rented",
    "rent_free_pct": "Lives rent free",
    "imd_score": "IMD score",
}

STRUCTURAL_GROUPS = {
    "Dwelling age": GROUPS["Dwelling age"],
    "EPC property category": GROUPS["EPC property category"],
    "EPC built form": GROUPS["EPC built form"],
    "Main fuel": GROUPS["Main fuel"],
    "Census tenure": GROUPS["Census tenure"],
}


def feature_group(feature: str) -> str:
    for group, features in GROUPS.items():
        if feature in features:
            return group
    raise KeyError(feature)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH, dtype={"lsoa21": "string"})
    features = [feature for values in GROUPS.values() for feature in values]
    required = ["below_c_rate", *features]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Analytical dataset is missing candidate columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Candidate audit contains missing values")

    candidate_summary = data[features].describe().T.reset_index(names="feature")
    candidate_summary.insert(
        1, "construct", candidate_summary["feature"].map(feature_group)
    )
    candidate_summary.insert(
        2, "display_label", candidate_summary["feature"].map(DISPLAY_LABELS)
    )
    candidate_summary["zero_share_lsoas"] = [
        int(data[feature].eq(0).sum()) for feature in candidate_summary["feature"]
    ]
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_variable_summary.csv", index=False)

    all_columns = ["below_c_rate", *features]
    correlation = data[all_columns].corr(method="spearman")
    correlation.to_csv(OUTPUT_DIR / "full_candidate_spearman.csv")

    p_values = pd.DataFrame(np.nan, index=all_columns, columns=all_columns)
    for first in all_columns:
        for second in all_columns:
            p_values.loc[first, second] = spearmanr(
                data[first], data[second], nan_policy="omit"
            ).pvalue
    p_values.to_csv(OUTPUT_DIR / "full_candidate_spearman_p_values.csv")

    outcome_rows = []
    for feature in features:
        result = spearmanr(data[feature], data["below_c_rate"], nan_policy="omit")
        outcome_rows.append(
            {
                "construct": feature_group(feature),
                "feature": feature,
                "display_label": DISPLAY_LABELS[feature],
                "n_lsoa": int(data[[feature, "below_c_rate"]].dropna().shape[0]),
                "spearman_rho_with_below_c_rate": result.statistic,
                "two_sided_p_value": result.pvalue,
                "mean_share_or_value": data[feature].mean(),
                "median_share_or_value": data[feature].median(),
                "zero_share_lsoas": int(data[feature].eq(0).sum()),
            }
        )
    outcome = pd.DataFrame(outcome_rows)
    outcome["absolute_rho"] = outcome["spearman_rho_with_below_c_rate"].abs()
    outcome.sort_values(["construct", "absolute_rho"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "candidate_outcome_correlations.csv", index=False
    )

    redundancy_rows = []
    for first_index, first in enumerate(features):
        for second in features[first_index + 1 :]:
            rho = correlation.loc[first, second]
            if abs(rho) >= 0.8:
                first_group = feature_group(first)
                second_group = feature_group(second)
                redundancy_rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "construct_1": first_group,
                        "construct_2": second_group,
                        "spearman_rho": rho,
                        "absolute_rho": abs(rho),
                        "same_compositional_group": first_group == second_group,
                    }
                )
    redundancy = pd.DataFrame(redundancy_rows)
    if not redundancy.empty:
        redundancy = redundancy.sort_values("absolute_rho", ascending=False)
    redundancy.to_csv(OUTPUT_DIR / "high_correlation_pairs_abs_ge_080.csv", index=False)

    composition_rows = []
    for group, group_features in STRUCTURAL_GROUPS.items():
        totals = data[group_features].sum(axis=1)
        composition_rows.append(
            {
                "construct": group,
                "n_categories": len(group_features),
                "mean_sum": totals.mean(),
                "minimum_sum": totals.min(),
                "maximum_sum": totals.max(),
                "maximum_absolute_deviation_from_one": (totals - 1).abs().max(),
            }
        )
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(OUTPUT_DIR / "compositional_sum_checks.csv", index=False)

    labels = [DISPLAY_LABELS[column] for column in all_columns]
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(15.2, 13.6))
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5.5},
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.15,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.72},
        ax=ax,
    )
    ax.set_title(
        "Spearman correlations among the outcome and complete candidate variables",
        pad=14,
    )
    ax.tick_params(axis="x", labelrotation=55, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "full_candidate_correlation.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    summary = {
        "n_lsoa": len(data),
        "n_candidate_features": len(features),
        "high_correlation_threshold": 0.8,
        "n_high_correlation_pairs": len(redundancy),
        "structural_composition_checks": composition.to_dict(orient="records"),
        "expanded_model_encoding": {
            "omitted_age_category": "mid_1930_1990_pct",
            "omitted_property_category": "house_pct",
            "omitted_built_form": "detached_pct",
            "omitted_main_fuel": "mains_gas_pct",
            "omitted_tenure": "owner_occupied_pct",
            "reason": "One category per exhaustive group is omitted to avoid deterministic redundancy in SHAP attribution.",
        },
    }
    (OUTPUT_DIR / "candidate_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
