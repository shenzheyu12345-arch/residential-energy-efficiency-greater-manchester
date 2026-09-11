"""Run prespecified robustness checks for the explanatory XGBoost model.

The primary performance estimate remains the nested, grouped validation produced by
``run_explainable_models.py``.  This script holds the selected hyperparameters fixed
and tests whether substantive SHAP patterns survive alternative feature sets,
quality filters and an alternative outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
from esda.moran import Moran
from libpysal.weights import Queen
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lsoa_analytical_dataset.csv"
BOUNDARIES_PATH = PROJECT_ROOT / "data" / "spatial" / "gm_lsoa_2021_bgc_v5.gpkg"
TUNING_PATH = PROJECT_ROOT / "outputs" / "models" / "final_xgb_tuning.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sensitivity"

MAIN_FEATURES = [
    "pre_1930_pct",
    "post_1990_pct",
    "flat_maisonette_pct",
    "terraced_pct",
    "median_floor_area",
    "electric_fuel_pct",
    "private_rented_pct",
    "social_rented_pct",
    "imd_score",
]

# Complete k-1 encoding for each exhaustive categorical group.  The omitted
# categories are 1930-1990 age, houses, detached built form, mains gas and owner
# occupation.  Their information is recoverable from the retained shares, while
# their simultaneous inclusion would create deterministic compositional redundancy.
EXPANDED_FEATURES = [
    "pre_1930_pct",
    "post_1990_pct",
    "flat_maisonette_pct",
    "bungalow_pct",
    "park_home_pct",
    "semi_detached_pct",
    "terraced_pct",
    "median_floor_area",
    "electric_fuel_pct",
    "other_fuel_pct",
    "private_rented_pct",
    "social_rented_pct",
    "shared_ownership_pct",
    "rent_free_pct",
    "imd_score",
]

PERCENT_FEATURES = {
    "pre_1930_pct",
    "post_1990_pct",
    "flat_maisonette_pct",
    "bungalow_pct",
    "park_home_pct",
    "semi_detached_pct",
    "terraced_pct",
    "electric_fuel_pct",
    "other_fuel_pct",
    "non_mains_gas_pct",
    "private_rented_pct",
    "social_rented_pct",
    "shared_ownership_pct",
    "rent_free_pct",
}

SCENARIOS = [
    {
        "scenario": "main_fixed_params",
        "description": "Main sample and features; fixed selected hyperparameters",
        "features": MAIN_FEATURES,
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
    },
    {
        "scenario": "exclude_social_rented",
        "description": "Drops social-rented share to test its correlation with IMD",
        "features": [f for f in MAIN_FEATURES if f != "social_rented_pct"],
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
    },
    {
        "scenario": "exclude_imd",
        "description": "Drops IMD score to test its correlation with tenure",
        "features": [f for f in MAIN_FEATURES if f != "imd_score"],
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
    },
    {
        "scenario": "non_mains_gas_substitution",
        "description": "Replaces electric main fuel with non-mains-gas fuel share",
        "features": [
            "non_mains_gas_pct" if f == "electric_fuel_pct" else f
            for f in MAIN_FEATURES
        ],
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
    },
    {
        "scenario": "coverage_025_to_125",
        "description": "Restricts EPC-to-Census household coverage to 0.25-1.25",
        "features": MAIN_FEATURES,
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
        "query": "epc_coverage_rate >= 0.25 and epc_coverage_rate <= 1.25",
    },
    {
        "scenario": "age_missing_le_030",
        "description": "Restricts EPC age-band missingness to at most 30%",
        "features": MAIN_FEATURES,
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
        "query": "age_missing_rate <= 0.30",
    },
    {
        "scenario": "mean_epc_score_outcome",
        "description": "Uses mean current EPC efficiency score as the outcome",
        "features": MAIN_FEATURES,
        "target": "mean_current_energy_efficiency",
        "target_multiplier": 1.0,
        "target_unit": "EPC score points",
    },
    {
        "scenario": "expanded_complete_categories",
        "description": (
            "Adds complete k-1 encodings of EPC property category, built form, "
            "main fuel and Census tenure to test omitted-category sensitivity"
        ),
        "features": EXPANDED_FEATURES,
        "target": "below_c_rate",
        "target_multiplier": 100.0,
        "target_unit": "percentage points",
    },
]


def make_model(params: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def prepare(data: pd.DataFrame, scenario: dict):
    subset = data.query(scenario["query"]).copy() if scenario.get("query") else data.copy()
    features = scenario["features"]
    X = subset[features].copy()
    shares = [feature for feature in features if feature in PERCENT_FEATURES]
    X[shares] = X[shares] * 100.0
    y = subset[scenario["target"]].to_numpy(dtype=float) * scenario["target_multiplier"]
    groups = subset["lad24_name"].astype("string").to_numpy()
    if X.isna().any().any() or np.isnan(y).any():
        raise ValueError(f"Missing model values in {scenario['scenario']}")
    if pd.Series(groups).nunique() != 10:
        raise ValueError(f"Expected 10 LAD groups in {scenario['scenario']}")
    return subset, X, y, groups


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "r2": r2_score(y, prediction),
        "mae": mean_absolute_error(y, prediction),
        "rmse": np.sqrt(mean_squared_error(y, prediction)),
    }


def residual_moran(
    subset: pd.DataFrame,
    residuals: np.ndarray,
    boundaries: gpd.GeoDataFrame,
    seed: int,
) -> dict:
    residual_frame = pd.DataFrame(
        {"lsoa21": subset["lsoa21"].astype("string"), "residual": residuals}
    )
    merged = boundaries.merge(
        residual_frame, on="lsoa21", how="inner", validate="one_to_one"
    )
    if len(merged) != len(subset):
        raise ValueError("Residual Moran merge lost LSOAs")
    projected = merged.to_crs(27700)
    weights = Queen.from_dataframe(projected, ids=projected["lsoa21"].tolist())
    weights.transform = "R"
    np.random.seed(seed)
    result = Moran(projected["residual"].to_numpy(), weights, permutations=999)
    return {
        "residual_moran_i": result.I,
        "residual_moran_expected_i": result.EI,
        "residual_moran_permutation_p": result.p_sim,
        "residual_moran_islands": len(weights.islands),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, dtype={"lsoa21": "string"})
    boundaries = gpd.read_file(BOUNDARIES_PATH)[["lsoa21", "geometry"]]
    boundaries["lsoa21"] = boundaries["lsoa21"].astype("string")
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    params = tuning["best_params"]

    performance_rows: list[dict] = []
    fold_rows: list[dict] = []
    shap_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for scenario_number, scenario in enumerate(SCENARIOS, start=1):
        name = scenario["scenario"]
        print(f"Sensitivity {scenario_number}/{len(SCENARIOS)}: {name}", flush=True)
        subset, X, y, groups = prepare(data, scenario)
        oof = np.full(len(subset), np.nan)

        for fold, (train_idx, test_idx) in enumerate(
            LeaveOneGroupOut().split(X, y, groups), start=1
        ):
            model = make_model(params, 42 + scenario_number * 100 + fold)
            model.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = model.predict(X.iloc[test_idx])
            held_out = str(pd.Series(groups[test_idx]).unique()[0])
            fold_rows.append(
                {
                    "scenario": name,
                    "fold": fold,
                    "held_out_lad": held_out,
                    "n": len(test_idx),
                    **metrics(y[test_idx], oof[test_idx]),
                }
            )

        overall = metrics(y, oof)
        spatial_diagnostics = residual_moran(
            subset,
            y - oof,
            boundaries,
            seed=42 + scenario_number,
        )
        performance_rows.append(
            {
                "scenario": name,
                "description": scenario["description"],
                "target": scenario["target"],
                "target_unit": scenario["target_unit"],
                "n_lsoa": len(subset),
                "n_lad": pd.Series(groups).nunique(),
                "n_features": X.shape[1],
                **overall,
                **spatial_diagnostics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "scenario": name,
                    "lsoa21": subset["lsoa21"].to_numpy(),
                    "lad24_name": groups,
                    "observed": y,
                    "predicted_oof": oof,
                    "residual_oof": y - oof,
                }
            )
        )

        final_model = make_model(params, 42 + scenario_number)
        final_model.fit(X, y)
        shap_values = np.asarray(shap.TreeExplainer(final_model).shap_values(X))
        for feature_index, feature in enumerate(X.columns):
            feature_values = X[feature].to_numpy(dtype=float)
            contribution = shap_values[:, feature_index]
            direction = pd.Series(feature_values).corr(
                pd.Series(contribution), method="spearman"
            )
            median = np.median(feature_values)
            shap_rows.append(
                {
                    "scenario": name,
                    "target": scenario["target"],
                    "target_unit": scenario["target_unit"],
                    "feature": feature,
                    "mean_absolute_shap": np.mean(np.abs(contribution)),
                    "spearman_feature_vs_shap": direction,
                    "mean_shap_below_feature_median": np.mean(
                        contribution[feature_values <= median]
                    ),
                    "mean_shap_above_feature_median": np.mean(
                        contribution[feature_values > median]
                    ),
                }
            )

    performance = pd.DataFrame(performance_rows)
    folds = pd.DataFrame(fold_rows)
    shap_summary = pd.DataFrame(shap_rows)
    shap_summary["importance_rank"] = (
        shap_summary.groupby("scenario")["mean_absolute_shap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    performance.to_csv(OUTPUT_DIR / "sensitivity_performance.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "sensitivity_fold_metrics.csv", index=False)
    shap_summary.sort_values(["scenario", "importance_rank"]).to_csv(
        OUTPUT_DIR / "sensitivity_shap_summary.csv", index=False
    )
    predictions.to_csv(OUTPUT_DIR / "sensitivity_oof_predictions.csv", index=False)
    metadata = {
        "purpose": "Robustness of substantive patterns, not replacement primary performance estimate",
        "validation": "Leave-one-local-authority-out with fixed final main-model hyperparameters",
        "hyperparameters": params,
        "scenarios": SCENARIOS,
    }
    (OUTPUT_DIR / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(performance.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
