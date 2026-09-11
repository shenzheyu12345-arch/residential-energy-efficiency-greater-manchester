"""Run grouped validation, XGBoost, Random Forest and SHAP analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / "outputs" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from esda.moran import Moran
from libpysal.weights import Queen
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RandomizedSearchCV
from xgboost import XGBRegressor


FEATURES = [
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

PERCENT_FEATURES = [
    "pre_1930_pct",
    "post_1990_pct",
    "flat_maisonette_pct",
    "terraced_pct",
    "electric_fuel_pct",
    "private_rented_pct",
    "social_rented_pct",
]

DISPLAY_LABELS = {
    "pre_1930_pct": "Pre-1930 dwellings (%)",
    "post_1990_pct": "Post-1990 dwellings (%)",
    "flat_maisonette_pct": "Flats/maisonettes (%)",
    "terraced_pct": "Terraced dwellings (%)",
    "median_floor_area": "Median floor area (m²)",
    "electric_fuel_pct": "Electric main fuel (%)",
    "private_rented_pct": "Private rented (%)",
    "social_rented_pct": "Social rented (%)",
    "imd_score": "IMD score",
}

PARAM_DISTRIBUTIONS = {
    "n_estimators": [150, 250, 400, 600, 800],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0, 30.0],
    "gamma": [0.0, 0.01, 0.1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "lsoa_analytical_dataset.csv",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=PROJECT_ROOT / "data" / "spatial" / "gm_lsoa_2021_bgc_v5.gpkg",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs"
    )
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=999)
    return parser.parse_args()


def prepare_data(data: pd.DataFrame):
    X = data[FEATURES].copy()
    X[PERCENT_FEATURES] = X[PERCENT_FEATURES] * 100
    y = data["below_c_rate"].to_numpy(dtype=float) * 100
    groups = data["lad24_name"].astype("string").to_numpy()
    if X.isna().any().any() or np.isnan(y).any():
        raise ValueError("Model data contains missing values")
    if pd.Series(groups).nunique() != 10:
        raise ValueError("Expected ten local-authority groups")
    return X, y, groups


def xgb_estimator(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )


def metric_row(model: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model,
        "n": len(y_true),
        "r2": r2_score(y_true, y_pred),
        "mae_percentage_points": mean_absolute_error(y_true, y_pred),
        "rmse_percentage_points": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def fit_search(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_iter: int,
    inner_splits: int,
) -> RandomizedSearchCV:
    cv = GroupKFold(n_splits=inner_splits)
    search = RandomizedSearchCV(
        estimator=xgb_estimator(seed),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search


def residual_moran(
    predictions: pd.DataFrame,
    boundaries_path: Path,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    boundaries = gpd.read_file(boundaries_path)[["lsoa21", "geometry"]]
    merged = boundaries.merge(predictions, on="lsoa21", how="inner", validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("Residual Moran spatial merge lost LSOAs")
    projected = merged.to_crs(27700)
    weights = Queen.from_dataframe(projected, ids=projected["lsoa21"].tolist())
    weights.transform = "R"
    rows = []
    np.random.seed(seed)
    for model in ["dummy", "random_forest", "xgboost"]:
        residual = projected["observed_below_c_pp"] - projected[f"predicted_{model}_pp"]
        moran = Moran(residual.to_numpy(), weights, permutations=permutations)
        rows.append(
            {
                "model": model,
                "moran_i": moran.I,
                "expected_i": moran.EI,
                "permutation_p_value": moran.p_sim,
                "z_sim": moran.z_sim,
                "permutations": permutations,
                "queen_islands": len(weights.islands),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    model_dir = args.output_root / "models"
    figures_dir = args.output_root / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data, dtype={"lsoa21": "string"})
    X, y, groups = prepare_data(data)
    outer = LeaveOneGroupOut()

    predictions = pd.DataFrame(
        {
            "lsoa21": data["lsoa21"],
            "lsoa21_name": data["lsoa21_name"],
            "lad24_name": data["lad24_name"],
            "observed_below_c_pp": y,
            "predicted_dummy_pp": np.nan,
            "predicted_random_forest_pp": np.nan,
            "predicted_xgboost_pp": np.nan,
        }
    )
    fold_metrics = []
    outer_params = []
    permutation_rows = []

    splits = list(outer.split(X, y, groups))
    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        held_out_lad = str(pd.Series(groups[test_idx]).unique()[0])
        print(
            f"Outer fold {fold_number}/{len(splits)}: held out {held_out_lad}",
            flush=True,
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)

        rf = RandomForestRegressor(
            n_estimators=600,
            max_features=0.8,
            min_samples_leaf=3,
            random_state=args.seed + fold_number,
            n_jobs=-1,
        ).fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        search = fit_search(
            X_train,
            y_train,
            groups_train,
            seed=args.seed + fold_number,
            n_iter=args.n_iter,
            inner_splits=args.inner_splits,
        )
        xgb = search.best_estimator_
        xgb_pred = xgb.predict(X_test)

        predictions.loc[test_idx, "predicted_dummy_pp"] = dummy_pred
        predictions.loc[test_idx, "predicted_random_forest_pp"] = rf_pred
        predictions.loc[test_idx, "predicted_xgboost_pp"] = xgb_pred

        for model_name, pred in [
            ("dummy", dummy_pred),
            ("random_forest", rf_pred),
            ("xgboost", xgb_pred),
        ]:
            row = metric_row(model_name, y_test, pred)
            row.update({"fold": fold_number, "held_out_lad": held_out_lad})
            fold_metrics.append(row)

        outer_params.append(
            {
                "fold": fold_number,
                "held_out_lad": held_out_lad,
                "best_inner_rmse": -search.best_score_,
                "best_params": search.best_params_,
            }
        )

        perm = permutation_importance(
            xgb,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=20,
            random_state=args.seed + fold_number,
            n_jobs=-1,
        )
        for feature, mean, std in zip(
            FEATURES, perm.importances_mean, perm.importances_std, strict=True
        ):
            permutation_rows.append(
                {
                    "fold": fold_number,
                    "held_out_lad": held_out_lad,
                    "feature": feature,
                    "importance_mean_mae_pp": mean,
                    "importance_std_mae_pp": std,
                }
            )

    if predictions.filter(like="predicted_").isna().any().any():
        raise ValueError("Some LSOAs did not receive out-of-fold predictions")

    performance = pd.DataFrame(
        [
            metric_row("dummy", y, predictions["predicted_dummy_pp"].to_numpy()),
            metric_row(
                "random_forest", y, predictions["predicted_random_forest_pp"].to_numpy()
            ),
            metric_row("xgboost", y, predictions["predicted_xgboost_pp"].to_numpy()),
        ]
    )
    performance.to_csv(model_dir / "model_performance_oof.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(model_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(model_dir / "oof_predictions.csv", index=False)
    with (model_dir / "outer_xgb_best_params.json").open("w", encoding="utf-8") as stream:
        json.dump(outer_params, stream, indent=2)

    permutation_folds = pd.DataFrame(permutation_rows)
    permutation_folds.to_csv(model_dir / "permutation_importance_by_fold.csv", index=False)
    permutation_summary = (
        permutation_folds.groupby("feature", as_index=False)
        .agg(
            mean_importance_mae_pp=("importance_mean_mae_pp", "mean"),
            sd_across_folds=("importance_mean_mae_pp", "std"),
            min_importance_mae_pp=("importance_mean_mae_pp", "min"),
            max_importance_mae_pp=("importance_mean_mae_pp", "max"),
        )
        .sort_values("mean_importance_mae_pp", ascending=False)
    )
    permutation_summary.to_csv(model_dir / "permutation_importance_summary.csv", index=False)

    moran = residual_moran(predictions, args.boundaries, args.permutations, args.seed)
    moran.to_csv(model_dir / "oof_residual_moran.csv", index=False)

    print("Tuning final XGBoost on all LSOAs", flush=True)
    final_search = fit_search(
        X,
        y,
        groups,
        seed=args.seed,
        n_iter=max(args.n_iter, 20),
        inner_splits=5,
    )
    final_model = final_search.best_estimator_
    final_model.save_model(model_dir / "final_xgboost_model.json")
    joblib.dump(final_model, model_dir / "final_xgboost_model.joblib")
    final_tuning = {
        "best_grouped_cv_rmse_pp": -final_search.best_score_,
        "best_params": final_search.best_params_,
        "features": FEATURES,
        "percentage_scaled_features": PERCENT_FEATURES,
        "target": "below_c_rate multiplied by 100",
        "grouping": "Local Authority District",
        "inner_splits": 5,
    }
    with (model_dir / "final_xgb_tuning.json").open("w", encoding="utf-8") as stream:
        json.dump(final_tuning, stream, indent=2)

    print("Computing TreeSHAP values", flush=True)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X)
    shap_frame = pd.DataFrame(shap_values.values, columns=[f"shap_{c}" for c in FEATURES])
    shap_frame.insert(0, "base_value_pp", np.asarray(shap_values.base_values))
    shap_frame.insert(0, "lsoa21", data["lsoa21"].to_numpy())
    shap_frame.to_csv(model_dir / "shap_values.csv", index=False)

    shap_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "display_label": [DISPLAY_LABELS[f] for f in FEATURES],
            "mean_absolute_shap_pp": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_pp", ascending=False)
    shap_importance.to_csv(model_dir / "shap_global_importance.csv", index=False)

    display_names = [DISPLAY_LABELS[f] for f in FEATURES]
    display_explanation = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=X.to_numpy(),
        feature_names=display_names,
    )
    shap.plots.bar(display_explanation, max_display=len(FEATURES), show=False)
    plt.title("Global SHAP importance for neighbourhood below-C rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_global_bar.png", bbox_inches="tight", dpi=300)
    plt.close()

    shap.plots.beeswarm(display_explanation, max_display=len(FEATURES), show=False)
    plt.title("SHAP summary: direction and magnitude of model contributions")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.close()

    top_features = shap_importance.head(4)["feature"].tolist()
    for rank, feature in enumerate(top_features, start=1):
        feature_index = FEATURES.index(feature)
        shap.dependence_plot(
            feature_index,
            shap_values.values,
            X.to_numpy(),
            feature_names=display_names,
            interaction_index="auto",
            show=False,
        )
        plt.title(f"SHAP dependence: {DISPLAY_LABELS[feature]}")
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"shap_dependence_{rank}_{feature}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    print("Computing SHAP interaction values", flush=True)
    interactions = np.asarray(explainer.shap_interaction_values(X))
    strength = np.abs(interactions).mean(axis=0)
    rows = []
    for i, first in enumerate(FEATURES):
        for j, second in enumerate(FEATURES):
            if i < j:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "mean_absolute_interaction_shap_pp": strength[i, j],
                    }
                )
    interaction_summary = pd.DataFrame(rows).sort_values(
        "mean_absolute_interaction_shap_pp", ascending=False
    )
    interaction_summary.to_csv(model_dir / "shap_interaction_strength.csv", index=False)
    top_pair = interaction_summary.iloc[0]
    first_index = FEATURES.index(top_pair["feature_1"])
    second_index = FEATURES.index(top_pair["feature_2"])
    shap.dependence_plot(
        first_index,
        shap_values.values,
        X.to_numpy(),
        feature_names=display_names,
        interaction_index=second_index,
        show=False,
    )
    plt.title(
        f"SHAP interaction: {DISPLAY_LABELS[top_pair['feature_1']]} × "
        f"{DISPLAY_LABELS[top_pair['feature_2']]}"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_top_interaction.png", bbox_inches="tight", dpi=300)
    plt.close()

    run_summary = {
        "performance": performance.to_dict(orient="records"),
        "residual_moran": moran.to_dict(orient="records"),
        "final_tuning": final_tuning,
        "shap_top_features": shap_importance.head(5).to_dict(orient="records"),
        "top_shap_interaction": {
            key: (float(value) if isinstance(value, (np.floating, float)) else value)
            for key, value in top_pair.to_dict().items()
        },
    }
    with (model_dir / "model_run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(run_summary, stream, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
