"""Render publication-ready SHAP figures from the saved final-model outputs."""

from __future__ import annotations

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
import pandas as pd
import shap


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


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "lsoa_analytical_dataset.csv"
    model_dir = PROJECT_ROOT / "outputs" / "models"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path, dtype={"lsoa21": "string"})
    saved = pd.read_csv(model_dir / "shap_values.csv", dtype={"lsoa21": "string"})
    merged = data[["lsoa21", *FEATURES]].merge(
        saved,
        on="lsoa21",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(data):
        raise ValueError("Saved SHAP values do not cover the complete analytical sample")

    feature_data = merged[FEATURES].copy()
    feature_data[PERCENT_FEATURES] = feature_data[PERCENT_FEATURES] * 100
    values = merged[[f"shap_{feature}" for feature in FEATURES]].to_numpy()
    base_values = merged["base_value_pp"].to_numpy()
    display_names = [DISPLAY_LABELS[feature] for feature in FEATURES]
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=feature_data.to_numpy(),
        feature_names=display_names,
    )

    shap.plots.beeswarm(
        explanation,
        max_display=len(FEATURES),
        plot_size=(9.2, 5.8),
        show=False,
    )
    figure = plt.gcf()
    axis = figure.axes[0]
    axis.set_title(
        "SHAP contributions to predicted Below-C rate",
        pad=12,
        weight="bold",
        fontsize=13,
    )
    axis.set_xlabel("SHAP contribution (percentage points)", fontsize=10.5)
    axis.tick_params(axis="both", labelsize=9.5)
    axis.axvline(0, color="#555555", linewidth=0.7, zorder=0)
    figure.subplots_adjust(left=0.30, right=0.93, bottom=0.14, top=0.88)
    figure.savefig(figures_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.close(figure)

    importance = pd.read_csv(model_dir / "shap_global_importance.csv")
    top_features = importance.head(4)["feature"].tolist()
    for rank, feature in enumerate(top_features, start=1):
        feature_index = FEATURES.index(feature)
        shap.dependence_plot(
            feature_index,
            values,
            feature_data.to_numpy(),
            feature_names=display_names,
            interaction_index="auto",
            show=False,
            alpha=0.72,
            dot_size=12,
        )
        figure = plt.gcf()
        figure.set_size_inches(7.8, 5.5)
        axis = figure.axes[0]
        axis.set_title(
            DISPLAY_LABELS[feature].replace(" (%)", ""),
            pad=11,
            weight="bold",
            fontsize=13,
        )
        axis.set_xlabel(DISPLAY_LABELS[feature], fontsize=10.5)
        axis.set_ylabel("SHAP contribution\n(percentage points)", fontsize=10.5)
        axis.tick_params(axis="both", labelsize=9.5)
        axis.axhline(0, color="#555555", linewidth=0.7, linestyle="--", zorder=0)
        axis.grid(alpha=0.2, linewidth=0.5)
        if len(figure.axes) > 1:
            colour_axis = figure.axes[1]
            colour_axis.set_ylabel(colour_axis.get_ylabel(), fontsize=10.5)
            colour_axis.tick_params(labelsize=9)
        figure.subplots_adjust(left=0.17, right=0.93, bottom=0.15, top=0.88)
        figure.savefig(
            figures_dir / f"shap_dependence_{rank}_{feature}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
