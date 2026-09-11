"""Run descriptive and spatial analysis for Greater Manchester LSOAs."""

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import seaborn as sns
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from matplotlib.patches import Patch


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

DISPLAY_LABELS = {
    "pre_1930_pct": "Pre-1930 dwellings",
    "post_1990_pct": "Post-1990 dwellings",
    "flat_maisonette_pct": "Flats/maisonettes",
    "terraced_pct": "Terraced dwellings",
    "median_floor_area": "Median floor area",
    "electric_fuel_pct": "Electric main fuel",
    "private_rented_pct": "Private rented",
    "social_rented_pct": "Social rented",
    "imd_score": "IMD score",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analytical-data",
        type=Path,
        default=root / "data" / "processed" / "lsoa_analytical_dataset.csv",
    )
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=root / "data" / "spatial" / "gm_lsoa_2021_bgc_v5.gpkg",
    )
    parser.add_argument(
        "--output-root", type=Path, default=root / "outputs"
    )
    parser.add_argument(
        "--spatial-output",
        type=Path,
        default=root / "data" / "processed" / "lsoa_spatial_analysis.gpkg",
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )


def save_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output: Path,
    cmap: str,
    legend_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.08,
        edgecolor="#ffffff",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"label": legend_label, "shrink": 0.66},
        ax=ax,
    )
    add_lad_context(gdf, ax)
    ax.set_title(title, pad=12)
    ax.set_axis_off()
    ax.text(
        0,
        -0.015,
        "Source: author's analysis; ONS 2021 LSOA BGC V5 boundaries.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def add_lad_context(gdf: gpd.GeoDataFrame, ax: plt.Axes) -> None:
    """Overlay LAD boundaries and unobtrusive labels for map orientation."""
    if "lad24_name" not in gdf.columns:
        return
    lad = gdf.dissolve(by="lad24_name")
    lad.boundary.plot(ax=ax, color="#4a4a4a", linewidth=0.55, zorder=3)
    for name, point in lad.geometry.representative_point().items():
        label = ax.text(
            point.x,
            point.y,
            str(name),
            ha="center",
            va="center",
            fontsize=6.8,
            color="#222222",
            zorder=4,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.2, foreground="white", alpha=0.9)]
        )


def main() -> None:
    args = parse_args()
    descriptive_dir = args.output_root / "descriptive"
    spatial_dir = args.output_root / "spatial"
    figures_dir = args.output_root / "figures"
    for directory in [descriptive_dir, spatial_dir, figures_dir, args.spatial_output.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    setup_style()
    data = pd.read_csv(args.analytical_data, dtype={"lsoa21": "string"})
    boundaries = gpd.read_file(args.boundaries)
    boundaries["lsoa21"] = boundaries["lsoa21"].astype("string")
    if boundaries["lsoa21"].duplicated().any():
        raise ValueError("Boundary file contains duplicate LSOA codes")

    gdf = boundaries.merge(data, on="lsoa21", how="left", validate="one_to_one")
    if len(gdf) != 1702 or gdf["below_c_rate"].isna().any():
        raise ValueError("Spatial merge did not produce 1,702 complete LSOAs")

    projected = gdf.to_crs(27700)
    weights = Queen.from_dataframe(projected, ids=projected["lsoa21"].tolist())
    weights.transform = "R"
    islands = list(weights.islands)

    np.random.seed(args.seed)
    global_rows = []
    for outcome in ["below_c_rate", "mean_current_energy_efficiency"]:
        values = projected[outcome].to_numpy(dtype=float)
        result = Moran(values, weights, permutations=args.permutations)
        global_rows.append(
            {
                "outcome": outcome,
                "n_lsoa": len(values),
                "moran_i": result.I,
                "expected_i": result.EI,
                "permutation_p_value": result.p_sim,
                "z_sim": result.z_sim,
                "permutations": args.permutations,
                "queen_islands": len(islands),
            }
        )
    global_moran = pd.DataFrame(global_rows)
    global_moran.to_csv(spatial_dir / "global_moran.csv", index=False)

    y = projected["below_c_rate"].to_numpy(dtype=float)
    local = Moran_Local(y, weights, permutations=args.permutations, seed=args.seed)
    p_fdr = fdr_bh(local.p_sim)
    quadrant_labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    significant = p_fdr < 0.05
    lisa_cluster = np.where(
        significant,
        pd.Series(local.q).map(quadrant_labels).to_numpy(),
        "Not significant",
    )
    projected["local_moran_i"] = local.Is
    projected["local_p_sim"] = local.p_sim
    projected["local_p_fdr"] = p_fdr
    projected["lisa_quadrant"] = pd.Series(local.q).map(quadrant_labels).to_numpy()
    projected["lisa_cluster"] = lisa_cluster

    lisa_columns = [
        "lsoa21",
        "lsoa21_name_x",
        "lad24_name",
        "below_c_rate",
        "local_moran_i",
        "local_p_sim",
        "local_p_fdr",
        "lisa_quadrant",
        "lisa_cluster",
    ]
    lisa = projected[lisa_columns].rename(columns={"lsoa21_name_x": "lsoa21_name"})
    lisa.to_csv(spatial_dir / "lisa_results.csv", index=False)

    # Coverage-restricted spatial sensitivity check. This mirrors the model
    # sensitivity sample and tests whether RQ1 clustering depends on LSOAs with
    # unusually low or high EPC-to-Census household coverage.
    coverage_subset = projected.loc[
        projected["epc_coverage_rate"].between(0.25, 1.25, inclusive="both")
    ].copy()
    coverage_weights = Queen.from_dataframe(
        coverage_subset, ids=coverage_subset["lsoa21"].tolist()
    )
    coverage_weights.transform = "R"
    coverage_y = coverage_subset["below_c_rate"].to_numpy(dtype=float)
    np.random.seed(args.seed)
    coverage_global = Moran(
        coverage_y, coverage_weights, permutations=args.permutations
    )
    coverage_local = Moran_Local(
        coverage_y,
        coverage_weights,
        permutations=args.permutations,
        seed=args.seed,
    )
    coverage_p_fdr = fdr_bh(coverage_local.p_sim)
    coverage_significant = coverage_p_fdr < 0.05
    coverage_clusters = np.where(
        coverage_significant,
        pd.Series(coverage_local.q).map(quadrant_labels).to_numpy(),
        "Not significant",
    )
    coverage_subset["coverage_local_moran_i"] = coverage_local.Is
    coverage_subset["coverage_local_p_sim"] = coverage_local.p_sim
    coverage_subset["coverage_local_p_fdr"] = coverage_p_fdr
    coverage_subset["coverage_lisa_quadrant"] = (
        pd.Series(coverage_local.q).map(quadrant_labels).to_numpy()
    )
    coverage_subset["coverage_lisa_cluster"] = coverage_clusters
    coverage_subset["same_cluster_label_as_main"] = (
        coverage_subset["coverage_lisa_cluster"] == coverage_subset["lisa_cluster"]
    )
    coverage_subset[
        [
            "lsoa21",
            "lad24_name",
            "epc_coverage_rate",
            "below_c_rate",
            "lisa_cluster",
            "coverage_local_moran_i",
            "coverage_local_p_sim",
            "coverage_local_p_fdr",
            "coverage_lisa_quadrant",
            "coverage_lisa_cluster",
            "same_cluster_label_as_main",
        ]
    ].to_csv(spatial_dir / "coverage_sensitivity_lisa_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "outcome": "below_c_rate",
                "n_lsoa": len(coverage_subset),
                "coverage_min": 0.25,
                "coverage_max": 1.25,
                "moran_i": coverage_global.I,
                "expected_i": coverage_global.EI,
                "permutation_p_value": coverage_global.p_sim,
                "z_sim": coverage_global.z_sim,
                "permutations": args.permutations,
                "queen_islands": len(coverage_weights.islands),
            }
        ]
    ).to_csv(spatial_dir / "coverage_sensitivity_global_moran.csv", index=False)

    descriptive = data[
        ["below_c_rate", "mean_current_energy_efficiency", "epc_coverage_rate", *FEATURES]
    ].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    descriptive.to_csv(descriptive_dir / "lsoa_summary_statistics.csv")

    by_imd = (
        data.groupby("imd_decile", as_index=False)
        .agg(
            n_lsoa=("lsoa21", "size"),
            mean_below_c_rate=("below_c_rate", "mean"),
            median_below_c_rate=("below_c_rate", "median"),
            mean_epc_score=("mean_current_energy_efficiency", "mean"),
        )
        .sort_values("imd_decile")
    )
    by_imd.to_csv(descriptive_dir / "outcomes_by_imd_decile.csv", index=False)

    by_lad = (
        data.groupby("lad24_name", as_index=False)
        .agg(
            n_lsoa=("lsoa21", "size"),
            n_epc=("n_epc", "sum"),
            mean_below_c_rate=("below_c_rate", "mean"),
            median_below_c_rate=("below_c_rate", "median"),
            mean_epc_score=("mean_current_energy_efficiency", "mean"),
            median_coverage=("epc_coverage_rate", "median"),
        )
        .sort_values("mean_below_c_rate", ascending=False)
    )
    by_lad.to_csv(descriptive_dir / "outcomes_by_local_authority.csv", index=False)

    corr = data[["below_c_rate", *FEATURES]].corr(method="spearman")
    corr.to_csv(descriptive_dir / "candidate_feature_spearman.csv")

    map_gdf = projected.to_crs(4326)
    map_gdf["below_c_rate_pct"] = map_gdf["below_c_rate"] * 100
    save_map(
        map_gdf,
        "below_c_rate_pct",
        "Residential energy inefficiency across Greater Manchester",
        figures_dir / "map_below_c_rate.png",
        "OrRd",
        "Dwellings below EPC C (%)",
        vmin=0,
        vmax=90,
    )
    save_map(
        map_gdf,
        "mean_current_energy_efficiency",
        "Mean current EPC score across Greater Manchester",
        figures_dir / "map_mean_epc_score.png",
        "YlGnBu",
        "Mean current EPC score",
    )
    save_map(
        map_gdf,
        "epc_coverage_rate",
        "EPC coverage relative to Census 2021 households",
        figures_dir / "map_epc_coverage.png",
        "viridis",
        "EPC dwellings / Census households",
        vmin=0,
        vmax=float(data["epc_coverage_rate"].quantile(0.99)),
    )

    colors = {
        "High-High": "#b2182b",
        "Low-Low": "#2166ac",
        "High-Low": "#ef8a62",
        "Low-High": "#67a9cf",
        "Not significant": "#d9d9d9",
    }
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    map_gdf.assign(_color=map_gdf["lisa_cluster"].map(colors)).plot(
        color=map_gdf["lisa_cluster"].map(colors),
        linewidth=0.08,
        edgecolor="#ffffff",
        ax=ax,
    )
    add_lad_context(map_gdf, ax)
    present_clusters = set(map_gdf["lisa_cluster"].dropna().astype(str))
    ax.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for label, color in colors.items()
            if label in present_clusters
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
        title="FDR-significant LISA cluster",
    )
    ax.set_title("Local spatial clusters of residential energy inefficiency", pad=12)
    ax.set_axis_off()
    ax.text(
        0,
        -0.015,
        "Local Moran's I, 999 permutations, Benjamini-Hochberg FDR p < 0.05.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "map_lisa_clusters.png", bbox_inches="tight")
    plt.close(fig)

    plot_data = data.copy()
    plot_data["below_c_rate_pct"] = plot_data["below_c_rate"] * 100
    decile_counts = plot_data["imd_decile"].value_counts().sort_index()
    decile_order = decile_counts.index.tolist()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.boxplot(
        data=plot_data,
        x="imd_decile",
        y="below_c_rate_pct",
        order=decile_order,
        color="#8db6d9",
        fliersize=1.5,
        linewidth=0.8,
        ax=ax,
    )
    decile_means = (
        plot_data.groupby("imd_decile")["below_c_rate_pct"]
        .mean()
        .reindex(decile_order)
    )
    ax.scatter(
        range(len(decile_order)),
        decile_means,
        marker="D",
        s=28,
        color="#202020",
        edgecolor="white",
        linewidth=0.45,
        label="Mean",
        zorder=4,
    )
    ax.set_xticks(range(len(decile_order)))
    ax.set_xticklabels(
        [f"{int(decile)}\n(n={int(decile_counts.loc[decile])})" for decile in decile_order]
    )
    ax.set_xlabel("IMD decile (1 = most deprived, 10 = least deprived)")
    ax.set_ylabel("Dwellings below EPC C (%)")
    ax.set_title("Residential energy inefficiency by deprivation decile")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "below_c_by_imd_decile.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["Below-C rate", *[DISPLAY_LABELS[feature] for feature in FEATURES]]
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Spearman correlations among outcome and candidate features", pad=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "candidate_feature_correlation.png", bbox_inches="tight")
    plt.close(fig)

    z = (y - y.mean()) / y.std(ddof=0)
    lag_z = weights.sparse @ z
    global_below_c = global_moran.loc[global_moran["outcome"].eq("below_c_rate")].iloc[0]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    sns.regplot(x=z, y=lag_z, scatter_kws={"s": 13, "alpha": 0.55}, ax=ax)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Standardised below-C rate")
    ax.set_ylabel("Spatial lag of standardised below-C rate")
    ax.set_title("Moran scatter plot for residential energy inefficiency")
    ax.text(
        0.03,
        0.97,
        f"Moran's I = {global_below_c['moran_i']:.3f}\n"
        f"Permutation p = {global_below_c['permutation_p_value']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "moran_scatter_below_c.png", bbox_inches="tight")
    plt.close(fig)

    projected.to_crs(4326).to_file(
        args.spatial_output, layer="lsoa_spatial_analysis", driver="GPKG"
    )
    run_summary = {
        "n_lsoa": len(projected),
        "queen_islands": islands,
        "global_moran": global_rows,
        "lisa_cluster_counts": {
            str(key): int(value)
            for key, value in pd.Series(lisa_cluster).value_counts().items()
        },
        "coverage_sensitivity": {
            "coverage_range": [0.25, 1.25],
            "n_lsoa": len(coverage_subset),
            "queen_islands": list(coverage_weights.islands),
            "global_moran_below_c": {
                "moran_i": coverage_global.I,
                "expected_i": coverage_global.EI,
                "permutation_p_value": coverage_global.p_sim,
            },
            "lisa_cluster_counts": {
                str(key): int(value)
                for key, value in pd.Series(coverage_clusters).value_counts().items()
            },
            "same_cluster_label_as_main": int(
                coverage_subset["same_cluster_label_as_main"].sum()
            ),
        },
    }
    with (spatial_dir / "spatial_analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(run_summary, stream, indent=2)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
