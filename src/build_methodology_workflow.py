"""Create the two-strand analytical workflow used in the methodology chapter."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "methodology_workflow.png"


COLORS = {
    "navy": "#173B57",
    "blue": "#2E74B5",
    "pale_blue": "#EAF2F8",
    "teal": "#2A7F86",
    "pale_teal": "#E8F4F3",
    "gold": "#C58B2A",
    "pale_gold": "#FBF2DF",
    "ink": "#26333D",
    "muted": "#65727C",
    "line": "#9AABB8",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, text, face, edge, fontsize=9.5, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight="bold" if bold else "normal",
        linespacing=1.18,
    )
    return patch


def arrow(ax, start, end, color=None, width=1.25):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=color or COLORS["line"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.4, 7.6), dpi=220)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Research design: two complementary analytical strands",
        ha="center",
        va="top",
        fontsize=16,
        weight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.5,
        0.925,
        "Spatial description establishes where inequalities occur; explainable modelling examines which neighbourhood characteristics are associated with them.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    box(
        ax,
        0.20,
        0.805,
        0.60,
        0.080,
        "Research questions\nRQ1 spatial patterns   |   RQ2 dwelling characteristics   |   RQ3 deprivation",
        COLORS["pale_gold"],
        COLORS["gold"],
        10,
        True,
    )
    box(
        ax,
        0.20,
        0.680,
        0.60,
        0.083,
        "Integrated LSOA dataset (n = 1,702)\nEPC + Census tenure + IMD + postcode lookup + LSOA boundaries",
        COLORS["pale_blue"],
        COLORS["blue"],
        10,
        True,
    )
    arrow(ax, (0.50, 0.805), (0.50, 0.765))

    ax.text(0.255, 0.635, "STRAND A  •  RQ1", ha="center", fontsize=10.5, weight="bold", color=COLORS["blue"])
    ax.text(0.745, 0.635, "STRAND B  •  RQ2 + RQ3", ha="center", fontsize=10.5, weight="bold", color=COLORS["teal"])
    arrow(ax, (0.43, 0.680), (0.255, 0.610), COLORS["blue"])
    arrow(ax, (0.57, 0.680), (0.745, 0.610), COLORS["teal"])

    left = [
        (0.075, 0.515, "Describe and map\nBelow-C rate, score and coverage"),
        (0.075, 0.385, "Test global clustering\nGlobal Moran's I"),
        (0.075, 0.255, "Locate clusters and outliers\nFDR-corrected Local Moran's I"),
    ]
    right = [
        (0.575, 0.515, "Explore X–Y relationships\nshape, direction and collinearity"),
        (0.575, 0.385, "Fit and validate models\nXGBoost + grouped CV; RF robustness"),
        (0.575, 0.255, "Explain fitted relationships\nSHAP + permutation importance + sensitivity"),
    ]
    for x, y, text in left:
        box(ax, x, y, 0.36, 0.085, text, COLORS["pale_blue"], COLORS["blue"], 9.4)
    for x, y, text in right:
        box(ax, x, y, 0.36, 0.085, text, COLORS["pale_teal"], COLORS["teal"], 9.4)

    for x in (0.255, 0.755):
        arrow(ax, (x, 0.515), (x, 0.472))
        arrow(ax, (x, 0.385), (x, 0.342))

    box(
        ax,
        0.20,
        0.085,
        0.60,
        0.105,
        "Integrated interpretation and policy implications\nWhere are inefficient neighbourhoods, which housing and deprivation patterns recur,\nand what does this imply for place-based retrofit targeting?",
        COLORS["pale_gold"],
        COLORS["gold"],
        10,
        True,
    )
    arrow(ax, (0.255, 0.255), (0.405, 0.190), COLORS["blue"])
    arrow(ax, (0.755, 0.255), (0.595, 0.190), COLORS["teal"])

    ax.text(
        0.5,
        0.025,
        "Interpretive boundary: model reliability is checked, but forecasting and causal estimation are not research objectives.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
        style="italic",
    )

    plt.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    build()
