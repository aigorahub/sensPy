"""Render static proof-object charts for the sensPy poster."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT / "poster"
CHART_DATA = POSTER / "chart_data"
CHARTS = POSTER / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from senspy import psy_fun  # noqa: E402

CANVAS = "#f4f0e6"
INK = "#17291f"
CORAL = "#c7563f"
GREEN = "#4c8a61"
SAGE = "#9aa79b"
MIST = "#e7ebe4"
CLAY = "#d69c8f"
GREY = "#8a918a"

plt.rcParams.update(
    {
        "figure.facecolor": CANVAS,
        "axes.facecolor": "#fbfaf6",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "axes.titleweight": "normal",
        "savefig.facecolor": CANVAS,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(CHARTS / name, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def protocol_coverage() -> None:
    rows = read_csv(CHART_DATA / "protocol_coverage.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(-0.6, 2.2)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.axis("off")

    ax.text(0.55, len(rows) - 0.05, "single", ha="center", va="bottom", fontsize=16)
    ax.text(1.55, len(rows) - 0.05, "double", ha="center", va="bottom", fontsize=16)

    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.45, y, row["display"], ha="left", va="center", fontsize=15.5)
        for j, key in enumerate(["single", "double"]):
            available = int(row[key])
            color = GREEN if available else "#ded8cd"
            edge = GREEN if available else "#c7c0b4"
            ax.add_patch(
                FancyBboxPatch(
                    (0.26 + j, y - 0.25),
                    0.58,
                    0.5,
                    boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=1.0,
                )
            )
            ax.text(
                0.55 + j,
                y,
                "yes" if available else "-",
                ha="center",
                va="center",
                fontsize=13.5,
                color="white" if available else GREY,
                weight="bold" if available else "normal",
            )
        ax.text(
            2.0,
            y,
            f"p0={float(row['p_guess']):.2g}",
            ha="left",
            va="center",
            fontsize=13,
            color=GREY,
        )

    save(fig, "protocol_coverage.png")


def psychometric_curves() -> None:
    fig, ax = plt.subplots(figsize=(8.1, 5.1))
    d = np.linspace(0, 4, 180)
    methods = [
        ("twoafc", "2-AFC", GREEN),
        ("triangle", "Triangle", CORAL),
        ("duotrio", "Duo-trio", SAGE),
        ("threeafc", "3-AFC", "#6f7f77"),
        ("tetrad", "Tetrad", "#b77768"),
    ]
    for method, label, color in methods:
        ax.plot(d, psy_fun(d, method=method), label=label, lw=2.8, color=color)

    ax.set_title("Common d-prime scale across sensory protocols", fontsize=19, pad=13)
    ax.set_xlabel("d-prime")
    ax.set_ylabel("proportion correct")
    ax.set_xlim(0, 4)
    ax.set_ylim(0.25, 1.01)
    ax.grid(True, color="#e8e0d5", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=13)
    save(fig, "psychometric_curves.png")


def test_inventory() -> None:
    rows = read_csv(CHART_DATA / "test_inventory.csv")
    top = sorted(rows, key=lambda r: int(r["test_functions"]), reverse=True)[:10]
    labels = [r["category"] for r in top][::-1]
    values = [int(r["test_functions"]) for r in top][::-1]
    colors = [CORAL if "sensr" in label else GREEN if "coverage" in label else SAGE for label in labels]

    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    bars = ax.barh(labels, values, color=colors, alpha=0.92)
    ax.set_title("Validation surface spans unit, coverage, and parity tests", fontsize=19, pad=13)
    ax.set_xlabel("test functions")
    ax.grid(axis="x", color="#e8e0d5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=12.5)

    save(fig, "test_inventory.png")


def roc_bridge() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fpr = np.linspace(0.001, 0.999, 240)
    z = stats.norm.ppf(fpr)
    for d_prime, color in [(0.8, SAGE), (1.5, GREEN), (2.2, CORAL)]:
        tpr = stats.norm.cdf(z + d_prime)
        auc = stats.norm.cdf(d_prime / np.sqrt(2))
        ax.plot(fpr, tpr, lw=2.8, color=color, label=f"d={d_prime:.1f}, AUC={auc:.2f}")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.3, label="chance")
    ax.set_title("ROC analysis remains in the Python workflow", fontsize=18.5, pad=13)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e8e0d5", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=12.5)
    save(fig, "roc_bridge.png")


def architecture_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    stages = [
        ("sensR\nreference", "gold standard\nR outputs"),
        ("golden\nfixtures", "portable parity\nchecks"),
        ("SciPy\nkernels", "optimize + stats\nnative Python"),
        ("dataclass\nresults", "typed estimates\nand intervals"),
        ("Plotly\nfigures", "interactive\nanalysis"),
    ]
    xs = np.linspace(0.05, 0.82, len(stages))
    w, h = 0.14, 0.52
    for i, ((title, sub), x) in enumerate(zip(stages, xs, strict=True)):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.26),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                transform=ax.transAxes,
                facecolor="#fbfaf6",
                edgecolor=CORAL if i in (0, 2) else "#c8c0b4",
                linewidth=1.6,
            )
        )
        ax.text(x + w / 2, 0.58, title, transform=ax.transAxes, ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x + w / 2, 0.40, sub, transform=ax.transAxes, ha="center", va="center", fontsize=11.2, color="#4f5c53")
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.012, 0.52),
                    (xs[i + 1] - 0.012, 0.52),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color=INK,
                    linewidth=1.2,
                )
            )
    ax.text(0.05, 0.12, "Validation is an architecture feature, not an afterthought.", transform=ax.transAxes, fontsize=14.5, color=CORAL, weight="bold")
    save(fig, "architecture_pipeline.png")


def main() -> None:
    protocol_coverage()
    psychometric_curves()
    test_inventory()
    roc_bridge()
    architecture_pipeline()
    print(f"[charts] wrote charts to {CHARTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
