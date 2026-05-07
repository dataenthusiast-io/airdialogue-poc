import matplotlib
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import config


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config.paths.plots_dir,
        filename,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=config.plots.dpi, bbox_inches="tight", format=config.plots.format)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(report: dict) -> None:
    cls = report["classification"]
    cm_dict = cls["confusion_matrix"]

    true_labels = list(config.classes)
    has_invalid = report["operational"]["schema_failures"] > 0
    pred_labels = true_labels + (["__invalid__"] if has_invalid else [])

    matrix = np.array([
        [cm_dict[tl].get(pl, 0) for pl in pred_labels]
        for tl in true_labels
    ], dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    plt.style.use(config.plots.style)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix_norm,
        annot=True,
        fmt=".2f",
        xticklabels=pred_labels,
        yticklabels=true_labels,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Confusion Matrix\nMacro F1: {cls['macro_f1']:.3f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_f1_per_class(report: dict) -> None:
    classes = list(config.classes)
    f1_values = [report["classification"]["per_class"][c]["f1"] for c in classes]
    colors = [getattr(config.plots.palette.classes, c) for c in classes]

    plt.style.use(config.plots.style)
    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(classes, f1_values, color=colors, width=0.5)
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )

    ax.axhline(
        y=config.plots.h1_threshold,
        color="red", linestyle="--", linewidth=1.2,
        label=f"H1 threshold ({config.plots.h1_threshold})",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.legend()
    plt.tight_layout()
    _save(fig, "f1_per_class.png")


def plot_token_distribution(df: pd.DataFrame) -> None:
    prompt_tokens = df["prompt_tokens"].dropna().tolist()
    median_val = float(np.median(prompt_tokens))

    plt.style.use(config.plots.style)
    fig, ax = plt.subplots(figsize=(5, 5))

    bp = ax.boxplot(
        [prompt_tokens],
        labels=["Prompt tokens"],
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )
    bp["boxes"][0].set_facecolor(config.plots.palette.primary)
    bp["boxes"][0].set_alpha(0.7)

    ax.annotate(
        f"median={median_val:.0f}",
        xy=(1, median_val),
        xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=9,
    )

    ax.set_ylabel("Token Count")
    ax.set_title("Prompt Token Distribution")
    plt.tight_layout()
    _save(fig, "token_distribution.png")


def plot_cost_projection(report: dict) -> None:
    benchmark = config.costs.industrial_scale_benchmark
    mean_cost = report["operational"]["mean_cost_usd"]

    x = np.linspace(0, benchmark, 500)
    y = x * mean_cost
    proj_val = benchmark * mean_cost

    plt.style.use(config.plots.style)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, y, color=config.plots.palette.primary, linewidth=2)
    ax.annotate(
        f"${proj_val:.2f}",
        xy=(benchmark, proj_val),
        xytext=(-50, 8), textcoords="offset points",
        color=config.plots.palette.primary, fontsize=9, fontweight="bold",
    )

    ax.set_xlabel("Number of Interactions")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title(f"Cost Projection up to {benchmark:,} Interactions")
    plt.tight_layout()
    _save(fig, "cost_projection.png")


def generate_all_plots(report: dict, df: pd.DataFrame) -> None:
    print("\nGenerating plots...")
    plot_confusion_matrix(report)
    plot_f1_per_class(report)
    plot_token_distribution(df)
    plot_cost_projection(report)
    print("All plots generated.")
