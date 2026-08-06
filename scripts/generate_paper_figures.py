"""Generate figure images (PNG) for the IEEE research paper.

Produces PNGs under myDocs/ieee_paper/figures/ using real project data
(human_dataset.csv, ai_dataset.csv, models/best_model.joblib) plus
schematic diagrams (architecture / pipeline / protocol) drawn with
matplotlib so the paper has actual embedded figures instead of text
placeholders.

Run:

    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER, ModelTrainer  # noqa: E402

FIG_DIR = PROJECT_ROOT / "myDocs" / "ieee_paper" / "figures"
HUMAN_CSV = PROJECT_ROOT / "dataset" / "human_dataset.csv"
AI_CSV = PROJECT_ROOT / "dataset" / "ai_dataset.csv"
BEST_MODEL = PROJECT_ROOT / "models" / "best_model.joblib"

# Column-width sizing for a two-column IEEE layout (~3.3 in usable width).
COL_W = 3.35
sns.set_theme(style="whitegrid", font_scale=0.75)
BOX_FC = "#eaf1fb"
BOX_EC = "#1f4e79"
ARROW_C = "#1f4e79"


def _new_ax(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    return fig, ax


def _box(ax, xy, w, h, text, *, fc=BOX_FC, ec=BOX_EC, fontsize=7.2) -> None:
    x, y = xy
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0b1f1a",
        wrap=True,
    )


def _arrow(ax, start, end, *, color=ARROW_C) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color,
    )
    ax.add_patch(arrow)


def fig1_architecture() -> Path:
    """System architecture: captions -> features -> model -> app."""
    fig, ax = _new_ax((COL_W, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")

    _box(ax, (0.3, 13.0), 4.2, 1.4, "COCO Human\nCaptions", fc="#eef7f0")
    _box(ax, (5.5, 13.0), 4.2, 1.4, "BLIP AI\nCaptions", fc="#fdf1e6")
    _box(ax, (2.6, 10.6), 4.8, 1.4, "Sentence-BERT\nEmbeddings")
    _box(ax, (2.6, 8.2), 4.8, 1.4, "Caption Diversity\n(1 - avg. cosine sim.)")
    _box(ax, (0.3, 5.8), 4.2, 1.4, "OpenCV Visual\nFeatures")
    _box(ax, (5.5, 5.8), 4.2, 1.4, "Ambiguity Label\n(Low / Med / High)")
    _box(ax, (2.6, 3.4), 4.8, 1.4, "Feature Vector (11-D)\n+ Random Forest / XGBoost")
    _box(ax, (2.6, 1.0), 4.8, 1.4, "SHAP Explanation")
    _box(ax, (0.1, -1.4), 4.6, 1.4, "FastAPI Service", fc="#eaf1fb")
    _box(ax, (5.3, -1.4), 4.6, 1.4, "React \u201cAmbiguity\nLens\u201d UI", fc="#eaf1fb")
    ax.set_ylim(-2.0, 15)

    _arrow(ax, (2.4, 13.0), (4.2, 12.0))
    _arrow(ax, (7.6, 13.0), (5.9, 12.0))
    _arrow(ax, (5.0, 10.6), (5.0, 9.6))
    _arrow(ax, (5.0, 8.2), (2.4, 6.5))
    _arrow(ax, (5.0, 8.2), (7.6, 7.2))
    _arrow(ax, (2.4, 5.8), (4.9, 4.8))
    _arrow(ax, (7.6, 5.8), (5.1, 4.8))
    _arrow(ax, (5.0, 3.4), (5.0, 2.4))
    _arrow(ax, (4.3, 1.0), (2.4, 0.0))
    _arrow(ax, (5.7, 1.0), (7.6, 0.0))

    fig.tight_layout(pad=0.2)
    out = FIG_DIR / "fig1_architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2_pipeline() -> Path:
    """Methodology pipeline: A-E stages."""
    fig, ax = _new_ax((COL_W, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    stages = [
        ("A. Caption\nAcquisition", "COCO human +\nBLIP AI captions"),
        ("B. Diversity\nScoring", "Sentence-BERT +\ncosine similarity"),
        ("C. Visual Feature\nExtraction", "OpenCV descriptors"),
        ("D. Ambiguity\nLabeling", "Threshold rule\n(0.35 / 0.65)"),
        ("E. Learning &\nExplanation", "RF / XGBoost +\nSHAP"),
    ]
    y = 10.2
    for i, (title, sub) in enumerate(stages):
        _box(ax, (0.4, y), 4.4, 1.5, title, fontsize=7.5)
        ax.text(5.2, y + 0.75, sub, ha="left", va="center", fontsize=6.6, color="#333333")
        if i < len(stages) - 1:
            _arrow(ax, (2.6, y), (2.6, y - 0.5))
        y -= 2.2

    fig.tight_layout(pad=0.2)
    out = FIG_DIR / "fig2_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_class_distribution() -> Path:
    df = pd.read_csv(HUMAN_CSV)
    counts = df["ambiguity_label"].value_counts().reindex(LABEL_ORDER).fillna(0)

    fig, ax = _new_ax((COL_W, 2.6))
    colors = ["#1f8a70", "#e0a45c", "#c45c26"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts.values) * 0.02,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Number of images", fontsize=8)
    ax.set_title("Human-labeled ambiguity class distribution (N={})".format(int(counts.sum())), fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5)
    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "fig3_class_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_correlation_heatmap() -> Path:
    df = pd.read_csv(HUMAN_CSV)
    corr = df[list(FEATURE_COLUMNS)].corr()

    fig, ax = _new_ax((COL_W, 3.4))
    short_labels = [c.replace("_", "\n") for c in FEATURE_COLUMNS]
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    ax.tick_params(axis="x", labelsize=5.4, rotation=90)
    ax.tick_params(axis="y", labelsize=5.4, rotation=0)
    ax.set_title("Feature correlation matrix (human dataset)", fontsize=8)
    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "fig4_correlation_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_experimental_protocol() -> Path:
    fig, ax = _new_ax((COL_W, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    steps = [
        "Load human_dataset.csv\n(11 features + label)",
        "Stratified 80/20\ntrain/test split",
        "Oversample minority\nclasses (train only)",
        "Grid-search CV\n(Random Forest / XGBoost)",
        "Evaluate on held-out\ntest set",
        "Compare human vs. AI\ncaption diversity",
        "Inspect SHAP\nexplanations",
    ]
    y = 11.0
    step_h = 1.35
    for i, text in enumerate(steps):
        _box(ax, (2.0, y), 6.0, step_h, text, fontsize=7.0)
        if i < len(steps) - 1:
            _arrow(ax, (5.0, y), (5.0, y - (1.6 - step_h)))
        y -= 1.6

    fig.tight_layout(pad=0.2)
    out = FIG_DIR / "fig5_experimental_protocol.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6_human_vs_ai_diversity() -> Path:
    human = pd.read_csv(HUMAN_CSV)
    human_mean = human["caption_diversity"].mean()
    human_std = human["caption_diversity"].std()

    if AI_CSV.is_file():
        ai = pd.read_csv(AI_CSV)
        ai_mean = ai["caption_diversity"].mean()
        ai_std = ai["caption_diversity"].std()
        ai_n = len(ai)
    else:
        ai_mean, ai_std, ai_n = 0.0, 0.0, 0

    fig, ax = _new_ax((COL_W, 2.6))
    labels = [f"Human\n(N={len(human)})", f"AI / BLIP\n(N={ai_n})"]
    means = [human_mean, ai_mean]
    stds = [human_std, ai_std]
    colors = ["#1f8a70", "#c45c26"]
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.35, color="#888888", linestyle="--", linewidth=0.8)
    ax.axhline(0.65, color="#888888", linestyle="--", linewidth=0.8)
    ax.text(1.55, 0.36, "Low/Med", fontsize=6, color="#666666")
    ax.text(1.55, 0.66, "Med/High", fontsize=6, color="#666666")
    ax.set_ylabel("Mean caption diversity", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Human vs. AI caption diversity", fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5)
    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "fig6_human_vs_ai_diversity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_confusion_and_importance() -> Path:
    trainer = ModelTrainer()
    df = trainer.load_dataset(HUMAN_CSV)
    X, y = trainer.prepare_data(df)
    _, X_test, _, y_test = trainer.split_data(X, y)

    model = joblib.load(BEST_MODEL)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABEL_ORDER))))

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 5.6), dpi=220)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar=False,
        annot_kws={"fontsize": 8},
        square=True,
    )
    axes[0].set_xlabel("Predicted", fontsize=7.5)
    axes[0].set_ylabel("True", fontsize=7.5)
    axes[0].set_title("(a) Confusion matrix (test set)", fontsize=8)
    axes[0].tick_params(axis="both", labelsize=7)

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        importances = importances.sort_values(ascending=True)
        axes[1].barh(
            [c.replace("_", " ") for c in importances.index],
            importances.values,
            color="#1f8a70",
        )
        axes[1].set_title("(b) Feature importance", fontsize=8)
        axes[1].tick_params(axis="both", labelsize=6.5)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Feature importance\nnot available", ha="center", va="center")

    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "fig7_confusion_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        fig1_architecture(),
        fig2_pipeline(),
        fig3_class_distribution(),
        fig4_correlation_heatmap(),
        fig5_experimental_protocol(),
        fig6_human_vs_ai_diversity(),
        fig7_confusion_and_importance(),
    ]
    for path in generated:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
