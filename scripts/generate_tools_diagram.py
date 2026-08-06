"""Generate a visual architecture / tool-stack diagram for the project.

Draws a colour-coded flowchart (matplotlib) showing every module and the
underlying library/tool it relies on, from raw COCO data all the way to
the React UI, and saves it as a PNG.

Run:

    python scripts/generate_tools_diagram.py

Produces: results/figures/tools_architecture_diagram.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "tools_architecture_diagram.png"

# Category -> colour (fill, edge)
CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "data": ("#DCEBFB", "#1F4E79"),
    "nlp": ("#E4D9F5", "#5B2C8D"),
    "vision": ("#FDE7CF", "#B15C00"),
    "pipeline": ("#E1F2E1", "#256029"),
    "model": ("#FBE1E1", "#A32020"),
    "explain": ("#FFF6CC", "#8A6D00"),
    "serve": ("#DDF2F0", "#0F6B63"),
    "ui": ("#E6E6FA", "#3B3B98"),
}

CATEGORY_LABELS = {
    "data": "Data Source",
    "nlp": "NLP / Embeddings",
    "vision": "Computer Vision",
    "pipeline": "Dataset Pipeline",
    "model": "Modeling",
    "explain": "Explainability",
    "serve": "Serving / API",
    "ui": "Frontend / UI",
}

# (x, y, width, height, title, subtitle(tools), category)
Box = tuple[float, float, float, float, str, str, str]

BOXES: list[Box] = [
    (0.5, 12.6, 3.2, 0.9, "COCO val2017", "images + human captions (dataset/)", "data"),

    (0.3, 11.0, 3.6, 0.9, "CocoDatasetLoader", "pycocotools \u00b7 Pillow", "data"),
    (4.3, 11.0, 3.7, 0.9, "BlipCaptionGenerator", "transformers (BLIP) \u00b7 torch", "nlp"),
    (8.4, 11.0, 3.9, 0.9, "OpenCVFeatureExtractor", "opencv-python \u00b7 numpy", "vision"),

    (0.3, 9.4, 4.5, 0.9, "SentenceEmbeddingGenerator", "sentence-transformers \u00b7 torch (SBERT)", "nlp"),
    (5.2, 9.4, 4.3, 0.9, "CaptionDiversityAnalyzer", "scikit-learn cosine_similarity \u00b7 numpy", "nlp"),

    (2.0, 7.8, 6.5, 0.9, "MLDatasetBuilder", "pandas  \u2192  human_dataset.csv / ai_dataset.csv", "pipeline"),

    (2.0, 6.2, 6.5, 0.9, "AmbiguityLabelGenerator", "pandas \u00b7 matplotlib  \u2192  Low / Medium / High", "pipeline"),

    (2.0, 4.6, 6.5, 0.9, "ModelTrainer", "scikit-learn (Random Forest) \u00b7 xgboost \u00b7 joblib", "model"),

    (2.0, 3.0, 6.5, 0.9, "SHAPExplainer", "shap (TreeExplainer) \u00b7 numba-stub", "explain"),

    (0.3, 1.2, 5.0, 1.0, "FastAPI Backend", "fastapi \u00b7 uvicorn \u00b7 pydantic-settings", "serve"),
    (5.7, 1.2, 6.6, 1.0, "React Frontend \u2013 Ambiguity Lens", "Vite \u00b7 React \u00b7 TypeScript \u00b7 Tailwind \u00b7 Recharts", "ui"),
]

ARROWS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (4, 6), (5, 6), (3, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10), (9, 11),
    (10, 11),
]


def _box_center(box: Box) -> tuple[float, float]:
    x, y, w, h, *_ = box
    return x + w / 2, y + h / 2


def _box_bottom(box: Box) -> tuple[float, float]:
    x, y, w, _h, *_ = box
    return x + w / 2, y


def _box_top(box: Box) -> tuple[float, float]:
    x, y, w, h, *_ = box
    return x + w / 2, y + h


def build_diagram() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 15))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0.5, 13.9)
    ax.axis("off")

    ax.text(
        6.4,
        13.6,
        "Explainable Image Ambiguity Prediction \u2014 Tool & Module Map",
        fontsize=17,
        fontweight="bold",
        ha="center",
        color="#1F1F1F",
    )

    for idx_a, idx_b in ARROWS:
        top_box, bottom_box = BOXES[idx_a], BOXES[idx_b]
        start = _box_bottom(top_box)
        end = _box_top(bottom_box)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#555555",
            connectionstyle="arc3,rad=0.0",
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
        ax.add_patch(arrow)

    for box in BOXES:
        x, y, w, h, title, subtitle, category = box
        fill, edge = CATEGORY_COLORS[category]
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(rect)
        cx, cy = _box_center(box)
        ax.text(
            cx,
            cy + 0.16,
            title,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="center",
            color="#1A1A1A",
            zorder=3,
        )
        ax.text(
            cx,
            cy - 0.20,
            subtitle,
            fontsize=8.7,
            ha="center",
            va="center",
            color="#3A3A3A",
            zorder=3,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=12,
            markerfacecolor=CATEGORY_COLORS[key][0],
            markeredgecolor=CATEGORY_COLORS[key][1],
            label=label,
        )
        for key, label in CATEGORY_LABELS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_diagram()
    print(f"Wrote {path}")
