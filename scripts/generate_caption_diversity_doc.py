"""One-off script to generate the Caption Diversity explanation document.

Run once with:

    python scripts/generate_caption_diversity_doc.py

Produces: myDocs/Caption_Diversity_Module_Explanation.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "Caption_Diversity_Module_Explanation.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x22, 0x22, 0x22)


def set_base_font(document: Document, name: str = "Calibri") -> None:
    style = document.styles["Normal"]
    style.font.name = name
    style.font.size = Pt(11)
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)


def add_title_page(document: Document) -> None:
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Caption Diversity as an Ambiguity Signal")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run(
        "Quantifying Caption Disagreement from Sentence-BERT Embeddings"
    )
    run2.italic = True
    run2.font.size = Pt(15)
    run2.font.color.rgb = DARK

    document.add_paragraph()
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(
        "Project: Explainable Image Ambiguity Prediction Using Human and "
        "AI-Generated Caption Diversity with Computer Vision Features\n"
        "Module: image_ambiguity.features.caption_diversity.CaptionDiversityAnalyzer\n"
        "Input: Sentence-BERT (all-MiniLM-L6-v2) caption embeddings"
    )
    meta_run.font.size = Pt(11)
    meta_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    document.add_page_break()


def add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = ACCENT


def add_body(document: Document, text: str) -> None:
    p = document.add_paragraph(text)
    p.paragraph_format.space_after = Pt(8)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def add_numbered(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Number")


def add_code(document: Document, code: str) -> None:
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    for i, line in enumerate(code.strip("\n").split("\n")):
        run = p.add_run(("\n" if i > 0 else "") + line)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0x0A, 0x0A, 0x0A)
    p.paragraph_format.space_after = Pt(10)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val
    document.add_paragraph()


def build() -> None:
    document = Document()
    set_base_font(document)
    add_title_page(document)

    # 1. Motivation
    add_heading(document, "1. Motivation")
    add_body(
        document,
        "An image is ambiguous when different, equally reasonable observers "
        "describe it in meaningfully different ways. In the COCO dataset, "
        "every image already carries five independent human captions, and "
        "this project additionally considers AI-generated captions. If "
        "those captions consistently agree, the image is probably visually "
        "unambiguous. If they diverge -- describing different subjects, "
        "actions, or interpretations -- that disagreement is itself "
        "evidence of ambiguity."
    )
    add_body(
        document,
        "The Caption Diversity module turns this qualitative intuition "
        "into a precise, reproducible numeric feature that can be fed into "
        "a machine learning model and, critically, explained."
    )

    # 2. What it computes
    add_heading(document, "2. What the Module Computes")
    add_body(
        document,
        "CaptionDiversityAnalyzer takes the Sentence-BERT embeddings for a "
        "set of captions belonging to one image (produced upstream by "
        "SentenceEmbeddingGenerator) and computes a full pairwise "
        "similarity profile plus summary statistics."
    )
    add_table(
        document,
        ["Metric", "Meaning"],
        [
            ["Pairwise cosine similarity", "Similarity between every unique pair of captions for the same image."],
            ["Average similarity", "Mean of all unique pairwise similarities -- the central diversity signal."],
            ["Minimum similarity", "The most disagreeing caption pair; flags the sharpest semantic split."],
            ["Maximum similarity", "The most agreeing caption pair; shows the strongest shared interpretation."],
            ["Standard deviation", "Spread of agreement across caption pairs -- consistent vs. polarized disagreement."],
            ["Caption diversity score", "1 - Average similarity -- the final ambiguity-facing feature."],
        ],
    )

    # 3. Formula
    add_heading(document, "3. Core Formulas")
    add_body(document, "Cosine similarity between two caption embeddings u and v:")
    add_code(document, "cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)")
    add_body(
        document,
        "For n captions there are n choose 2 unique pairs. Average "
        "similarity is the mean over exactly those unique pairs (the upper "
        "triangle of the similarity matrix, excluding the diagonal, which "
        "is always 1.0 by definition):"
    )
    add_code(
        document,
        "average_similarity = mean( cosine_similarity(i, j) )  for all i < j",
    )
    add_body(document, "Caption diversity is then defined as the complement of agreement:")
    add_code(document, "Diversity = 1 - Average Similarity")
    add_body(
        document,
        "This keeps the score bounded in a familiar, interpretable [0, 1] "
        "range whenever embeddings are non-negative-cosine (as is typical "
        "for normalized sentence embeddings): 0 means captions are "
        "essentially identical in meaning; values approaching 1 mean "
        "captions describe the image in largely unrelated ways."
    )

    # 4. Worked example
    add_heading(document, "4. Worked Example (from this codebase)")
    add_body(
        document,
        "Running python src/caption_diversity.py on COCO image 391895 "
        "(5 human captions describing a man on a moped / motorcycle) "
        "produces the following measured result:"
    )
    add_table(
        document,
        ["Metric", "Value"],
        [
            ["Captions", "5"],
            ["Unique pairs", "10"],
            ["Average Similarity", "0.49"],
            ["Minimum Similarity", "0.32"],
            ["Maximum Similarity", "0.60"],
            ["Standard Deviation", "0.09"],
            ["Caption Diversity", "0.51"],
        ],
    )
    add_body(
        document,
        "Interpretation: an average pairwise similarity of 0.49 means "
        "these five human descriptions, while all broadly correct, differ "
        "enough in wording, framing, and emphasis (moped vs. motorcycle, "
        "man vs. young person, foreground vs. background detail) that the "
        "resulting diversity score of 0.51 flags this image as having "
        "moderate captioning disagreement -- a candidate signal of "
        "ambiguity for the downstream prediction model."
    )

    # 5. Implementation
    add_heading(document, "5. Implementation in This Codebase")
    add_body(
        document,
        "CaptionDiversityAnalyzer "
        "(src/image_ambiguity/features/caption_diversity.py) is a "
        "reusable, stateless class: it holds no per-image state, so a "
        "single instance can score every image in a dataset."
    )
    add_table(
        document,
        ["Method", "Responsibility"],
        [
            ["pairwise_cosine_similarity(embeddings)", "L2-normalizes embeddings and returns the full (n, n) cosine similarity matrix."],
            ["compute(embeddings)", "Extracts unique pairs, computes average / min / max / std and the diversity score; returns a DiversityMetrics record."],
            ["to_dict(metrics, ...)", "Serializes metrics (plus optional image_id and captions) into a JSON-ready dictionary."],
            ["save_json(metrics, path, ...)", "Writes the full metrics payload, including the pairwise matrix, to a .json file."],
            ["save_csv(metrics, path, ...)", "Writes a flat one-row summary (ideal for aggregating across many images) to .csv."],
            ["visualize(metrics, path, ...)", "Renders and optionally saves a pairwise similarity heatmap annotated with per-cell values."],
        ],
    )
    add_code(
        document,
        """
analyzer = CaptionDiversityAnalyzer()
metrics = analyzer.compute(embeddings)   # embeddings: (n_captions, 384)

print(metrics.average_similarity)  # 0.49
print(metrics.diversity_score)     # 0.51

analyzer.save_json(metrics, "results/diversity/diversity_391895.json",
                    image_id=391895, captions=captions)
analyzer.save_csv(metrics, "results/diversity/diversity_391895.csv",
                   image_id=391895)
analyzer.visualize(metrics, "results/figures/similarity_391895.png",
                    captions=captions)
""",
    )
    add_body(
        document,
        "The class raises clear, typed errors for malformed input (fewer "
        "than two captions, non-2D arrays, NaN/Inf values) and logs every "
        "computation and saved artifact through the project's shared "
        "logging configuration, keeping it consistent with the rest of the "
        "pipeline's production-oriented design."
    )

    # 6. Position in pipeline
    add_heading(document, "6. Position in the Overall Pipeline")
    add_numbered(
        document,
        [
            "CocoDatasetLoader retrieves captions for an image id.",
            "SentenceEmbeddingGenerator encodes each caption into a "
            "384-dimensional Sentence-BERT vector (batched, GPU/CPU-aware).",
            "CaptionDiversityAnalyzer.compute() converts those embeddings "
            "into pairwise similarities and a single diversity score.",
            "The diversity score, along with related statistics (min, "
            "max, std), becomes one input feature -- alongside computer "
            "vision features -- to the ambiguity prediction model.",
            "save_json / save_csv persist the metrics for reproducibility "
            "and later aggregation across the full dataset; visualize() "
            "produces the qualitative heatmap used for figures and slides.",
        ],
    )

    # 7. Why explainable
    add_heading(document, "7. Why This Is Explainable")
    add_body(
        document,
        "Unlike opaque deep features, a caption diversity score of 0.51 "
        "has a direct, human-readable explanation: 'these captions "
        "disagreed about roughly half of their described content.' "
        "When a SHAP attribution shows that diversity_score was the "
        "leading contributor to a high ambiguity prediction, that "
        "explanation can be paired with the underlying similarity heatmap "
        "and the raw captions themselves, letting a reviewer verify the "
        "model's reasoning against the original text -- fulfilling the "
        "explainability requirement central to this project."
    )

    # 8. Talking points
    add_heading(document, "8. Conference Talking Points (Summary)")
    add_bullets(
        document,
        [
            "Caption diversity converts free-text disagreement into a "
            "single interpretable number: Diversity = 1 - Average "
            "Similarity, computed over Sentence-BERT embeddings.",
            "We compute the full pairwise cosine similarity matrix, then "
            "summarize it with average, minimum, maximum, and standard "
            "deviation -- capturing both the overall agreement level and "
            "its consistency.",
            "On a real COCO example (5 captions), we measured an average "
            "similarity of 0.49, giving a diversity score of 0.51 -- a "
            "moderate ambiguity signal grounded in real caption text.",
            "Every result is exported to JSON and CSV for reproducibility, "
            "and visualized as an annotated similarity heatmap for "
            "qualitative inspection.",
            "Because the score is derived from directly readable caption "
            "comparisons, it integrates naturally with SHAP-based "
            "explainability, letting predictions be traced back to "
            "specific, human-verifiable evidence.",
        ],
    )

    # 9. References
    add_heading(document, "9. References")
    add_bullets(
        document,
        [
            "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence "
            "Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
            "Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in "
            "Context. ECCV 2014. (source of caption annotations used in "
            "this project)",
            "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to "
            "Interpreting Model Predictions (SHAP). NeurIPS 2017.",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
