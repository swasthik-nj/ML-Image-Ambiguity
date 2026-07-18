"""One-off script to generate the Ambiguity Label Generation explanation document.

Run once with:

    python scripts/generate_label_generation_doc.py

Produces: myDocs/Ambiguity_Label_Generation_Explanation.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "Ambiguity_Label_Generation_Explanation.docx"

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
    run = title.add_run("Turning Diversity Scores into Ambiguity Labels")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run(
        "Rule-Based Low / Medium / High Ambiguity Labeling for the Human Dataset"
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
        "Module: image_ambiguity.pipeline.label_generator.AmbiguityLabelGenerator\n"
        "Output: dataset/human_dataset.csv (with ambiguity_label column)"
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
        "human_dataset.csv already stores a continuous caption_diversity "
        "score (0 = captions fully agree, 1 = captions fully disagree) "
        "alongside six OpenCV image statistics. A continuous score is "
        "useful for regression, but many downstream uses -- dashboards, "
        "stratified sampling, classification baselines, human review "
        "triage -- need a small number of interpretable categories "
        "instead of a raw float."
    )
    add_body(
        document,
        "The Ambiguity Label Generation module converts the continuous "
        "caption_diversity score into three human-readable categories -- "
        "Low, Medium, and High ambiguity -- using fixed, documented "
        "thresholds. It then reports how many images fall into each "
        "bucket and visualizes the split, turning a single numeric "
        "column into a labeled, analysis-ready dataset."
    )

    # 2. Rules
    add_heading(document, "2. Labeling Rules")
    add_body(
        document,
        "The rule is a simple threshold cut over caption_diversity, "
        "applied independently to every image (no dependence between "
        "rows, no model to train, fully deterministic and reproducible):"
    )
    add_table(
        document,
        ["Diversity range", "Label", "Interpretation"],
        [
            ["0.00 - 0.35", "Low", "Captions largely agree; the image likely has one clear, dominant reading."],
            ["0.35 - 0.65", "Medium", "Captions partially disagree; some room for differing interpretations."],
            ["0.65 - 1.00", "High", "Captions strongly disagree; the image is likely genuinely ambiguous."],
        ],
    )
    add_body(
        document,
        "Boundaries are treated as: lower-inclusive, upper-exclusive for "
        "Low and Medium (e.g. 0.35 itself falls into Medium, not Low), "
        "and the High bucket closes the range at 1.0 inclusive. Any row "
        "with a missing (NaN) caption_diversity value is labeled "
        "'Unknown' rather than silently dropped or mis-bucketed."
    )
    add_code(
        document,
        """
def label_value(self, diversity: float) -> str:
    if diversity < self.low_max:      # < 0.35
        return "Low"
    if diversity < self.medium_max:   # < 0.65
        return "Medium"
    return "High"                     # >= 0.65
""",
    )

    # 3. What the module produces
    add_heading(document, "3. What the Module Produces")
    add_numbered(
        document,
        [
            "A labeled DataFrame: the original 12 columns from "
            "human_dataset.csv plus a new ambiguity_label column "
            "(an ordered category: Low < Medium < High).",
            "A statistics dictionary: overall label counts and "
            "percentages, plus per-label diversity mean / std / min / "
            "max, saved to results/labels/label_statistics.json.",
            "A two-panel plot: a histogram of caption_diversity with "
            "the two threshold lines drawn on it, alongside a bar chart "
            "of label counts, saved to "
            "results/figures/ambiguity_label_distribution.png.",
            "An updated CSV: the labeled dataset is written back to "
            "dataset/human_dataset.csv, so every downstream step can "
            "read ambiguity_label directly.",
        ],
    )

    # 4. Implementation
    add_heading(document, "4. Implementation in This Codebase")
    add_table(
        document,
        ["Method", "Responsibility"],
        [
            ["label_value(diversity)", "Classifies a single float into Low / Medium / High (raises on NaN)."],
            ["label_dataframe(df)", "Vectorized version: appends the ambiguity_label categorical column to a copy of the input DataFrame; NaNs become 'Unknown'."],
            ["compute_statistics(df)", "Returns label counts/percentages and per-label diversity mean/std/min/max as a plain dict."],
            ["save_statistics_json(stats, path)", "Writes the statistics dict to a JSON file."],
            ["save_csv(df, path)", "Writes the labeled DataFrame to CSV, creating parent directories as needed."],
            ["plot_distribution(df, path)", "Renders the histogram + bar-chart figure and saves it as PNG."],
        ],
    )
    add_code(
        document,
        """
generator = AmbiguityLabelGenerator(low_max=0.35, medium_max=0.65)

labeled_df = generator.label_dataframe(df)          # + ambiguity_label
stats = generator.compute_statistics(labeled_df)    # counts, %s, per-label stats

generator.save_csv(labeled_df, "dataset/human_dataset.csv")
generator.save_statistics_json(stats, "results/labels/label_statistics.json")
generator.plot_distribution(labeled_df, "results/figures/ambiguity_label_distribution.png")
""",
    )
    add_body(
        document,
        "The thresholds (low_max, medium_max) are constructor arguments, "
        "not hard-coded constants, so the same class can be reused to "
        "test alternative cut points (e.g. quartile-based thresholds) "
        "without touching the labeling logic itself. The class validates "
        "that 0 < low_max < medium_max < 1 at construction time to catch "
        "misconfiguration early."
    )

    # 5. Worked example
    add_heading(document, "5. Worked Example (from this codebase)")
    add_body(
        document,
        "Running python src/generate_labels.py against the 300-row "
        "human_dataset.csv produced:"
    )
    add_table(
        document,
        ["Label", "Count", "Percentage"],
        [
            ["Low", "109", "36.33%"],
            ["Medium", "187", "62.33%"],
            ["High", "4", "1.33%"],
        ],
    )
    add_body(
        document,
        "Overall caption_diversity across the 300 images had mean "
        "0.3847, std 0.1060, min 0.1313, and max 0.7196 -- consistent "
        "with the label split above: most images cluster in the Medium "
        "band just above the Low/Medium boundary, very few reach the "
        "High-ambiguity tail past 0.65."
    )
    add_body(
        document,
        "For example, image_id 301061 has caption_diversity 0.2595 "
        "(labeled Low), while image_id 382030 has caption_diversity "
        "0.5716 (labeled Medium) -- both correctly bucketed by the "
        "fixed threshold rule."
    )

    # 6. Why fixed thresholds (not learned)
    add_heading(document, "6. Why Fixed Thresholds Instead of a Learned Model")
    add_body(
        document,
        "A learned classifier (e.g. k-means or a trained threshold) "
        "would require its own labels to validate against -- but "
        "ambiguity labels are exactly what this project is trying to "
        "produce. Fixed, human-chosen thresholds (0.35 / 0.65) instead "
        "give a transparent, defensible starting point: every bucket "
        "boundary is stated up front, is easy to justify in a paper or "
        "presentation, and can be swapped for data-driven cut points "
        "(e.g. quartiles of the observed distribution) later without "
        "changing the surrounding pipeline."
    )

    # 7. Explainability
    add_heading(document, "7. Why This Design Supports Explainability")
    add_body(
        document,
        "Because the label is a deterministic function of one already-"
        "documented column (caption_diversity, itself defined as "
        "1 - average cosine similarity across captions), every labeled "
        "row can be explained in one sentence: 'this image is High "
        "ambiguity because its five human captions only agreed with "
        "each other 32% of the time on average.' There is no opaque "
        "model between the evidence (captions) and the label -- the "
        "threshold rule is the entire explanation."
    )

    # 8. Talking points
    add_heading(document, "8. Conference Talking Points (Summary)")
    add_bullets(
        document,
        [
            "AmbiguityLabelGenerator converts the continuous "
            "caption_diversity score into three interpretable buckets "
            "-- Low (<0.35), Medium (0.35-0.65), High (>=0.65) -- using "
            "a fully transparent, deterministic threshold rule.",
            "On the 300-image human_dataset.csv, the split was 109 Low "
            "(36.3%), 187 Medium (62.3%), and 4 High (1.3%), matching "
            "the observed diversity distribution (mean 0.38).",
            "Missing diversity values are labeled 'Unknown' rather than "
            "dropped, keeping every sampled image in the dataset.",
            "Thresholds are configurable constructor arguments, so the "
            "same rule-based approach can be re-tuned without changing "
            "any downstream code.",
            "Because the rule is a one-line function of an already-"
            "documented column, every label is traceable back to the "
            "exact captions that produced it -- no black-box model "
            "sits between evidence and label.",
        ],
    )

    # 9. References
    add_heading(document, "9. References")
    add_bullets(
        document,
        [
            "Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in "
            "Context. ECCV 2014.",
            "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence "
            "Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
            "McKinney, W. (2010). Data Structures for Statistical "
            "Computing in Python (pandas). SciPy 2010.",
            "Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. "
            "Computing in Science & Engineering.",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
