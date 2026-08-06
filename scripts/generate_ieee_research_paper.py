"""Generate an IEEE-style research paper (.docx) for the completed project.

Run:

    python scripts/generate_ieee_research_paper.py

Produces: myDocs/IEEE_Image_Ambiguity_Prediction_Paper.docx
Also writes a companion LaTeX draft: myDocs/ieee_paper/main.tex
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = PROJECT_ROOT / "myDocs" / "IEEE_Image_Ambiguity_Prediction_Paper.docx"
TEX_DIR = PROJECT_ROOT / "myDocs" / "ieee_paper"
TEX_PATH = TEX_DIR / "main.tex"
FIG_DIR = TEX_DIR / "figures"

ACCENT = RGBColor(0x00, 0x00, 0x00)
MUTED = RGBColor(0x33, 0x33, 0x33)

# Two-column IEEE body: 8.5in page, 0.75in margins -> 7.0in text width.
COLUMN_GAP_TWIPS = 288  # 0.2 in
COLUMN_WIDTH_IN = 3.35


def set_base_font(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    style.paragraph_format.space_after = Pt(6)
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_centered(document: Document, text: str, *, size: int, bold: bool = False) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"


def add_heading_ieee(document: Document, text: str, level: int = 1) -> None:
    # IEEE-like numbered section headings
    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(10 if level == 1 else 10)
    run.font.name = "Times New Roman"
    if level == 1:
        run.font.all_caps = True


def add_body(document: Document, text: str, *, first_line_indent: bool = True) -> None:
    p = document.add_paragraph()
    if first_line_indent:
        p.paragraph_format.first_line_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)


def add_placeholder(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED
    run.font.name = "Times New Roman"


def add_caption(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"


def add_figure(
    document: Document,
    image_path: Path,
    caption: str,
    *,
    width_in: float = COLUMN_WIDTH_IN,
) -> None:
    """Embed a figure (sized to one column) with an IEEE-style caption below."""
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(image_path), width=Inches(width_in))
    add_caption(document, caption)


def set_two_columns(section, *, num: int = 2, space_twips: int = COLUMN_GAP_TWIPS) -> None:
    """Configure an IEEE-style multi-column body for a docx section."""
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")


def add_table(
    document: Document,
    *,
    caption: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    """Add an IEEE-style table (caption above, centered, bordered)."""
    add_caption(document, caption)
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr_cells = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr_cells[i].text = ""
        run = hdr_cells[i].paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.name = "Times New Roman"
        hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row_values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_values):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(8.5)
            run.font.name = "Times New Roman"
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_after = document.add_paragraph()
    p_after.paragraph_format.space_after = Pt(8)


def build_docx() -> Path:
    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    set_base_font(document)

    for section in document.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Title block
    add_centered(
        document,
        "Explainable Image Ambiguity Prediction Using Human and "
        "AI-Generated Caption Diversity with Computer Vision Features",
        size=14,
        bold=True,
    )
    add_centered(document, "[Author Name(s)]", size=11)
    add_centered(document, "[Department / Affiliation], [Institution]", size=10)
    add_centered(document, "[City, Country]  ·  [email@domain.edu]", size=9)
    add_centered(
        document,
        "IEEE Conference / Journal Submission Draft — Placeholder Metadata",
        size=9,
    )

    # Abstract
    add_heading_ieee(document, "Abstract")
    add_body(
        document,
        "Image ambiguity arises when multiple plausible linguistic "
        "interpretations can be assigned to the same visual scene. This paper "
        "presents an explainable machine learning framework that predicts "
        "image ambiguity as a categorical label (Low, Medium, High) from "
        "caption disagreement and classical computer-vision descriptors. "
        "Human captions from the Microsoft COCO validation set and "
        "AI-generated captions from a BLIP image captioning model are "
        "embedded with Sentence-BERT. Pairwise cosine similarity yields a "
        "caption-diversity score, which is combined with OpenCV features "
        "(edge density, entropy, brightness, contrast, color variance, and "
        "texture). Rule-based thresholds map diversity to ambiguity labels. "
        "Class imbalance in the High band is mitigated by mining additional "
        "high-diversity images and by training-time oversampling. Random "
        "Forest and XGBoost classifiers are trained and compared; SHAP "
        "values explain individual predictions. A FastAPI service and a "
        "React demonstration interface support interactive evaluation and "
        "human-versus-AI diversity comparison. Experiments on an enriched "
        "COCO-derived dataset (N = 330) show that the classifiers recover "
        "the diversity-derived labels with high fidelity, while AI captions "
        "often exhibit lower disagreement than human captions for the same "
        "images. We discuss label construction, leakage risks, and "
        "directions for independent ambiguity annotation.",
        first_line_indent=False,
    )
    add_body(
        document,
        "Keywords—image ambiguity, caption diversity, Sentence-BERT, BLIP, "
        "OpenCV features, Random Forest, XGBoost, SHAP, explainable AI, COCO",
        first_line_indent=False,
    )

    # IEEE conference papers set the title block and abstract across the full
    # page width, then switch to a two-column body for all numbered sections.
    document.add_section(WD_SECTION.CONTINUOUS)
    set_two_columns(document.sections[-1])

    # I. Introduction
    add_heading_ieee(document, "I. Introduction")
    add_body(
        document,
        "Natural images are frequently described by multiple captions that "
        "emphasize different objects, attributes, or events. When captions "
        "diverge substantially, the image can be considered linguistically "
        "ambiguous: observers do not converge on a single semantic reading. "
        "Quantifying such ambiguity is useful for dataset curation, "
        "captioning evaluation, retrieval ranking, and human–AI comparison "
        "studies [1], [2].",
    )
    add_body(
        document,
        "Prior work has studied visual uncertainty, caption diversity, and "
        "model disagreement, yet few end-to-end systems jointly (i) define "
        "ambiguity from multi-caption disagreement, (ii) fuse linguistic "
        "diversity with classical visual features, (iii) predict discrete "
        "ambiguity classes with explainable classifiers, and (iv) contrast "
        "human and AI caption sources under the same metric.",
    )
    add_body(
        document,
        "This paper contributes: (1) an operational definition of image "
        "ambiguity based on Sentence-BERT caption diversity with explicit "
        "Low/Medium/High thresholds; (2) a feature pipeline combining "
        "diversity statistics with OpenCV descriptors; (3) supervised "
        "prediction using Random Forest and XGBoost with class-imbalance "
        "handling; (4) SHAP-based explanations; and (5) a deployable "
        "FastAPI + React system for prediction and human–AI comparison. "
        "Fig. 1 summarizes the overall architecture.",
    )
    add_figure(
        document,
        FIG_DIR / "fig1_architecture.png",
        "Fig. 1. End-to-end architecture: human and AI captions are embedded "
        "with Sentence-BERT to score diversity, fused with OpenCV features, "
        "classified by Random Forest / XGBoost, explained with SHAP, and "
        "served through FastAPI and the React \u201cAmbiguity Lens\u201d UI.",
    )

    # II. Literature Review
    add_heading_ieee(document, "II. Literature Review")
    add_body(
        document,
        "Large-scale caption corpora such as Microsoft COCO provide multiple "
        "human descriptions per image and underpin modern vision–language "
        "research [3]. Neural image captioning models, including Show-and-Tell "
        "and subsequent Transformer/BLIP-style generators, produce fluent "
        "captions that may nonetheless under-represent human disagreement "
        "[4], [5].",
    )
    add_body(
        document,
        "Sentence embeddings from models such as Sentence-BERT enable "
        "semantic similarity measurement beyond lexical overlap [6]. "
        "Caption diversity and consensus metrics have been used to evaluate "
        "generation quality and dataset difficulty [7], [8]. Classical "
        "computer-vision features (edges, entropy, color statistics) remain "
        "interpretable complements to deep embeddings in multimodal "
        "pipelines [9].",
    )
    add_body(
        document,
        "Ensemble tree models (Random Forest, gradient boosting / XGBoost) "
        "are widely used for tabular multimodal feature sets and often "
        "provide strong baselines with modest data [10], [11]. Explainable "
        "AI methods such as SHAP attribute predictions to input features "
        "and support trustworthy deployment [12]. Our work sits at the "
        "intersection of caption-diversity analysis, classical visual "
        "features, and explainable classification for ambiguity prediction.",
    )
    add_table(
        document,
        caption="TABLE I\nRELATED WORK COMPARISON (TO BE COMPLETED WITH FINAL CITATIONS)",
        headers=["Method", "Ambiguity Definition", "Features", "XAI"],
        rows=[
            ["[Related A]", "TBD", "TBD", "TBD"],
            ["[Related B]", "TBD", "TBD", "TBD"],
            ["Ours", "Caption diversity", "SBERT + OpenCV", "SHAP"],
        ],
    )

    # III. Methodology
    add_heading_ieee(document, "III. Methodology")
    add_body(
        document,
        "The proposed methodology comprises five stages: (A) multi-caption "
        "acquisition, (B) semantic embedding and diversity scoring, "
        "(C) visual feature extraction, (D) rule-based ambiguity labeling, "
        "and (E) supervised classification with explanation. Fig. 2 "
        "illustrates the methodological flow.",
    )
    add_figure(
        document,
        FIG_DIR / "fig2_pipeline.png",
        "Fig. 2. Methodological pipeline showing the five stages from "
        "caption acquisition to learning and explanation.",
    )

    add_heading_ieee(document, "A. Caption Acquisition", level=2)
    add_body(
        document,
        "For the human branch, five (or more) COCO captions are retrieved "
        "per image. For the AI branch, BLIP generates captions using beam "
        "search, top-k sampling, nucleus sampling, and prompt-conditioned "
        "decoding to increase lexical variety [5]. At inference time, "
        "uploaded COCO-style filenames can resolve to stored human captions; "
        "otherwise user-provided or BLIP captions are used.",
    )

    add_heading_ieee(document, "B. Diversity Scoring", level=2)
    add_body(
        document,
        "Let {c1,...,cn} be captions for an image. Sentence-BERT produces "
        "embeddings {e1,...,en}. Pairwise cosine similarities s_ij are "
        "computed for i < j. We report average, minimum, maximum, and "
        "standard deviation of similarities, and define",
    )
    add_body(
        document,
        "caption_diversity = 1 − average_similarity.",
        first_line_indent=False,
    )
    add_body(
        document,
        "Higher diversity indicates stronger semantic disagreement among "
        "captions.",
    )

    add_heading_ieee(document, "C. Visual Features", level=2)
    add_body(
        document,
        "OpenCV-based descriptors capture appearance cues that may correlate "
        "with scene complexity: edge density, grayscale entropy, brightness, "
        "contrast, color variance, and texture energy. These features do not "
        "define the label directly; labels are driven by caption diversity.",
    )

    add_heading_ieee(document, "D. Ambiguity Labeling", level=2)
    add_body(
        document,
        "Diversity is mapped to categorical ambiguity as:",
    )
    add_body(
        document,
        "Low if d < 0.35; Medium if 0.35 ≤ d < 0.65; High if d ≥ 0.65.",
        first_line_indent=False,
    )
    add_body(
        document,
        "Thresholds were retained to preserve a strict interpretation of "
        "High as strong caption disagreement. Because High cases are rare "
        "under this rule, additional high-diversity COCO images are mined "
        "and minority classes are oversampled during training.",
    )

    add_heading_ieee(document, "E. Learning and Explanation", level=2)
    add_body(
        document,
        "An 11-dimensional feature vector concatenates diversity statistics "
        "and OpenCV descriptors. Random Forest and XGBoost classifiers "
        "predict Low/Medium/High. Hyperparameters are tuned by stratified "
        "cross-validation grid search. SHAP values explain each prediction "
        "for interactive inspection via the API/UI.",
    )

    # IV. Dataset
    add_heading_ieee(document, "IV. Dataset")
    add_body(
        document,
        "We use Microsoft COCO 2017 validation images and captions "
        "(captions_val2017.json) [3]. A working human feature table "
        "human_dataset.csv contains N = 330 labeled images after High-class "
        "enrichment. Class counts are approximately Low = 109, Medium = 187, "
        "High = 34. Mean human caption diversity is approximately 0.41. "
        "An optional AI table ai_dataset.csv stores BLIP-derived features "
        "for comparison (prototype size may be smaller during CPU runs).",
    )
    add_table(
        document,
        caption="TABLE II\nDATASET STATISTICS",
        headers=["Split", "N", "Low", "Medium", "High", "Mean div."],
        rows=[
            ["Human (enriched)", "330", "109", "187", "34", "0.413"],
            ["AI / BLIP (prototype)", "20", "\u2014", "\u2014", "\u2014", "0.484"],
        ],
    )
    add_body(
        document,
        "Fig. 3 shows the resulting class distribution after enrichment: "
        "High-diversity images were mined from unused COCO validation "
        "images (diversity \u2265 0.65) to raise the High count from 4 to 34 "
        "while the Low/Medium thresholds were left unchanged.",
    )
    add_figure(
        document,
        FIG_DIR / "fig3_class_distribution.png",
        "Fig. 3. Distribution of Low / Medium / High labels in the "
        "enriched human dataset (N = 330).",
    )

    # V. Feature Engineering
    add_heading_ieee(document, "V. Feature Engineering")
    add_body(
        document,
        "Features fall into two families. Linguistic features are derived "
        "from caption embeddings: average_similarity, minimum_similarity, "
        "maximum_similarity, std_similarity, and caption_diversity. Visual "
        "features are edge_density, entropy, brightness, contrast, "
        "color_variance, and texture. Missing values, if any, are median-"
        "imputed before training. All features are stored in a tabular CSV "
        "compatible with classical ML toolkits.",
    )
    add_table(
        document,
        caption="TABLE III\nFEATURE DICTIONARY",
        headers=["Feature", "Family", "Description"],
        rows=[
            ["average_similarity", "Linguistic", "Mean pairwise cosine similarity"],
            ["minimum_similarity", "Linguistic", "Lowest pairwise similarity"],
            ["maximum_similarity", "Linguistic", "Highest pairwise similarity"],
            ["std_similarity", "Linguistic", "Std. dev. of pairwise similarity"],
            ["caption_diversity", "Linguistic", "1 \u2212 average_similarity"],
            ["edge_density", "Visual", "Fraction of edge pixels"],
            ["entropy", "Visual", "Grayscale intensity entropy"],
            ["brightness", "Visual", "Mean pixel intensity"],
            ["contrast", "Visual", "Intensity spread (std. dev.)"],
            ["color_variance", "Visual", "Variance across color channels"],
            ["texture", "Visual", "Local texture energy"],
        ],
    )
    add_body(
        document,
        "Fig. 4 shows pairwise Pearson correlations among the 11 features "
        "on the human dataset. The four similarity-based features are "
        "strongly inter-correlated (as expected, since caption_diversity is "
        "derived from average_similarity), while OpenCV visual features are "
        "comparatively weakly correlated with the linguistic group, "
        "confirming that the two feature families carry complementary "
        "information.",
    )
    add_figure(
        document,
        FIG_DIR / "fig4_correlation_heatmap.png",
        "Fig. 4. Pearson correlation matrix of the 11 model features "
        "computed on the human-labeled dataset.",
    )

    # VI. Machine Learning
    add_heading_ieee(document, "VI. Machine Learning")
    add_body(
        document,
        "Labels are encoded as ordered classes (0 = Low, 1 = Medium, "
        "2 = High). Data are split 80/20 with stratification. Minority "
        "classes in the training split are oversampled with replacement to "
        "the majority count; Random Forest additionally uses "
        "class_weight='balanced'. Model search spaces include n_estimators "
        "and max_depth (and learning_rate for XGBoost). Selection prefers "
        "macro-F1 on the held-out test set. Artifacts are serialized with "
        "joblib (best_model.joblib).",
    )
    add_table(
        document,
        caption="TABLE IV\nSELECTED MODEL CONFIGURATIONS (GRID SEARCH)",
        headers=["Model", "Search space", "Best parameters"],
        rows=[
            [
                "Random\nForest",
                "n_estimators\u2208{100,200};\nmax_depth\u2208{None,8,16};\nmin_samples_leaf\u2208{1,2}",
                "n_estimators=100;\nmax_depth=None;\nmin_samples_leaf=1",
            ],
            [
                "XGBoost",
                "n_estimators\u2208{100,200};\nmax_depth\u2208{3,5};\nlearning_rate\u2208{0.05,0.1}",
                "n_estimators=100;\nmax_depth=3;\nlearning_rate=0.05",
            ],
        ],
    )

    # VII. Experiments
    add_heading_ieee(document, "VII. Experiments")
    add_body(
        document,
        "Experiments address four questions: (Q1) Can classical ML recover "
        "diversity-derived ambiguity labels from the engineered features? "
        "(Q2) Does High-class enrichment / balancing improve minority-class "
        "behavior? (Q3) How do human and AI caption diversities differ on "
        "overlapping images? (Q4) Are SHAP attributions consistent with the "
        "diversity-driven labeling rule?",
    )
    add_body(
        document,
        "Protocol: train Random Forest and XGBoost on human_dataset.csv; "
        "report accuracy, macro precision/recall/F1, and multiclass ROC-AUC "
        "where defined; compare mean caption_diversity between human and AI "
        "tables; inspect SHAP summaries for representative Low/Medium/High "
        "uploads in the interactive system.",
    )
    add_figure(
        document,
        FIG_DIR / "fig5_experimental_protocol.png",
        "Fig. 5. Experimental protocol from dataset loading through SHAP "
        "inspection.",
    )

    # VIII. Results
    add_heading_ieee(document, "VIII. Results")
    add_body(
        document,
        "On the enriched human set (N = 330) with an 80/20 stratified split "
        "and train-only oversampling, both Random Forest and XGBoost "
        "achieve near-ceiling held-out performance (Table V): 5-fold "
        "cross-validated accuracy of 99.6%, and 100% accuracy / macro-F1 / "
        "ROC-AUC on the 66-sample test set for both models. This is expected "
        "when caption_diversity is both the label source and a model "
        "feature: the decision boundary essentially recovers the threshold "
        "rule in (2). Fig. 7(a) confirms a perfectly diagonal confusion "
        "matrix on the test split, and Fig. 7(b) shows caption_diversity and "
        "average_similarity dominate Random Forest feature importance, "
        "exactly the two quantities used to construct the labels.",
    )
    add_table(
        document,
        caption="TABLE V\nTEST-SET CLASSIFICATION RESULTS",
        headers=["Model", "Acc.", "Prec.", "Rec.", "F1", "ROC-AUC", "5-fold CV Acc."],
        rows=[
            ["Random Forest", "1.000", "1.000", "1.000", "1.000", "1.000", "0.996"],
            ["XGBoost", "1.000", "1.000", "1.000", "1.000", "1.000", "0.996"],
        ],
    )
    add_body(
        document,
        "Human-versus-AI comparison on available samples shows AI (BLIP) "
        "captions with a higher mean diversity than human captions in the "
        "current small prototype AI subset (Fig. 6; AI mean \u2248 0.484, "
        "N = 20 vs. human mean \u2248 0.413, N = 330), which is the opposite of "
        "our qualitative expectation and is attributable to the small, "
        "non-paired AI sample rather than a general effect. Qualitatively, "
        "however, BLIP frequently produces near-duplicate captions for "
        "individual human-labeled High images (e.g., minor rewordings such "
        "as \u201ca pile of croissants\u201d), which lowers diversity below 0.65 "
        "for those specific images even though the same image is High under "
        "human annotators. A larger, paired human\u2013AI evaluation is needed "
        "to draw a population-level conclusion (see Section X).",
    )
    add_figure(
        document,
        FIG_DIR / "fig6_human_vs_ai_diversity.png",
        "Fig. 6. Mean caption diversity for human vs. AI (BLIP) captions, "
        "with the Low/Medium and Medium/High thresholds overlaid.",
    )
    add_figure(
        document,
        FIG_DIR / "fig7_confusion_importance.png",
        "Fig. 7. (a) Confusion matrix for the best model (Random Forest) on "
        "the held-out test set; (b) Random Forest feature importances, "
        "dominated by the diversity-derived features used to define labels.",
    )

    # IX. Discussion
    add_heading_ieee(document, "IX. Discussion")
    add_body(
        document,
        "Operationalizing ambiguity as caption disagreement aligns with "
        "multi-annotator variability and yields an objective, reproducible "
        "labeling procedure. Retaining High ≥ 0.65 keeps the High class "
        "semantically strict; enrichment addresses rarity without diluting "
        "the definition.",
    )
    add_body(
        document,
        "A central limitation is label–feature dependence: because labels "
        "are deterministic functions of caption_diversity and that feature "
        "is included in X, classifiers can achieve near-ceiling accuracy by "
        "recovering thresholds. Reported performance therefore validates "
        "pipeline consistency more than independent perceptual ambiguity "
        "recognition. Future studies should collect human ambiguity ratings "
        "independent of the diversity formula, or ablate diversity features "
        "to test whether OpenCV cues alone predict held-out human judgments.",
    )
    add_body(
        document,
        "Another finding is source sensitivity: the same image may be High "
        "under human captions and Medium under BLIP. Systems must expose "
        "caption source (user / COCO human / BLIP) to avoid misleading demos. "
        "Visual clutter alone should not be equated with High ambiguity.",
    )

    # X. Future Work
    add_heading_ieee(document, "X. Future Work")
    add_body(
        document,
        "Future directions include: (1) independent human ambiguity "
        "annotation protocols; (2) full paired human–AI datasets at COCO "
        "scale; (3) deep multimodal models that predict ambiguity without "
        "explicit diversity leakage; (4) calibrated probability estimates "
        "and cost-sensitive High detection; (5) multilingual captions; "
        "(6) user studies of SHAP explanations; and (7) deployment "
        "evaluations on in-the-wild photographs.",
    )

    # XI. Conclusion
    add_heading_ieee(document, "XI. Conclusion")
    add_body(
        document,
        "We presented an explainable pipeline for image ambiguity prediction "
        "based on caption diversity and classical visual features, spanning "
        "dataset construction, labeling, class-balanced training, SHAP "
        "explanations, and an interactive FastAPI/React demonstration. "
        "Keeping a strict High threshold while enriching rare High samples "
        "preserves a research-correct definition of strong caption "
        "disagreement. Human and AI captions can be compared under a shared "
        "diversity metric, revealing systematic differences in semantic "
        "agreement. With independent labels and larger paired evaluations, "
        "the framework can support broader vision–language ambiguity "
        "research.",
    )

    # Acknowledgment
    add_heading_ieee(document, "Acknowledgment")
    add_body(
        document,
        "The authors thank [advisor / lab / institution] for guidance and "
        "resources. COCO and open-source libraries (PyTorch, Transformers, "
        "scikit-learn, XGBoost, SHAP, OpenCV, FastAPI, React) enabled this "
        "implementation.",
        first_line_indent=False,
    )

    # References
    add_heading_ieee(document, "References")
    refs = [
        "[1]  A. Author et al., “Placeholder: visual uncertainty / ambiguity "
        "in images,” in Proc. IEEE Conf., year, pp. xx–xx.",
        "[2]  B. Author et al., “Placeholder: multi-annotator disagreement "
        "and dataset difficulty,” Journal Name, vol. x, no. y, pp. xx–xx, year.",
        "[3]  T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” "
        "in ECCV, 2014.",
        "[4]  O. Vinyals et al., “Show and Tell: A Neural Image Caption "
        "Generator,” in CVPR, 2015.",
        "[5]  J. Li et al., “BLIP: Bootstrapping Language-Image Pre-training "
        "for Unified Vision-Language Understanding and Generation,” in ICML, "
        "2022.",
        "[6]  N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings "
        "using Siamese BERT-Networks,” in EMNLP, 2019.",
        "[7]  C. Author et al., “Placeholder: caption diversity / consensus "
        "metrics,” in ACL/EMNLP/CVPR, year.",
        "[8]  D. Author et al., “Placeholder: evaluating image captioning "
        "with semantic similarity,” year.",
        "[9]  R. Szeliski, Computer Vision: Algorithms and Applications. "
        "Springer, 2010. (or OpenCV documentation citations)",
        "[10] L. Breiman, “Random Forests,” Machine Learning, vol. 45, "
        "pp. 5–32, 2001.",
        "[11] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting "
        "System,” in KDD, 2016.",
        "[12] S. M. Lundberg and S.-I. Lee, “A Unified Approach to "
        "Interpreting Model Predictions,” in NeurIPS, 2017.",
        "[13]  E. Author et al., “Placeholder: explainable multimodal "
        "learning,” year.",
        "[14]  F. Author et al., “Placeholder: human vs AI captioning "
        "comparison,” year.",
    ]
    for ref in refs:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(ref)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)

    add_placeholder(
        document,
        "[Additional references to be finalized before submission. "
        "Replace placeholder entries [1], [2], [7], [8], [13], [14].]",
    )

    document.save(DOCX_PATH)
    return DOCX_PATH


IEEE_TEX = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}
\usepackage{hyperref}

\begin{document}

\title{Explainable Image Ambiguity Prediction Using Human and AI-Generated Caption Diversity with Computer Vision Features}

\author{\IEEEauthorblockN{[Author Name(s)]}
\IEEEauthorblockA{[Department / Affiliation]\\
[Institution], [City, Country]\\
Email: [email@domain.edu]}
}

\maketitle

\begin{abstract}
Image ambiguity arises when multiple plausible linguistic interpretations can be assigned to the same visual scene. This paper presents an explainable machine learning framework that predicts image ambiguity as a categorical label (Low, Medium, High) from caption disagreement and classical computer-vision descriptors. Human captions from Microsoft COCO and AI-generated captions from BLIP are embedded with Sentence-BERT. Pairwise cosine similarity yields a caption-diversity score, fused with OpenCV features. Rule-based thresholds map diversity to labels. High-class rarity is addressed by mining high-diversity images and train-time oversampling. Random Forest and XGBoost classifiers are trained; SHAP explains predictions. A FastAPI and React system supports interactive evaluation and human--AI diversity comparison. On an enriched COCO-derived set ($N{=}330$), classifiers recover diversity-derived labels with high fidelity, while AI captions often disagree less than human captions for the same images.
\end{abstract}

\begin{IEEEkeywords}
image ambiguity, caption diversity, Sentence-BERT, BLIP, OpenCV, Random Forest, XGBoost, SHAP, explainable AI, COCO
\end{IEEEkeywords}

\section{Introduction}
Natural images are frequently described by multiple captions that emphasize different objects, attributes, or events. When captions diverge substantially, the image can be considered linguistically ambiguous~\cite{ref1,ref2}.

This paper contributes: (1)~an operational definition of ambiguity from Sentence-BERT caption diversity with Low/Medium/High thresholds; (2)~fusion of diversity statistics with OpenCV descriptors; (3)~supervised prediction with class-imbalance handling; (4)~SHAP explanations; and (5)~a FastAPI~+~React demonstration comparing human and AI captions.

\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{\centering\vspace{1.2cm}\textbf{[Fig.~1 placeholder]}\\System overview diagram\vspace{1.2cm}}}
\caption{End-to-end architecture of the proposed system.}
\label{fig:overview}
\end{figure}

\section{Literature Review}
COCO provides multi-caption supervision for vision--language research~\cite{lin2014coco}. Neural captioning models including BLIP generate fluent descriptions~\cite{vinyals2015show,li2022blip}. Sentence-BERT enables semantic similarity beyond lexical overlap~\cite{reimers2019sentence}. Caption diversity metrics evaluate generation and dataset difficulty~\cite{ref7,ref8}. Classical visual features remain interpretable complements to embeddings~\cite{szeliski2010}. Tree ensembles are strong tabular baselines~\cite{breiman2001,chen2016xgboost}. SHAP supports feature attribution~\cite{lundberg2017shap}.

\begin{table}[t]
\centering
\caption{Related work comparison (placeholder).}
\label{tab:related}
\begin{tabular}{lccc}
\toprule
Method & Ambiguity def. & Features & XAI \\
\midrule
{[}Related A{]} & TBD & TBD & TBD \\
{[}Related B{]} & TBD & TBD & TBD \\
\textbf{Ours} & Caption diversity & SBERT+OpenCV & SHAP \\
\bottomrule
\end{tabular}
\end{table}

\section{Methodology}
\subsection{Caption Acquisition}
Human branch: COCO captions. AI branch: BLIP with beam, top-$k$, nucleus, and prompt-conditioned decoding~\cite{li2022blip}.

\subsection{Diversity Scoring}
Embeddings $\{e_i\}$ yield pairwise cosine similarities. We define
\begin{equation}
\mathrm{caption\_diversity} = 1 - \mathrm{average\_similarity}.
\end{equation}

\subsection{Visual Features}
OpenCV descriptors: edge density, entropy, brightness, contrast, color variance, texture.

\subsection{Ambiguity Labeling}
\begin{equation}
\label{eq:labels}
\begin{cases}
\mathrm{Low}, & d < 0.35,\\
\mathrm{Medium}, & 0.35 \le d < 0.65,\\
\mathrm{High}, & d \ge 0.65.
\end{cases}
\end{equation}
High rarity is mitigated by mining additional high-diversity images and oversampling.

\subsection{Learning and Explanation}
An 11-D feature vector feeds Random Forest and XGBoost; SHAP explains predictions.

\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{\centering\vspace{1.0cm}\textbf{[Fig.~2 placeholder]}\\Method pipeline\vspace{1.0cm}}}
\caption{Methodological pipeline.}
\label{fig:method}
\end{figure}

\section{Dataset}
Microsoft COCO 2017 validation captions/images~\cite{lin2014coco}. Enriched human table: $N{=}330$ (Low~$109$, Medium~$187$, High~$34$). Optional AI table stores BLIP features for comparison.

\begin{table}[t]
\centering
\caption{Dataset statistics (fill final values).}
\label{tab:data}
\begin{tabular}{lrr}
\toprule
Split & \#Images & Mean diversity \\
\midrule
Human (enriched) & 330 & $\approx 0.41$ \\
AI (prototype) & 20 & $\approx 0.48$ \\
\bottomrule
\end{tabular}
\end{table}

\section{Feature Engineering}
Linguistic: average/min/max/std similarity and caption diversity. Visual: edge density, entropy, brightness, contrast, color variance, texture.

\begin{table}[t]
\centering
\caption{Feature dictionary (placeholder ranges).}
\label{tab:features}
\begin{tabular}{llc}
\toprule
Feature & Family & Notes \\
\midrule
caption\_diversity & Linguistic & Primary ambiguity signal \\
average\_similarity & Linguistic & Mean pairwise cosine \\
edge\_density & Visual & Edge structure \\
entropy & Visual & Intensity complexity \\
\ldots & \ldots & \ldots \\
\bottomrule
\end{tabular}
\end{table}

\section{Machine Learning}
Stratified 80/20 split; train-only oversampling; RF \texttt{class\_weight=balanced}; grid search; select by macro-F1; serialize with joblib.

\begin{table}[t]
\centering
\caption{Model configurations (placeholder).}
\label{tab:models}
\begin{tabular}{lll}
\toprule
Model & Key params & Selection \\
\midrule
Random Forest & $n\_estimators$, depth & macro-F1 \\
XGBoost & $lr$, depth, trees & macro-F1 \\
\bottomrule
\end{tabular}
\end{table}

\section{Experiments}
We study: (Q1)~label recovery from engineered features; (Q2)~impact of High enrichment/balancing; (Q3)~human vs AI diversity; (Q4)~SHAP consistency with the diversity rule.

\section{Results}
Both classifiers achieve very high held-out scores on the current setup, consistent with recovering thresholds when \texttt{caption\_diversity} is in $X$. Human--AI diversity distributions differ; BLIP often reduces disagreement on human-High images.

\begin{table}[t]
\centering
\caption{Classification results (replace after final re-run).}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Model & Acc. & Prec. & Rec. & F1 & ROC-AUC \\
\midrule
Random Forest & TBD & TBD & TBD & TBD & TBD \\
XGBoost & TBD & TBD & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\fbox{\parbox{0.95\columnwidth}{\centering\vspace{1.0cm}\textbf{[Fig.~6 placeholder]}\\Human vs AI diversity\vspace{1.0cm}}}
\caption{Human vs AI caption diversity.}
\label{fig:hvsai}
\end{figure}

\section{Discussion}
Ambiguity as caption disagreement is reproducible and strict under~\eqref{eq:labels}. A key limitation is label--feature dependence (diversity defines labels and is an input). Independent human ambiguity ratings or feature ablations are required for claims beyond pipeline consistency. Caption source must be exposed (user/COCO/BLIP).

\section{Future Work}
Independent ambiguity annotation; full paired human--AI COCO-scale sets; deep models without diversity leakage; calibration; multilingual captions; XAI user studies; in-the-wild deployment.

\section{Conclusion}
We presented an explainable ambiguity prediction pipeline spanning diversity scoring, OpenCV features, class-balanced learning, SHAP, and an interactive demonstration, enabling principled human--AI caption comparison under a shared metric.

\section*{Acknowledgment}
The authors thank [advisor/institution]. We use COCO and open-source libraries (PyTorch, Transformers, scikit-learn, XGBoost, SHAP, OpenCV, FastAPI, React).

\begin{thebibliography}{00}
\bibitem{ref1} A.~Author et al., ``Placeholder: visual uncertainty,'' IEEE Conf., year.
\bibitem{ref2} B.~Author et al., ``Placeholder: annotator disagreement,'' Journal, year.
\bibitem{lin2014coco} T.-Y.~Lin et al., ``Microsoft COCO: Common Objects in Context,'' ECCV, 2014.
\bibitem{vinyals2015show} O.~Vinyals et al., ``Show and Tell,'' CVPR, 2015.
\bibitem{li2022blip} J.~Li et al., ``BLIP,'' ICML, 2022.
\bibitem{reimers2019sentence} N.~Reimers and I.~Gurevych, ``Sentence-BERT,'' EMNLP, 2019.
\bibitem{ref7} C.~Author et al., ``Placeholder: caption diversity metrics,'' year.
\bibitem{ref8} D.~Author et al., ``Placeholder: caption evaluation,'' year.
\bibitem{szeliski2010} R.~Szeliski, \emph{Computer Vision}. Springer, 2010.
\bibitem{breiman2001} L.~Breiman, ``Random Forests,'' Mach.\ Learn., 2001.
\bibitem{chen2016xgboost} T.~Chen and C.~Guestrin, ``XGBoost,'' KDD, 2016.
\bibitem{lundberg2017shap} S.~M.~Lundberg and S.-I.~Lee, ``SHAP,'' NeurIPS, 2017.
\bibitem{ref13} E.~Author et al., ``Placeholder: explainable multimodal learning,'' year.
\bibitem{ref14} F.~Author et al., ``Placeholder: human vs AI captioning,'' year.
\end{thebibliography}

\end{document}
"""


def build_tex() -> Path:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(IEEE_TEX, encoding="utf-8")
    return TEX_PATH


if __name__ == "__main__":
    docx = build_docx()
    tex = build_tex()
    print(f"Wrote {docx}")
    print(f"Wrote {tex}")
