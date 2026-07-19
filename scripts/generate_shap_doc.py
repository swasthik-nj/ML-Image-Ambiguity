"""One-off script to generate the SHAP Explainability explanation document.

Run once with:

    python scripts/generate_shap_doc.py

Produces: myDocs/SHAP_Explainability_Explanation.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "SHAP_Explainability_Explanation.docx"

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
    run = title.add_run("Opening the Black Box with SHAP")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run(
        "Feature Importance, Summary, Waterfall, and Force Plots for the "
        "Random Forest Ambiguity Classifier"
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
        "Module: image_ambiguity.explainability.shap_explainer.SHAPExplainer\n"
        "Output: results/figures/shap_*.png, results/explanations/shap_*.json"
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
        "Random Forest and XGBoost gave this project two working "
        "ambiguity classifiers, but neither model explains itself: a "
        "prediction of 'High ambiguity' is just a class label unless "
        "something can point to which of the eleven input features "
        "actually drove that decision, and by how much. For a research "
        "project centered on explainability, this is not optional -- it "
        "is the point."
    )
    add_body(
        document,
        "SHAP (SHapley Additive exPlanations) fills that gap. It assigns "
        "every feature of every prediction a signed contribution value "
        "-- how much that feature pushed the model's output up or down "
        "from a baseline -- grounded in cooperative game theory, so the "
        "contributions are guaranteed to sum exactly to the difference "
        "between the model's baseline output and its actual prediction."
    )

    # 2. What SHAP is
    add_heading(document, "2. What SHAP Is")
    add_body(
        document,
        "SHAP values are based on Shapley values from cooperative game "
        "theory: treat each feature as a 'player' contributing to a "
        "'payout' (the model's prediction), and fairly split that payout "
        "among the players based on their average marginal contribution "
        "across every possible subset (coalition) of features. This "
        "fair-division property is what makes SHAP values additive and "
        "comparable across features, rows, and even different models."
    )
    add_body(
        document,
        "Computing exact Shapley values requires evaluating every "
        "feature subset (2^11 = 2048 combinations for this project's "
        "11 features) -- expensive in general, but shap.TreeExplainer "
        "exploits the internal structure of decision trees to compute "
        "exact SHAP values for tree ensembles (Random Forest, XGBoost, "
        "LightGBM, ...) in low-order polynomial time instead, which is "
        "why this project's SHAPExplainer wraps TreeExplainer rather "
        "than the slower, model-agnostic KernelExplainer."
    )

    # 3. Why SHAP for this project
    add_heading(document, "3. Why SHAP Was Chosen for This Project")
    add_bullets(
        document,
        [
            "Model-agnostic within the tree family: the same "
            "SHAPExplainer class works unmodified for both the Random "
            "Forest and XGBoost models trained earlier, since both are "
            "supported by TreeExplainer.",
            "Consistent, additive attributions: contributions for a row "
            "always sum to (prediction - base value), so 'why did the "
            "model say High ambiguity' has one precise, numeric answer "
            "rather than a qualitative feature-importance guess.",
            "Both global and local views from one computation: the same "
            "shap_values object powers dataset-wide feature importance "
            "(Top Feature Importance, Summary plot) and single-row "
            "explanations (Waterfall, Force) without recomputing "
            "anything.",
            "Directly corroborates the labeling rule: because "
            "ambiguity_label is a deterministic function of "
            "caption_diversity, SHAP should -- and does -- surface "
            "caption_diversity and average_similarity as by far the "
            "most important features, giving a built-in sanity check on "
            "the whole pipeline.",
        ],
    )

    # 4. Implementation
    add_heading(document, "4. Implementation in This Codebase")
    add_table(
        document,
        ["Method", "Responsibility"],
        [
            ["compute_shap_values(X)", "Runs shap.TreeExplainer(model)(X); returns a shap.Explanation with values shaped (n_samples, n_features, n_classes)."],
            ["top_feature_importance(shap_values, class_index, top_n)", "Ranks features by mean |SHAP value|, aggregated across classes or for one class."],
            ["plot_top_feature_importance(importance_df, path)", "Renders and saves a horizontal bar chart of the ranking."],
            ["summary_plot(shap_values, X, class_index, path)", "Renders and saves a SHAP beeswarm/summary plot for one class."],
            ["waterfall_plot(shap_values, row_index, class_index, path)", "Renders and saves a waterfall plot decomposing one prediction."],
            ["force_plot(shap_values, row_index, class_index, path)", "Renders and saves a static (matplotlib) force plot for one prediction."],
            ["explain_prediction(shap_values, X, row_index, class_index)", "Builds a PredictionExplanation: predicted class, probability, and the top contributing features with direction."],
        ],
    )
    add_code(
        document,
        """
explainer = SHAPExplainer(
    model,                       # trained RandomForestClassifier
    feature_names=FEATURE_COLUMNS,
    class_names=["Low", "Medium", "High"],
)
shap_values = explainer.compute_shap_values(X)   # (300, 11, 3)

importance = explainer.top_feature_importance(shap_values, top_n=10)
explainer.plot_top_feature_importance(importance, "shap_top_feature_importance.png")
explainer.summary_plot(shap_values, X, class_index=2, path="shap_summary_high.png")

explanation = explainer.explain_prediction(shap_values, X, row_index=187)
print(explanation.summary_text())
""",
    )

    # 5. The four deliverables
    add_heading(document, "5. The Four Required Outputs")
    add_numbered(
        document,
        [
            "Summary Plot: a beeswarm chart for one class (High, by "
            "default) showing every image's SHAP value for every "
            "feature at once -- color-coded by that feature's raw "
            "value -- revealing both which features matter and in "
            "which direction.",
            "Waterfall Plot: for one chosen prediction, a single-image "
            "step-by-step breakdown showing exactly how each feature "
            "pushed the model's output up or down from its baseline to "
            "its final probability.",
            "Force Plot: the same single-prediction attribution "
            "rendered as one compressed, color-coded bar (pink pushes "
            "the prediction higher, blue pushes it lower) -- a more "
            "compact, presentation-friendly alternative to the "
            "waterfall view.",
            "Top Feature Importance: a bar chart ranking all eleven "
            "features by their mean absolute SHAP value across every "
            "sampled image and class, answering 'which features matter "
            "most, overall?' in one glance.",
        ],
    )

    # 6. Worked example
    add_heading(document, "6. Worked Example (from this codebase)")
    add_body(
        document,
        "Running python src/shap_analysis.py against the trained Random "
        "Forest model and all 300 labeled images produced the following "
        "Top Feature Importance ranking:"
    )
    add_table(
        document,
        ["Rank", "Feature", "Mean |SHAP value|"],
        [
            ["1", "average_similarity", "0.1273"],
            ["2", "caption_diversity", "0.1269"],
            ["3", "minimum_similarity", "0.0336"],
            ["4", "maximum_similarity", "0.0213"],
            ["5", "std_similarity", "0.0118"],
            ["6", "brightness", "0.0031"],
            ["7", "entropy", "0.0026"],
            ["8", "contrast", "0.0019"],
            ["9", "edge_density", "0.0014"],
            ["10", "color_variance", "0.0011"],
        ],
    )
    add_body(
        document,
        "The five caption-diversity features together dwarf all six "
        "OpenCV features combined -- exactly as expected, since "
        "ambiguity_label is derived directly from caption_diversity "
        "(see Section 8)."
    )
    add_body(
        document,
        "One prediction was explained in detail: image_id 425227 was "
        "predicted High ambiguity with 72% probability. Its top "
        "contributing features were:"
    )
    add_table(
        document,
        ["Feature", "Value", "SHAP Contribution", "Effect"],
        [
            ["caption_diversity", "0.719", "+0.207", "increased High probability"],
            ["average_similarity", "0.281", "+0.174", "increased High probability"],
            ["minimum_similarity", "-0.0001", "+0.153", "increased High probability"],
            ["std_similarity", "0.359", "+0.095", "increased High probability"],
            ["entropy", "6.325", "+0.025", "increased High probability"],
        ],
    )
    add_body(
        document,
        "These five contributions, plus the remaining six smaller ones "
        "and the model's base rate (0.0115), sum exactly to the final "
        "predicted probability of 0.72 -- the additive guarantee that "
        "sets SHAP apart from ordinary feature-importance scores."
    )

    # 7. Rendering notes
    add_heading(document, "7. A Practical Note on Multiclass Models")
    add_body(
        document,
        "Because the ambiguity classifier predicts three classes "
        "(Low/Medium/High) rather than a single score, shap.TreeExplainer "
        "returns one SHAP value per feature per class -- an array shaped "
        "(n_images, 11 features, 3 classes) rather than a flat 2D table. "
        "Every plot in this module therefore requires an explicit "
        "class_index: the Summary plot defaults to explaining the "
        "'High' class (the most actionable one for flagging ambiguous "
        "images), while the Waterfall and Force plots explain whichever "
        "class the model actually predicted for that specific image."
    )

    # 8. Label-provenance caveat
    add_heading(document, "8. Reading the Results: A Built-In Sanity Check")
    add_body(
        document,
        "The dominance of average_similarity and caption_diversity in "
        "the Top Feature Importance ranking is not a surprise -- it is "
        "a confirmation. ambiguity_label was constructed as a fixed "
        "threshold rule over caption_diversity (see the earlier "
        "Ambiguity Label Generation module), so a correctly trained "
        "model should recover exactly this relationship. SHAP makes "
        "that recovery visible and quantifiable rather than a "
        "hidden implementation detail, which is itself a valuable "
        "explainability result: it proves the model learned the "
        "intended signal rather than spurious correlations in the "
        "OpenCV features."
    )

    # 9. Environment note
    add_heading(document, "9. Implementation Note: A Blocked Dependency")
    add_body(
        document,
        "shap depends on numba for a few internal helper functions "
        "(only used by hierarchical-clustering-based plot orderings, "
        "not by TreeExplainer itself). On this development machine, "
        "numba's native extensions are blocked by a Windows Application "
        "Control policy, which would otherwise make shap entirely "
        "unimportable. A small compatibility shim "
        "(image_ambiguity.explainability.numba_stub) detects this "
        "failure and substitutes a no-op, pure-Python stand-in for the "
        "handful of numba functions shap needs at import time -- "
        "restoring full functionality for every feature used in this "
        "project with no change to correctness, only to the (irrelevant, "
        "at this data scale) JIT speedup."
    )

    # 10. Talking points
    add_heading(document, "10. Conference Talking Points (Summary)")
    add_bullets(
        document,
        [
            "SHAP grounds feature attribution in Shapley values from "
            "cooperative game theory, guaranteeing that every "
            "prediction's feature contributions sum exactly to its "
            "final output -- not just a heuristic ranking.",
            "TreeExplainer computes exact SHAP values for the Random "
            "Forest classifier efficiently, exploiting tree structure "
            "instead of the exponential brute-force Shapley "
            "computation.",
            "Four complementary views were generated: a Top Feature "
            "Importance ranking (global), a class-level Summary plot "
            "(global + directional), and Waterfall/Force plots for one "
            "specific prediction (local).",
            "average_similarity and caption_diversity dominate every "
            "ranking, correctly recovering the fixed threshold rule "
            "used to construct ambiguity_label -- a concrete, "
            "quantitative sanity check on the entire pipeline.",
            "One prediction (image 425227, 72% High ambiguity) was "
            "fully decomposed feature-by-feature, demonstrating exactly "
            "how much each input pushed the model toward its final "
            "answer.",
        ],
    )

    # 11. References
    add_heading(document, "11. References")
    add_bullets(
        document,
        [
            "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to "
            "Interpreting Model Predictions. NeurIPS 2017.",
            "Lundberg, S. M. et al. (2020). From Local Explanations to "
            "Global Understanding with Explainable AI for Trees. "
            "Nature Machine Intelligence, 2, 56-67.",
            "Shapley, L. S. (1953). A Value for n-Person Games. "
            "Contributions to the Theory of Games, 2(28), 307-317.",
            "Breiman, L. (2001). Random Forests. Machine Learning, "
            "45(1), 5-32.",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
