"""Generate documentation for the visual-gate + stable-BLIP update.

Run:

    python scripts/generate_visual_gate_doc.py

Produces: myDocs/Visual_Gate_Stable_BLIP_Update.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "Visual_Gate_Stable_BLIP_Update.docx"

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
    run = title.add_run("Visual-Gate and Stable-BLIP Update")
    run.bold = True
    run.font.size = Pt(26)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run(
        "Fixing inflated ambiguity on simple images during BLIP inference"
    )
    run2.italic = True
    run2.font.size = Pt(13)
    run2.font.color.rgb = DARK

    document.add_paragraph()
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(
        "Project: Explainable Image Ambiguity Prediction\n"
        "Date: August 2026\n"
        "Scope: Inference-only; label thresholds unchanged"
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
        p = document.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)


def add_code(document: Document, text: str) -> None:
    p = document.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(10)


def build() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    set_base_font(document)
    add_title_page(document)

    add_heading(document, "1. What problem this update solves")
    add_body(
        document,
        "When users upload non-COCO images (for example an apple on a white "
        "background), BLIP was configured to produce many varied captions "
        "using beam search, top-k sampling, nucleus sampling, and multiple "
        "text prompts. Even when all captions described the same object, "
        "Sentence-BERT diversity could rise into the Medium or High range. "
        "The Random Forest then predicted Medium/High ambiguity even though "
        "the scene was visually and semantically simple.",
    )
    add_body(
        document,
        "This update does not change the Low / Medium / High definition "
        "(diversity < 0.35, 0.35–0.65, ≥ 0.65). It changes only how BLIP "
        "captions are generated and filtered before diversity is computed "
        "at inference time.",
    )

    add_heading(document, "2. Root cause (why it happened)")
    add_bullets(
        document,
        [
            "Diverse BLIP decoding invents wording variety, not semantic "
            "disagreement.",
            "Diversity = 1 − average pairwise SBERT cosine similarity treats "
            "paraphrases as disagreement when vectors are not identical.",
            "OpenCV features were already in the model, but caption diversity "
            "still dominated when BLIP captions were noisy.",
            "Stale sample captions in the UI could also override the image "
            "when BLIP was off (fixed earlier in PredictionPage.tsx).",
        ],
    )

    add_heading(document, "3. Solution overview")
    add_body(
        document,
        "A visual simplicity gate uses existing OpenCV features as an "
        "additional signal. It selects the BLIP generation path; it never "
        "hardcodes Low/Medium/High.",
    )
    add_bullets(
        document,
        [
            "Visually simple image → stable BLIP (beam search only) + semantic "
            "caption clustering → diversity computed on cluster representatives.",
            "Visually complex image → diverse BLIP (beam, top-k, nucleus, "
            "prompted sampling) → diversity on raw captions (clustering off "
            "by default).",
            "Final 11-D feature vector still feeds the same Random Forest; "
            "SHAP explanations unchanged.",
        ],
    )

    add_heading(document, "4. Files changed and why")
    add_heading(document, "4.1 New modules", level=2)
    add_bullets(
        document,
        [
            "src/image_ambiguity/features/visual_simplicity.py — votes on "
            "edge_density, entropy, texture, contrast, color_variance, and "
            "optional brightness to decide is_simple.",
            "src/image_ambiguity/features/caption_clustering.py — merges "
            "SBERT-near-duplicate captions (threshold 0.85 default) before "
            "diversity.",
        ],
    )
    add_heading(document, "4.2 BLIP generation", level=2)
    add_bullets(
        document,
        [
            "src/image_ambiguity/features/blip_captions.py — added generate_stable(), "
            "generate_for_mode(), stable prompted beam captions, and "
            "flatten_strategy_captions().",
        ],
    )
    add_heading(document, "4.3 Inference wiring", level=2)
    add_bullets(
        document,
        [
            "backend/app/services/inference.py — assess_image_simplicity(), "
            "generate_captions(mode), maybe_cluster_captions(), updated "
            "resolve_captions() and extract_features(); API returns "
            "visual_simplicity and caption_pipeline metadata.",
        ],
    )
    add_heading(document, "4.4 Configuration", level=2)
    add_bullets(
        document,
        [
            "configs/default.yaml — visual_gate_* and blip_* settings.",
            "src/image_ambiguity/config.py — Settings fields with IAP_ env "
            "override support.",
        ],
    )
    add_heading(document, "4.5 Tests", level=2)
    add_bullets(
        document,
        [
            "tests/unit/test_visual_gate.py — simplicity voting and clustering.",
            "tests/unit/test_blip_captions.py — updated for prompted output.",
            "tests/unit/test_config.py — new settings load correctly.",
        ],
    )

    add_heading(document, "5. How the visual gate works")
    add_body(
        document,
        "OpenCV features are extracted first. Six criteria vote whether the "
        "image looks simple (low edges, low entropy, low texture, low contrast, "
        "low color variance, optionally high brightness). The image is simple "
        "when at least 3 votes pass AND at least 50% of criteria pass. "
        "Thresholds are configurable; defaults were chosen from medians in "
        "human_dataset.csv (Low class tends to have lower edge density and "
        "texture than High).",
    )
    add_code(
        document,
        "visual_gate_edge_density_max: 0.040\n"
        "visual_gate_entropy_max: 7.25\n"
        "visual_gate_texture_max: 900.0\n"
        "visual_gate_contrast_max: 55.0\n"
        "visual_gate_color_variance_max: 3200.0\n"
        "visual_gate_brightness_min: 165.0\n"
        "visual_gate_min_simple_votes: 3\n"
        "visual_gate_min_vote_fraction: 0.5",
    )

    add_heading(document, "6. Stable vs diverse BLIP")
    add_heading(document, "6.1 Stable path (simple images)", level=2)
    add_bullets(
        document,
        [
            "Beam search only (no random sampling).",
            "Optional mild prompts: \"\" and \"a photo of\".",
            "Fewer, more consistent captions.",
            "Clustering ON: merge paraphrases before diversity.",
        ],
    )
    add_heading(document, "6.2 Diverse path (complex images)", level=2)
    add_bullets(
        document,
        [
            "Beam + top-k + nucleus + prompted sampling (previous behaviour).",
            "Higher temperature / top_p for variation.",
            "Clustering OFF by default so genuine disagreement is preserved.",
        ],
    )

    add_heading(document, "7. End-to-end inference flow")
    add_code(
        document,
        "Image upload\n"
        "  → OpenCV features\n"
        "  → visual_simplicity assessment (is_simple?)\n"
        "  → caption source: user | COCO human | BLIP\n"
        "  → if BLIP: stable or diverse mode\n"
        "  → optional semantic clustering\n"
        "  → SBERT diversity (avg/min/max/std)\n"
        "  → 11-D feature vector\n"
        "  → Random Forest prediction\n"
        "  → SHAP explanation",
    )

    add_heading(document, "8. What was NOT changed")
    add_bullets(
        document,
        [
            "Label thresholds: Low < 0.35, Medium 0.35–0.65, High ≥ 0.65.",
            "FEATURE_COLUMNS for training (11 features).",
            "COCO human caption preference for val2017 filenames.",
            "No forced Low prediction for simple images.",
        ],
    )

    add_heading(document, "9. How to run and verify")
    add_code(
        document,
        "$env:PYTHONPATH=\"src;.\"\n"
        "python -m pytest tests/unit/test_visual_gate.py "
        "tests/unit/test_blip_captions.py tests/unit/test_config.py -q\n\n"
        "# Restart API after code changes\n"
        "uvicorn app:app --reload",
    )
    add_body(
        document,
        "Manual checks: upload a simple apple on white (expect is_simple=true, "
        "blip_mode=stable, lower diversity); upload a face/rock illusion "
        "(expect is_simple=false, blip_mode=diverse). Inspect API fields "
        "visual_simplicity and caption_pipeline in the JSON response.",
    )

    add_heading(document, "10. Tuning without retraining")
    add_bullets(
        document,
        [
            "Raise blip_cluster_similarity_threshold (e.g. 0.88) to merge more "
            "paraphrases.",
            "Loosen visual_gate_* if too many complex images are marked simple.",
            "Set IAP_VISUAL_GATE_ENABLED=false to disable gating entirely.",
            "Set blip_cluster_when_complex=true to cluster on complex images too.",
        ],
    )

    document.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
