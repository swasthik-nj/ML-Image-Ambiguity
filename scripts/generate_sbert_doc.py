"""One-off script to generate the Sentence-BERT explanation document.

Run once with:

    python scripts/generate_sbert_doc.py

Produces: myDocs/Sentence-BERT_Caption_Diversity_Explanation.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "Sentence-BERT_Caption_Diversity_Explanation.docx"

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
    run = title.add_run("Sentence-BERT for Caption Diversity")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run(
        "A Feature-Extraction Component for Explainable Image Ambiguity Prediction"
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
        "Module: image_ambiguity.features.sentence_embeddings.SentenceEmbeddingGenerator\n"
        "Model: sentence-transformers/all-MiniLM-L6-v2"
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

    # 1. What is Sentence-BERT
    add_heading(document, "1. What Is Sentence-BERT?")
    add_body(
        document,
        "Sentence-BERT (SBERT) is a modification of the pretrained BERT/RoBERTa "
        "family of transformer models, fine-tuned specifically to produce "
        "semantically meaningful sentence embeddings that can be compared "
        "directly using cosine similarity or Euclidean distance."
    )
    add_body(
        document,
        "Vanilla BERT produces contextual token embeddings, but pooling them "
        "naively (e.g. averaging the last layer, or using the [CLS] token) "
        "yields sentence vectors that perform poorly on semantic similarity "
        "tasks and are computationally expensive to compare pairwise, "
        "because BERT requires both sentences to be passed through the "
        "network together (cross-encoding) to get an accurate similarity "
        "score. For n sentences, this means O(n^2) full forward passes."
    )
    add_body(
        document,
        "SBERT solves this with a twin-tower (Siamese) network architecture: "
        "two identical BERT encoders with shared weights independently embed "
        "each sentence into a fixed-size vector. During fine-tuning "
        "(Reimers and Gurevych, 2019), the model is trained on tasks such as "
        "Natural Language Inference (NLI) and Semantic Textual Similarity "
        "(STS) so that semantically similar sentences end up close together "
        "in vector space, and dissimilar sentences end up far apart."
    )
    add_body(
        document,
        "Because embeddings are computed once per sentence (O(n) instead of "
        "O(n^2)), SBERT is far more efficient for large-scale similarity, "
        "clustering, and retrieval tasks -- exactly the setting needed to "
        "compare many image captions against each other."
    )

    # 2. Why this model
    add_heading(document, "2. Why all-MiniLM-L6-v2?")
    add_body(
        document,
        "This project uses the sentence-transformers/all-MiniLM-L6-v2 "
        "checkpoint, a distilled 6-layer MiniLM model fine-tuned on over 1 "
        "billion sentence pairs. It offers the best trade-off between "
        "embedding quality and computational cost among widely used SBERT "
        "checkpoints."
    )
    add_table(
        document,
        ["Property", "Value"],
        [
            ["Architecture", "6-layer MiniLM (distilled from BERT)"],
            ["Embedding dimension", "384"],
            ["Max sequence length", "256 word pieces"],
            ["Training data", "~1B sentence pairs (multiple NLI/STS datasets)"],
            ["Relative speed", "~5x faster than base BERT-large encoders"],
            ["Typical use", "Semantic search, clustering, paraphrase / diversity scoring"],
        ],
    )
    add_body(
        document,
        "Its small size (384-dim vectors, ~22M parameters) makes CPU "
        "inference practical for research iteration, while still ranking "
        "among the strongest general-purpose sentence encoders on the "
        "MTEB / SentEval benchmarks."
    )

    # 3. Role in the project
    add_heading(document, "3. Role in This Project")
    add_body(
        document,
        "The central research hypothesis is that images which are inherently "
        "ambiguous produce captions -- from human annotators and/or "
        "AI captioning models -- that disagree with each other more than "
        "captions for unambiguous images. Caption diversity is therefore "
        "used as a weak, learnable signal for image ambiguity, which is "
        "then combined with computer vision features and made explainable."
    )
    add_body(
        document,
        "Sentence-BERT is the component that converts a set of free-text "
        "captions for a single image into comparable numeric vectors, so "
        "that 'diversity' can be measured mathematically rather than "
        "qualitatively."
    )
    add_numbered(
        document,
        [
            "COCO captions are loaded per image via CocoDatasetLoader "
            "(5 human captions per image in the COCO dataset), optionally "
            "supplemented with AI-generated captions.",
            "Each caption is encoded into a 384-dimensional embedding using "
            "SentenceEmbeddingGenerator (this module).",
            "Pairwise cosine similarity / distance is computed between all "
            "caption embeddings belonging to the same image.",
            "Diversity statistics (e.g. mean pairwise distance, variance, "
            "centroid dispersion) become input features for the ambiguity "
            "prediction model, alongside CV-derived features.",
            "SHAP-based explainability is later applied to attribute the "
            "model's ambiguity score back to individual features, including "
            "caption-diversity features derived from these embeddings.",
        ],
    )

    # 4. Implementation
    add_heading(document, "4. Implementation in This Codebase")
    add_body(
        document,
        "The SentenceEmbeddingGenerator class "
        "(src/image_ambiguity/features/sentence_embeddings.py) wraps the "
        "sentence-transformers library with production-oriented behavior: "
        "explicit lifecycle methods, GPU/CPU auto-detection, batch "
        "inference, structured logging, and reproducible NumPy persistence."
    )
    add_table(
        document,
        ["Method", "Responsibility"],
        [
            ["load_model()", "Loads the SentenceTransformer checkpoint onto the resolved device (CPU/CUDA)."],
            ["generate_embeddings(captions)", "Validates input, runs batched model.encode(), returns a float32 NumPy array of shape (n, 384)."],
            ["save_embeddings(embeddings, path, captions=...)", "Persists embeddings (+ optional captions) as a compressed .npz archive."],
            ["load_embeddings(path)", "Reloads a previously saved .npz / .npy archive as a NumPy array."],
            ["resolve_device(device)", "Auto-selects 'cuda' when available, else falls back to 'cpu'."],
        ],
    )

    add_heading(document, "4.1 Batched, Device-Aware Inference", level=2)
    add_body(
        document,
        "Rather than encoding captions one at a time, generate_embeddings() "
        "delegates to SentenceTransformer.encode() with an explicit "
        "batch_size (default 32). This lets the underlying framework stack "
        "multiple captions into a single tensor and process them in one "
        "forward pass, which is dramatically more efficient on a GPU and "
        "still faster on CPU due to reduced Python-level overhead."
    )
    add_code(
        document,
        """
generator = SentenceEmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="auto",       # -> "cuda" if available, else "cpu"
    batch_size=32,
)
generator.load_model()

captions = [
    "A man is in a kitchen making pizzas.",
    "A baker is working in the kitchen rolling dough.",
    "A person standing by a stove in a kitchen.",
]
embeddings = generator.generate_embeddings(captions)
# embeddings.shape == (3, 384), dtype == float32
""",
    )
    add_body(
        document,
        "Device selection is automatic and safe: if PyTorch or CUDA is "
        "unavailable, the module logs a warning and transparently falls "
        "back to CPU rather than failing, which keeps the pipeline portable "
        "across laptops, workstations, and GPU servers."
    )

    add_heading(document, "4.2 Reproducible Persistence", level=2)
    add_body(
        document,
        "Embeddings are saved with save_embeddings() as compressed .npz "
        "archives that store the embedding matrix alongside the source "
        "captions and metadata (model name, device). This keeps every "
        "experiment auditable: any downstream diversity score can be traced "
        "back to the exact captions and model checkpoint that produced it."
    )
    add_code(
        document,
        """
path = generator.save_embeddings(
    embeddings, "results/embeddings/image_397133.npz", captions=captions
)
reloaded = generator.load_embeddings(path)
""",
    )

    # 5. Diversity metrics
    add_heading(document, "5. From Embeddings to an Ambiguity Signal")
    add_body(
        document,
        "Once captions for an image are embedded, pairwise cosine similarity "
        "is the primary comparison operator:"
    )
    add_code(
        document,
        "cosine_similarity(u, v) = (u . v) / (||u|| * ||v||)",
    )
    add_body(
        document,
        "Because embeddings are L2-normalized by default in this module "
        "(normalize_embeddings=True), cosine similarity reduces to a simple "
        "dot product, which is both faster to compute and numerically "
        "stable for downstream feature engineering."
    )
    add_bullets(
        document,
        [
            "Mean pairwise cosine distance across all caption pairs for an "
            "image -- higher values suggest captions describe the image "
            "differently, a proxy for semantic ambiguity.",
            "Variance / spread of embeddings around their centroid -- "
            "captures how tightly captions cluster in meaning space.",
            "Maximum pairwise distance -- flags the single most divergent "
            "caption pair, useful for qualitative error analysis.",
            "Human-vs-AI caption divergence -- comparing the centroid of "
            "human captions against AI-generated captions can isolate "
            "disagreement introduced by the captioning model itself, versus "
            "disagreement inherent to the image.",
        ],
    )

    # 6. Why this matters for explainability
    add_heading(document, "6. Why This Supports Explainability")
    add_body(
        document,
        "A key design goal of the project is that ambiguity predictions must "
        "be explainable, not just accurate. Caption-diversity features "
        "derived from Sentence-BERT are inherently interpretable: a SHAP "
        "attribution showing that 'mean caption distance' was the dominant "
        "contributor to a high ambiguity score is directly explainable to a "
        "human reviewer -- unlike raw pixel-level or opaque deep features. "
        "This makes Sentence-BERT embeddings a bridge between free-text "
        "human judgment and a quantifiable, model-ready feature."
    )

    # 7. Talking points
    add_heading(document, "7. Conference Talking Points (Summary)")
    add_bullets(
        document,
        [
            "Sentence-BERT converts free-text captions into comparable "
            "vectors in O(n) time via a Siamese BERT architecture, versus "
            "O(n^2) for cross-encoding BERT.",
            "We use all-MiniLM-L6-v2: 384-dim embeddings, ~5x faster than "
            "base BERT, strong performance on semantic similarity "
            "benchmarks -- ideal for iterative research.",
            "Our SentenceEmbeddingGenerator wraps this model with "
            "production practices: automatic GPU/CPU selection, batched "
            "inference, structured logging, input validation, and "
            "reproducible .npz persistence.",
            "Caption embeddings let us quantify 'how much captions "
            "disagree' for an image -- our core hypothesis is that this "
            "disagreement correlates with visual ambiguity.",
            "These diversity features are explicitly interpretable, "
            "enabling transparent SHAP-based explanations of the final "
            "ambiguity prediction -- fulfilling the explainability goal in "
            "the project title.",
        ],
    )

    # 8. References
    add_heading(document, "8. References")
    add_bullets(
        document,
        [
            "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence "
            "Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019.",
            "Wang, W. et al. (2020). MiniLM: Deep Self-Attention "
            "Distillation for Task-Agnostic Compression of Pre-Trained "
            "Transformers. NeurIPS 2020.",
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
