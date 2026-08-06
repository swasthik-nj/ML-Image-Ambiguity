"""Generate the full interview-preparation document for this project.

Covers all 25 sections requested for technical + HR interview prep:
elevator pitch, problem statement, solution, tech stack, architecture,
data design, folder structure, request flow, features, auth, API design,
challenges, performance, security, testing, deployment, future work,
learnings, 30 interview Q&A, HR Q&A, 2/5/10-minute explanations, STAR,
and common mistakes.

Run:

    python scripts/generate_interview_prep_doc.py

Produces: myDocs/Interview_Preparation_Guide.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "myDocs" / "Interview_Preparation_Guide.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x55, 0x55, 0x55)


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def set_base_font(document: Document, name: str = "Calibri") -> None:
    style = document.styles["Normal"]
    style.font.name = name
    style.font.size = Pt(11)
    rpr = style.element.get_or_add_rPr()
    r_fonts = rpr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(r_fonts)
    r_fonts.set(qn("w:eastAsia"), name)


def add_title_page(document: Document) -> None:
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Interview Preparation Guide")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ACCENT

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = subtitle.add_run("Explainable Image Ambiguity Prediction")
    run2.italic = True
    run2.font.size = Pt(15)
    run2.font.color.rgb = DARK

    document.add_paragraph()
    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(
        "Prepared for technical + HR interview rounds\n"
        "Stack: COCO . Sentence-BERT . BLIP . OpenCV . Random Forest / "
        "XGBoost . SHAP . FastAPI . React (Ambiguity Lens)\n"
        "Covers: elevator pitch, architecture, data design, request flow, "
        "APIs, challenges, security, 30 interview questions, HR questions, "
        "2/5/10-minute explanations, STAR story, and common mistakes."
    )
    meta_run.font.size = Pt(11)
    meta_run.font.color.rgb = MUTED
    document.add_page_break()


def add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = ACCENT if level == 1 else ACCENT2


def add_body(document: Document, text: str) -> None:
    p = document.add_paragraph(text)
    p.paragraph_format.space_after = Pt(8)


def add_quote(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    run.italic = True
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        p = document.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(3)


def add_numbered(document: Document, items: list[str]) -> None:
    for item in items:
        p = document.add_paragraph(item, style="List Number")
        p.paragraph_format.space_after = Pt(3)


def add_code(document: Document, code: str) -> None:
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    for i, line in enumerate(code.strip("\n").split("\n")):
        run = p.add_run(("\n" if i > 0 else "") + line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x0A, 0x0A, 0x0A)
    p.paragraph_format.space_after = Pt(10)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr_cells[i].text = ""
        run = hdr_cells[i].paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(9.5)
    for row_values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_values):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9.5)
    p_after = document.add_paragraph()
    p_after.paragraph_format.space_after = Pt(10)


def add_qa(document: Document, question: str, answer: str) -> None:
    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(question)
    run.bold = True
    run.font.color.rgb = ACCENT2
    add_body(document, answer)


# --------------------------------------------------------------------------
# Document build
# --------------------------------------------------------------------------
def build() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    set_base_font(document)
    add_title_page(document)

    # 1 ------------------------------------------------------------------
    add_heading(document, "1. Elevator Pitch (30-60 seconds)")
    add_quote(
        document,
        "\"I built an explainable machine learning system called Image "
        "Ambiguity Prediction. It looks at an image and multiple captions "
        "describing it, and predicts how ambiguous that image is - Low, "
        "Medium, or High - based on how much the captions disagree with "
        "each other. It's for researchers and ML practitioners who care "
        "about dataset quality and caption evaluation - for example, if "
        "you're building an image-captioning dataset and want to flag "
        "images where human describers can't agree on what's happening. "
        "Existing captioning benchmarks just average caption quality; they "
        "don't quantify disagreement. My system turns caption diversity "
        "into a measurable, explainable signal using Sentence-BERT "
        "embeddings, classical computer vision features, and a Random "
        "Forest / XGBoost classifier, and it explains every prediction "
        "with SHAP. I also built a side-by-side comparison between human "
        "captions and AI-generated (BLIP) captions, which shows AI models "
        "often agree with themselves more than humans do - a subtle but "
        "important finding.\"",
    )

    # 2 ------------------------------------------------------------------
    add_heading(document, "2. Problem Statement")
    add_heading(document, "Real-world problem", level=2)
    add_body(
        document,
        "Not all images are equally easy to describe. Some images have one "
        "obvious interpretation; others are genuinely ambiguous (a "
        "cluttered scene, an action mid-motion, an object that could be "
        "several things). When humans caption the same image multiple "
        "different ways, that disagreement is signal - it tells you the "
        "image is inherently ambiguous - but almost nobody measures it "
        "directly.",
    )
    add_heading(document, "Why existing solutions fall short", level=2)
    add_bullets(
        document,
        [
            "Captioning metrics (BLEU, CIDEr, METEOR) measure how close a "
            "generated caption is to reference captions - they say nothing "
            "about whether the references agree with each other.",
            "Most 'difficulty' filters for vision-language datasets rely on "
            "manual review or crude heuristics (caption length, word "
            "overlap), not semantic disagreement.",
            "There is very little tooling that treats ambiguity as a "
            "first-class, measurable, explainable label you can train a "
            "classifier on.",
        ],
    )
    add_heading(document, "Motivation", level=2)
    add_body(
        document,
        "I wanted an objective, reproducible way to say 'this image is "
        "ambiguous' that doesn't rely on subjective human judgment calls, "
        "and I wanted the answer to be explainable (not a black box), so "
        "it could actually be trusted and used to curate or filter "
        "datasets.",
    )

    # 3 ------------------------------------------------------------------
    add_heading(document, "3. Solution")
    add_body(document, "How it solves the problem:")
    add_numbered(
        document,
        [
            "Take multiple captions for an image (human-written from COCO, "
            "or AI-generated via BLIP).",
            "Embed each caption with Sentence-BERT, compute pairwise "
            "cosine similarity, and derive a single caption_diversity "
            "score (1 - average_similarity).",
            "Add classical OpenCV visual features (edge density, entropy, "
            "brightness, contrast, color variance, texture) so the model "
            "also has visual context.",
            "Apply a rule-based threshold to turn the diversity score into "
            "a label: Low / Medium / High.",
            "Train a Random Forest / XGBoost classifier on these 11 "
            "features to predict that label from an image + captions.",
            "Explain every prediction with SHAP, so you can see exactly "
            "which feature pushed the prediction toward High or Low.",
            "Wrap it all in a FastAPI backend and a React demo UI so "
            "anyone can upload an image and see the result live.",
        ],
    )
    add_heading(document, "User workflow (from the UI)", level=2)
    add_code(
        document,
        """
User opens Ambiguity Lens UI
        |
        v
Uploads an image
        |
        v
Filename matches a known COCO image? --Yes--> Auto-load 5 human COCO captions --+
        |No                                                                     |
        v                                                                       |
User types captions OR enables BLIP AI captions ----------------------------->--+
        |
        v
Backend extracts features (Sentence-BERT diversity + OpenCV)
        |
        v
Model predicts Low / Medium / High
        |
        v
SHAP explains the prediction
        |
        v
UI shows label, confidence, charts, and explanation
""",
    )

    # 4 ------------------------------------------------------------------
    add_heading(document, "4. Tech Stack")
    add_body(document, "Grouped by category, with rationale for each choice.")
    add_table(
        document,
        ["Category", "Technology", "Why chosen / role"],
        [
            ["Frontend", "React 19 + TypeScript", "Type safety, component reuse; renders Upload/Predict/Compare pages"],
            ["Frontend build", "Vite 6", "Instant HMR, fast builds, simpler config than Webpack"],
            ["Styling", "Tailwind CSS v4", "Utility-first, fast prototyping, no separate CSS files"],
            ["Charts", "Recharts", "Declarative React charting for probabilities and SHAP bars"],
            ["Routing", "React Router v7", "Client-side routing for a multi-page SPA"],
            ["Backend framework", "FastAPI", "Async, auto OpenAPI/Swagger docs, native Pydantic validation"],
            ["ASGI server", "Uvicorn", "Standard production ASGI server for FastAPI"],
            ["Validation/Config", "Pydantic + pydantic-settings", "Type-safe schemas; layered config (env > .env > YAML > defaults)"],
            ["NLP embeddings", "Sentence-Transformers (all-MiniLM-L6-v2)", "Fast, small, well-validated for semantic similarity"],
            ["AI captioning", "BLIP (Salesforce/blip-image-captioning-base)", "Open-source, runs locally, supports multiple decoding strategies"],
            ["DL runtime", "PyTorch", "Backend for Sentence-Transformers and BLIP"],
            ["Classical CV", "OpenCV", "Fast, battle-tested feature extraction (edges, entropy, texture)"],
            ["ML models", "scikit-learn (Random Forest) + XGBoost", "Strong tabular baselines, interpretable, no GPU needed"],
            ["Explainability", "SHAP", "Model-agnostic, theoretically grounded per-prediction attribution"],
            ["Data handling", "pandas, NumPy", "Standard tabular manipulation for feature/label CSVs"],
            ["Data source", "COCO 2017 (via pycocotools)", "Industry-standard multi-caption image dataset"],
            ["Model persistence", "joblib", "Standard scikit-learn/XGBoost artifact serialization"],
            ["Storage", "Flat CSV / JSON / joblib files", "Simplest reproducible format; no concurrent multi-user writes"],
            ["Auth", "None (by design)", "Stateless public research/demo API, no private user data"],
            ["Version control", "Git", "Track project history"],
            ["Testing", "pytest, pytest-cov", "Standard Python testing with coverage"],
            ["Linting", "ruff", "Fast, combines flake8 + isort + more"],
            ["Docs generation", "python-docx, matplotlib, seaborn", "Auto-generate reports, figures, and this guide"],
        ],
    )
    add_body(
        document,
        "Note: there is no database and no authentication in this "
        "project. It's a research/ML pipeline, not a CRUD SaaS app - "
        "being upfront about this is safer in interviews than inventing "
        "a JWT flow that doesn't exist.",
    )

    # 5 ------------------------------------------------------------------
    add_heading(document, "5. System Architecture")
    add_body(
        document,
        "Two halves: an offline/training pipeline that builds the dataset "
        "and trains models, and an online inference system (FastAPI + "
        "React) that serves predictions.",
    )
    add_code(
        document,
        """
OFFLINE TRAINING PIPELINE (CLI scripts)
----------------------------------------
COCO val2017 images + captions
        -> CocoDatasetLoader
        -> Sentence-BERT embeddings -> CaptionDiversityAnalyzer
        -> OpenCVFeatureExtractor (in parallel)
        -> MLDatasetBuilder -> human_dataset.csv / ai_dataset.csv
        -> AmbiguityLabelGenerator (Low/Medium/High)
        -> enrich_high_diversity.py (mine more High samples)
        -> ModelTrainer (RF + XGBoost, class balancing)
        -> best_model.joblib

ONLINE SYSTEM
----------------------------------------
React "Ambiguity Lens" UI  <-- HTTP/JSON -->  FastAPI Backend
                                                    |
                                +-------------------+-------------------+
                                |                   |                   |
                     AmbiguityInferenceService   Compare Service    Uploads store
                                |                   |             (results/uploads)
                +---------------+---------------+   |
                |               |               |   v
           BLIP captions   Sentence-BERT   OpenCV   human_dataset.csv /
                |               |               |   ai_dataset.csv
                +-------> best_model.joblib <----+
                                |
                                v
                          SHAP Explainer
""",
    )
    add_heading(document, "Component roles", level=2)
    add_bullets(
        document,
        [
            "Client: React SPA (Vite dev server / static build) - talks to "
            "the API via fetch, proxied through /api.",
            "Server: FastAPI app (backend/app/main.py) - CORS-enabled, "
            "mounts one router (backend/app/api/routes.py).",
            "'Database': none. State lives in CSV feature tables, a "
            "joblib model loaded once into memory, and per-upload JSON "
            "metadata files on disk.",
            "Authentication: none - public, stateless endpoints.",
            "Third-party services: none at runtime - BLIP and "
            "Sentence-BERT models are downloaded once from Hugging Face "
            "and run locally.",
            "File storage: local filesystem - dataset/, results/uploads/, "
            "models/.",
        ],
    )

    # 6 ------------------------------------------------------------------
    add_heading(document, "6. Database Design (Data Model)")
    add_body(
        document,
        "Honest framing: this project does NOT use a relational or NoSQL "
        "database. It uses flat files as its data layer:",
    )
    add_table(
        document,
        ["'Table'", "Real file", "Purpose"],
        [
            ["Images", "dataset/val2017/*.jpg + captions_val2017.json", "Source images + human captions"],
            ["Human features", "dataset/human_dataset.csv", "One row per image: 11 features + label"],
            ["AI features", "dataset/ai_dataset.csv", "Same schema, BLIP-derived"],
            ["Upload metadata", "results/uploads/{upload_id}.json", "One JSON file per uploaded image"],
            ["Model artifacts", "models/*.joblib", "Serialized trained classifiers"],
            ["Metrics", "results/metrics/training_metrics.json", "Training run results"],
        ],
    )
    add_heading(document, "Why this design, not a DB", level=2)
    add_bullets(
        document,
        [
            "It's a research pipeline, not a multi-user transactional app "
            "- there's no concurrent writes, no need for ACID guarantees.",
            "CSV keeps the dataset portable and diffable - open it in "
            "Excel, pandas, or Git and see exactly what changed.",
            "Avoids infrastructure overhead (no DB server) for something "
            "that is fundamentally a batch ML pipeline.",
        ],
    )
    add_heading(document, "Conceptual ER model (if formalized as tables)", level=2)
    add_code(
        document,
        """
IMAGE (image_id PK, file_name, width, height)
   |\\
   | \\ 1-to-many
   |  \\
   |   CAPTION (caption_id PK, image_id FK, text, source[human|blip])
   |
   | 1-to-1
   v
FEATURE_ROW (image_id PK/FK, average_similarity, minimum_similarity,
             maximum_similarity, std_similarity, caption_diversity,
             edge_density, entropy, brightness, contrast,
             color_variance, texture)
   |
   | 1-to-1
   v
AMBIGUITY_LABEL (image_id PK/FK, label[Low|Medium|High])

UPLOAD (upload_id PK, image_id FK nullable, filename, created_at)
""",
    )
    add_body(
        document,
        "Primary key: image_id (COCO's native integer ID), used as the "
        "join key everywhere. caption_diversity is the single most "
        "important column - it is both a feature and the source of the "
        "label (discussed deliberately as a limitation in Section 12/18).",
    )

    # 7 ------------------------------------------------------------------
    add_heading(document, "7. Folder Structure")
    add_code(
        document,
        """
Image Ambiguity Prediction/
|-- backend/app/
|   |-- main.py            FastAPI app, CORS, /health, / root
|   |-- schemas.py         Pydantic request/response models
|   |-- api/routes.py      /upload /features /predict /explain /compare
|   `-- services/
|       |-- inference.py   feature extraction + prediction + SHAP
|       `-- compare.py     human vs AI diversity comparison
|-- src/
|   |-- image_ambiguity/            installable core ML package
|   |   |-- config.py               Settings (env > .env > YAML > defaults)
|   |   |-- data/coco_loader.py     COCO caption/image loading
|   |   |-- features/               sentence_embeddings, caption_diversity,
|   |   |                           cv_features, blip_captions
|   |   |-- pipeline/               dataset_builder, label_generator,
|   |   |                           ai_captions
|   |   |-- models/trainer.py       train/tune/evaluate RF + XGBoost
|   |   `-- explainability/         SHAP wrapper
|   `-- create_dataset.py, generate_labels.py, train.py,
|       enrich_high_diversity.py, create_ai_dataset.py   CLI scripts
|-- frontend/src/                    React + Vite + TS UI ("Ambiguity Lens")
|   |-- api/client.ts               fetch wrappers for the backend API
|   |-- pages/                      Home, Prediction, Comparison, About
|   `-- types.ts                    shared TS types mirroring backend schemas
|-- configs/default.yaml            default runtime configuration
|-- dataset/                        COCO images/annotations + generated CSVs
|-- models/                         trained model artifacts (.joblib)
|-- results/                        uploads, metrics, captions cache, figures
|-- notebooks/                      exploratory research notebooks
|-- tests/unit, tests/integration, tests/fixtures
|-- scripts/                        one-off doc/figure generators
|-- myDocs/                         generated documentation
|-- app.py                          entrypoint: uvicorn app:app
`-- requirements.txt, pyproject.toml, pytest.ini
""",
    )
    add_body(
        document,
        "Summary line: it's a layered architecture - a pure-Python ML "
        "core package (src/image_ambiguity) that knows nothing about "
        "HTTP, a thin FastAPI layer (backend/) that wraps it for the web, "
        "and a separate React frontend (frontend/) that only talks to "
        "the API - so the UI or the web framework could be swapped "
        "without touching the ML logic.",
    )

    # 8 ------------------------------------------------------------------
    add_heading(document, "8. Request Flow: 'Predict & Explain' Feature")
    add_body(document, "This is the core feature: upload an image, get an explained ambiguity prediction.")
    add_numbered(
        document,
        [
            "User clicks Predict after choosing a file.",
            "Frontend validation - checks a file is selected, and if "
            "manual captions are used, that there are >= 2 (diversity "
            "needs at least a pair).",
            "Upload request - image bytes go to POST /upload; the "
            "backend saves the file, detects if the filename matches a "
            "COCO id pattern, and pre-attaches human captions if so.",
            "Explain request - POST /explain with the upload_id. The "
            "service picks captions in priority order: user-typed -> "
            "COCO human (if available) -> BLIP-generated.",
            "Feature extraction - captions go through Sentence-BERT -> "
            "diversity score; the image goes through OpenCV -> 6 visual "
            "features.",
            "Prediction - the 11-feature vector is fed into the loaded "
            "Random Forest/XGBoost model, returning class probabilities.",
            "Explanation - SHAP computes per-feature contribution to that "
            "specific prediction.",
            "Response & UI update - the frontend renders the label with "
            "a color, a confidence bar, class-probability chart, and a "
            "SHAP contribution chart.",
        ],
    )
    add_code(
        document,
        """
User -> UI: selects image file
UI -> UI: client-side validation
UI -> API: POST /upload (multipart file)
API -> Service: save_upload(filename, bytes)
Service -> Service: validate extension, write to disk
Service -> COCO lookup: parse_coco_image_id(filename)
COCO lookup --> Service: captions (if matched)
Service --> API: upload_id, coco_captions
API --> UI: 201 Created

UI -> API: POST /explain (upload_id, captions?, force_blip?)
API -> Service: resolve_captions(...)
   [user captions] OR [cached COCO human captions] OR [BLIP generate_all()]
Service -> Sentence-BERT: embed captions
Service -> Service: pairwise cosine similarity -> caption_diversity
Service -> OpenCV: extract(image) -> 6 visual features
Service -> Model: predict_proba(11-feature vector)
Service -> SHAP: compute_shap_values(feature_row)
Service --> API: label, confidence, probabilities, shap_explanation
API --> UI: 200 OK
UI -> UI: render label, confidence, charts, SHAP chart, caption source
""",
    )

    # 9 ------------------------------------------------------------------
    add_heading(document, "9. Core Features")
    features = [
        (
            "Caption Diversity Scoring",
            "Quantify how much captions disagree.",
            "Sentence-BERT embeddings -> pairwise cosine similarity -> "
            "1 - average_similarity.",
            "sentence-transformers, PyTorch.",
            "Deciding which similarity statistic to use (avg vs min vs "
            "max) - settled on average, with min/max/std stored as extra "
            "features.",
            "Use a learned/weighted diversity metric instead of a simple "
            "formula.",
        ),
        (
            "OpenCV Visual Feature Extraction",
            "Give the model visual context beyond language.",
            "Edge density (Canny), entropy, brightness, contrast, color "
            "variance, texture energy.",
            "OpenCV.",
            "Keeping features scale-invariant across different image "
            "resolutions.",
            "Add deep visual embeddings (e.g. CLIP image embeddings) "
            "alongside classical features.",
        ),
        (
            "BLIP AI Caption Generation",
            "Generate AI captions to compare against human captions.",
            "Beam search + top-k + nucleus sampling + prompt-conditioned "
            "generation, deduplicated.",
            "Hugging Face transformers, BLIP model.",
            "BLIP tends to generate near-duplicate captions (low "
            "diversity) even for genuinely ambiguous images - a real bug "
            "I diagnosed and fixed (see Sections 12/24).",
            "Try larger BLIP variants or multiple captioning models to "
            "increase caption diversity naturally.",
        ),
        (
            "Rule-Based Ambiguity Labeling",
            "Turn a continuous diversity score into a categorical label.",
            "0.00-0.35 Low, 0.35-0.65 Medium, 0.65-1.00 High.",
            "Pure Python rule in label_generator.py.",
            "Choosing thresholds that are meaningful, and keeping them "
            "fixed as a research decision even when the High class was "
            "rare.",
            "Learn thresholds from human ambiguity ratings instead of "
            "fixed cutoffs.",
        ),
        (
            "Classification (Random Forest / XGBoost)",
            "Predict the label from the 11-feature vector for new "
            "images.",
            "Stratified 80/20 split, grid-search CV, train-only "
            "oversampling, class_weight=balanced for RF.",
            "scikit-learn, XGBoost.",
            "Severe class imbalance (only 4 High samples originally) - "
            "fixed by mining more high-diversity COCO images and "
            "oversampling.",
            "Try calibrated probabilities and cost-sensitive learning for "
            "the rare High class.",
        ),
        (
            "SHAP Explainability",
            "Make every prediction explainable, not a black box.",
            "TreeExplainer computes per-feature SHAP values per "
            "prediction; top contributions rendered as a chart.",
            "SHAP.",
            "Making SHAP output digestible for non-technical UI users - "
            "solved with a plain-English summary string + bar chart.",
            "Add SHAP waterfall/force plots for richer visual "
            "explanation.",
        ),
        (
            "Human vs AI Diversity Comparison",
            "Show that AI captions and human captions produce "
            "systematically different diversity distributions.",
            "/compare endpoint reads both CSVs and returns mean diversity "
            "+ histograms.",
            "pandas, Recharts.",
            "Small/unpaired AI sample size limits statistical strength - "
            "called out explicitly rather than overstated.",
            "Build a full paired dataset (same images, both caption "
            "sources) at COCO scale.",
        ),
        (
            "Dataset Enrichment & Class Balancing",
            "Fix the rare High class without changing the definition of "
            "High.",
            "enrich_high_diversity.py mines unused COCO images >= 0.65 "
            "diversity; trainer oversamples minority classes in the "
            "training split only.",
            "pandas, scikit-learn resample.",
            "Balancing without leaking synthetic/duplicated rows into the "
            "test set.",
            "Collect genuinely new human-annotated high-ambiguity images "
            "instead of oversampling existing ones.",
        ),
    ]
    for name, purpose, impl, tech, challenge, improvement in features:
        add_heading(document, name, level=2)
        add_bullets(
            document,
            [
                f"Purpose: {purpose}",
                f"Implementation: {impl}",
                f"Technologies used: {tech}",
                f"Challenge: {challenge}",
                f"Possible improvement: {improvement}",
            ],
        )

    # 10 -----------------------------------------------------------------
    add_heading(document, "10. Authentication")
    add_body(
        document,
        "Honest answer: there is no authentication in this project, and "
        "that's a design choice, not an oversight - this is a public, "
        "stateless research/demo API with no user accounts and no "
        "private data to protect.",
    )
    add_quote(
        document,
        "\"I deliberately didn't add authentication because the "
        "project's scope is a public research demo. If I were to turn "
        "this into a real product, I would add registration/login "
        "(email + password hashed with bcrypt/argon2), JWT access + "
        "refresh tokens, role-based authorization, and route protection "
        "via a FastAPI dependency (Depends(get_current_user)) - the same "
        "dependency-injection pattern already used for "
        "Depends(get_service).\"",
    )

    # 11 -----------------------------------------------------------------
    add_heading(document, "11. API Design")
    add_body(document, "Base URL: http://127.0.0.1:8000 (Swagger docs at /docs).")
    api_rows = [
        ["GET /health", "Liveness probe", "-", "{status, environment}"],
        ["GET /", "Root metadata + endpoint index", "-", "project, docs, health, endpoints[]"],
        ["POST /upload", "Store an uploaded image", "multipart file (jpg/png/webp/bmp)", "upload_id, size, dims, coco_image_id?, coco_captions?"],
        ["POST /features", "Extract diversity + CV features", "file or upload_id, captions?, force_blip?", "features, caption_source"],
        ["POST /predict", "Predict ambiguity label", "same as /features", "predicted_ambiguity, confidence, probabilities"],
        ["POST /explain", "Predict + SHAP explanation", "same as /predict, top_n", "prediction + shap_explanation + summary"],
        ["GET /compare", "Human vs AI diversity comparison", "-", "mean diversity, counts, histograms per source"],
    ]
    add_table(document, ["Endpoint", "Purpose", "Request", "Response"], api_rows)
    add_heading(document, "Validation & error handling", level=2)
    add_bullets(
        document,
        [
            "Every response is typed with Pydantic models (schemas.py); "
            "FastAPI validates and documents them automatically.",
            "Empty files, unsupported extensions -> 400.",
            "Missing upload/model file -> 404 / clear FileNotFoundError.",
            "All domain errors are converted to HTTP errors in one "
            "central place (_http_error), instead of scattering "
            "try/except HTTPException everywhere.",
            "Swagger's literal 'string' placeholder is sanitized so it "
            "isn't parsed as a real caption or upload_id.",
        ],
    )

    # 12 -----------------------------------------------------------------
    add_heading(document, "12. Challenges")
    challenges = [
        (
            "Technical: BLIP captions vs human captions producing "
            "different labels",
            "A COCO image labeled High (human diversity ~0.80) scored "
            "Medium (~0.40) when BLIP generated near-duplicate captions "
            "for the same image.",
            "Backend auto-detects COCO filenames and prefers cached human "
            "captions unless force_blip is set; every response now "
            "exposes caption_source; BLIP decoding diversified (higher "
            "temperature, more return sequences, multiple prompts).",
        ),
        (
            "Design: Class imbalance in the High ambiguity label",
            "Only 4 of ~300 images naturally fell into High; models "
            "couldn't learn that class well.",
            "Kept the 0.65 threshold unchanged (research correctness), "
            "mined more genuinely high-diversity COCO images (4 -> 34), "
            "added train-split-only oversampling + class_weight=balanced.",
        ),
        (
            "Performance/Correctness: Missing XGBoost causing 500 errors",
            "API tried to load an XGBoost model but the package wasn't "
            "installed, causing an unhandled crash.",
            "Fallback chain: best_model.joblib -> random_forest.joblib -> "
            "xgboost.joblib, with individual load failures logged instead "
            "of crashing on the first one.",
        ),
        (
            "API/UX: Swagger's default 'string' placeholder",
            "FastAPI Swagger pre-fills optional fields with the literal "
            "text 'string', causing 422 errors when treated as real "
            "input.",
            "Added explicit sanitization that treats 'string', 'null', "
            "empty strings as 'not provided'.",
        ),
        (
            "Security-adjacent: file handling",
            "Accepting arbitrary uploaded files is a risk surface.",
            "Extension whitelist, server-generated filenames "
            "(uuid4().hex, never user input), Pillow validates the file "
            "actually opens as an image before accepting it.",
        ),
    ]
    for title, problem, solution in challenges:
        add_heading(document, title, level=2)
        add_bullets(document, [f"Problem: {problem}", f"Solution: {solution}"])

    # 13 -----------------------------------------------------------------
    add_heading(document, "13. Performance Optimizations")
    add_bullets(
        document,
        [
            "Model loaded once, not per-request - cached via lru_cache on "
            "the service singleton.",
            "Batch embedding generation - Sentence-BERT embeds all "
            "captions for an image in one batched call.",
            "BLIP caption caching - results/captions/blip_caption_cache.json "
            "avoids regenerating captions across runs.",
            "CPU/GPU auto-detection - resolve_device('auto') picks CUDA "
            "if available, else CPU.",
            "Frontend code splitting - Vite + React Router naturally "
            "splits bundles per route.",
            "Avoiding recompute - upload metadata is written once and "
            "reused across /features, /predict, /explain for the same "
            "upload_id.",
            "Honest gap: no HTTP response caching, no CDN, no GPU batching "
            "across concurrent requests yet.",
        ],
    )

    # 14 -----------------------------------------------------------------
    add_heading(document, "14. Security")
    add_table(
        document,
        ["Concern", "Status", "Explanation"],
        [
            ["Authentication", "Not implemented", "Public demo/research API by design"],
            ["Authorization", "Not implemented", "No per-user resources to protect"],
            ["Password hashing", "N/A", "No passwords exist"],
            ["Input validation", "Implemented", "Pydantic schemas + custom sanitizers for captions/IDs"],
            ["File upload validation", "Implemented", "Extension whitelist, Pillow decode check, server-generated filenames"],
            ["SQL Injection", "N/A", "No SQL database exists"],
            ["XSS", "Mitigated by default", "React escapes rendered text by default; no dangerouslySetInnerHTML"],
            ["CSRF", "Low risk currently", "No cookies/sessions used for auth yet; would matter once sessions are added"],
            ["CORS", "Implemented", "Explicit cors_origins allow-list, not a wildcard with credentials"],
            ["Rate limiting", "Not implemented", "Would add via slowapi or a reverse proxy in production"],
            ["Secrets management", "Implemented", ".env file (gitignored) + pydantic-settings, never hardcoded"],
        ],
    )

    # 15 -----------------------------------------------------------------
    add_heading(document, "15. Testing")
    add_bullets(
        document,
        [
            "Structure: tests/unit/, tests/integration/, "
            "tests/fixtures/images/ - pytest + pytest-cov via pytest.ini.",
            "Unit tests: diversity score math, label threshold "
            "boundaries, config loading precedence.",
            "Integration tests: end-to-end feature-row building from a "
            "fixture image + captions.",
            "Manual/API testing: FastAPI's Swagger UI (/docs) plus manual "
            "UI testing in the React app.",
            "Edge cases covered: <2 captions (400), missing/corrupted "
            "image (404/400), missing trained model (clear "
            "FileNotFoundError), Swagger 'string' placeholders "
            "normalized.",
            "Gap: no automated end-to-end (frontend + backend) tests yet "
            "(e.g. Playwright/Cypress) - currently manually verified.",
        ],
    )

    # 16 -----------------------------------------------------------------
    add_heading(document, "16. Deployment")
    add_body(document, "Current state (honest): runs locally only - no cloud deployment yet.")
    add_bullets(
        document,
        [
            "Backend: uvicorn app:app --reload locally, with PYTHONPATH "
            "including src.",
            "Frontend: npm run dev (Vite) proxying /api to the backend; "
            "npm run build produces a static dist/ bundle.",
            "Environment variables: .env (gitignored) + .env.example "
            "documents IAP_* settings.",
            "Build process: pip install -r requirements.txt && pip "
            "install -e . ; npm install && npm run build.",
        ],
    )
    add_heading(document, "If deployed for real", level=2)
    add_numbered(
        document,
        [
            "Containerize backend with Docker (multi-stage build).",
            "Host the static frontend build on Vercel/Netlify/S3+CloudFront.",
            "Host the FastAPI container on Render/Railway/AWS ECS/Fly.io.",
            "Store model artifacts in S3, downloaded at container startup.",
            "Add a GitHub Actions CI pipeline: pytest + ruff on PRs, build "
            "and push Docker image on merge to main.",
            "No CI/CD exists yet - that is an honest current gap.",
        ],
    )

    # 17 -----------------------------------------------------------------
    add_heading(document, "17. Future Improvements")
    add_table(
        document,
        ["Improvement", "Why useful", "How I'd implement it"],
        [
            ["Independent human ambiguity ratings", "Removes label-leakage (diversity both defines and predicts the label)", "Small annotation study rating images 1-5, compare against diversity-derived label"],
            ["Larger paired human/AI dataset", "Statistically stronger human-vs-AI comparison", "Run BLIP over the full COCO val2017 set, not just a sample"],
            ["Dockerize + CI/CD", "Reproducible deploys, catch regressions automatically", "Dockerfile + GitHub Actions running pytest/ruff on every PR"],
            ["Real database for uploads/history", "Enables history, analytics, multi-user support", "Swap flat JSON files for PostgreSQL + SQLAlchemy"],
            ["Authentication", "Needed for a multi-tenant product", "JWT auth via OAuth2PasswordBearer + Depends()-injected current user"],
            ["Deep visual embeddings", "Richer visual signal beyond classical features", "Add CLIP image embeddings as extra model features"],
            ["Calibrated probabilities", "Confidence scores would be trustworthy, not leakage-inflated", "Platt scaling / isotonic regression on an independently-labeled set"],
            ["Rate limiting", "Protect a public deployment from abuse", "slowapi middleware or an API gateway"],
            ["Multilingual captions", "Broader applicability", "Multilingual Sentence-BERT + multilingual captioning model"],
        ],
    )

    # 18 -----------------------------------------------------------------
    add_heading(document, "18. Learnings")
    add_heading(document, "Technical concepts learned", level=2)
    add_bullets(
        document,
        [
            "How Sentence-BERT embeddings + cosine similarity turn "
            "unstructured text into a single interpretable numeric "
            "signal.",
            "How SHAP attributes predictions to features using Shapley "
            "values, wired into a live API instead of just a notebook.",
            "FastAPI's dependency-injection pattern (Depends()) for "
            "clean, testable service singletons.",
            "Why train-only oversampling matters - oversampling before "
            "the split leaks duplicated rows into the test set and "
            "inflates metrics.",
            "Layered configuration design (env > .env > YAML > code "
            "defaults) using pydantic-settings.",
        ],
    )
    add_heading(document, "Mistakes made (owned, not hidden)", level=2)
    add_bullets(
        document,
        [
            "Label leakage: caption_diversity is literally the formula "
            "used to create the label, and it is also a feature fed into "
            "the model - near-100% accuracy mostly proves the pipeline is "
            "internally consistent, not that the model learned something "
            "an independent human judge would agree with. Now explicitly "
            "called out in the results discussion.",
            "Not committing to Git often enough - working files got "
            "deleted at points and had to be reconstructed. Lesson: "
            "commit early and often, especially before destructive "
            "cleanup.",
        ],
    )
    add_body(
        document,
        "What I'd do differently: commit after every meaningful change; "
        "keep an independent, leakage-free validation set from day one "
        "instead of retrofitting the caveat later.",
    )

    # 19 -----------------------------------------------------------------
    add_heading(document, "19. Interview Questions (30, with ideal answers)")

    add_heading(document, "Beginner", level=2)
    beginner_qa = [
        ("What does your project do, in one sentence?",
         "It predicts how much captions disagree about an image (Low/"
         "Medium/High ambiguity) and explains why, using caption "
         "embeddings, visual features, and SHAP."),
        ("What is Sentence-BERT and why did you use it?",
         "A model that turns sentences into fixed-length vectors such "
         "that semantically similar sentences have similar vectors (via "
         "cosine similarity). Raw word overlap (like BLEU) doesn't "
         "capture meaning-level agreement between captions, but "
         "embedding similarity does."),
        ("What's the difference between your model's features and the "
         "label?",
         "Features are the 11 numeric inputs (4 similarity stats + "
         "diversity + 6 OpenCV values). The label is the categorical "
         "output (Low/Medium/High) derived from thresholding the "
         "caption_diversity feature."),
        ("Why Random Forest and XGBoost instead of a neural network?",
         "With only ~300 labeled rows and 11 tabular features, tree "
         "ensembles are a stronger, more interpretable baseline than a "
         "neural net, which would need far more data to avoid "
         "overfitting."),
        ("What is SHAP and why did you use it instead of just feature "
         "importance?",
         "SHAP explains individual predictions, not just global feature "
         "importance across the whole model. It's based on Shapley "
         "values from cooperative game theory, so contributions are "
         "theoretically fair and consistent."),
        ("What database do you use?",
         "None - flat CSV files for training data and joblib files for "
         "model artifacts, since this is a single-writer research "
         "pipeline, not a concurrent multi-user app."),
        ("How does the frontend talk to the backend?",
         "The React app makes fetch calls to REST endpoints (/upload, "
         "/explain, etc.), proxied through /api in development via "
         "Vite's proxy config."),
        ("What's OpenCV used for here?",
         "Extracting classical visual descriptors - edge density, "
         "entropy, brightness, contrast, color variance, texture - that "
         "give the model visual context beyond just caption text."),
        ("What does 'diversity' mean numerically in your project?",
         "1 - average_similarity, where average_similarity is the mean "
         "pairwise cosine similarity between all caption embeddings for "
         "one image. Higher value = captions disagree more."),
        ("Why is there no login/authentication?",
         "It's a stateless public demo with no private user data - auth "
         "wasn't a scope requirement. I know how I'd add it (JWT + "
         "Depends()-based route protection) if needed."),
    ]
    for q, a in beginner_qa:
        add_qa(document, q, a)

    add_heading(document, "Intermediate", level=2)
    intermediate_qa = [
        ("How did you handle class imbalance?",
         "Two-pronged: mined additional naturally high-diversity COCO "
         "images to raise the rare High class count, and oversampled "
         "minority classes only in the training split (never the test "
         "set) plus used class_weight=balanced for Random Forest."),
        ("Why is oversampling only applied to the training split "
         "important?",
         "If you oversample before splitting, duplicated rows can end "
         "up in both train and test, letting the model memorize test "
         "rows it already saw in training - inflating reported accuracy "
         "in a way that doesn't reflect real generalization."),
        ("What's the risk of using caption_diversity as both a label "
         "source and a model feature?",
         "Label leakage - the model can trivially learn to threshold "
         "that one feature and get near-perfect accuracy, which mostly "
         "validates pipeline consistency rather than proving the model "
         "learned a deeper, independent notion of ambiguity."),
        ("How do you decide which captions to use - human, AI, or "
         "user-provided?",
         "Priority order: explicit user-provided captions (>=2) -> "
         "cached human COCO captions (if filename matches and BLIP isn't "
         "forced) -> BLIP-generated captions as the fallback. The API "
         "response exposes which source was used."),
        ("Why did BLIP sometimes give a different ambiguity label than "
         "human captions for the same image?",
         "BLIP tends to generate near-paraphrases of the same caption "
         "repeatedly, so its captions agree with each other more than "
         "diverse human annotators do - lowering the diversity score for "
         "the same image, purely because of the caption source."),
        ("How does your API handle a missing trained model file "
         "gracefully?",
         "It tries a fallback chain of model files (best_model -> "
         "random_forest -> xgboost), logs each failure, and raises a "
         "clear error telling the user to run the training script or "
         "install a missing dependency, instead of an unhandled 500."),
        ("How would you scale this to handle many concurrent prediction "
         "requests?",
         "Run multiple Uvicorn workers behind a load balancer, ensure "
         "the model is loaded once per worker (already true via the "
         "singleton pattern), and consider batch inference if GPU "
         "throughput becomes the bottleneck."),
        ("What's the difference between /predict and /explain?",
         "/predict returns just the label, confidence, and features. "
         "/explain does everything /predict does plus computes SHAP "
         "values and a human-readable summary - strictly more "
         "expensive, so I split them."),
        ("How do you validate uploaded files are actually images?",
         "Whitelist allowed extensions, then attempt to open the file "
         "with Pillow - if it's not a valid image, that throws and gets "
         "converted into a 400 error instead of crashing downstream "
         "feature extraction."),
        ("What would you change about your labeling thresholds?",
         "I'd want to validate 0.35/0.65 against real human ambiguity "
         "judgments rather than picking them by inspection - right now "
         "they're a reasonable but somewhat arbitrary research decision."),
    ]
    for q, a in intermediate_qa:
        add_qa(document, q, a)

    add_heading(document, "Advanced", level=2)
    advanced_qa = [
        ("Walk me through what happens end-to-end when SHAP explains a "
         "Random Forest prediction.",
         "SHAP's TreeExplainer walks each decision tree and computes, "
         "for each feature, its marginal contribution to moving the "
         "prediction away from a baseline value, in a way that satisfies "
         "fairness properties (efficiency, symmetry, additivity) from "
         "cooperative game theory. For multi-class RF you get one SHAP "
         "value per feature per class."),
        ("Why is class_weight='balanced' not sufficient alone to fix "
         "imbalance, and why add oversampling too?",
         "class_weight='balanced' reweights the loss so minority-class "
         "errors count more, but doesn't create new information - the "
         "model still sees only a handful of true High examples. "
         "Oversampling gives the minority class real influence on tree "
         "splits themselves, not just loss weighting."),
        ("What's a subtle failure mode in using cosine similarity "
         "between Sentence-BERT embeddings as your only diversity "
         "signal?",
         "It can't distinguish 'captions disagree about content' from "
         "'captions describe genuinely different but complementary "
         "aspects of the same scene' - both would lower similarity even "
         "without semantic contradiction."),
        ("How would you redesign this system for 100,000 images per "
         "day?",
         "Move from synchronous request/response to an async job queue "
         "(Celery/RQ + Redis) - accept the upload, return a job ID, "
         "process BLIP/embedding/prediction in background GPU-backed "
         "workers, and let the client poll or use websockets for the "
         "result. Also move storage to S3 and CSVs to a proper data "
         "warehouse."),
        ("Your test accuracy is 100%. Is that a good result?",
         "Not on its own. Because the label is a deterministic function "
         "of one of the input features, the classifier can reach "
         "near-ceiling accuracy by re-deriving the threshold rule, not "
         "by learning generalizable ambiguity. I present it transparently "
         "alongside that caveat."),
        ("How do you keep the ML core package decoupled from the "
         "FastAPI layer?",
         "src/image_ambiguity has zero imports from backend/ or FastAPI "
         "- it's a plain Python library. backend/app/services/"
         "inference.py is the only adapter that imports both, so the ML "
         "core could be reused in a CLI script or a different web "
         "framework unchanged."),
        ("What would an ablation study look like to validate whether "
         "OpenCV features actually help?",
         "Train two models - one with only linguistic features, one with "
         "all 11 - and compare macro-F1 on an independent, non-leaky "
         "validation set (ideally human-rated ambiguity, not "
         "diversity-derived labels) to isolate the visual features' "
         "contribution."),
        ("How does your layered configuration system resolve conflicts "
         "between sources?",
         "Priority, highest wins: process env vars (IAP_*) > .env file > "
         "configs/default.yaml > hardcoded Pydantic defaults - "
         "implemented via pydantic-settings' settings_customise_sources "
         "hook to insert a custom YAML source into the resolution "
         "chain."),
        ("Is there any risk of shared state causing incorrect results "
         "under concurrent requests?",
         "The inference service is stateless per-request except for the "
         "loaded model and embedding model, which are read-only after "
         "loading - concurrent requests are safe since PyTorch/"
         "scikit-learn predict calls are thread-safe for inference."),
        ("How would you detect model/data drift in production for this "
         "system?",
         "Log the distribution of caption_diversity and predicted-label "
         "proportions over time; if predicted-label proportions shift "
         "significantly from the training distribution, or human "
         "feedback disagrees at a rising rate, that signals drift "
         "requiring retraining or threshold re-calibration."),
    ]
    for q, a in advanced_qa:
        add_qa(document, q, a)

    # 20 -----------------------------------------------------------------
    add_heading(document, "20. HR Questions (with sample answers)")
    hr_qa = [
        ("Why did you build this?",
         "\"I was interested in the gap between how captioning models "
         "are evaluated (BLEU/CIDEr-style similarity to references) and "
         "a question nobody was directly measuring: how much do the "
         "reference captions themselves disagree? I wanted to build a "
         "full pipeline - from raw COCO data to an explainable, deployed "
         "prediction - to prove I could take a research idea and make it "
         "usable, not just leave it in a notebook.\""),
        ("What was your biggest challenge?",
         "\"Realizing partway through that my label and one of my "
         "model's features were mathematically the same thing - "
         "caption_diversity defines the label and is fed into the "
         "classifier. It would have been easy to just report the "
         "near-100% accuracy and move on, but I made myself dig into why "
         "that number was so high, document the leakage honestly, and "
         "design around it instead of hiding it.\""),
        ("What would you improve?",
         "\"Two things: get real, independent human ambiguity ratings so "
         "my evaluation isn't self-referential, and containerize + add "
         "CI/CD so the whole thing is one command to deploy instead of "
         "manual setup.\""),
        ("What was your role?",
         "\"I designed and built the entire pipeline solo - the ML "
         "feature engineering, model training, the FastAPI backend, and "
         "the React frontend - plus wrote the documentation and a full "
         "IEEE-format research paper summarizing the work.\""),
        ("What did you learn?",
         "\"Beyond the technical skills - Sentence-BERT, SHAP, FastAPI's "
         "dependency injection - the biggest lesson was about "
         "intellectual honesty in ML: it's tempting to report a great "
         "accuracy number, but understanding why a metric looks good "
         "(and being upfront about label leakage) is what actually makes "
         "the work trustworthy.\""),
        ("How do you handle disagreement in a team, based on this "
         "project?",
         "\"Even working solo, I 'disagreed with myself' - I initially "
         "wanted to lower the ambiguity threshold to get more High "
         "examples, but stepped back and decided the research-correct "
         "move was to keep the definition strict and fix the data "
         "instead. I'd apply the same instinct in a team: don't "
         "compromise on correctness for a convenient short-term "
         "metric.\""),
    ]
    for q, a in hr_qa:
        add_qa(document, q, a)

    # 21 -----------------------------------------------------------------
    add_heading(document, "21. 2-Minute Explanation")
    add_quote(
        document,
        "\"My project predicts how ambiguous an image is - meaning, how "
        "much different captions disagree about what's in it - and "
        "explains that prediction. I use Sentence-BERT to embed multiple "
        "captions per image and compute a diversity score from their "
        "cosine similarity, add classical OpenCV visual features like "
        "edge density and entropy, and feed all of that into a Random "
        "Forest / XGBoost classifier that outputs Low, Medium, or High "
        "ambiguity. Every prediction is explained with SHAP, so you can "
        "see exactly which feature drove the decision. I built this as a "
        "full pipeline: a training side that builds the dataset from "
        "COCO images and captions, and a live side - a FastAPI backend "
        "plus a React frontend - where you can upload any image, get an "
        "AI-generated or human caption set, and see the prediction and "
        "explanation instantly. One interesting finding: I can also "
        "compare human-written captions against AI-generated ones (using "
        "BLIP) under the exact same diversity metric, and I found AI "
        "captions often paraphrase themselves rather than genuinely "
        "disagreeing, which lowers their measured diversity even for "
        "images humans found highly ambiguous.\"",
    )

    # 22 -----------------------------------------------------------------
    add_heading(document, "22. 5-Minute Explanation")
    add_quote(
        document,
        "\"The motivation was that image captioning research measures "
        "caption quality against references, but nobody really measures "
        "caption disagreement - how much references contradict each "
        "other - even though that's a strong signal about how genuinely "
        "ambiguous an image is.\n\n"
        "So I built a pipeline in three layers. First, the feature "
        "layer: for each image, I take its captions - either "
        "human-written from Microsoft COCO, or AI-generated with BLIP - "
        "and embed them with Sentence-BERT. I compute pairwise cosine "
        "similarity across all caption pairs and derive a "
        "caption_diversity score as one minus the average similarity. I "
        "also extract six classical OpenCV features from the image "
        "itself - edge density, entropy, brightness, contrast, color "
        "variance, texture - so the model has visual context, not just "
        "language.\n\n"
        "Second, the labeling and modeling layer: I apply a fixed "
        "threshold rule (below 0.35 is Low, 0.35 to 0.65 is Medium, "
        "above 0.65 is High) to turn the diversity score into a "
        "categorical label, and I train Random Forest and XGBoost "
        "classifiers on the 11-feature vector to predict that label. I "
        "ran into a real class-imbalance problem here - only 4 out of "
        "roughly 300 images naturally fell into 'High' - so I mined more "
        "high-diversity COCO images to enrich that class and used "
        "train-split-only oversampling plus balanced class weights, "
        "being careful never to leak duplicated rows into the test "
        "set.\n\n"
        "Third, the serving layer: a FastAPI backend exposes /upload, "
        "/predict, /explain, and /compare endpoints. /explain also runs "
        "SHAP to attribute the prediction to specific features, and I "
        "built a React frontend called 'Ambiguity Lens' so you can "
        "upload an image, choose or generate captions, and see the "
        "label, confidence, and SHAP explanation live.\n\n"
        "One of the more interesting engineering problems I solved: "
        "BLIP-generated AI captions for the same image that COCO labels "
        "as 'High' ambiguity often score 'Medium,' because BLIP tends to "
        "produce near-duplicate paraphrases rather than genuinely "
        "different interpretations. So I made the backend "
        "caption-source-aware - it auto-detects known COCO images and "
        "prefers cached human captions unless you explicitly force BLIP "
        "- and I expose which caption source was used in every API "
        "response, so results are never misleading about why a label "
        "came out the way it did.\"",
    )

    # 23 -----------------------------------------------------------------
    add_heading(document, "23. 10-Minute Deep Dive (for a senior engineer)")
    add_quote(
        document,
        "\"I'll walk through this from architecture down to a specific "
        "hard problem I solved.\n\n"
        "Architecture: there are two clearly separated halves. An "
        "offline training pipeline - pure Python, no web framework "
        "dependency - that goes from raw COCO images and captions "
        "through feature engineering to a saved model artifact. And an "
        "online serving system: a FastAPI backend and a React/TypeScript "
        "frontend. The critical design decision is that my ML core "
        "package, image_ambiguity, imports nothing from FastAPI or the "
        "web layer. backend/app/services/inference.py is the only "
        "adapter that imports both sides - so I can reuse the exact same "
        "feature extraction and model logic in a CLI script, a Jupyter "
        "notebook, or a completely different web framework without "
        "touching the ML code.\n\n"
        "Feature engineering: each image gets an 11-dimensional feature "
        "vector - four Sentence-BERT-derived similarity statistics plus "
        "a diversity score, and six OpenCV visual descriptors. The "
        "diversity score is 1 minus average_similarity across all "
        "caption pairs for that image, computed with Sentence-BERT "
        "embeddings and cosine similarity.\n\n"
        "The labeling problem: I use a fixed threshold rule to convert "
        "that diversity score into a categorical Low/Medium/High label. "
        "This is a deliberate research decision - I chose to keep the "
        "threshold fixed even when it created a severe class imbalance "
        "(early on, only 4 of ~300 images were 'High'), because lowering "
        "the threshold to get more High examples would have diluted what "
        "'High' actually means. Instead, I fixed the data: I wrote a "
        "script that scans unused COCO validation images, computes "
        "diversity for each, and appends genuinely high-diversity ones - "
        "bringing High from 4 to 34 samples - combined with train-split-"
        "only oversampling in my ModelTrainer class. I want to stress "
        "'train-split-only' because the more common mistake is "
        "oversampling before the split, which duplicates rows into both "
        "train and test and inflates your reported metrics without "
        "improving real generalization. I explicitly split first, then "
        "oversample only X_train/y_train.\n\n"
        "A real bug I want to highlight, because it's a good example of "
        "debugging a subtle cross-layer issue: after fixing the class "
        "imbalance, users reported that a known High-ambiguity COCO "
        "image showed up as 'Medium' in the live demo. I traced it to "
        "the caption source: the UI defaulted to generating BLIP "
        "captions rather than using the dataset's human captions, and "
        "BLIP kept producing five near-paraphrases of the same caption "
        "for that image, which drove its measured diversity down to "
        "about 0.40, well under the 0.65 High threshold, even though the "
        "same image scored 0.80 with its original human captions. This "
        "wasn't a model bug at all - the model was working correctly on "
        "the features it was given; the problem was an inconsistent "
        "caption source between training data and live demo. I fixed it "
        "architecturally: the backend now detects COCO-style filenames, "
        "looks up and caches the original human captions, and uses them "
        "by default unless the caller explicitly passes force_blip=true. "
        "Every API response now also returns a caption_source field "
        "(user, coco_human, or blip), so the UI - and anyone reading the "
        "response - always knows exactly which caption source produced "
        "a given label.\n\n"
        "On evaluation honesty: my classifiers hit near-100% test "
        "accuracy. I don't present that uncritically - because "
        "caption_diversity is both the source of the label and a "
        "feature fed into the model, high accuracy mostly proves the "
        "classifier can recover a threshold rule, not that it has "
        "learned an independent notion of ambiguity that would "
        "generalize to human perceptual ratings. I call this out "
        "explicitly in my results discussion and frame it as the "
        "primary direction for future work.\n\n"
        "Explainability: every prediction goes through SHAP's "
        "TreeExplainer, giving per-feature, per-prediction attributions "
        "- which, unsurprisingly given the leakage point above, are "
        "dominated by caption_diversity and average_similarity. That "
        "SHAP output itself became useful evidence for diagnosing the "
        "leakage issue, since it made the label-feature relationship "
        "visually obvious rather than something I had to infer from "
        "metrics alone.\"",
    )

    # 24 -----------------------------------------------------------------
    add_heading(document, "24. STAR Format - Major Challenge")
    add_qa(
        document,
        "Situation:",
        "After I fixed the class-imbalance problem in my dataset and "
        "retrained the model, a user testing the live demo reported that "
        "a COCO image I knew was labeled 'High' ambiguity in my training "
        "data was being predicted as 'Medium' in the UI. This looked "
        "like a regression right after I'd just fixed something.",
    )
    add_qa(
        document,
        "Task:",
        "I needed to figure out whether this was a model bug, a data "
        "bug, or something else entirely - and fix it without breaking "
        "the research-correctness of my labeling thresholds, which I'd "
        "deliberately decided not to compromise on.",
    )
    add_qa(
        document,
        "Action:",
        "I traced the prediction pipeline step by step instead of "
        "assuming it was the model. I logged the actual captions being "
        "used for that specific upload and discovered the UI was "
        "defaulting to BLIP-generated captions rather than the dataset's "
        "original human COCO captions. BLIP had generated five "
        "near-identical paraphrases of essentially the same caption for "
        "that image, which drove the measured caption diversity down "
        "from ~0.80 (human captions) to ~0.40 (BLIP captions) - correctly "
        "landing it in 'Medium' for those specific captions. The model "
        "wasn't wrong at all; the input data had silently changed. I "
        "fixed this at the architecture level: I added logic to detect "
        "when an uploaded filename matches a known COCO image, "
        "automatically load and cache its original human captions, and "
        "use them by default unless the caller explicitly requests BLIP. "
        "I also added a caption_source field to every API response "
        "(user, coco_human, or blip) so this ambiguity in caption "
        "provenance could never again silently masquerade as a model "
        "problem. Separately, I improved BLIP's own decoding strategy "
        "(higher temperature, more return sequences, prompt-conditioned "
        "generation) to reduce near-duplicate captions when BLIP is "
        "intentionally used.",
    )
    add_qa(
        document,
        "Result:",
        "The 'High' COCO image now correctly predicts 'High' when using "
        "its human captions, and the UI/API are fully transparent about "
        "which caption source produced any given result - turning a "
        "confusing 'the model is broken' bug report into a "
        "well-understood, documented behavior (human vs. AI caption "
        "agreement), which actually became one of the more interesting "
        "findings I wrote up in my research paper's discussion section.",
    )

    # 25 -----------------------------------------------------------------
    add_heading(document, "25. Common Mistakes (and how to avoid them)")
    mistakes = [
        (
            "Overclaiming metrics without understanding why they're "
            "high.",
            "Candidates say '99% accuracy!' without realizing a feature "
            "and the label are correlated by construction. Fix: "
            "proactively explain any near-perfect metric and what it "
            "does/doesn't prove.",
        ),
        (
            "Inventing auth/DB details that don't exist in the project.",
            "If your project has no login system, don't describe a fake "
            "JWT flow - one follow-up question exposes it instantly. "
            "Fix: say what's real, then say what you'd add and why.",
        ),
        (
            "Confusing 'feature' with 'label' when explaining ML "
            "pipelines.",
            "Many candidates can't clearly separate what goes into the "
            "model from what the model predicts. Fix: always be ready to "
            "state the difference in one sentence.",
        ),
        (
            "Not knowing your own thresholds/numbers.",
            "If you say 'High ambiguity,' you should instantly know it "
            "means diversity >= 0.65 - vague answers signal you didn't "
            "internalize your own design decisions.",
        ),
        (
            "Treating gaps (no DB, no auth, no tests for X) as something "
            "to hide rather than explain.",
            "Interviewers ask about gaps on purpose to see how you "
            "handle imperfection. Fix: acknowledge the gap in one "
            "sentence, then pivot to what you'd do about it.",
        ),
        (
            "Forgetting to mention why they chose a technology, only "
            "what they used.",
            "'I used FastAPI' is weaker than explaining the concrete "
            "reason (auto validation + docs) that mattered for this "
            "project.",
        ),
        (
            "Not being able to trace a single request end-to-end.",
            "Candidates freeze when asked 'walk me through exactly what "
            "happens when I click this button.' Fix: rehearse the "
            "Request Flow section until it's automatic.",
        ),
        (
            "Downplaying real bugs they fixed.",
            "The BLIP-vs-human-caption bug in this project is excellent "
            "interview material - it shows debugging skill and "
            "architectural thinking. Always have 1-2 concrete bugs ready "
            "in STAR format.",
        ),
    ]
    for title, tip in mistakes:
        add_heading(document, title, level=2)
        add_body(document, tip)

    document.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
