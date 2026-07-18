# Explainable Image Ambiguity Prediction

Research codebase for:

**Explainable Image Ambiguity Prediction Using Human and AI-Generated Caption Diversity with Computer Vision Features.**

The project follows a clean layered architecture:

| Layer | Location | Responsibility |
|---|---|---|
| Domain / ML core | `src/image_ambiguity/` | Data, features, models, explainability, pipeline |
| Config | `configs/`, `src/.../config.py` | YAML + environment settings |
| API | `backend/` | FastAPI service over the domain package |
| UI | `frontend/` | Demo / visualization client |
| Experiments | `notebooks/` | Exploratory research |
| Artifacts | `models/`, `results/` | Trained weights, metrics, figures, logs |
| Data | `dataset/` | COCO images + captions (local, gitignored) |

## Requirements

- Python **3.12+**
- Virtual environment (`.venv` recommended)
- COCO 2017 validation images + caption annotations under `dataset/`

## Project structure

```text
.
├── backend/                 # FastAPI application
├── configs/                 # YAML configs
├── dataset/                 # COCO data (local)
├── frontend/                # UI scaffold
├── models/                  # Saved model artifacts
├── notebooks/               # Research notebooks
├── results/                 # Metrics, figures, logs
├── src/image_ambiguity/     # Installable Python package
│   ├── config.py
│   ├── logging_config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   ├── pipeline/
│   └── utils/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```powershell
# from repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# environment variables
copy .env.example .env
```

Expected data layout:

```text
dataset/
├── annotations/
│   └── captions_val2017.json
└── val2017/
    └── *.jpg
```

## Configuration

Settings are loaded by `image_ambiguity.config.get_settings()` with this priority:

1. Process env vars (`IAP_*`)
2. `.env`
3. `configs/default.yaml`
4. Code defaults

Examples:

```powershell
$env:IAP_SAMPLE_SIZE = "1000"
$env:IAP_LOG_LEVEL = "DEBUG"
```

## Quickstart

### Load COCO captions

```python
from image_ambiguity import CocoDatasetLoader
from image_ambiguity.config import get_settings
from image_ambiguity.logging_config import setup_logging

setup_logging()
settings = get_settings()

loader = CocoDatasetLoader(settings.annotation_file, settings.image_dir)
loader.load_annotations()

sample = loader.get_random_sample(seed=settings.random_seed)
print(sample["image_id"], sample["captions"])
```

### Run API

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Run tests

```powershell
pytest -v
```

## Design notes

- **Domain logic** lives in `src/image_ambiguity` and must not import FastAPI/UI code.
- **Backend** depends inward on the domain package (services/API adapters).
- **Notebooks** may import the package for exploration but should not own production logic.
- **Artifacts** (`models/`, `results/`) are writable outputs and mostly gitignored.

## License

Research / academic use — update this section before public release.
