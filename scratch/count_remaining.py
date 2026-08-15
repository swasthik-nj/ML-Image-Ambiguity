import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_ambiguity.config import get_settings
from image_ambiguity.data.coco_loader import CocoDatasetLoader
from image_ambiguity.features.caption_diversity import CaptionDiversityAnalyzer
from image_ambiguity.features.sentence_embeddings import SentenceEmbeddingGenerator

settings = get_settings()

existing = pd.read_csv(PROJECT_ROOT / "dataset" / "human_dataset.csv")
existing_ids = set(existing["image_id"].astype(int).tolist())

loader = CocoDatasetLoader(settings.annotation_file, settings.image_dir)
loader.load_annotations()
all_ids = list(loader.coco.getImgIds())
unused = [img_id for img_id in all_ids if img_id not in existing_ids]

print(f"Total unused images: {len(unused)}")

captions_by_image = {}
for img_id in unused:
    caps = loader.get_captions(int(img_id))
    if len(caps) >= 2:
        captions_by_image[int(img_id)] = caps

embedder = SentenceEmbeddingGenerator(
    model_name=settings.sentence_model_name,
    device=settings.device,
    batch_size=settings.embedding_batch_size,
)
embedder.load_model()
analyzer = CaptionDiversityAnalyzer()

flat = []
spans = {}
for img_id, caps in captions_by_image.items():
    start = len(flat)
    flat.extend(caps)
    spans[img_id] = (start, len(flat))

embeddings = embedder.generate_embeddings(flat)
scored = []
for img_id, (start, end) in spans.items():
    metrics = analyzer.compute(embeddings[start:end])
    scored.append((img_id, float(metrics.diversity_score)))

high = [s for id, s in scored if s >= 0.65]
medium = [s for id, s in scored if 0.35 <= s < 0.65]
low = [s for id, s in scored if s < 0.35]

print(f"\nBreakdown of the remaining {len(scored)} usable images:")
print(f"High Ambiguity (>= 0.65): {len(high)}")
print(f"Medium Ambiguity (0.35 - 0.65): {len(medium)}")
print(f"Low Ambiguity (< 0.35): {len(low)}")
