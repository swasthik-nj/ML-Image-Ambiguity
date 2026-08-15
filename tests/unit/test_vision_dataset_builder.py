import pandas as pd
import pytest
from pathlib import Path
import tempfile
import numpy as np

from image_ambiguity.pipeline.vision_dataset_builder import VisionOnlyDatasetBuilder, VISION_FEATURE_COLUMNS

@pytest.fixture
def dummy_human_dataset(tmp_path):
    df = pd.DataFrame({
        "image_id": [1, 2, 3],
        "caption_diversity": [0.1, 0.4, 0.8],
        "edge_density": [0.05, 0.06, np.nan],  # one missing value to test median imputation
        "entropy": [7.0, 7.5, 7.8],
        "brightness": [100.0, 110.0, 120.0],
        "contrast": [50.0, 60.0, 70.0],
        "color_variance": [1000.0, 2000.0, 3000.0],
        "texture": [100.0, 200.0, 300.0],
        "ambiguity_label": ["Low", "Medium", "High"]
    })
    path = tmp_path / "dummy_human_dataset.csv"
    df.to_csv(path, index=False)
    return path

def test_vision_dataset_builder_filters_columns_and_imputes(dummy_human_dataset, tmp_path):
    builder = VisionOnlyDatasetBuilder(input_csv=dummy_human_dataset)
    df = builder.build()
    
    # Assert caption diversity is dropped
    assert "caption_diversity" not in df.columns
    
    # Assert required columns are present
    expected_cols = ["image_id"] + list(VISION_FEATURE_COLUMNS) + ["ambiguity_label"]
    assert list(df.columns) == expected_cols
    
    # Assert missing value was imputed with median (0.055)
    assert not df["edge_density"].isna().any()
    assert np.isclose(df.loc[2, "edge_density"], 0.055)

def test_vision_dataset_builder_missing_required_col(tmp_path):
    # Missing texture
    df = pd.DataFrame({
        "image_id": [1],
        "edge_density": [0.05],
        "entropy": [7.0],
        "brightness": [100.0],
        "contrast": [50.0],
        "color_variance": [1000.0],
        "ambiguity_label": ["Low"]
    })
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    
    builder = VisionOnlyDatasetBuilder(input_csv=path)
    with pytest.raises(KeyError, match="Input dataset is missing required vision columns"):
        builder.build()
