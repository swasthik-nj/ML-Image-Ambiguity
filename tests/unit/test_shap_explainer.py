"""Unit tests for :class:`SHAPExplainer`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from image_ambiguity.explainability.shap_explainer import (
    PredictionExplanation,
    SHAPExplainer,
)
from image_ambiguity.models.trainer import FEATURE_COLUMNS, LABEL_ORDER


def _make_synthetic_dataset(n_per_class: int = 15, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = []
    centers = {0: 0.15, 1: 0.5, 2: 0.85}
    for label, center in centers.items():
        for _ in range(n_per_class):
            row = {col: float(rng.normal(center, 0.05)) for col in FEATURE_COLUMNS}
            row["label"] = label
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    X = df[list(FEATURE_COLUMNS)]
    y = df["label"]
    return X, y


@pytest.fixture(scope="module")
def trained_model() -> RandomForestClassifier:
    X, y = _make_synthetic_dataset()
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    X, _ = _make_synthetic_dataset()
    return X


@pytest.fixture(scope="module")
def explainer(trained_model: RandomForestClassifier) -> SHAPExplainer:
    return SHAPExplainer(
        trained_model, feature_names=FEATURE_COLUMNS, class_names=list(LABEL_ORDER)
    )


@pytest.fixture(scope="module")
def shap_values(explainer: SHAPExplainer, dataset: pd.DataFrame):
    return explainer.compute_shap_values(dataset)


class TestComputeShapValues:
    def test_multiclass_shape(self, shap_values, dataset: pd.DataFrame) -> None:
        values = np.asarray(shap_values.values)
        assert values.shape == (len(dataset), len(FEATURE_COLUMNS), 3)


class TestTopFeatureImportance:
    def test_aggregate_importance_sorted_and_limited(
        self, explainer: SHAPExplainer, shap_values
    ) -> None:
        result = explainer.top_feature_importance(shap_values, top_n=5)

        assert list(result.columns) == ["feature", "importance"]
        assert len(result) == 5
        assert (result["importance"].diff().dropna() <= 0).all()

    def test_per_class_importance_differs_from_aggregate(
        self, explainer: SHAPExplainer, shap_values
    ) -> None:
        aggregate = explainer.top_feature_importance(shap_values, top_n=11)
        per_class = explainer.top_feature_importance(
            shap_values, class_index=2, top_n=11
        )
        # Not required to differ in ranking, but values themselves should
        # generally not be identical between an aggregate and single class.
        assert not np.allclose(
            aggregate.sort_values("feature")["importance"].to_numpy(),
            per_class.sort_values("feature")["importance"].to_numpy(),
        )

    def test_top_n_caps_row_count(self, explainer: SHAPExplainer, shap_values) -> None:
        result = explainer.top_feature_importance(shap_values, top_n=3)
        assert len(result) == 3


class TestPlotTopFeatureImportance:
    def test_saves_png(
        self, explainer: SHAPExplainer, shap_values, tmp_path: Path
    ) -> None:
        importance_df = explainer.top_feature_importance(shap_values, top_n=5)
        path = explainer.plot_top_feature_importance(
            importance_df, tmp_path / "importance.png"
        )
        assert path is not None and path.exists()

    def test_returns_none_without_path(
        self, explainer: SHAPExplainer, shap_values
    ) -> None:
        importance_df = explainer.top_feature_importance(shap_values, top_n=5)
        path = explainer.plot_top_feature_importance(importance_df, path=None)
        assert path is None


class TestSummaryPlot:
    def test_saves_png_for_given_class(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = explainer.summary_plot(
            shap_values, dataset, class_index=2, path=tmp_path / "summary.png"
        )
        assert path is not None and path.exists()

    def test_raises_without_class_index_for_multiclass(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            explainer.summary_plot(shap_values, dataset, class_index=None, path=None)


class TestWaterfallPlot:
    def test_saves_png(
        self, explainer: SHAPExplainer, shap_values, tmp_path: Path
    ) -> None:
        path = explainer.waterfall_plot(
            shap_values, row_index=0, class_index=0, path=tmp_path / "waterfall.png"
        )
        assert path is not None and path.exists()

    def test_raises_without_class_index_for_multiclass(
        self, explainer: SHAPExplainer, shap_values
    ) -> None:
        with pytest.raises(ValueError):
            explainer.waterfall_plot(shap_values, row_index=0, class_index=None, path=None)


class TestForcePlot:
    def test_saves_png(
        self, explainer: SHAPExplainer, shap_values, tmp_path: Path
    ) -> None:
        path = explainer.force_plot(
            shap_values, row_index=0, class_index=1, path=tmp_path / "force.png"
        )
        assert path is not None and path.exists()


class TestExplainPrediction:
    def test_returns_expected_structure(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame
    ) -> None:
        explanation = explainer.explain_prediction(
            shap_values, dataset, row_index=0, class_index=2, top_n=4
        )

        assert isinstance(explanation, PredictionExplanation)
        assert explanation.predicted_class == "High"
        assert 0.0 <= explanation.predicted_probability <= 1.0
        assert len(explanation.contributions) == 4
        magnitudes = [abs(c["shap_value"]) for c in explanation.contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_default_class_index_matches_model_prediction(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame, trained_model
    ) -> None:
        row_index = 0
        explanation = explainer.explain_prediction(shap_values, dataset, row_index)
        predicted_code = int(trained_model.predict(dataset.iloc[[row_index]])[0])
        assert explanation.predicted_class == LABEL_ORDER[predicted_code]

    def test_to_dict_roundtrip(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame
    ) -> None:
        explanation = explainer.explain_prediction(
            shap_values, dataset, row_index=1, class_index=0
        )
        payload = explanation.to_dict()
        assert payload["predicted_class"] == "Low"
        assert "contributions" in payload

    def test_summary_text_mentions_direction(
        self, explainer: SHAPExplainer, shap_values, dataset: pd.DataFrame
    ) -> None:
        explanation = explainer.explain_prediction(
            shap_values, dataset, row_index=0, class_index=2
        )
        text = explanation.summary_text()
        assert "Predicted class: High" in text
        assert "increased" in text or "decreased" in text


class TestRepr:
    def test_repr_contains_feature_count(self, explainer: SHAPExplainer) -> None:
        assert "SHAPExplainer" in repr(explainer)
        assert str(len(FEATURE_COLUMNS)) in repr(explainer)
