"""Unit tests for :class:`ModelTrainer`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from image_ambiguity.models.trainer import (
    FEATURE_COLUMNS,
    LABEL_ORDER,
    ModelResult,
    ModelTrainer,
)


def _make_synthetic_dataset(n_per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic, separable dataset with balanced classes."""
    rng = np.random.default_rng(seed)
    rows = []
    # Give each class a distinct center so classifiers can learn something
    # real, rather than relying purely on chance.
    centers = {"Low": 0.15, "Medium": 0.5, "High": 0.85}
    for label, center in centers.items():
        for _ in range(n_per_class):
            diversity = float(np.clip(rng.normal(center, 0.03), 0.0, 1.0))
            row = {col: float(rng.normal(center, 0.05)) for col in FEATURE_COLUMNS}
            row["caption_diversity"] = diversity
            row["ambiguity_label"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "image_id", range(1, len(df) + 1))
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def dataset() -> pd.DataFrame:
    return _make_synthetic_dataset()


@pytest.fixture
def trainer() -> ModelTrainer:
    return ModelTrainer(test_size=0.2, cv_folds=3, random_state=0)


class TestLoadDataset:
    def test_missing_file_raises(self, trainer: ModelTrainer, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            trainer.load_dataset(tmp_path / "missing.csv")

    def test_loads_existing_csv(
        self, trainer: ModelTrainer, dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "data.csv"
        dataset.to_csv(path, index=False)
        loaded = trainer.load_dataset(path)
        assert len(loaded) == len(dataset)


class TestPrepareData:
    def test_extracts_features_and_encodes_labels(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        assert list(X.columns) == list(FEATURE_COLUMNS)
        assert len(X) == len(y) == len(dataset)
        assert set(y.unique()) == {0, 1, 2}

    def test_drops_unknown_label_rows(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        df = dataset.copy()
        df.loc[0, "ambiguity_label"] = "Unknown"
        X, y = trainer.prepare_data(df)
        assert len(X) == len(dataset) - 1

    def test_missing_feature_column_raises(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        df = dataset.drop(columns=["entropy"])
        with pytest.raises(KeyError):
            trainer.prepare_data(df)

    def test_missing_target_column_raises(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        df = dataset.drop(columns=["ambiguity_label"])
        with pytest.raises(KeyError):
            trainer.prepare_data(df)

    def test_imputes_missing_feature_values(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        df = dataset.copy()
        df.loc[0, "brightness"] = np.nan
        X, _ = trainer.prepare_data(df)
        assert not X["brightness"].isna().any()

    def test_all_unknown_labels_raises(self, trainer: ModelTrainer) -> None:
        df = pd.DataFrame(
            {col: [0.1, 0.2] for col in FEATURE_COLUMNS}
            | {"ambiguity_label": ["Unknown", "Unknown"]}
        )
        with pytest.raises(ValueError):
            trainer.prepare_data(df)


class TestSplitData:
    def test_split_respects_test_size(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        total = len(X)
        assert len(X_test) == pytest.approx(total * 0.2, abs=2)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_falls_back_when_stratify_impossible(self, trainer: ModelTrainer) -> None:
        X = pd.DataFrame({col: np.random.rand(10) for col in FEATURE_COLUMNS})
        y = pd.Series([0] * 9 + [1])  # class "1" has only one member
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        assert len(X_train) + len(X_test) == 10


class TestResolveCvFolds:
    def test_returns_none_for_singleton_class(self, trainer: ModelTrainer) -> None:
        y = pd.Series([0, 0, 0, 1])
        assert trainer._resolve_cv_folds(y) is None

    def test_caps_folds_at_min_class_count(self, trainer: ModelTrainer) -> None:
        y = pd.Series([0] * 10 + [1] * 10 + [2] * 2)
        folds = trainer._resolve_cv_folds(y)
        assert folds == 2

    def test_uses_requested_folds_when_sufficient(self, trainer: ModelTrainer) -> None:
        y = pd.Series([0] * 10 + [1] * 10 + [2] * 10)
        folds = trainer._resolve_cv_folds(y)
        assert folds == trainer.cv_folds


class TestCrossValidateBaseline:
    def test_skips_when_class_too_small(self, trainer: ModelTrainer) -> None:
        X = pd.DataFrame({col: np.random.rand(4) for col in FEATURE_COLUMNS})
        y = pd.Series([0, 0, 0, 1])
        score, folds = trainer.cross_validate_baseline("random_forest", X, y)
        assert score is None
        assert folds is None

    def test_returns_score_for_valid_data(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        score, folds = trainer.cross_validate_baseline("random_forest", X, y)
        assert score is not None
        assert 0.0 <= score <= 1.0
        assert folds is not None and folds >= 2


class TestTuneHyperparameters:
    def test_random_forest_returns_fitted_model(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        model, params, score, folds = trainer.tune_hyperparameters(
            "random_forest", X_train, y_train
        )
        assert isinstance(model, RandomForestClassifier)
        assert isinstance(params, dict) and params
        assert score is not None and 0.0 <= score <= 1.0
        assert folds is not None
        # Fitted model should be usable for prediction immediately.
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_xgboost_returns_fitted_model(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        X_train, _, y_train, _ = trainer.split_data(X, y)
        model, params, score, folds = trainer.tune_hyperparameters(
            "xgboost", X_train, y_train
        )
        assert params
        assert score is not None
        assert folds is not None

    def test_falls_back_to_default_when_too_few_samples(
        self, trainer: ModelTrainer
    ) -> None:
        X = pd.DataFrame({col: np.random.rand(5) for col in FEATURE_COLUMNS})
        y = pd.Series([0, 0, 0, 0, 1])
        model, params, score, folds = trainer.tune_hyperparameters(
            "random_forest", X, y
        )
        assert params == {}
        assert score is None
        assert folds is None
        assert model.predict(X) is not None


class TestEvaluate:
    def test_metrics_have_expected_keys_and_ranges(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

        metrics = trainer.evaluate(model, X_test, y_test)

        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0
        assert metrics["roc_auc"] is None or 0.0 <= metrics["roc_auc"] <= 1.0

    def test_separable_synthetic_data_achieves_high_accuracy(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        X, y = trainer.prepare_data(dataset)
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

        metrics = trainer.evaluate(model, X_test, y_test)
        assert metrics["accuracy"] >= 0.8


class TestSelectBest:
    def test_picks_highest_metric(self) -> None:
        results = {
            "random_forest": ModelResult(
                name="random_forest",
                best_params={},
                cv_folds_used=3,
                baseline_cv_accuracy=0.8,
                tuned_cv_score=0.82,
                metrics={"accuracy": 0.91, "precision": 0.9, "recall": 0.9, "f1": 0.9, "roc_auc": 0.95},
                model=object(),
            ),
            "xgboost": ModelResult(
                name="xgboost",
                best_params={},
                cv_folds_used=3,
                baseline_cv_accuracy=0.75,
                tuned_cv_score=0.78,
                metrics={"accuracy": 0.85, "precision": 0.84, "recall": 0.84, "f1": 0.84, "roc_auc": 0.9},
                model=object(),
            ),
        }
        best = ModelTrainer.select_best(results, metric="accuracy")
        assert best.name == "random_forest"

    def test_raises_on_empty_results(self) -> None:
        with pytest.raises(ValueError):
            ModelTrainer.select_best({})


class TestSaveLoadModel:
    def test_roundtrip(self, trainer: ModelTrainer, dataset: pd.DataFrame, tmp_path: Path) -> None:
        X, y = trainer.prepare_data(dataset)
        model = RandomForestClassifier(random_state=0).fit(X, y)

        path = trainer.save_model(model, tmp_path / "model.joblib")
        assert path.exists()

        reloaded = trainer.load_model(path)
        np.testing.assert_array_equal(reloaded.predict(X), model.predict(X))

    def test_load_missing_file_raises(self, trainer: ModelTrainer, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            trainer.load_model(tmp_path / "missing.joblib")


class TestTrainEndToEnd:
    def test_train_returns_both_models_with_metrics(
        self, trainer: ModelTrainer, dataset: pd.DataFrame
    ) -> None:
        results = trainer.train(dataset)

        assert set(results) == {"random_forest", "xgboost"}
        for result in results.values():
            assert isinstance(result, ModelResult)
            assert result.metrics["accuracy"] >= 0.5
            assert result.model is not None

    def test_invalid_test_size_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelTrainer(test_size=1.5)

    def test_invalid_cv_folds_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelTrainer(cv_folds=1)

    def test_repr(self, trainer: ModelTrainer) -> None:
        assert "ModelTrainer" in repr(trainer)
