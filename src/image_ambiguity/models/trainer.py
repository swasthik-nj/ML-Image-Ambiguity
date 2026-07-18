"""Train and evaluate classical ML classifiers on the ambiguity dataset.

Trains a :class:`~sklearn.ensemble.RandomForestClassifier` and an
:class:`~xgboost.XGBClassifier` on ``human_dataset.csv`` to predict the
rule-based ``ambiguity_label`` (Low / Medium / High) from the eleven
numeric caption-diversity and OpenCV features. Performs an 80/20
train/test split, baseline k-fold cross-validation, grid-search
hyperparameter tuning, and reports Accuracy / Precision / Recall / F1 /
ROC-AUC on the held-out test set. The best-performing model (by test
accuracy) is persisted with ``joblib``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, timed

logger = get_logger("models.trainer")

LABEL_ORDER: tuple[str, ...] = ("Low", "Medium", "High")

FEATURE_COLUMNS: tuple[str, ...] = (
    "average_similarity",
    "minimum_similarity",
    "maximum_similarity",
    "std_similarity",
    "caption_diversity",
    "edge_density",
    "entropy",
    "brightness",
    "contrast",
    "color_variance",
    "texture",
)

TARGET_COLUMN = "ambiguity_label"

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}

DISPLAY_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


@dataclass
class ModelResult:
    """Container for one trained model's tuning + evaluation results."""

    name: str
    best_params: dict[str, Any]
    cv_folds_used: int | None
    baseline_cv_accuracy: float | None
    tuned_cv_score: float | None
    metrics: dict[str, float | None]
    model: Any = field(repr=False)

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name)


class ModelTrainer:
    """Train, tune, and evaluate classifiers that predict ambiguity labels.

    Args:
        test_size: Fraction of data held out for testing (default 0.2, i.e. 80/20).
        cv_folds: Requested number of cross-validation folds (auto-reduced
            when a class has too few samples).
        scoring: Scoring metric used to select hyperparameters during
            :class:`~sklearn.model_selection.GridSearchCV`.
        random_state: Seed used for splitting and both estimators.
    """

    FEATURE_COLUMNS = FEATURE_COLUMNS
    TARGET_COLUMN = TARGET_COLUMN
    LABEL_ORDER = LABEL_ORDER

    def __init__(
        self,
        *,
        test_size: float = 0.2,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ) -> None:
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

        self.test_size = test_size
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def load_dataset(self, path: str | Path) -> pd.DataFrame:
        """Load the merged feature dataset from CSV.

        Args:
            path: Path to ``human_dataset.csv``.

        Returns:
            Loaded dataframe.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. Run "
                "'python src/create_dataset.py' and "
                "'python src/generate_labels.py' first."
            )
        return pd.read_csv(csv_path)

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extract features (X) and integer-encoded target (y) from the dataset.

        Rows whose label is not one of :data:`LABEL_ORDER` (e.g. ``"Unknown"``
        from missing diversity scores) are dropped. Missing feature values
        are imputed with the column median.

        Args:
            df: Raw dataset containing feature and label columns.

        Returns:
            Tuple of ``(X, y)`` where ``y`` holds integer class codes
            (0=Low, 1=Medium, 2=High per :data:`LABEL_ORDER`).

        Raises:
            KeyError: If required feature or target columns are missing.
            ValueError: If no labeled rows remain after filtering.
        """
        missing_features = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing_features:
            raise KeyError(f"Missing feature column(s): {missing_features}")
        if self.TARGET_COLUMN not in df.columns:
            raise KeyError(f"Missing target column '{self.TARGET_COLUMN}'")

        data = df[df[self.TARGET_COLUMN].isin(self.LABEL_ORDER)].copy()
        if data.empty:
            raise ValueError("No rows with a valid ambiguity_label to train on")

        X = data[list(self.FEATURE_COLUMNS)].astype(float)
        if X.isna().any().any():
            logger.warning("Imputing missing feature values with column medians")
            X = X.fillna(X.median())

        codes = pd.Categorical(
            data[self.TARGET_COLUMN], categories=self.LABEL_ORDER, ordered=True
        ).codes
        y = pd.Series(codes, index=data.index, name="label_code")

        logger.info(
            "Prepared %s rows, %s features, class distribution=%s",
            len(X),
            X.shape[1],
            {self.LABEL_ORDER[k]: int(v) for k, v in pd.Series(y).value_counts().items()},
        )
        return X, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform an 80/20 (or configured) stratified train/test split.

        Falls back to a non-stratified split if any class has too few
        members to stratify.

        Args:
            X: Feature matrix.
            y: Integer-encoded target.

        Returns:
            ``(X_train, X_test, y_train, y_test)``.
        """
        try:
            return train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError as exc:
            logger.warning(
                "Stratified split failed (%s); falling back to a random split", exc
            )
            return train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

    # ------------------------------------------------------------------
    # Cross-validation / tuning helpers
    # ------------------------------------------------------------------

    def _resolve_cv_folds(self, y: pd.Series) -> int | None:
        """Return a safe fold count, or ``None`` if CV is not feasible."""
        counts = pd.Series(y).value_counts()
        if counts.empty:
            return None
        min_count = int(counts.min())
        if min_count < 2:
            logger.warning(
                "Smallest class has only %s sample(s); skipping cross-validation",
                min_count,
            )
            return None
        return max(2, min(self.cv_folds, min_count))

    def _build_estimator(self, name: str, **overrides: Any) -> Any:
        if name == "random_forest":
            params: dict[str, Any] = {"random_state": self.random_state, "n_jobs": -1}
            params.update(overrides)
            return RandomForestClassifier(**params)
        if name == "xgboost":
            from xgboost import XGBClassifier

            params = {
                "random_state": self.random_state,
                "eval_metric": "mlogloss",
                "n_jobs": -1,
            }
            params.update(overrides)
            return XGBClassifier(**params)
        raise ValueError(f"Unknown model name: {name!r}")

    def cross_validate_baseline(
        self, name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[float | None, int | None]:
        """Run baseline k-fold cross-validation with default hyperparameters.

        Args:
            name: ``"random_forest"`` or ``"xgboost"``.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Tuple of ``(mean_accuracy, folds_used)``; both ``None`` if a
            class has too few samples to cross-validate.
        """
        folds = self._resolve_cv_folds(y_train)
        if folds is None:
            return None, None

        estimator = self._build_estimator(name)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        with timed(f"cross_validate_baseline:{name}"):
            scores = cross_val_score(
                estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
        mean_score = float(np.mean(scores))
        logger.info(
            "%s baseline %s-fold CV accuracy: %.4f (+/- %.4f)",
            name,
            folds,
            mean_score,
            float(np.std(scores)),
        )
        return mean_score, folds

    def tune_hyperparameters(
        self, name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[Any, dict[str, Any], float | None, int | None]:
        """Grid-search hyperparameters for the named model.

        Falls back to fitting a default-parameter estimator (no search) if
        the smallest class has too few samples to cross-validate.

        Args:
            name: ``"random_forest"`` or ``"xgboost"``.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            ``(best_estimator, best_params, best_cv_score, folds_used)``.
        """
        folds = self._resolve_cv_folds(y_train)
        if folds is None:
            estimator = self._build_estimator(name)
            with timed(f"fit_default:{name}"):
                estimator.fit(X_train, y_train)
            logger.warning(
                "%s: hyperparameter tuning skipped (too few samples per class); "
                "fitted with default parameters",
                name,
            )
            return estimator, {}, None, None

        base_estimator = self._build_estimator(name)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            base_estimator,
            param_grid=PARAM_GRIDS[name],
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            refit=True,
        )
        with warnings.catch_warnings(), timed(f"tune_hyperparameters:{name}"):
            warnings.simplefilter("ignore", category=UserWarning)
            search.fit(X_train, y_train)

        logger.info(
            "%s tuned (%s-fold, scoring=%s): best_score=%.4f best_params=%s",
            name,
            folds,
            self.scoring,
            search.best_score_,
            search.best_params_,
        )
        return search.best_estimator_, dict(search.best_params_), float(search.best_score_), folds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float | None]:
        """Compute Accuracy / Precision / Recall / F1 / ROC-AUC on held-out data.

        Precision, Recall, and F1 use macro-averaging across the three
        ambiguity classes. ROC-AUC uses a macro-averaged one-vs-rest
        strategy and is set to ``None`` if it cannot be computed (e.g. a
        class is entirely absent from the test split).

        Args:
            model: A fitted classifier exposing ``predict`` (and ideally
                ``predict_proba``).
            X_test: Held-out features.
            y_test: Held-out integer labels.

        Returns:
            Dictionary with keys ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``roc_auc``.
        """
        predictions = model.predict(X_test)

        metrics: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "roc_auc": None,
        }

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X_test)
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                        labels=model.classes_,
                    )
                )
            except ValueError as exc:
                logger.warning("Could not compute ROC-AUC: %s", exc)

        return metrics

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict[str, ModelResult]:
        """Run the full pipeline: split, cross-validate, tune, and evaluate.

        Args:
            df: Dataset containing feature and ``ambiguity_label`` columns.

        Returns:
            Mapping of model name (``"random_forest"``, ``"xgboost"``) to
            its :class:`ModelResult`.
        """
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        logger.info(
            "Split: %s train / %s test (%.0f/%.0f)",
            len(X_train),
            len(X_test),
            (1 - self.test_size) * 100,
            self.test_size * 100,
        )

        results: dict[str, ModelResult] = {}
        for name in PARAM_GRIDS:
            baseline_acc, baseline_folds = self.cross_validate_baseline(
                name, X_train, y_train
            )
            best_model, best_params, tuned_score, tuned_folds = self.tune_hyperparameters(
                name, X_train, y_train
            )
            metrics = self.evaluate(best_model, X_test, y_test)

            results[name] = ModelResult(
                name=name,
                best_params=best_params,
                cv_folds_used=tuned_folds or baseline_folds,
                baseline_cv_accuracy=baseline_acc,
                tuned_cv_score=tuned_score,
                metrics=metrics,
                model=best_model,
            )
            logger.info(
                "%s test metrics: %s", DISPLAY_NAMES.get(name, name), metrics
            )

        return results

    @staticmethod
    def select_best(results: dict[str, ModelResult], metric: str = "accuracy") -> ModelResult:
        """Pick the model with the highest value of ``metric`` on the test set.

        Args:
            results: Output of :meth:`train`.
            metric: Metric key to compare (default ``"accuracy"``).

        Returns:
            The best-performing :class:`ModelResult`.
        """
        if not results:
            raise ValueError("results is empty; nothing to select from")
        return max(
            results.values(),
            key=lambda result: result.metrics.get(metric) or float("-inf"),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, model: Any, path: str | Path) -> Path:
        """Persist a fitted model to disk with ``joblib``.

        Args:
            model: Fitted estimator.
            path: Destination ``.joblib`` path.

        Returns:
            Resolved path written to disk.
        """
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".joblib")
        ensure_dir(destination.parent)
        joblib.dump(model, destination)
        logger.info("Saved model to %s", destination)
        return destination.resolve()

    def load_model(self, path: str | Path) -> Any:
        """Load a previously saved model with ``joblib``.

        Args:
            path: Path to a ``.joblib`` file.

        Returns:
            The deserialized estimator.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Model file not found: {source}")
        return joblib.load(source)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(test_size={self.test_size}, "
            f"cv_folds={self.cv_folds}, scoring={self.scoring!r})"
        )
