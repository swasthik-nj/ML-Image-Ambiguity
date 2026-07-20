"""SHAP-based explainability for the tree-ensemble ambiguity classifiers.

Wraps :class:`shap.TreeExplainer` to produce a Top Feature Importance
chart, a class-level Summary (beeswarm) plot, and single-prediction
Waterfall / Force plots for a fitted tree model -- primarily
:class:`~sklearn.ensemble.RandomForestClassifier`, but any
``TreeExplainer``-compatible model (e.g. XGBoost) works too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from image_ambiguity.explainability.numba_stub import install_numba_stub
from image_ambiguity.logging_config import get_logger
from image_ambiguity.utils.common import ensure_dir, timed

install_numba_stub()
import shap  # noqa: E402  (must be imported after the numba stub is installed)

logger = get_logger("explainability.shap_explainer")


@dataclass
class PredictionExplanation:
    """Structured, human-readable explanation for a single prediction."""

    row_index: int
    predicted_class: str
    predicted_probability: float
    base_value: float
    contributions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "row_index": self.row_index,
            "predicted_class": self.predicted_class,
            "predicted_probability": self.predicted_probability,
            "base_value": self.base_value,
            "contributions": self.contributions,
        }

    def summary_text(self, top_n: int = 5) -> str:
        """Render a short, plain-language explanation.

        Args:
            top_n: Maximum number of contributing features to mention.

        Returns:
            Multi-line human-readable summary.
        """
        lines = [
            f"Predicted class: {self.predicted_class} "
            f"(probability={self.predicted_probability:.2%})"
        ]
        for contribution in self.contributions[:top_n]:
            direction = "increased" if contribution["shap_value"] > 0 else "decreased"
            lines.append(
                f"  - {contribution['feature']} = {contribution['value']:.4f}  "
                f"{direction} the '{self.predicted_class}' probability by "
                f"{abs(contribution['shap_value']):.4f}"
            )
        return "\n".join(lines)


class SHAPExplainer:
    """Explain a fitted tree-ensemble classifier's predictions with SHAP.

    Supports scikit-learn's :class:`~sklearn.ensemble.RandomForestClassifier`
    (and any other model compatible with :class:`shap.TreeExplainer`,
    e.g. XGBoost).

    Args:
        model: A fitted tree-based classifier.
        feature_names: Names of the feature columns, in the same order
            used to fit ``model``.
        class_names: Optional display names for each class index (e.g.
            ``["Low", "Medium", "High"]``); used only for readable output.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.class_names = list(class_names) if class_names is not None else None
        self._explainer: Any = None

    def _class_label(self, class_index: int) -> str:
        if self.class_names and 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return str(class_index)

    def _ensure_explainer(self) -> Any:
        if self._explainer is None:
            with timed("shap.TreeExplainer:init"):
                self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def compute_shap_values(self, X: pd.DataFrame) -> Any:
        """Compute SHAP values for every row in ``X``.

        Args:
            X: Feature matrix, columns matching ``feature_names``.

        Returns:
            A :class:`shap.Explanation` whose ``values`` array has shape
            ``(n_samples, n_features)`` for single-output models or
            ``(n_samples, n_features, n_classes)`` for multiclass models.
        """
        explainer = self._ensure_explainer()
        with timed(f"shap.compute_shap_values:n={len(X)}"):
            return explainer(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def top_feature_importance(
        self, shap_values: Any, *, class_index: int | None = None, top_n: int = 10
    ) -> pd.DataFrame:
        """Rank features by mean absolute SHAP value.

        Args:
            shap_values: Output of :meth:`compute_shap_values`.
            class_index: Restrict ranking to one class; if ``None``,
                average absolute SHAP value across all classes.
            top_n: Number of top features to keep.

        Returns:
            DataFrame with columns ``feature`` and ``importance``,
            sorted descending, limited to ``top_n`` rows.
        """
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            importance = (
                np.abs(values[:, :, class_index]).mean(axis=0)
                if class_index is not None
                else np.abs(values).mean(axis=(0, 2))
            )
        else:
            importance = np.abs(values).mean(axis=0)

        frame = pd.DataFrame({"feature": self.feature_names, "importance": importance})
        return (
            frame.sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def plot_top_feature_importance(
        self,
        importance_df: pd.DataFrame,
        path: str | Path | None = None,
        *,
        title: str = "Top Feature Importance (mean |SHAP value|)",
        show: bool = False,
    ) -> Path | None:
        """Render a horizontal bar chart of feature importances.

        Args:
            importance_df: Output of :meth:`top_feature_importance`.
            path: Optional destination ``.png`` path.
            title: Plot title.
            show: If ``True``, display interactively in addition to saving.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        plt = _get_pyplot(show)

        ordered = importance_df.iloc[::-1]  # smallest-to-largest for horizontal bars
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(ordered["feature"], ordered["importance"], color="#4c72b0")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title)
        fig.tight_layout()

        return _save_or_show(fig, path, show)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def summary_plot(
        self,
        shap_values: Any,
        X: pd.DataFrame,
        *,
        class_index: int | None = None,
        path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """Render a SHAP beeswarm/summary plot for one class.

        Args:
            shap_values: Output of :meth:`compute_shap_values`.
            X: The feature matrix used to compute ``shap_values``.
            class_index: Which class to plot; required when
                ``shap_values`` is multiclass (3D).
            path: Optional destination ``.png`` path.
            show: If ``True``, display interactively in addition to saving.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.

        Raises:
            ValueError: If ``class_index`` is omitted for multiclass values.
        """
        plt = _get_pyplot(show)

        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            if class_index is None:
                raise ValueError("class_index is required for multiclass shap_values")
            plot_values = values[:, :, class_index]
        else:
            plot_values = values

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(plot_values, X, feature_names=self.feature_names, show=False)
        return _save_or_show(fig, path, show)

    def waterfall_plot(
        self,
        shap_values: Any,
        row_index: int,
        *,
        class_index: int | None = None,
        path: str | Path | None = None,
        max_display: int = 10,
        show: bool = False,
    ) -> Path | None:
        """Render a SHAP waterfall plot explaining a single prediction.

        Args:
            shap_values: Output of :meth:`compute_shap_values`.
            row_index: Position (0-based) of the row to explain.
            class_index: Which class's attribution to plot; required
                when ``shap_values`` is multiclass.
            path: Optional destination ``.png`` path.
            max_display: Maximum number of individual features to show.
            show: If ``True``, display interactively in addition to saving.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        plt = _get_pyplot(show)
        row = self._slice_row(shap_values, row_index, class_index)

        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(row, show=False, max_display=max_display)
        return _save_or_show(fig, path, show)

    def force_plot(
        self,
        shap_values: Any,
        row_index: int,
        *,
        class_index: int | None = None,
        path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """Render a static (matplotlib) SHAP force plot for one prediction.

        Args:
            shap_values: Output of :meth:`compute_shap_values`.
            row_index: Position (0-based) of the row to explain.
            class_index: Which class's attribution to plot; required
                when ``shap_values`` is multiclass.
            path: Optional destination ``.png`` path.
            show: If ``True``, display interactively in addition to saving.

        Returns:
            Resolved save path when ``path`` is provided, else ``None``.
        """
        plt = _get_pyplot(show)
        row = self._slice_row(shap_values, row_index, class_index)

        shap.plots.force(row, matplotlib=True, show=False)
        fig = plt.gcf()
        return _save_or_show(fig, path, show)

    # ------------------------------------------------------------------
    # Single-prediction explanation
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        shap_values: Any,
        X: pd.DataFrame,
        row_index: int,
        *,
        class_index: int | None = None,
        top_n: int = 5,
    ) -> PredictionExplanation:
        """Build a structured, human-readable explanation for one row.

        Args:
            shap_values: Output of :meth:`compute_shap_values`.
            X: The feature matrix used to compute ``shap_values``.
            row_index: Position (0-based) of the row to explain.
            class_index: Class to explain; defaults to the model's own
                predicted class for this row.
            top_n: Number of top contributing features to include.

        Returns:
            A :class:`PredictionExplanation` for the row.
        """
        row_features = X.iloc[row_index]

        if class_index is None:
            class_index = int(self.model.predict(X.iloc[[row_index]])[0])

        values = np.asarray(shap_values.values)
        base_values = np.asarray(shap_values.base_values)
        if values.ndim == 3:
            row_shap = values[row_index, :, class_index]
            base_value = float(base_values[row_index, class_index])
        else:
            row_shap = values[row_index, :]
            base_value = float(base_values[row_index])

        probability = float("nan")
        if hasattr(self.model, "predict_proba"):
            probability = float(
                self.model.predict_proba(X.iloc[[row_index]])[0][class_index]
            )

        order = np.argsort(-np.abs(row_shap))[:top_n]
        contributions = [
            {
                "feature": self.feature_names[i],
                "value": float(row_features.iloc[i]),
                "shap_value": float(row_shap[i]),
            }
            for i in order
        ]

        explanation = PredictionExplanation(
            row_index=row_index,
            predicted_class=self._class_label(class_index),
            predicted_probability=probability,
            base_value=base_value,
            contributions=contributions,
        )
        logger.info(
            "Explained row %s: predicted_class=%s probability=%.4f",
            row_index,
            explanation.predicted_class,
            probability,
        )
        return explanation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _slice_row(self, shap_values: Any, row_index: int, class_index: int | None) -> Any:
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            if class_index is None:
                raise ValueError("class_index is required for multiclass shap_values")
            return shap_values[row_index, :, class_index]
        return shap_values[row_index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={len(self.feature_names)})"


def _get_pyplot(show: bool) -> Any:
    import matplotlib

    if not show and matplotlib.get_backend().lower() != "agg":
        try:
            matplotlib.use("Agg", force=False)
        except Exception:  # noqa: BLE001 - backend choice is best-effort
            pass
    import matplotlib.pyplot as plt

    return plt


def _save_or_show(fig: Any, path: str | Path | None, show: bool) -> Path | None:
    plt = _get_pyplot(show)

    saved: Path | None = None
    if path is not None:
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".png")
        ensure_dir(destination.parent)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        saved = destination.resolve()
        logger.info("Saved plot to %s", saved)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved
