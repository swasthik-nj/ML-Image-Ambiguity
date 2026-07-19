"""SHAP-based explainability for the trained ambiguity classifiers.

Deliberately not re-exported from the top-level ``image_ambiguity``
package: importing this subpackage triggers a probe of the real
``numba`` package (installing a pure-Python stub if it cannot load,
see :mod:`image_ambiguity.explainability.numba_stub`) and imports
``shap``, both of which add noticeable startup cost that unrelated
modules (COCO loading, feature extraction, training, ...) should not
have to pay.
"""

from image_ambiguity.explainability.numba_stub import install_numba_stub
from image_ambiguity.explainability.shap_explainer import (
    PredictionExplanation,
    SHAPExplainer,
)

__all__ = ["PredictionExplanation", "SHAPExplainer", "install_numba_stub"]
