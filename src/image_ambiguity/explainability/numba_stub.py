"""Compatibility shim so ``shap`` works when ``numba`` cannot load.

Some locked-down environments (e.g. Windows machines under an
Application Control / WDAC policy) block numba's native extensions
from loading at all, raising an ``ImportError`` on ``import numba``.
``shap`` unconditionally imports numba at package init time -- for a
handful of hierarchical-clustering helpers used by only a few plotting
utilities -- which means the entire ``shap`` package becomes
unimportable even though the ``TreeExplainer`` path used in this
project never touches numba directly.

This module installs a minimal, pure-Python stand-in for the ``numba``
package (only if the real one cannot be imported) so that ``import
shap`` succeeds. The stand-in makes ``@njit`` a no-op decorator, so the
handful of numba-accelerated helper functions inside shap simply run
as plain Python -- slower, but perfectly correct for the small
datasets used in this project.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

from image_ambiguity.logging_config import get_logger

logger = get_logger("explainability.numba_stub")

_stub_active: bool | None = None


def install_numba_stub() -> bool:
    """Ensure ``import numba`` (and therefore ``import shap``) succeeds.

    Idempotent: safe to call multiple times; only performs the real
    import probe once per process.

    Returns:
        ``True`` if the no-op stub was installed in place of a real,
        working ``numba``; ``False`` if the genuine package imported
        successfully and no stub was needed.
    """
    global _stub_active
    if _stub_active is not None:
        return _stub_active

    try:
        import numba  # noqa: F401

        _stub_active = False
        return False
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
        logger.warning(
            "Real numba could not be imported (%s); installing a "
            "pure-Python stub so shap remains usable",
            exc,
        )

    # Remove any partially-initialized numba modules left behind by the
    # failed import above before installing the stand-in.
    for name in list(sys.modules):
        if name == "numba" or name.startswith("numba."):
            del sys.modules[name]

    stub = _build_stub_module()
    sys.modules["numba"] = stub
    sys.modules["numba.typed"] = stub.typed
    _stub_active = True
    return True


def _identity_decorator(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[..., Any]:
    """Mimic ``numba.njit``/``jit``: usable both as ``@njit`` and ``@njit(...)``."""
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return _wrap


def _build_stub_module() -> types.ModuleType:
    stub = types.ModuleType("numba")
    stub.njit = _identity_decorator
    stub.jit = _identity_decorator
    stub.vectorize = _identity_decorator
    stub.generated_jit = _identity_decorator
    stub.prange = range

    typed_stub = types.ModuleType("numba.typed")
    typed_stub.List = list
    typed_stub.Dict = dict
    stub.typed = typed_stub
    return stub
