"""ASGI entrypoint for ``uvicorn app:app --reload``.

Re-exports the FastAPI application defined in ``backend.app.main``.
"""

from __future__ import annotations

from backend.app.main import app

__all__ = ["app"]
