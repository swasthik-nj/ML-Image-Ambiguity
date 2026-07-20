"""Unit tests for the numba compatibility shim."""

from __future__ import annotations

import sys

from image_ambiguity.explainability.numba_stub import install_numba_stub


class TestInstallNumbaStub:
    def test_import_numba_succeeds_after_install(self) -> None:
        install_numba_stub()
        import numba  # noqa: F401  (must not raise)

        assert "numba" in sys.modules

    def test_idempotent_across_multiple_calls(self) -> None:
        first = install_numba_stub()
        second = install_numba_stub()
        assert first == second

    def test_njit_usable_as_bare_decorator(self) -> None:
        install_numba_stub()
        import numba

        @numba.njit
        def add_one(x: int) -> int:
            return x + 1

        assert add_one(4) == 5

    def test_njit_usable_with_call_arguments(self) -> None:
        install_numba_stub()
        import numba

        @numba.njit(cache=True)
        def double(x: int) -> int:
            return x * 2

        assert double(3) == 6

    def test_typed_list_is_constructible(self) -> None:
        install_numba_stub()
        import numba.typed

        values = numba.typed.List([1, 2, 3])
        assert list(values) == [1, 2, 3]

    def test_prange_behaves_like_range(self) -> None:
        install_numba_stub()
        import numba

        assert list(numba.prange(3)) == [0, 1, 2]
