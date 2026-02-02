"""Tests for nrtk.interfaces._plugfigurable fault-tolerant plugin discovery."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nrtk.interfaces._plugfigurable import Plugfigurable, _safe_discover_via_entrypoints

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entrypoint(
    *,
    load_result: types.ModuleType | None = None,
    load_error: Exception | None = None,
) -> MagicMock:
    """Create a fake entrypoint whose .load() returns *load_result* or raises *load_error*."""
    ep = MagicMock()
    if load_error is not None:
        ep.load.side_effect = load_error
    else:
        ep.load.return_value = load_result
    return ep


class _StubType:
    """Dummy type returned by _collect_types_in_module."""


# ---------------------------------------------------------------------------
# _safe_discover_via_entrypoints
# ---------------------------------------------------------------------------

_MODULE = "nrtk.interfaces._plugfigurable"


@patch(f"{_MODULE}.get_ns_entrypoints")
@patch(f"{_MODULE}._collect_types_in_module", return_value={_StubType})
def test_safe_discover_skips_broken_entrypoint(
    mock_collect: MagicMock,
    mock_get_eps: MagicMock,
) -> None:
    """A broken entrypoint is skipped while healthy ones are still discovered."""
    good_module = types.ModuleType("good_plugin")
    good_ep = _make_entrypoint(load_result=good_module)
    bad_ep = _make_entrypoint(load_error=TypeError("boom"))

    mock_get_eps.return_value = [bad_ep, good_ep]

    result = _safe_discover_via_entrypoints("some.namespace")

    assert _StubType in result
    mock_collect.assert_called_once_with(good_module)


@patch(f"{_MODULE}.get_ns_entrypoints")
@patch(f"{_MODULE}._collect_types_in_module")
def test_safe_discover_logs_broken_entrypoint(
    mock_collect: MagicMock,
    mock_get_eps: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A broken entrypoint triggers a debug log message."""
    bad_ep = _make_entrypoint(load_error=RuntimeError("import failed"))
    mock_get_eps.return_value = [bad_ep]

    with caplog.at_level("DEBUG", logger=_MODULE):
        _safe_discover_via_entrypoints("some.namespace")

    assert "Skipping broken entrypoint" in caplog.text
    mock_collect.assert_not_called()


# ---------------------------------------------------------------------------
# Plugfigurable.get_impls  (broken EP doesn't block subclass discovery)
# ---------------------------------------------------------------------------


class _ConcretePlugfigurable(Plugfigurable):
    """Minimal concrete subclass used to exercise get_impls."""

    def get_config(self) -> dict[str, Any]:
        return {}  # pragma: no cover


@patch(f"{_MODULE}.discover_via_env_var", return_value=set())
@patch(f"{_MODULE}.get_ns_entrypoints")
@patch(f"{_MODULE}.discover_via_subclasses", return_value={_ConcretePlugfigurable})
@patch(f"{_MODULE}.filter_plugin_types", side_effect=lambda _cls, candidates: candidates)
def test_get_impls_tolerates_broken_entrypoint(
    mock_filter: MagicMock,  # noqa: ARG001 - required by @patch decorator order
    mock_subclasses: MagicMock,  # noqa: ARG001 - required by @patch decorator order
    mock_get_eps: MagicMock,
    mock_env: MagicMock,  # noqa: ARG001 - required by @patch decorator order
) -> None:
    """get_impls still returns subclass-discovered types when an entrypoint is broken."""
    bad_ep = _make_entrypoint(load_error=TypeError("boom"))
    mock_get_eps.return_value = [bad_ep]

    impls = _ConcretePlugfigurable.get_impls()

    assert _ConcretePlugfigurable in impls
    mock_get_eps.assert_called_once()
