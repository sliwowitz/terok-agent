# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""First-run ``setup`` / ``uninstall`` and the run-time preflight gate.

Covers:

- ``_handle_setup`` phase ordering + opt-out flags + ``--check`` mode
- ``_handle_uninstall`` reverse-order teardown + ``--keep-images``
- ``_preflight_or_exit`` TTY gating — non-TTY without ``--yes`` refuses
  interactive mode and points the operator at setup

Individual preflight checks are covered in ``test_preflight.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from terok_executor.commands import (
    _handle_setup,
    _handle_uninstall,
    _preflight_or_exit,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def setup_spies():
    """Replace every phase (sandbox, images) with a MagicMock so order is observable."""
    with (
        patch("terok_sandbox.commands._handle_sandbox_setup") as sandbox_setup,
        patch("terok_sandbox.commands._handle_sandbox_uninstall") as sandbox_uninstall,
        patch("terok_executor.commands._build_images_with_banner") as build_images,
        patch("terok_executor.commands._remove_images") as remove_images,
    ):
        yield {
            "sandbox_setup": sandbox_setup,
            "sandbox_uninstall": sandbox_uninstall,
            "build_images": build_images,
            "remove_images": remove_images,
        }


# ── Setup flow ────────────────────────────────────────────────────────


class TestHandleSetup:
    """``terok-executor setup`` orchestrates sandbox + image-build."""

    def test_default_runs_sandbox_setup_then_image_build(self, setup_spies) -> None:
        _handle_setup()
        setup_spies["sandbox_setup"].assert_called_once_with(root=False)
        setup_spies["build_images"].assert_called_once()

    def test_root_flag_propagates_to_sandbox_setup(self, setup_spies) -> None:
        _handle_setup(root=True)
        setup_spies["sandbox_setup"].assert_called_once_with(root=True)

    def test_no_sandbox_skips_sandbox_setup(self, setup_spies) -> None:
        _handle_setup(no_sandbox=True)
        setup_spies["sandbox_setup"].assert_not_called()
        setup_spies["build_images"].assert_called_once()

    def test_no_images_skips_image_build(self, setup_spies) -> None:
        _handle_setup(no_images=True)
        setup_spies["sandbox_setup"].assert_called_once()
        setup_spies["build_images"].assert_not_called()

    def test_check_mode_does_not_install(self, setup_spies) -> None:
        """``--check`` reports without touching anything; exits 0 when ready."""
        with (
            patch("terok_executor.commands._print_setup_status") as print_status,
        ):
            _handle_setup(check=True)
        print_status.assert_called_once()
        setup_spies["sandbox_setup"].assert_not_called()
        setup_spies["build_images"].assert_not_called()


# ── Uninstall flow ────────────────────────────────────────────────────


class TestHandleUninstall:
    """``terok-executor uninstall`` mirrors setup in reverse."""

    def test_default_removes_images_then_sandbox(self, setup_spies) -> None:
        order: list[str] = []
        setup_spies["remove_images"].side_effect = lambda _base: order.append("images")
        setup_spies["sandbox_uninstall"].side_effect = lambda root: order.append("sandbox")

        _handle_uninstall()

        assert order == ["images", "sandbox"]

    def test_keep_images_preserves_image_cache(self, setup_spies) -> None:
        _handle_uninstall(keep_images=True)
        setup_spies["remove_images"].assert_not_called()
        setup_spies["sandbox_uninstall"].assert_called_once_with(root=False)

    def test_no_sandbox_skips_sandbox_teardown(self, setup_spies) -> None:
        _handle_uninstall(no_sandbox=True)
        setup_spies["remove_images"].assert_called_once()
        setup_spies["sandbox_uninstall"].assert_not_called()

    def test_root_flag_propagates_to_sandbox_uninstall(self, setup_spies) -> None:
        _handle_uninstall(root=True)
        setup_spies["sandbox_uninstall"].assert_called_once_with(root=True)


# ── Preflight gate ────────────────────────────────────────────────────


class TestPreflightOrExit:
    """The TTY-aware gatekeeper wrapping ``run_preflight``."""

    def test_no_preflight_short_circuits(self) -> None:
        """``--no-preflight`` returns True without running any check."""
        with patch("terok_executor.preflight.run_preflight") as mock_rp:
            result = _preflight_or_exit(
                "claude", base="ubuntu:24.04", family=None, assume_yes=False, skip_preflight=True
            )
        assert result is True
        mock_rp.assert_not_called()

    def test_non_tty_without_yes_refuses(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Non-TTY + no ``--yes`` → False, with a pointer to setup."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("terok_executor.preflight.run_preflight") as mock_rp,
        ):
            mock_stdin.isatty.return_value = False
            result = _preflight_or_exit(
                "claude", base="ubuntu:24.04", family=None, assume_yes=False, skip_preflight=False
            )
        assert result is False
        mock_rp.assert_not_called()
        assert "terok-executor setup" in capsys.readouterr().err

    def test_non_tty_with_yes_runs_preflight(self) -> None:
        """Non-TTY is fine when ``--yes`` drives the prompts."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("terok_executor.preflight.run_preflight", return_value=True) as mock_rp,
        ):
            mock_stdin.isatty.return_value = False
            result = _preflight_or_exit(
                "claude", base="ubuntu:24.04", family=None, assume_yes=True, skip_preflight=False
            )
        assert result is True
        mock_rp.assert_called_once()
        assert mock_rp.call_args.kwargs["assume_yes"] is True

    def test_tty_runs_preflight(self) -> None:
        """TTY without ``--yes`` still runs preflight interactively."""
        with (
            patch("sys.stdin") as mock_stdin,
            patch("terok_executor.preflight.run_preflight", return_value=True) as mock_rp,
        ):
            mock_stdin.isatty.return_value = True
            result = _preflight_or_exit(
                "claude", base="ubuntu:24.04", family=None, assume_yes=False, skip_preflight=False
            )
        assert result is True
        mock_rp.assert_called_once()
        assert mock_rp.call_args.kwargs["interactive"] is True
