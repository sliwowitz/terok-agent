# SPDX-FileCopyrightText: 2025 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests — verify the package is importable and wired correctly."""

import importlib


class TestPackageImport:
    """Basic import checks."""

    def test_import_terok_agent(self) -> None:
        """Package root is importable."""
        mod = importlib.import_module("terok_agent")
        assert hasattr(mod, "__version__")

    def test_import_cli(self) -> None:
        """CLI module is importable."""
        mod = importlib.import_module("terok_agent.cli")
        assert hasattr(mod, "main")

    def test_version_is_string(self) -> None:
        """__version__ is a non-empty string."""
        from terok_agent import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
