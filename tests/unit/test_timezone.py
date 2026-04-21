# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for host timezone detection.

Each source (``$TZ``, ``/etc/timezone``, ``/etc/localtime`` symlink) is
exercised by swapping the module's ``Path`` for a stub that routes the
two hardcoded probes to test-owned files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_executor._util import _timezone
from terok_executor._util._timezone import detect_host_timezone


@pytest.fixture
def no_tz_env(monkeypatch):
    """Clear ``$TZ`` so fallback sources are reached."""
    monkeypatch.delenv("TZ", raising=False)


def _route_paths(timezone_file: Path | None, localtime_target: Path | None):
    """Return a ``Path`` stub that swaps in test files for the two probed locations.

    ``timezone_file=None`` makes the ``/etc/timezone`` read raise; likewise
    ``localtime_target=None`` makes the ``/etc/localtime`` resolve raise.
    """
    missing = Path("/this/does/not/exist")

    class _LocaltimeStub:
        @staticmethod
        def resolve():
            if localtime_target is None:
                raise OSError("missing")
            return localtime_target

    def _factory(arg):
        if arg == "/etc/timezone":
            return timezone_file if timezone_file is not None else missing
        if arg == "/etc/localtime":
            return _LocaltimeStub()
        raise AssertionError(f"unexpected Path({arg!r}) in timezone probe")

    return _factory


class TestEnvVar:
    """``$TZ`` wins over every other source."""

    def test_env_var_wins_over_filesystem(self, monkeypatch):
        monkeypatch.setenv("TZ", "Europe/Prague")
        with patch.object(_timezone, "Path", side_effect=AssertionError("should not read")):
            assert detect_host_timezone() == "Europe/Prague"


class TestEtcTimezone:
    """``/etc/timezone`` supplies the zone as a single line on Debian/Ubuntu."""

    def test_reads_zone_name(self, no_tz_env, tmp_path):
        zone_file = tmp_path / "timezone"
        zone_file.write_text("Europe/Prague\n")
        with patch.object(_timezone, "Path", side_effect=_route_paths(zone_file, None)):
            assert detect_host_timezone() == "Europe/Prague"

    def test_blank_file_falls_through_to_localtime(self, no_tz_env, tmp_path):
        blank = tmp_path / "timezone"
        blank.write_text("   \n")
        target = Path("/usr/share/zoneinfo/UTC")
        with patch.object(_timezone, "Path", side_effect=_route_paths(blank, target)):
            assert detect_host_timezone() == "UTC"


class TestEtcLocaltime:
    """``/etc/localtime`` symlink supplies the zone via its resolved target."""

    def test_resolves_symlink_target(self, no_tz_env):
        target = Path("/usr/share/zoneinfo/Asia/Tokyo")
        with patch.object(_timezone, "Path", side_effect=_route_paths(None, target)):
            assert detect_host_timezone() == "Asia/Tokyo"

    def test_macos_layout(self, no_tz_env):
        """macOS symlinks into ``/var/db/timezone/zoneinfo/<zone>``."""
        target = Path("/var/db/timezone/zoneinfo/America/New_York")
        with patch.object(_timezone, "Path", side_effect=_route_paths(None, target)):
            assert detect_host_timezone() == "America/New_York"


class TestUnresolvable:
    """When every source is silent the helper returns ``None`` — never guesses."""

    def test_all_sources_missing(self, no_tz_env):
        with patch.object(_timezone, "Path", side_effect=_route_paths(None, None)):
            assert detect_host_timezone() is None

    def test_localtime_target_not_in_zoneinfo(self, no_tz_env):
        """A ``/etc/localtime`` that points outside a ``zoneinfo/`` dir is unusable."""
        target = Path("/some/random/path")
        with patch.object(_timezone, "Path", side_effect=_route_paths(None, target)):
            assert detect_host_timezone() is None
