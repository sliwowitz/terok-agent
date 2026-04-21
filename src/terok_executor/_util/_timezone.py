# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Detects the host's IANA timezone for propagation into containers.

Returned as a plain string (``"Europe/Prague"``, ``"UTC"``, …) suitable
for use as a ``TZ`` env var inside the container — glibc resolves it
against ``/usr/share/zoneinfo`` without needing the host's filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

_ZONEINFO_MARKER = "/zoneinfo/"


def detect_host_timezone() -> str | None:
    """Return the host's IANA timezone name, or ``None`` if it can't be detected.

    Tried in order:

    1. ``$TZ`` — the user's explicit override.
    2. ``/etc/timezone`` — Debian/Ubuntu convention, single-line zone name.
    3. ``/etc/localtime`` symlink — systemd-family hosts (and macOS) symlink
       this into the zoneinfo database; the zone name is the path suffix
       after the ``zoneinfo/`` component.

    Returns ``None`` on hosts that expose none of the above (containers with
    only a copied-in ``/etc/localtime`` file, for instance), letting the
    caller fall back to the image default rather than guessing.
    """
    if tz := os.environ.get("TZ"):
        return tz

    try:
        if zone := Path("/etc/timezone").read_text(encoding="utf-8").strip():
            return zone
    except OSError:
        pass

    try:
        target = Path("/etc/localtime").resolve().as_posix()
    except OSError:
        return None
    if _ZONEINFO_MARKER in target:
        return target.split(_ZONEINFO_MARKER, 1)[1]
    return None
