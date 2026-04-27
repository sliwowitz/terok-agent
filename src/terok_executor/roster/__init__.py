# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Loads agent and tool definitions from layered YAML config into a queryable roster.

Delegates to [`.loader`][] for YAML deserialization and roster construction,
and to [`.config_stack`][] for generic layered config resolution.
"""

from .loader import (
    AgentRoster,
    MountDef,
    SidecarSpec,
    VaultRoute,
    ensure_vault_routes,
    get_roster,
    load_roster,
    parse_agent_selection,
)

__all__ = [
    "AgentRoster",
    "MountDef",
    "SidecarSpec",
    "VaultRoute",
    "ensure_vault_routes",
    "get_roster",
    "load_roster",
    "parse_agent_selection",
]
