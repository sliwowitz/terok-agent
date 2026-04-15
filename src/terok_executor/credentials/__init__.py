# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Authenticates agents and vaults their credentials into sandboxed containers.

Delegates to :mod:`.auth` for auth provider registry and container-based
auth flows, :mod:`.extractors` for per-provider credential file parsing,
:mod:`.vault_commands` for vault CLI lifecycle, and :mod:`.vault_config`
for post-auth config file patching.
"""
