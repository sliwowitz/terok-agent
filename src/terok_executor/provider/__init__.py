# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""AI provider behavior — provider definitions, headless modes, wrapper generation, instructions.

Delegates to `.providers` for the agent provider registry and environment
collection, `.wrappers` for shell wrapper generation, `.headless`
for headless command construction and config resolution, `.config` for
provider-aware config value extraction, `.instructions` for per-provider
instruction resolution, and `.agents` for agent config directory
preparation and wrapper scripts.
"""
