# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Container lifecycle — image building, environment assembly, agent launch.

Delegates to `.build` for Dockerfile rendering and ``podman build``,
`.env` for container environment and volume assembly, and
`.runner` for the high-level agent runner that composes all three.
"""
