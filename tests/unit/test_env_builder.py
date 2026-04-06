# SPDX-FileCopyrightText: 2026 Jiri Vyskocil
# SPDX-License-Identifier: Apache-2.0

"""Tests for the container environment assembly function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from terok_agent.env_builder import (
    ContainerEnvResult,
    ContainerEnvSpec,
    _resolve_git_identity,
    _shared_config_mounts,
    assemble_container_env,
)
from terok_agent.roster import get_roster


@pytest.fixture
def roster():
    """Return the live agent roster (loaded from bundled YAML)."""
    return get_roster()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a pre-created workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def base_spec(workspace: Path) -> ContainerEnvSpec:
    """Minimal spec with only required fields."""
    return ContainerEnvSpec(
        task_id="test-123",
        provider_name="claude",
        workspace_host_path=workspace,
    )


# ---------------------------------------------------------------------------
# assemble_container_env — base env
# ---------------------------------------------------------------------------


class TestBaseEnv:
    """Verify base environment variables are always set."""

    def test_task_id(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.env["TASK_ID"] == "test-123"

    def test_repo_root(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.env["REPO_ROOT"] == "/workspace"

    def test_git_reset_mode(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.env["GIT_RESET_MODE"] == "none"

    def test_claude_config_dir(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.env["CLAUDE_CONFIG_DIR"] == "/home/dev/.claude"

    def test_returns_frozen_result(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert isinstance(result, ContainerEnvResult)


# ---------------------------------------------------------------------------
# Git identity
# ---------------------------------------------------------------------------


class TestGitIdentity:
    """Verify git identity resolution from spec fields or roster fallback."""

    def test_identity_from_roster_provider(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["GIT_AUTHOR_NAME"] == "Claude"
        assert result.env["GIT_AUTHOR_EMAIL"] == "noreply@anthropic.com"
        # Standalone default: committer = author (no human specified)
        assert result.env["GIT_COMMITTER_NAME"] == "Claude"

    def test_explicit_identity_overrides_roster(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            git_author_name="Human Author",
            git_author_email="human@example.com",
            git_committer_name="AI Committer",
            git_committer_email="ai@example.com",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["GIT_AUTHOR_NAME"] == "Human Author"
        assert result.env["GIT_AUTHOR_EMAIL"] == "human@example.com"
        assert result.env["GIT_COMMITTER_NAME"] == "AI Committer"
        assert result.env["GIT_COMMITTER_EMAIL"] == "ai@example.com"

    def test_committer_defaults_to_author(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            git_author_name="Custom",
            git_author_email="custom@test.com",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["GIT_COMMITTER_NAME"] == "Custom"
        assert result.env["GIT_COMMITTER_EMAIL"] == "custom@test.com"

    def test_unknown_provider_uses_fallback(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="nonexistent",
            workspace_host_path=workspace,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["GIT_AUTHOR_NAME"] == "AI Agent"


# ---------------------------------------------------------------------------
# Authorship env
# ---------------------------------------------------------------------------


class TestAuthorship:
    """Verify authorship mode and human identity env vars."""

    def test_defaults(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.env["TEROK_GIT_AUTHORSHIP"] == "agent"
        assert result.env["HUMAN_GIT_NAME"] == "Nobody"
        assert result.env["HUMAN_GIT_EMAIL"] == "nobody@localhost"

    def test_custom_authorship(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            authorship="agent-human",
            human_name="Jane Doe",
            human_email="jane@example.com",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["TEROK_GIT_AUTHORSHIP"] == "agent-human"
        assert result.env["HUMAN_GIT_NAME"] == "Jane Doe"
        assert result.env["HUMAN_GIT_EMAIL"] == "jane@example.com"


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------


class TestRepoSetup:
    """Verify repository env vars and branch."""

    def test_code_repo(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            code_repo="http://gate@host:9418/repo",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["CODE_REPO"] == "http://gate@host:9418/repo"

    def test_clone_from(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            clone_from="http://gate@host:9418/mirror",
            code_repo="https://github.com/user/repo",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["CLONE_FROM"] == "http://gate@host:9418/mirror"
        assert result.env["CODE_REPO"] == "https://github.com/user/repo"

    def test_branch(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            branch="feat/my-branch",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["GIT_BRANCH"] == "feat/my-branch"

    def test_no_branch_omits_key(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert "GIT_BRANCH" not in result.env

    def test_no_code_repo_omits_key(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert "CODE_REPO" not in result.env


# ---------------------------------------------------------------------------
# Workspace volume
# ---------------------------------------------------------------------------


class TestWorkspaceVolume:
    """Verify workspace volume mount."""

    def test_workspace_mounted_with_exclusive_label(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        ws_mount = f"{base_spec.workspace_host_path}:/workspace:Z"
        assert ws_mount in result.volumes


# ---------------------------------------------------------------------------
# Shared config mounts
# ---------------------------------------------------------------------------


class TestSharedConfigMounts:
    """Verify roster-derived shared config mounts."""

    def test_claude_config_mounted(self, base_spec, roster, tmp_path):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=base_spec.workspace_host_path,
            envs_dir=tmp_path / "mounts",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        mount_str = " ".join(result.volumes)
        assert "_claude-config" in mount_str
        assert "/home/dev/.claude:z" in mount_str

    def test_shared_mounts_use_lowercase_z(self, base_spec, roster, tmp_path):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=base_spec.workspace_host_path,
            envs_dir=tmp_path / "mounts",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        shared = [v for v in result.volumes if ":z" in v and ":Z" not in v]
        assert len(shared) > 0

    def test_host_dirs_created(self, base_spec, roster, tmp_path):
        mounts_base = tmp_path / "mounts"
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=base_spec.workspace_host_path,
            envs_dir=mounts_base,
        )
        assemble_container_env(spec, roster, proxy_bypass=True)
        assert (mounts_base / "_claude-config").is_dir()


# ---------------------------------------------------------------------------
# Agent config mount
# ---------------------------------------------------------------------------


class TestAgentConfigMount:
    """Verify agent config directory mount."""

    def test_agent_config_mounted_when_set(self, workspace, roster, tmp_path):
        cfg_dir = tmp_path / "agent-config"
        cfg_dir.mkdir()
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            agent_config_dir=cfg_dir,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        expected = f"{cfg_dir}:/home/dev/.terok:Z"
        assert expected in result.volumes

    def test_no_agent_config_when_none(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert not any("/home/dev/.terok" in v for v in result.volumes)


# ---------------------------------------------------------------------------
# Unrestricted mode
# ---------------------------------------------------------------------------


class TestUnrestrictedMode:
    """Verify unrestricted/auto-approve env injection."""

    def test_unrestricted_sets_env(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            unrestricted=True,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.env["TEROK_UNRESTRICTED"] == "1"

    def test_restricted_omits_env(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            unrestricted=False,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert "TEROK_UNRESTRICTED" not in result.env


# ---------------------------------------------------------------------------
# Shared task directory
# ---------------------------------------------------------------------------


class TestSharedTaskDir:
    """Verify shared task directory mount and env var."""

    def test_shared_dir_mounted_when_set(self, workspace, roster, tmp_path):
        shared = tmp_path / "shared"
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            shared_dir=shared,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert f"{shared}:/shared:z" in result.volumes
        assert result.env["TEROK_SHARED_DIR"] == "/shared"

    def test_shared_dir_custom_mount(self, workspace, roster, tmp_path):
        shared = tmp_path / "data"
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            shared_dir=shared,
            shared_mount="/data/ipc",
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert f"{shared}:/data/ipc:z" in result.volumes
        assert result.env["TEROK_SHARED_DIR"] == "/data/ipc"

    def test_shared_dir_created(self, workspace, roster, tmp_path):
        shared = tmp_path / "new-shared"
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            shared_dir=shared,
        )
        assemble_container_env(spec, roster, proxy_bypass=True)
        assert shared.is_dir()

    def test_no_shared_dir_by_default(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert "TEROK_SHARED_DIR" not in result.env
        assert not any("/shared" in v for v in result.volumes)


# ---------------------------------------------------------------------------
# Extra volumes
# ---------------------------------------------------------------------------


class TestExtraVolumes:
    """Verify caller-provided extra volumes are appended."""

    def test_extra_volumes_appended(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            extra_volumes=("/host/ssh:/home/dev/.ssh:z",),
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert "/host/ssh:/home/dev/.ssh:z" in result.volumes


# ---------------------------------------------------------------------------
# Credential proxy
# ---------------------------------------------------------------------------


class TestCredentialProxy:
    """Verify credential proxy token injection."""

    def test_proxy_bypass_skips_injection(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert "TEROK_PROXY_PORT" not in result.env

    def test_proxy_not_running_returns_no_tokens(self, base_spec, roster):
        with (
            patch("terok_sandbox.is_proxy_socket_active", return_value=False),
            patch("terok_sandbox.is_proxy_running", return_value=False),
        ):
            result = assemble_container_env(base_spec, roster, proxy_bypass=False)
        assert "TEROK_PROXY_PORT" not in result.env

    def test_proxy_running_injects_tokens(self, workspace, roster, tmp_path):
        from terok_sandbox import CredentialDB, SandboxConfig

        cfg = SandboxConfig(state_dir=tmp_path, credentials_dir=tmp_path / "credentials")
        cfg.proxy_db_path.parent.mkdir(parents=True, exist_ok=True)
        db = CredentialDB(cfg.proxy_db_path)
        db.store_credential("default", "claude", {"type": "api_key", "key": "sk-test"})
        db.close()

        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            credential_scope="test-project",
        )

        with (
            patch("terok_sandbox.is_proxy_socket_active", return_value=False),
            patch("terok_sandbox.is_proxy_running", return_value=True),
            patch("terok_sandbox.SandboxConfig", return_value=cfg),
        ):
            result = assemble_container_env(spec, roster, proxy_bypass=False)

        assert "ANTHROPIC_API_KEY" in result.env
        assert result.env["ANTHROPIC_API_KEY"].startswith("terok-p-")


# ---------------------------------------------------------------------------
# Task dir
# ---------------------------------------------------------------------------


class TestTaskDir:
    """Verify task_dir resolution."""

    def test_explicit_task_dir(self, workspace, roster, tmp_path):
        td = tmp_path / "my-task"
        td.mkdir()
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            task_dir=td,
        )
        result = assemble_container_env(spec, roster, proxy_bypass=True)
        assert result.task_dir == td

    def test_auto_creates_temp_dir(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        assert result.task_dir.exists()
        assert "terok-agent-test-123" in str(result.task_dir)


# ---------------------------------------------------------------------------
# OpenCode provider env
# ---------------------------------------------------------------------------


class TestOpenCodeEnv:
    """Verify OpenCode provider env vars from roster."""

    def test_opencode_vars_present(self, base_spec, roster):
        result = assemble_container_env(base_spec, roster, proxy_bypass=True)
        # Blablador/KISSKI contribute TEROK_OC_* vars
        oc_vars = [k for k in result.env if k.startswith("TEROK_OC_")]
        assert len(oc_vars) > 0


# ---------------------------------------------------------------------------
# _resolve_git_identity (unit)
# ---------------------------------------------------------------------------


class TestResolveGitIdentityUnit:
    """Unit tests for the internal git identity resolver."""

    def test_spec_fields_take_precedence(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
            git_author_name="Override",
        )
        identity = _resolve_git_identity(spec, roster)
        assert identity["GIT_AUTHOR_NAME"] == "Override"
        # Committer defaults to author when not specified
        assert identity["GIT_COMMITTER_NAME"] == "Override"

    def test_roster_fallback(self, workspace, roster):
        spec = ContainerEnvSpec(
            task_id="t1",
            provider_name="claude",
            workspace_host_path=workspace,
        )
        identity = _resolve_git_identity(spec, roster)
        assert identity["GIT_AUTHOR_NAME"] == "Claude"


# ---------------------------------------------------------------------------
# _shared_config_mounts (unit)
# ---------------------------------------------------------------------------


class TestSharedConfigMountsUnit:
    """Unit tests for the internal shared mount builder."""

    def test_creates_host_dirs(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        assert len(mounts) > 0
        assert (tmp_path / "_claude-config").is_dir()

    def test_deduplicates_by_host_dir(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        host_dirs = [m.split(":")[0] for m in mounts]
        assert len(host_dirs) == len(set(host_dirs))

    def test_all_use_shared_label(self, roster, tmp_path):
        mounts = _shared_config_mounts(roster, tmp_path)
        for m in mounts:
            assert m.endswith(":z"), f"Expected :z label, got: {m}"
