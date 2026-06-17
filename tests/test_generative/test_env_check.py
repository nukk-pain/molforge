# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import molforge.generative._env_check as env_check_module  # noqa: E402

default_install_command = env_check_module.default_install_command
probe_reinvent_environment = env_check_module.probe_reinvent_environment


def test_default_install_command_prefers_mac_path_for_arm64() -> None:
    assert (
        default_install_command(system="darwin", machine="arm64")
        == "python install.py cpu"
    )
    assert (
        default_install_command(system="linux", machine="x86_64")
        == "python install.py cpu"
    )


def test_probe_reinvent_environment_reports_missing_prior(monkeypatch) -> None:
    monkeypatch.delenv("REINVENT4_PRIOR_PATH", raising=False)
    monkeypatch.delenv("REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.delenv("MOLFORGE_REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.setattr(
        env_check_module,
        "DEFAULT_REPO_PRIOR_PATH",
        Path("/tmp/definitely-missing-reinvent.prior"),
    )
    monkeypatch.setattr(
        env_check_module,
        "EXTERNAL_REINVENT_CLI",
        Path("/tmp/definitely-missing-reinvent-cli"),
    )
    monkeypatch.setattr(
        env_check_module,
        "EXTERNAL_REINVENT_PACKAGE_INIT",
        Path("/tmp/definitely-missing-reinvent-package/__init__.py"),
    )

    env = probe_reinvent_environment(which_resolver=lambda _: None)

    assert env.ready is False
    assert "missing_reinvent_cli" in env.blocking_reasons
    assert "missing_reinvent_prior" in env.blocking_reasons


def test_probe_reinvent_environment_accepts_repo_local_external_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    external_cli = tmp_path / "bin" / "reinvent"
    external_cli.parent.mkdir(parents=True, exist_ok=True)
    external_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    external_pkg = tmp_path / "site-packages" / "reinvent" / "__init__.py"
    external_pkg.parent.mkdir(parents=True, exist_ok=True)
    external_pkg.write_text("__version__ = 'test'\n", encoding="utf-8")
    prior_path = tmp_path / "vendor" / "reinvent.prior"
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text("prior\n", encoding="utf-8")

    monkeypatch.delenv("REINVENT4_PRIOR_PATH", raising=False)
    monkeypatch.delenv("REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.delenv("MOLFORGE_REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.setattr(env_check_module, "EXTERNAL_REINVENT_CLI", external_cli)
    monkeypatch.setattr(
        env_check_module, "EXTERNAL_REINVENT_PACKAGE_INIT", external_pkg
    )
    monkeypatch.setattr(env_check_module, "DEFAULT_REPO_PRIOR_PATH", prior_path)

    env = probe_reinvent_environment(which_resolver=lambda _: None)

    assert env.ready is True
    assert env.cli_path == str(external_cli)
    assert env.package_origin == str(external_pkg)
    assert env.prior_path == str(prior_path)
