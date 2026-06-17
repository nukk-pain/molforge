from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .reinvent4_poc import AUTHORITATIVE_CLI_COMMAND

WhichResolver = Callable[[str], str | None]
DEFAULT_PRIOR_ENV_VARS = (
    "REINVENT4_PRIOR_PATH",
    "REINVENT_PRIOR_PATH",
    "MOLFORGE_REINVENT_PRIOR_PATH",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_REINVENT_CLI = REPO_ROOT / ".uv" / "phase0-reinvent4-mac" / "bin" / "reinvent"
EXTERNAL_REINVENT_PACKAGE_INIT = (
    REPO_ROOT
    / ".uv"
    / "phase0-reinvent4-mac"
    / "lib"
    / "python3.11"
    / "site-packages"
    / "reinvent"
    / "__init__.py"
)
DEFAULT_REPO_PRIOR_PATH = REPO_ROOT / "vendor" / "reinvent_priors" / "reinvent.prior"


@dataclass(frozen=True, slots=True)
class ReinventEnvironment:
    platform: str
    machine: str
    install_command: str
    sampling_command: str
    package_found: bool
    package_origin: str | None
    cli_path: str | None
    prior_path: str | None
    ready: bool
    blocking_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_reinvent_environment(
    *,
    prior_path: str | Path | None = None,
    which_resolver: WhichResolver = shutil.which,
) -> ReinventEnvironment:
    system = platform.system().lower()
    machine = platform.machine().lower()
    install_command = default_install_command(system=system, machine=machine)
    cli_path = which_resolver("reinvent") or _resolve_external_cli_path()
    package_spec = importlib.util.find_spec("reinvent")
    package_origin = None if package_spec is None else package_spec.origin
    if package_origin is None and EXTERNAL_REINVENT_PACKAGE_INIT.exists():
        package_origin = str(EXTERNAL_REINVENT_PACKAGE_INIT)
    resolved_prior = resolve_prior_path(prior_path)
    blocking_reasons: list[str] = []
    if package_origin is None:
        blocking_reasons.append("missing_reinvent_package")
    if cli_path is None:
        blocking_reasons.append("missing_reinvent_cli")
    if resolved_prior is None:
        blocking_reasons.append("missing_reinvent_prior")
    return ReinventEnvironment(
        platform=system,
        machine=machine,
        install_command=install_command,
        sampling_command=AUTHORITATIVE_CLI_COMMAND,
        package_found=package_origin is not None,
        package_origin=package_origin,
        cli_path=cli_path,
        prior_path=None if resolved_prior is None else str(resolved_prior),
        ready=not blocking_reasons,
        blocking_reasons=blocking_reasons,
    )


def _resolve_external_cli_path() -> str | None:
    if EXTERNAL_REINVENT_CLI.exists():
        return str(EXTERNAL_REINVENT_CLI)
    return None


def resolve_prior_path(prior_path: str | Path | None) -> Path | None:
    if prior_path is not None:
        candidate = Path(prior_path).expanduser()
        return candidate if candidate.exists() else None
    for env_name in DEFAULT_PRIOR_ENV_VARS:
        value = os.environ.get(env_name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.exists():
            return candidate
    if DEFAULT_REPO_PRIOR_PATH.exists():
        return DEFAULT_REPO_PRIOR_PATH
    return None


def default_install_command(*, system: str, machine: str) -> str:
    return "python install.py cpu"
