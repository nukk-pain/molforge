from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path

VINA_INSTALL_HINT = "conda install -c bioconda vina or pip install vina"
FPOCKET_INSTALL_HINT = "conda install -c bioconda fpocket"


@dataclass(frozen=True, slots=True)
class EnvReport:
    vina_path: str | None
    fpocket_path: str | None
    meeko_available: bool
    geometry_fallback: bool

    @property
    def vina_available(self) -> bool:
        return self.vina_path is not None

    @property
    def fpocket_available(self) -> bool:
        return self.fpocket_path is not None


def which_binary(name: str) -> Path | None:
    resolved = shutil.which(name)
    return None if resolved is None else Path(resolved)


def check_env() -> EnvReport:
    vina_path = which_binary("vina")
    fpocket_path = which_binary("fpocket")
    return EnvReport(
        vina_path=None if vina_path is None else str(vina_path),
        fpocket_path=None if fpocket_path is None else str(fpocket_path),
        meeko_available=_can_import_meeko(),
        geometry_fallback=fpocket_path is None,
    )


def require_vina_binary() -> Path:
    vina_path = which_binary("vina")
    if vina_path is None:
        raise RuntimeError(
            f"AutoDock Vina binary not found. Install it with {VINA_INSTALL_HINT}."
        )
    return vina_path


def require_meeko_available() -> None:
    try:
        importlib.import_module("meeko")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Meeko and its import-time dependencies are not available. Install docking dependencies with "
            "`uv sync --extra docking`."
        ) from exc


def _can_import_meeko() -> bool:
    try:
        importlib.import_module("meeko")
        return True
    except ModuleNotFoundError:
        return False
