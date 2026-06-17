from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DiffSBDDBackend",
    "MolforgeGenerativeModule",
    "REINVENT4Backend",
    "build_default_backend",
    "generate_molecules",
]


def __getattr__(name: str) -> Any:
    if name == "DiffSBDDBackend":
        return getattr(import_module(".diffsbdd_stub", __name__), name)
    if name in {
        "MolforgeGenerativeModule",
        "build_default_backend",
        "generate_molecules",
    }:
        return getattr(import_module(".module", __name__), name)
    if name == "REINVENT4Backend":
        return getattr(import_module(".reinvent", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
