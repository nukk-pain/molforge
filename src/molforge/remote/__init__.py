from __future__ import annotations

import os

from .backend import JobHandle, JobResult, JobSpec, LocalMockBackend, RemoteGPUBackend
from .modal_backend import ModalBackend

__all__ = [
    "JobHandle",
    "JobResult",
    "JobSpec",
    "LocalMockBackend",
    "ModalBackend",
    "RemoteGPUBackend",
    "build_remote_backend",
]


def build_remote_backend() -> RemoteGPUBackend:
    backend_name = (os.environ.get("MOLFORGE_REMOTE_GPU_BACKEND") or "modal").strip()
    if backend_name == "local-mock":
        return LocalMockBackend()
    if backend_name == "modal":
        return ModalBackend()
    raise ValueError(
        "Unsupported MOLFORGE_REMOTE_GPU_BACKEND. Expected 'modal' or 'local-mock'."
    )
