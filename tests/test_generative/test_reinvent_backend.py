# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import BindingPocket, ProteinStructure, StructureSource  # noqa: E402
from molforge.generative.reinvent import REINVENT4Backend  # noqa: E402


def build_pocket() -> BindingPocket:
    return BindingPocket(
        structure=ProteinStructure(
            gene="CXCR4",
            uniprot="P61073",
            pdb_path="/tmp/cxcr4.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=88.0,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(12.0, 12.0, 12.0),
        druggability_score=0.7,
        residues=["ASP97"],
    )


def test_reinvent_backend_writes_run_artifacts_and_filters_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "prior.ckpt"
    prior_path.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setenv("REINVENT4_PRIOR_PATH", str(prior_path))

    def fake_spec(name: str):
        if name == "reinvent":
            return type("Spec", (), {"origin": "/fake/reinvent/__init__.py"})()
        return None

    monkeypatch.setattr("importlib.util.find_spec", fake_spec)

    def fake_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        (cwd / "sampling.log").write_text("ok\n", encoding="utf-8")
        (cwd / "generated.smi").write_text(
            "CCN\nCCO\nCCC\nCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    backend = REINVENT4Backend(
        workspace_root=tmp_path,
        reference_smiles=["CCO"],
        prior_path=prior_path,
        command_executor=fake_executor,
        which_resolver=lambda _: "/fake/bin/reinvent",
    )

    molecules = backend.generate(build_pocket(), n=5)

    assert len(molecules) >= 2
    assert molecules[0].pocket_ref is not None
    assert backend.last_run_artifacts is not None
    metadata = json.loads(
        Path(backend.last_run_artifacts.metadata_path).read_text(encoding="utf-8")
    )
    assert metadata["pocket_semantics"] == "accepted-not-conditioned-v1"
    command_log = Path(backend.last_run_artifacts.command_log_path).read_text(
        encoding="utf-8"
    )
    assert "resolved_cli_path=/fake/bin/reinvent" in command_log
    assert metadata["attempts"]


def test_reinvent_backend_uses_oversample_factor_in_sampling_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "prior.ckpt"
    prior_path.write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setenv("REINVENT4_PRIOR_PATH", str(prior_path))
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: (
            type("Spec", (), {"origin": "/fake/reinvent/__init__.py"})()
            if name == "reinvent"
            else None
        ),
    )

    def fake_executor(
        command: Sequence[str],
        timeout_seconds: float,
        cwd: Path | None,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        config_text = (cwd / "sampling.toml").read_text(encoding="utf-8")
        assert "num_smiles = 25" in config_text
        (cwd / "sampling.log").write_text("ok\n", encoding="utf-8")
        (cwd / "generated.smi").write_text("CCN\nCCO\nCCC\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    backend = REINVENT4Backend(
        workspace_root=tmp_path,
        reference_smiles=["CNC"],
        prior_path=prior_path,
        command_executor=fake_executor,
        which_resolver=lambda _: "/fake/bin/reinvent",
        observed_pass_rate=0.2,
    )

    _ = backend.generate(build_pocket(), n=5)


def test_reinvent_backend_refuses_to_fake_live_run_when_environment_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REINVENT4_PRIOR_PATH", raising=False)
    monkeypatch.delenv("REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.delenv("MOLFORGE_REINVENT_PRIOR_PATH", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    # v2 env_check has on-disk fallbacks (`vendor/reinvent_priors/`,
    # `.uv/phase0-reinvent4-mac/`). Force them to missing so the "environment
    # is missing" scenario is reproducible on a dev repo that has them.
    monkeypatch.setattr(
        "molforge.generative._env_check.DEFAULT_REPO_PRIOR_PATH",
        tmp_path / "missing-prior.prior",
    )
    monkeypatch.setattr(
        "molforge.generative._env_check._resolve_external_cli_path",
        lambda: None,
    )
    monkeypatch.setattr(
        "molforge.generative._env_check.EXTERNAL_REINVENT_PACKAGE_INIT",
        tmp_path / "missing-package" / "__init__.py",
    )

    backend = REINVENT4Backend(
        workspace_root=tmp_path,
        reference_smiles=["CCO"],
        which_resolver=lambda _: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend.generate(build_pocket(), n=10)

    assert "missing_reinvent_package" in str(exc_info.value)
    assert backend.last_run_artifacts is not None
    metadata = json.loads(
        Path(backend.last_run_artifacts.metadata_path).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "blocked"
