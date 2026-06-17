from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Protocol, cast

from contracts.schema import BindingPocket, GeneratedMolecule, GenerativeBackend

from .backend import (
    GenerationArtifacts,
    GenerationResult,
    validate_backend,
    write_generation_result,
)
from .reinvent import REINVENT4Backend


class HasRunArtifacts(Protocol):
    run_dir: str


VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_LIBRARY_PATH = VENDOR_DIR / "reference_smiles.smi"
SAMPLING_TEMPLATE_PATH = VENDOR_DIR / "sampling.template.toml"
ROOT_REFERENCE_LIBRARY_PATH = REPO_ROOT / "vendor" / "ref_drug_like.smi"


class MoleculeStore(Protocol):
    def save_molecule(self, run_id: str, molecule: GeneratedMolecule) -> int: ...


def load_vendor_reference_smiles() -> list[str]:
    library_path = ROOT_REFERENCE_LIBRARY_PATH
    if not library_path.exists():
        library_path = REFERENCE_LIBRARY_PATH
    return [
        line.strip()
        for line in library_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_sampling_template() -> str:
    return SAMPLING_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_default_backend(
    *,
    output_dir: str | Path,
    prior_path: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> REINVENT4Backend:
    return REINVENT4Backend(
        workspace_root=Path(output_dir),
        reference_smiles=load_vendor_reference_smiles(),
        prior_path=prior_path,
        timeout_seconds=timeout_seconds,
        sampling_template=load_sampling_template(),
    )


class MolforgeGenerativeModule:
    def __init__(self, backend: GenerativeBackend) -> None:
        self.backend = validate_backend(backend)

    def run(
        self,
        pockets: list[BindingPocket],
        *,
        store: MoleculeStore,
        run_id: str,
        n_per_pocket: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]:
        if not pockets:
            return []

        sampled = self.backend.generate(
            pocket=pockets[0],
            n=n_per_pocket,
            seed_smiles=seed_smiles,
        )
        molecules: list[GeneratedMolecule] = []
        for pocket in pockets:
            seen: set[str] = set()
            for molecule in sampled:
                if molecule.smiles in seen:
                    continue
                seen.add(molecule.smiles)
                copied = replace(molecule, pocket_ref=pocket)
                _ = store.save_molecule(run_id, copied)
                molecules.append(copied)
        return molecules


def generate_molecules(
    pocket: BindingPocket,
    *,
    output_dir: str | Path,
    n: int = 100,
    seed_smiles: str | None = None,
    backend: GenerativeBackend | None = None,
) -> GenerationResult:
    output_path = Path(output_dir)
    _ = output_path.mkdir(parents=True, exist_ok=True)
    backend_instance = validate_backend(
        backend or build_default_backend(output_dir=output_path)
    )
    molecules = backend_instance.generate(pocket=pocket, n=n, seed_smiles=seed_smiles)
    summary_path = output_path / "generation-summary.json"
    molecules_path = output_path / "generated-molecules.json"
    _ = molecules_path.write_text(
        json.dumps(
            [
                {
                    "smiles": item.smiles,
                    "qed": item.qed,
                    "sa_score": item.sa_score,
                    "novelty": item.novelty,
                    "backend": item.backend,
                }
                for item in molecules
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    backend_run_dir: str | None = None
    run_artifacts = getattr(backend_instance, "last_run_artifacts", None)
    if run_artifacts is not None:
        backend_run_dir = cast(HasRunArtifacts, run_artifacts).run_dir

    artifacts = GenerationArtifacts(
        output_dir=str(output_path),
        summary_path=str(summary_path),
        molecules_path=str(molecules_path),
        backend_run_dir=backend_run_dir,
    )
    result = GenerationResult(
        backend=backend_instance.name,
        requested_count=n,
        returned_count=len(molecules),
        pocket_gene=pocket.structure.gene,
        artifacts=artifacts,
        molecules=molecules,
    )
    write_generation_result(summary_path, result)
    return result
