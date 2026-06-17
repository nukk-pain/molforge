from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from contracts.schema import BindingPocket, GeneratedMolecule, GenerativeBackend


@dataclass(frozen=True, slots=True)
class GenerationArtifacts:
    output_dir: str
    summary_path: str
    molecules_path: str
    backend_run_dir: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    backend: str
    requested_count: int
    returned_count: int
    pocket_gene: str
    artifacts: GenerationArtifacts
    molecules: list[GeneratedMolecule]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "requested_count": self.requested_count,
            "returned_count": self.returned_count,
            "pocket_gene": self.pocket_gene,
            "artifacts": self.artifacts.to_dict(),
            "molecules": [
                serialize_generated_molecule(item) for item in self.molecules
            ],
        }


def validate_backend(backend: GenerativeBackend) -> GenerativeBackend:
    if not getattr(backend, "name", "").strip():
        raise ValueError("Generative backend instances must expose a non-empty name.")
    if not callable(getattr(backend, "generate", None)):
        raise ValueError("Generative backend instances must implement generate().")
    return backend


def ensure_generated_molecules(
    molecules: list[GeneratedMolecule],
    *,
    backend_name: str,
    pocket: BindingPocket,
) -> list[GeneratedMolecule]:
    normalized: list[GeneratedMolecule] = []
    for molecule in molecules:
        normalized.append(
            GeneratedMolecule(
                smiles=molecule.smiles,
                qed=molecule.qed,
                sa_score=molecule.sa_score,
                novelty=molecule.novelty,
                backend=molecule.backend or backend_name,
                pocket_ref=molecule.pocket_ref or pocket,
            )
        )
    return normalized


def serialize_generated_molecule(molecule: GeneratedMolecule) -> dict[str, object]:
    payload: dict[str, object] = {
        "smiles": molecule.smiles,
        "qed": molecule.qed,
        "sa_score": molecule.sa_score,
        "novelty": molecule.novelty,
        "backend": molecule.backend,
        "pocket_ref": None,
    }
    if molecule.pocket_ref is not None:
        payload["pocket_ref"] = serialize_binding_pocket(molecule.pocket_ref)
    return payload


def serialize_binding_pocket(pocket: BindingPocket) -> dict[str, object]:
    return {
        "structure": {
            "gene": pocket.structure.gene,
            "uniprot": pocket.structure.uniprot,
            "pdb_path": pocket.structure.pdb_path,
            "source": pocket.structure.source.value,
            "confidence": pocket.structure.confidence,
        },
        "center_xyz": list(pocket.center_xyz),
        "size_xyz": list(pocket.size_xyz),
        "druggability_score": pocket.druggability_score,
        "residues": list(pocket.residues),
    }


def write_generation_result(path: Path, result: GenerationResult) -> None:
    _ = path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
