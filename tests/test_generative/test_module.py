# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (
    BindingPocket,
    GeneratedMolecule,
    ProteinStructure,
    StructureSource,
)  # noqa: E402
from molforge.generative.module import (  # noqa: E402
    MolforgeGenerativeModule,
    generate_molecules,
    load_vendor_reference_smiles,
)


class FakeBackend:
    name = "reinvent4"
    last_run_artifacts = type("Artifacts", (), {"run_dir": "/tmp/fake-run"})()

    def generate(
        self, pocket: BindingPocket, n: int = 100, seed_smiles: str | None = None
    ):
        return [
            GeneratedMolecule(
                smiles="CCN",
                qed=0.8,
                sa_score=2.1,
                novelty=0.7,
                backend="reinvent4",
                pocket_ref=pocket,
            )
        ]


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[GeneratedMolecule] = []

    def save_molecule(self, run_id: str, molecule: GeneratedMolecule) -> int:
        _ = run_id
        self.saved.append(molecule)
        return len(self.saved)


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
        size_xyz=(10.0, 10.0, 10.0),
        druggability_score=0.7,
        residues=["ASP97"],
    )


def test_load_vendor_reference_smiles_reads_seed_library() -> None:
    reference_smiles = load_vendor_reference_smiles()

    assert "CCO" in reference_smiles
    assert len(reference_smiles) >= 5


def test_generate_molecules_writes_summary_and_molecule_artifacts(
    tmp_path: Path,
) -> None:
    result = generate_molecules(
        build_pocket(),
        output_dir=tmp_path,
        n=10,
        backend=FakeBackend(),
    )

    summary_payload = json.loads(
        (tmp_path / "generation-summary.json").read_text(encoding="utf-8")
    )
    molecules_payload = json.loads(
        (tmp_path / "generated-molecules.json").read_text(encoding="utf-8")
    )

    assert result.returned_count == 1
    assert summary_payload["backend"] == "reinvent4"
    assert molecules_payload[0]["smiles"] == "CCN"


def test_module_run_fans_out_and_persists_per_pocket() -> None:
    module = MolforgeGenerativeModule(FakeBackend())
    store = FakeStore()
    pocket_a = build_pocket()
    pocket_b = BindingPocket(
        structure=ProteinStructure(
            gene="TGFB1",
            uniprot="P01137",
            pdb_path="/tmp/tgfb1.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=77.0,
        ),
        center_xyz=(4.0, 5.0, 6.0),
        size_xyz=(10.0, 10.0, 10.0),
        druggability_score=0.5,
        residues=["TYR1"],
    )

    molecules = module.run([pocket_a, pocket_b], store=store, run_id="run-1")

    assert len(molecules) == 2
    assert len(store.saved) == 2
    assert store.saved[0].pocket_ref is not None
    assert store.saved[1].pocket_ref is not None
    assert (
        store.saved[0].pocket_ref.structure.gene
        != store.saved[1].pocket_ref.structure.gene
    )
