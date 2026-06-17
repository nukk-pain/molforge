# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contracts.schema import (  # noqa: E402
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    DockingPose,
    GeneratedMolecule,
    Ligand,
    OffTargetHit,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)
from molforge.cli import main  # noqa: E402
from molforge.core.pipeline import run_pipeline as core_run_pipeline  # noqa: E402


def build_target() -> TargetCandidate:
    return TargetCandidate(
        gene="CXCR4",
        score=0.92,
        disease="scar pain",
        ncbi_id=7852,
        uniprot_id="P61073",
        evidence=[],
        pathway=["chemokine_signaling"],
        extra=None,
    )


def build_pocket(target: TargetCandidate) -> BindingPocket:
    return BindingPocket(
        structure=ProteinStructure(
            gene=target.gene,
            uniprot=target.uniprot_id,
            pdb_path="/tmp/cxcr4.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=88.0,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 10.0, 10.0),
        druggability_score=0.7,
        residues=["ASP97"],
    )


class FakeDockingRunner:
    def __init__(self, target: TargetCandidate) -> None:
        self.target = target
        self.pocket = build_pocket(target)

    def run_target(
        self, target: TargetCandidate, *, store, run_id: str, top_n: int = 50
    ):
        _ = store, run_id
        predictions = []
        for index in range(top_n):
            smiles = f"CCO{index}"
            predictions.append(
                AffinityPrediction(
                    ligand_smiles=smiles,
                    target_gene=target.gene,
                    vina_score=-(8.5 - (index * 0.05)),
                    affinity_log_ki=None,
                    affinity_confidence=None,
                    pose_ref=DockingPose(
                        ligand_smiles=smiles,
                        pocket=self.pocket,
                        pose_pdb_path=f"/tmp/{smiles}.pdbqt",
                        vina_score=-(8.5 - (index * 0.05)),
                        rank=1,
                    ),
                )
            )
        return predictions

    def score_ligands(
        self,
        pocket: BindingPocket,
        ligands: list[Ligand],
        *,
        store,
        run_id: str,
        top_n: int = 50,
    ):
        _ = store, run_id
        scored = []
        for index, ligand in enumerate(ligands[:top_n], start=1):
            scored.append(
                AffinityPrediction(
                    ligand_smiles=ligand.smiles,
                    target_gene=pocket.structure.gene,
                    vina_score=-(7.0 - (index * 0.03)),
                    affinity_log_ki=None,
                    affinity_confidence=None,
                    pose_ref=DockingPose(
                        ligand_smiles=ligand.smiles,
                        pocket=pocket,
                        pose_pdb_path=f"/tmp/{ligand.smiles}.pdbqt",
                        vina_score=-(7.0 - (index * 0.03)),
                        rank=1,
                    ),
                )
            )
        return scored


class FakeGenerativeModule:
    def run(
        self,
        pockets: list[BindingPocket],
        *,
        store,
        run_id: str,
        n_per_pocket: int = 100,
        seed_smiles: str | None = None,
    ):
        _ = seed_smiles
        molecules = []
        for index in range(min(5, n_per_pocket)):
            molecule = GeneratedMolecule(
                smiles=f"CCN{index}",
                qed=0.8,
                sa_score=2.0,
                novelty=0.7,
                backend="reinvent4",
                pocket_ref=pockets[0],
            )
            _ = store.save_molecule(run_id, molecule)
            molecules.append(molecule)
        return molecules


class FakeADMETModule:
    def run(
        self,
        molecules,
        *,
        affinity_map,
        store,
        run_id: str,
        target: TargetCandidate,
        top_n: int = 10,
        enable_live_chembl: bool = False,
        enable_evebio: bool = False,
        provenance: dict[str, object] | None = None,
    ):
        _ = enable_live_chembl, enable_evebio
        ranked = []
        for index, molecule in enumerate(molecules[:top_n], start=1):
            ligand = (
                molecule
                if isinstance(molecule, Ligand)
                else Ligand(
                    smiles=molecule.smiles, source=f"generative:{molecule.backend}"
                )
            )
            admet = ADMETProfile(
                ligand_smiles=ligand.smiles,
                endpoints={"herg": 0.1, "ames": 0.1},
                liability_flags=[],
            )
            candidate = RankedCandidate(
                ligand=ligand,
                target=target,
                affinity=affinity_map[ligand.smiles],
                admet=admet,
                off_targets=[
                    OffTargetHit(
                        ligand_smiles=ligand.smiles,
                        off_target_gene="KCNH2",
                        similarity=0.2,
                        severity="low",
                    )
                ],
                composite_score=1.0 - (index * 0.05),
                rank=index,
                provenance={"run_id": run_id, **(provenance or {})},
            )
            _ = store.save_admet_profile(run_id, admet)
            _ = store.save_ranking(run_id, candidate)
            ranked.append(candidate)
        return ranked


def test_phase5_cli_e2e_writes_ranked_candidates(tmp_path: Path, monkeypatch) -> None:
    fixture_path = tmp_path / "targets.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "gene": {
                        "symbol": "CXCR4",
                        "ncbi_id": 7852,
                        "uniprot_id": "P61073",
                    },
                    "score": 0.92,
                    "evidence": [],
                    "pathway": ["chemokine_signaling"],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "run.json"
    target = build_target()

    def fake_run_pipeline(
        candidates,
        *,
        store,
        top_n=10,
        enable_live_chembl=False,
        enable_evebio=False,
    ):
        return core_run_pipeline(
            candidates,
            store=store,
            top_n=top_n,
            enable_live_chembl=enable_live_chembl,
            enable_evebio=enable_evebio,
            docking_runner=FakeDockingRunner(target),
            generative_module=FakeGenerativeModule(),
            admet_module=FakeADMETModule(),
        )

    monkeypatch.setattr("molforge.cli.run_pipeline", fake_run_pipeline)

    exit_code = main(
        [
            "run",
            str(fixture_path),
            "--disease",
            "scar pain",
            "--top",
            "10",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["candidate_count"] >= 10
    assert len(payload["candidates"]) >= 10
    assert payload["completed_at"] is not None
    assert all(candidate["rank"] >= 1 for candidate in payload["candidates"])
    assert all(candidate["affinity"] is not None for candidate in payload["candidates"])
