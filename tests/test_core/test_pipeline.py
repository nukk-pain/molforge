# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from molforge.core.pipeline import load_binding_pocket, run_generate_stage, run_pipeline  # noqa: E402
from molforge.core.store import MolforgeStore  # noqa: E402
from molforge.core.writer import write_ranked_candidates  # noqa: E402


class FakeBackend:
    name = "reinvent4"

    def generate(
        self, pocket: BindingPocket, n: int = 100, seed_smiles: str | None = None
    ):
        _ = n, seed_smiles
        return [
            GeneratedMolecule(
                smiles="CCN",
                qed=0.8,
                sa_score=2.0,
                novelty=0.6,
                backend="reinvent4",
                pocket_ref=pocket,
            )
        ]


class FakeDockingRunner:
    def __init__(self, *, include_generated: bool = True) -> None:
        self.include_generated = include_generated

    def run_target(
        self, target: TargetCandidate, *, store, run_id: str, top_n: int = 50
    ):
        _ = store, run_id, top_n
        return [build_affinity_prediction(target, smiles="CCO", vina_score=-8.1)]

    def score_ligands(
        self,
        pocket: BindingPocket,
        ligands: list[Ligand],
        *,
        store,
        run_id: str,
        top_n: int = 50,
    ):
        _ = store, run_id, top_n
        if not self.include_generated:
            return []
        return [
            AffinityPrediction(
                ligand_smiles=ligands[0].smiles,
                target_gene=pocket.structure.gene,
                vina_score=-7.2,
                affinity_log_ki=None,
                affinity_confidence=None,
                pose_ref=DockingPose(
                    ligand_smiles=ligands[0].smiles,
                    pocket=pocket,
                    pose_pdb_path="/tmp/generated-pose.pdbqt",
                    vina_score=-7.2,
                    rank=1,
                ),
            )
        ]


class FakeGenerativeModule:
    def run(
        self,
        pockets: list[BindingPocket],
        *,
        store,
        run_id: str,
        n_per_pocket: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]:
        _ = n_per_pocket, seed_smiles
        molecule = GeneratedMolecule(
            smiles="CCN",
            qed=0.81,
            sa_score=2.0,
            novelty=0.72,
            backend="reinvent4",
            pocket_ref=pockets[0],
        )
        _ = store.save_molecule(run_id, molecule)
        return [molecule]


class FakeBlockedGenerativeModule:
    def run(
        self,
        pockets: list[BindingPocket],
        *,
        store,
        run_id: str,
        n_per_pocket: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]:
        _ = pockets, store, run_id, n_per_pocket, seed_smiles
        raise RuntimeError("missing_reinvent_package")


class FakeADMETModule:
    def __init__(self) -> None:
        self.last_enable_live_chembl: bool | None = None
        self.last_enable_evebio: bool | None = None

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
        self.last_enable_live_chembl = enable_live_chembl
        self.last_enable_evebio = enable_evebio
        ranked: list[RankedCandidate] = []
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
                endpoints={"hERG": 0.1, "AMES": 0.2},
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
                composite_score=round(1.0 - (index * 0.1), 6),
                rank=index,
                provenance={"run_id": run_id, **(provenance or {})},
            )
            _ = store.save_admet_profile(run_id, admet)
            _ = store.save_ranking(run_id, candidate)
            ranked.append(candidate)
        return ranked


def test_run_pipeline_propagates_enable_live_chembl_to_admet_module() -> None:
    target = build_target_candidate()
    admet_module = FakeADMETModule()

    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            enable_live_chembl=True,
            docking_runner=FakeDockingRunner(),
            generative_module=FakeGenerativeModule(),
            admet_module=admet_module,
        )

    assert len(run.candidates) == 2
    assert admet_module.last_enable_live_chembl is True
    assert run.candidates[0].provenance["live_chembl_enabled"] is True


def test_run_pipeline_propagates_enable_evebio_to_admet_module() -> None:
    target = build_target_candidate()
    admet_module = FakeADMETModule()

    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            enable_evebio=True,
            docking_runner=FakeDockingRunner(),
            generative_module=FakeGenerativeModule(),
            admet_module=admet_module,
        )

    assert len(run.candidates) == 2
    assert admet_module.last_enable_evebio is True
    assert run.candidates[0].provenance["evebio_enabled"] is True


def build_target_candidate() -> TargetCandidate:
    return TargetCandidate(
        gene="TGFB1",
        score=0.87,
        disease="ALS",
        ncbi_id=7040,
        uniprot_id=None,
        evidence=[],
        pathway=["SMAD3"],
        extra=None,
    )


def build_affinity_prediction(
    target: TargetCandidate,
    *,
    smiles: str,
    vina_score: float,
) -> AffinityPrediction:
    pocket = BindingPocket(
        structure=ProteinStructure(
            gene=target.gene,
            uniprot=target.uniprot_id,
            pdb_path="/tmp/tgfb1.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=88.1,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.9,
        residues=["ASP97"],
    )
    pose = DockingPose(
        ligand_smiles=smiles,
        pocket=pocket,
        pose_pdb_path=f"/tmp/{smiles}.pdbqt",
        vina_score=vina_score,
        rank=1,
    )
    return AffinityPrediction(
        ligand_smiles=smiles,
        target_gene=target.gene,
        vina_score=vina_score,
        affinity_log_ki=None,
        affinity_confidence=None,
        pose_ref=pose,
    )


def test_run_pipeline_integrates_docking_generative_and_admet() -> None:
    target = build_target_candidate()

    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            docking_runner=FakeDockingRunner(),
            generative_module=FakeGenerativeModule(),
            admet_module=FakeADMETModule(),
        )

    assert run.input_target == target
    assert len(run.candidates) == 2
    assert run.completed_at is not None
    assert {candidate.ligand.smiles for candidate in run.candidates} == {"CCO", "CCN"}
    assert run.candidates[0].provenance["stage"] == "phase5_pipeline"


def test_run_pipeline_degrades_to_docking_only_when_generation_is_unavailable() -> None:
    target = build_target_candidate()

    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            docking_runner=FakeDockingRunner(include_generated=False),
            generative_module=FakeBlockedGenerativeModule(),
            admet_module=FakeADMETModule(),
        )

    assert len(run.candidates) == 1
    assert run.candidates[0].ligand.smiles == "CCO"
    assert (
        run.candidates[0].provenance["generative_status"]
        == "skipped:missing_reinvent_package"
    )


class FakeFailingDockingRunner:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc if exc is not None else RuntimeError("vina_binary_missing")

    def run_target(self, target, *, store, run_id, top_n=50):
        _ = target, store, run_id, top_n
        raise self.exc

    def score_ligands(self, pocket, ligands, *, store, run_id, top_n=50):
        _ = pocket, ligands, store, run_id, top_n
        return []


def test_run_pipeline_emits_diagnostics_when_docking_fails(
    tmp_path, monkeypatch
) -> None:
    """When docking stage returns 0, a diagnostics.json must surface the reason."""
    monkeypatch.chdir(tmp_path)
    target = build_target_candidate()
    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            docking_runner=FakeFailingDockingRunner(),
            generative_module=FakeBlockedGenerativeModule(),
            admet_module=FakeADMETModule(),
        )
    assert len(run.candidates) == 0
    diag_path = tmp_path / "archive" / "runs" / run.run_id / "diagnostics.json"
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    assert payload["docking_candidate_count"] == 0
    assert payload["admet_status"] == "skipped:empty_affinity_map"
    assert "failed:RuntimeError" in payload["docking_status"]
    assert "vina_binary_missing" in payload["docking_status"]


def test_run_pipeline_diagnoses_filenotfound_from_docking(
    tmp_path, monkeypatch
) -> None:
    """FileNotFoundError (missing vina binary) is recoverable → diagnostics emitted."""
    monkeypatch.chdir(tmp_path)
    target = build_target_candidate()
    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            docking_runner=FakeFailingDockingRunner(
                exc=FileNotFoundError("vina not on PATH")
            ),
            generative_module=FakeBlockedGenerativeModule(),
            admet_module=FakeADMETModule(),
        )
    assert len(run.candidates) == 0
    payload = json.loads(
        (tmp_path / "archive" / "runs" / run.run_id / "diagnostics.json").read_text()
    )
    assert "failed:FileNotFoundError" in payload["docking_status"]


def test_run_pipeline_reraises_programmer_errors_from_docking() -> None:
    """AttributeError / TypeError must NOT be swallowed — they're programmer bugs."""
    target = build_target_candidate()
    with MolforgeStore("sqlite:///:memory:") as store:
        try:
            run_pipeline(
                [target],
                store=store,
                top_n=5,
                docking_runner=FakeFailingDockingRunner(
                    exc=AttributeError("contract drift")
                ),
                generative_module=FakeBlockedGenerativeModule(),
                admet_module=FakeADMETModule(),
            )
        except AttributeError as exc:
            assert "contract drift" in str(exc)
        else:
            raise AssertionError("AttributeError must propagate — not be swallowed")


def test_run_pipeline_rejects_multiple_targets() -> None:
    target = build_target_candidate()

    with MolforgeStore("sqlite:///:memory:") as store:
        try:
            run_pipeline(
                [target, target],
                store=store,
                top_n=5,
                docking_runner=FakeDockingRunner(),
                generative_module=FakeGenerativeModule(),
                admet_module=FakeADMETModule(),
            )
        except ValueError as exc:
            assert "exactly one TargetCandidate" in str(exc)
        else:
            raise AssertionError(
                "Expected multiple-target pipeline invocation to fail."
            )


def test_write_ranked_candidates_round_trip(tmp_path: Path) -> None:
    target = build_target_candidate()

    with MolforgeStore("sqlite:///:memory:") as store:
        run = run_pipeline(
            [target],
            store=store,
            top_n=5,
            docking_runner=FakeDockingRunner(),
            generative_module=FakeGenerativeModule(),
            admet_module=FakeADMETModule(),
        )
        output_path = tmp_path / "ranked.json"
        write_ranked_candidates(run, output_path)
        loaded_run = store.load_run(run.run_id)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert payload["provenance"]["candidate_count"] == 2
    assert len(payload["candidates"]) == 2
    assert loaded_run == run


def test_load_binding_pocket_reads_contract_payload(tmp_path: Path) -> None:
    pocket_path = tmp_path / "pocket.json"
    pocket_path.write_text(
        json.dumps(
            {
                "structure": {
                    "gene": "CXCR4",
                    "uniprot": "P61073",
                    "pdb_path": "/tmp/cxcr4.pdb",
                    "source": "alphafold_db",
                    "confidence": 88.1,
                },
                "center_xyz": [1.0, 2.0, 3.0],
                "size_xyz": [10.0, 11.0, 12.0],
                "druggability_score": 0.9,
                "residues": ["ASP97"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pocket = load_binding_pocket(pocket_path)

    assert pocket.structure == ProteinStructure(
        gene="CXCR4",
        uniprot="P61073",
        pdb_path="/tmp/cxcr4.pdb",
        source=StructureSource.ALPHAFOLD_DB,
        confidence=88.1,
    )


def test_run_generate_stage_delegates_to_module(tmp_path: Path) -> None:
    pocket = BindingPocket(
        structure=ProteinStructure(
            gene="CXCR4",
            uniprot="P61073",
            pdb_path="/tmp/cxcr4.pdb",
            source=StructureSource.ALPHAFOLD_DB,
            confidence=88.1,
        ),
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.9,
        residues=["ASP97"],
    )

    result = run_generate_stage(pocket, output_dir=tmp_path, backend=FakeBackend())

    assert result.backend == "reinvent4"
    assert result.returned_count == 1
