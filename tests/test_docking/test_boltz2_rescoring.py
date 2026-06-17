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
    AffinityPrediction,
    BindingPocket,
    DockingPose,
    ProteinStructure,
    StructureSource,
)  # noqa: E402
from molforge.docking.boltz2 import BOLTZ2_OUTPUT_FILENAME  # noqa: E402
from molforge.docking.rescore import rescore_predictions  # noqa: E402
from molforge.remote.backend import JobHandle, JobResult  # noqa: E402


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.handles: list[JobHandle] = []

    def submit(self, job):
        handle = JobHandle(handle_id=f"job-{len(self.handles) + 1}", provider=self.name)
        self.handles.append(handle)
        return handle

    def fetch_result(self, handle: JobHandle) -> JobResult:
        _ = handle
        payload = [
            {
                "ligand_smiles": f"ligand-{index}",
                "affinity_log_ki": -7.0 - index,
                "affinity_confidence": 0.8,
            }
            for index in range(10)
        ]
        return JobResult(
            success=True,
            stdout="",
            stderr="",
            output_files={BOLTZ2_OUTPUT_FILENAME: json.dumps(payload).encode("utf-8")},
            elapsed=1.0,
            cost_estimate_usd=0.25,
        )


def test_rescore_predictions_populates_boltz_fields(tmp_path: Path) -> None:
    protein_path = tmp_path / "protein.pdb"
    protein_path.write_text("ATOM\n", encoding="utf-8")
    structure = ProteinStructure(
        gene="CXCR4",
        uniprot="P61073",
        pdb_path=str(protein_path),
        source=StructureSource.ALPHAFOLD_DB,
        confidence=90.0,
    )
    pocket = BindingPocket(
        structure=structure,
        center_xyz=(1.0, 2.0, 3.0),
        size_xyz=(10.0, 11.0, 12.0),
        druggability_score=0.8,
        residues=["ASP97"],
    )
    predictions = [
        AffinityPrediction(
            ligand_smiles=f"ligand-{index}",
            target_gene="CXCR4",
            vina_score=-8.0 + index,
            pose_ref=DockingPose(
                ligand_smiles=f"ligand-{index}",
                pocket=pocket,
                pose_pdb_path=str(tmp_path / f"pose-{index}.pdbqt"),
                vina_score=-8.0 + index,
                rank=1,
            ),
        )
        for index in range(10)
    ]

    rescored, total_cost = rescore_predictions(predictions, backend=FakeBackend())

    assert len(rescored) == 10
    assert total_cost == 0.25
    assert all(item.affinity_log_ki is not None for item in rescored)
    assert all(item.affinity_confidence == 0.8 for item in rescored)
