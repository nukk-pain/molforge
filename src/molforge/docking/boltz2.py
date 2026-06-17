from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from molforge.remote import JobResult, JobSpec

BOLTZ2_INPUT_FILENAME = "boltz2-input.json"
BOLTZ2_OUTPUT_FILENAME = "boltz2-output.json"
DEFAULT_BOLTZ2_IMAGE = os.environ.get("MOLFORGE_BOLTZ2_IMAGE", "boltz2-runtime")


def build_boltz2_job_spec(
    *,
    protein_pdb_path: Path,
    ligands: list[str],
    timeout_seconds: int = 900,
) -> JobSpec:
    payload = {
        "protein_pdb_path": protein_pdb_path.name,
        "ligands": ligands,
    }
    return JobSpec(
        image=DEFAULT_BOLTZ2_IMAGE,
        args=[
            "python",
            "-m",
            "molforge.docking.boltz2",
            "--input",
            BOLTZ2_INPUT_FILENAME,
            "--output",
            BOLTZ2_OUTPUT_FILENAME,
        ],
        input_files={
            protein_pdb_path.name: protein_pdb_path.read_bytes(),
            BOLTZ2_INPUT_FILENAME: (
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        },
        timeout_seconds=timeout_seconds,
        env={
            key: value
            for key, value in os.environ.items()
            if key.startswith("MOLFORGE_")
        },
    )


def parse_boltz2_job_result(job_result: JobResult) -> dict[str, tuple[float, float]]:
    raw_output = job_result.output_files.get(BOLTZ2_OUTPUT_FILENAME)
    if raw_output is None:
        raise ValueError("Boltz-2 job did not produce boltz2-output.json.")
    payload = json.loads(raw_output.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Boltz-2 output payload must be a list.")
    scores: dict[str, tuple[float, float]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Boltz-2 output entry must be an object.")
        ligand_smiles = str(item.get("ligand_smiles") or "").strip()
        if not ligand_smiles:
            raise ValueError("Boltz-2 output entry is missing ligand_smiles.")
        scores[ligand_smiles] = (
            float(item["affinity_log_ki"]),
            float(item["affinity_confidence"]),
        )
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m molforge.docking.boltz2")
    _ = parser.add_argument("--input", required=True)
    _ = parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if os.environ.get("MOLFORGE_BOLTZ2_MOCK") != "1":
        raise RuntimeError(
            "Local Boltz-2 execution is disabled. Use the remote backend or set MOLFORGE_BOLTZ2_MOCK=1 for deterministic mock execution."
        )

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ligands = payload.get("ligands")
    if not isinstance(ligands, list):
        raise ValueError("Boltz-2 input JSON must include a ligands list.")

    output_payload = [_mock_affinity_record(str(ligand)) for ligand in ligands]
    Path(args.output).write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _mock_affinity_record(ligand_smiles: str) -> dict[str, object]:
    digest = hashlib.sha256(ligand_smiles.encode("utf-8")).digest()
    affinity_bucket = int.from_bytes(digest[:2], "big") % 500
    confidence_bucket = int.from_bytes(digest[2:4], "big") % 100
    return {
        "ligand_smiles": ligand_smiles,
        "affinity_log_ki": round(-10.0 + (affinity_bucket / 100.0), 3),
        "affinity_confidence": round(confidence_bucket / 100.0, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
