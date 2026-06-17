from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from contracts.schema import (
    ADMETProfile,
    AffinityPrediction,
    BindingPocket,
    BIOCOMPUTE_SCHEMA_VERSION,
    DockingPose,
    EvidenceItem,
    GeneratedMolecule,
    Ligand,
    OffTargetHit,
    PipelineRun,
    ProteinStructure,
    RankedCandidate,
    StructureSource,
    TargetCandidate,
)

CREATE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        input_target_json TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        schema_version TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS structures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        structure_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS poses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        pose_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS molecules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        molecule_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS admet_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        admet_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rankings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        rank INTEGER NOT NULL,
        ranking_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        component TEXT PRIMARY KEY,
        version TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)


class MolforgeStore:
    db_path: str
    connection: sqlite3.Connection

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = normalize_db_path(db_path)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._configure_connection()
        self._create_schema()

    def __enter__(self) -> MolforgeStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def create_run(self, input_target: TargetCandidate, config_hash: str) -> str:
        run_id = str(uuid4())
        started_at = utc_now()
        _ = self.connection.execute(
            """
            INSERT INTO runs (run_id, input_target_json, config_hash, started_at, completed_at, schema_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                dumps_payload(input_target),
                config_hash,
                started_at,
                None,
                BIOCOMPUTE_SCHEMA_VERSION,
            ),
        )
        self.connection.commit()
        return run_id

    def complete_run(self, run_id: str, completed_at: str | None = None) -> None:
        cursor = self.connection.execute(
            "UPDATE runs SET completed_at = ? WHERE run_id = ?",
            (completed_at or utc_now(), run_id),
        )
        self.connection.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Run {run_id} not found.")

    def save_structure(self, run_id: str, structure: ProteinStructure) -> int:
        return self._insert_json_row("structures", "structure_json", run_id, structure)

    def save_pose(self, run_id: str, pose: DockingPose) -> int:
        return self._insert_json_row("poses", "pose_json", run_id, pose)

    def save_molecule(self, run_id: str, molecule: Ligand | GeneratedMolecule) -> int:
        return self._insert_json_row("molecules", "molecule_json", run_id, molecule)

    def save_admet_profile(self, run_id: str, admet_profile: ADMETProfile) -> int:
        return self._insert_json_row(
            "admet_profiles", "admet_json", run_id, admet_profile
        )

    def save_ranking(self, run_id: str, ranking: RankedCandidate) -> int:
        cursor = self.connection.execute(
            "INSERT INTO rankings (run_id, rank, ranking_json) VALUES (?, ?, ?)",
            (run_id, ranking.rank, dumps_payload(ranking)),
        )
        self.connection.commit()
        return require_lastrowid(cursor)

    def load_run(self, run_id: str) -> PipelineRun:
        run_row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise ValueError(f"Run {run_id} not found.")

        ranking_rows = self.connection.execute(
            "SELECT ranking_json FROM rankings WHERE run_id = ? ORDER BY rank ASC",
            (run_id,),
        ).fetchall()
        candidates = [
            parse_ranked_candidate(json.loads(row["ranking_json"]))
            for row in ranking_rows
        ]

        return PipelineRun(
            run_id=str(run_row["run_id"]),
            input_target=parse_target_candidate(
                json.loads(run_row["input_target_json"])
            ),
            started_at=str(run_row["started_at"]),
            completed_at=None
            if run_row["completed_at"] is None
            else str(run_row["completed_at"]),
            candidates=candidates,
            config_hash=str(run_row["config_hash"]),
            schema_version=str(run_row["schema_version"]),
        )

    def _configure_connection(self) -> None:
        _ = self.connection.execute("PRAGMA foreign_keys=ON")
        _ = self.connection.execute("PRAGMA busy_timeout=5000")
        _ = self.connection.execute("PRAGMA journal_mode=WAL")

    def _create_schema(self) -> None:
        for statement in CREATE_STATEMENTS:
            _ = self.connection.execute(statement)
        _ = self.connection.execute(
            """
            INSERT INTO schema_version (component, version, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(component) DO UPDATE SET
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            ("contracts", BIOCOMPUTE_SCHEMA_VERSION, utc_now()),
        )
        self.connection.commit()

    def _insert_json_row(
        self,
        table_name: str,
        payload_column: str,
        run_id: str,
        payload: object,
    ) -> int:
        cursor = self.connection.execute(
            f"INSERT INTO {table_name} (run_id, {payload_column}) VALUES (?, ?)",
            (run_id, dumps_payload(payload)),
        )
        self.connection.commit()
        return require_lastrowid(cursor)


def normalize_db_path(db_path: str | Path) -> str:
    if isinstance(db_path, Path):
        return str(db_path)
    if db_path == "sqlite:///:memory:":
        return ":memory:"
    return db_path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def dumps_payload(payload: object) -> str:
    return json.dumps(serialize(payload), sort_keys=True)


def serialize(value: object) -> object:
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {str(key): serialize(item) for key, item in vars(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value


def parse_target_candidate(payload: dict[str, object]) -> TargetCandidate:
    evidence = [
        parse_evidence_item(item)
        for item in expect_list(payload.get("evidence"), "TargetCandidate.evidence")
    ]
    pathway = [
        expect_str(item, "TargetCandidate.pathway entry")
        for item in expect_list(payload.get("pathway"), "TargetCandidate.pathway")
    ]
    extra = payload.get("extra")
    return TargetCandidate(
        gene=expect_str(payload.get("gene"), "TargetCandidate.gene"),
        score=expect_float(payload.get("score"), "TargetCandidate.score"),
        disease=optional_str(payload.get("disease")),
        ncbi_id=optional_int(payload.get("ncbi_id")),
        uniprot_id=optional_str(payload.get("uniprot_id")),
        evidence=evidence,
        pathway=pathway,
        extra=extra if isinstance(extra, dict) else None,
    )


def parse_evidence_item(payload: object) -> EvidenceItem:
    data = expect_dict(payload, "EvidenceItem")
    return EvidenceItem(
        source=expect_str(data.get("source"), "EvidenceItem.source"),
        description=expect_str(data.get("description"), "EvidenceItem.description"),
        confidence=expect_float(data.get("confidence"), "EvidenceItem.confidence"),
    )


def parse_protein_structure(payload: object) -> ProteinStructure:
    data = expect_dict(payload, "ProteinStructure")
    return ProteinStructure(
        gene=expect_str(data.get("gene"), "ProteinStructure.gene"),
        uniprot=optional_str(data.get("uniprot")),
        pdb_path=expect_str(data.get("pdb_path"), "ProteinStructure.pdb_path"),
        source=StructureSource(
            expect_str(data.get("source"), "ProteinStructure.source")
        ),
        confidence=optional_float(data.get("confidence")),
    )


def parse_binding_pocket(payload: object) -> BindingPocket:
    data = expect_dict(payload, "BindingPocket")
    return BindingPocket(
        structure=parse_protein_structure(data["structure"]),
        center_xyz=parse_xyz(data["center_xyz"]),
        size_xyz=parse_xyz(data["size_xyz"]),
        druggability_score=optional_float(data.get("druggability_score")),
        residues=[
            expect_str(item, "BindingPocket.residues entry")
            for item in expect_list(data.get("residues"), "BindingPocket.residues")
        ],
    )


def parse_docking_pose(payload: object) -> DockingPose:
    data = expect_dict(payload, "DockingPose")
    return DockingPose(
        ligand_smiles=expect_str(
            data.get("ligand_smiles"), "DockingPose.ligand_smiles"
        ),
        pocket=parse_binding_pocket(data["pocket"]),
        pose_pdb_path=expect_str(
            data.get("pose_pdb_path"), "DockingPose.pose_pdb_path"
        ),
        vina_score=expect_float(data.get("vina_score"), "DockingPose.vina_score"),
        rank=expect_int(data.get("rank"), "DockingPose.rank"),
    )


def parse_affinity_prediction(payload: object | None) -> AffinityPrediction | None:
    if payload is None:
        return None
    data = expect_dict(payload, "AffinityPrediction")
    pose_ref = data.get("pose_ref")
    return AffinityPrediction(
        ligand_smiles=expect_str(
            data.get("ligand_smiles"), "AffinityPrediction.ligand_smiles"
        ),
        target_gene=expect_str(
            data.get("target_gene"), "AffinityPrediction.target_gene"
        ),
        vina_score=expect_float(
            data.get("vina_score"), "AffinityPrediction.vina_score"
        ),
        affinity_log_ki=optional_float(data.get("affinity_log_ki")),
        affinity_confidence=optional_float(data.get("affinity_confidence")),
        pose_ref=None if pose_ref is None else parse_docking_pose(pose_ref),
    )


def parse_ligand(payload: object) -> Ligand:
    data = expect_dict(payload, "Ligand")
    return Ligand(
        smiles=expect_str(data.get("smiles"), "Ligand.smiles"),
        source=expect_str(data.get("source"), "Ligand.source"),
        chembl_id=optional_str(data.get("chembl_id")),
    )


def parse_admet_profile(payload: object) -> ADMETProfile:
    data = expect_dict(payload, "ADMETProfile")
    endpoints = expect_dict(data["endpoints"], "ADMETProfile.endpoints")
    return ADMETProfile(
        ligand_smiles=expect_str(
            data.get("ligand_smiles"), "ADMETProfile.ligand_smiles"
        ),
        endpoints={
            str(key): expect_float(value, f"ADMETProfile.endpoints[{key}]")
            for key, value in endpoints.items()
        },
        liability_flags=[
            expect_str(item, "ADMETProfile.liability_flags entry")
            for item in expect_list(
                data.get("liability_flags"), "ADMETProfile.liability_flags"
            )
        ],
    )


def parse_off_target_hit(payload: object) -> OffTargetHit:
    data = expect_dict(payload, "OffTargetHit")
    return OffTargetHit(
        ligand_smiles=expect_str(
            data.get("ligand_smiles"), "OffTargetHit.ligand_smiles"
        ),
        off_target_gene=expect_str(
            data.get("off_target_gene"), "OffTargetHit.off_target_gene"
        ),
        similarity=expect_float(data.get("similarity"), "OffTargetHit.similarity"),
        severity=expect_str(data.get("severity"), "OffTargetHit.severity"),
    )


def parse_ranked_candidate(payload: dict[str, object]) -> RankedCandidate:
    return RankedCandidate(
        ligand=parse_ligand(payload["ligand"]),
        target=parse_target_candidate(
            expect_dict(payload["target"], "TargetCandidate")
        ),
        affinity=parse_affinity_prediction(payload.get("affinity")),
        admet=parse_admet_profile(payload["admet"]),
        off_targets=[
            parse_off_target_hit(item)
            for item in expect_list(
                payload.get("off_targets"), "RankedCandidate.off_targets"
            )
        ],
        composite_score=expect_float(
            payload.get("composite_score"), "RankedCandidate.composite_score"
        ),
        rank=expect_int(payload.get("rank"), "RankedCandidate.rank"),
        provenance=expect_dict(payload["provenance"], "RankedCandidate.provenance"),
    )


def parse_xyz(payload: object) -> tuple[float, float, float]:
    values = list(payload) if isinstance(payload, (list, tuple)) else None
    if values is None or len(values) != 3:
        raise ValueError("Expected xyz payload with exactly 3 numeric values.")
    return (float(values[0]), float(values[1]), float(values[2]))


def expect_dict(payload: object, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} payload to be an object.")
    return {str(key): value for key, value in payload.items()}


def expect_list(payload: object, label: str) -> list[object]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"Expected {label} payload to be a list.")
    return list(payload)


def expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected {label} to be a string.")
    return value


def expect_int(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Expected {label} to be an integer.")
    return value


def expect_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected {label} to be numeric.")
    return float(value)


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return expect_int(value, "optional integer")


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return expect_float(value, "optional float")


def require_lastrowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite insert did not return a row id.")
    return int(cursor.lastrowid)
