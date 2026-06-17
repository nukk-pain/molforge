from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

BIOCOMPUTE_SCHEMA_VERSION = "2026-04-17"  # biocompute commit 80faca5 기준
BIOCOMPUTE_SCHEMA_VERSION_FIELD = "schema_version"
BIOCOMPUTE_CANDIDATES_FIELD = "candidates"


@dataclass
class EvidenceItem:
    """Flattened evidence row from the biocompute bare-array export."""

    source: str
    description: str
    confidence: float


@dataclass
class TargetCandidate:
    """biocompute bare-array export shape used by molforge v1 loader.

    Legacy upstream payloads are bare arrays. Versioned migration payloads wrap
    that same array in {"schema_version": ..., "candidates": [...]} so molforge
    can fail fast on future contract drift without breaking existing exports.
    """

    gene: str
    score: float
    disease: str | None = None
    ncbi_id: int | None = None
    uniprot_id: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    pathway: list[str] = field(default_factory=list)
    extra: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Stage 2 — Structure + Docking
# ---------------------------------------------------------------------------


class StructureSource(str, Enum):
    ALPHAFOLD_DB = "alphafold_db"
    OPENFOLD3_LOCAL = "openfold3_local"  # v2 — GPU required


@dataclass
class ProteinStructure:
    gene: str
    uniprot: str | None
    pdb_path: str  # local cached PDB
    source: StructureSource
    confidence: float | None = None  # pLDDT mean or equivalent


@dataclass
class BindingPocket:
    structure: ProteinStructure
    center_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    druggability_score: float | None = None  # fpocket score if available
    residues: list[str] = field(default_factory=list)  # e.g. ["ASP123", "TYR145"]


@dataclass
class DockingPose:
    ligand_smiles: str
    pocket: BindingPocket
    pose_pdb_path: str
    vina_score: float  # kcal/mol
    rank: int  # within the set for this ligand


@dataclass
class AffinityPrediction:
    """Stage 2 affinity record.

    In the CPU-first MVP, `vina_score` is the only guaranteed field. Boltz-2
    affinity fields remain optional until the remote GPU path is restored.
    """

    ligand_smiles: str
    target_gene: str
    vina_score: float
    affinity_log_ki: float | None = None  # predicted log(Ki) from Boltz-2
    affinity_confidence: float | None = None
    pose_ref: DockingPose | None = None  # top-rank pose tied to this affinity


# ---------------------------------------------------------------------------
# Stage 3 — Generative
# ---------------------------------------------------------------------------


@dataclass
class GeneratedMolecule:
    """Generated molecule candidate.

    SMILES are expected to be canonicalized and RDKit-valid before they enter
    downstream ranking.
    """

    smiles: str
    qed: float
    sa_score: float
    novelty: float  # 1 - max Tanimoto to nearest ChEMBL/training set
    backend: str  # "reinvent4" | "diffsbdd" (future)
    pocket_ref: BindingPocket | None = None


class GenerativeBackend(Protocol):
    """Plugin interface.

    v1 supports REINVENT4 sampling only (no RL training). Returned SMILES must
    pass `rdkit.Chem.MolFromSmiles` with a non-None result.
    """

    name: str

    def generate(
        self,
        pocket: BindingPocket,
        n: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]: ...


# ---------------------------------------------------------------------------
# Stage 4 — ADMET + Ranking
# ---------------------------------------------------------------------------


@dataclass
class Ligand:
    """Unified ligand handle.

    SMILES are expected to be canonicalized and RDKit-valid before being stored
    as a ligand reference.
    """

    smiles: str
    source: str  # "chembl_fda" | "generative:reinvent4" | "user" | ...
    chembl_id: str | None = None


@dataclass
class ADMETProfile:
    ligand_smiles: str
    # ADMET-AI v2 returns 41 endpoints — stored as dict to avoid schema churn.
    endpoints: dict[str, float]  # endpoint_name -> score/probability
    liability_flags: list[str] = field(
        default_factory=list
    )  # e.g. ["hERG_high", "hepatotox"]


@dataclass
class OffTargetHit:
    ligand_smiles: str
    off_target_gene: str
    similarity: float  # max Tanimoto to known ligand of this off-target
    severity: str  # "low" | "medium" | "high"


@dataclass
class RankedCandidate:
    """Final pipeline output — one row per candidate molecule."""

    ligand: Ligand
    target: TargetCandidate
    affinity: AffinityPrediction | None  # v1 keeps Vina-only affinity possible
    admet: ADMETProfile
    off_targets: list[OffTargetHit]
    composite_score: float  # Boltz affinity absent 시 vina_score 정규화 기반
    rank: int
    provenance: dict[str, object]  # run_id, stage versions, timestamps


# ---------------------------------------------------------------------------
# Top-level pipeline output
# ---------------------------------------------------------------------------


@dataclass
class PipelineRun:
    run_id: str
    input_target: TargetCandidate
    started_at: str  # ISO 8601
    completed_at: str | None
    candidates: list[RankedCandidate]
    config_hash: str
    schema_version: str = BIOCOMPUTE_SCHEMA_VERSION
