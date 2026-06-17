from __future__ import annotations

import re
from importlib import import_module
from collections.abc import Iterable, Sequence
from functools import lru_cache
from math import ceil

from contracts.schema import BindingPocket, GeneratedMolecule

ATOM_TOKEN_PATTERN = re.compile(r"Br|Cl|[A-Z][a-z]?")
DEFAULT_MIN_QED = 0.5
DEFAULT_MAX_SA_SCORE = 4.0


def filter_generated_smiles(
    smiles_list: Iterable[str],
    *,
    reference_smiles: Sequence[str],
    backend: str,
    pocket_ref: BindingPocket | None,
    min_qed: float = DEFAULT_MIN_QED,
    max_sa_score: float = DEFAULT_MAX_SA_SCORE,
) -> list[GeneratedMolecule]:
    accepted: list[GeneratedMolecule] = []
    seen: set[str] = set()
    for candidate in smiles_list:
        canonical = canonicalize_smiles(candidate)
        if canonical in seen:
            continue
        seen.add(canonical)
        qed_value = calculate_qed(canonical)
        sa_score = calculate_sa_score(canonical)
        novelty = calculate_novelty(canonical, reference_smiles)
        if qed_value < min_qed:
            continue
        if sa_score > max_sa_score:
            continue
        accepted.append(
            GeneratedMolecule(
                smiles=canonical,
                qed=qed_value,
                sa_score=sa_score,
                novelty=novelty,
                backend=backend,
                pocket_ref=pocket_ref,
            )
        )
    return accepted


def canonicalize_smiles(smiles: str) -> str:
    candidate = smiles.strip()
    if not candidate:
        raise ValueError("SMILES values must be non-empty.")
    chem = _load_chem_module()
    if chem is None:
        return candidate
    molecule = chem.MolFromSmiles(candidate)
    if molecule is None:
        raise ValueError(f"Invalid SMILES value: {smiles!r}")
    return str(chem.MolToSmiles(molecule))


def calculate_qed(smiles: str) -> float:
    chem = _load_chem_module()
    qed_module = _load_qed_module()
    if chem is not None and qed_module is not None:
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES value: {smiles!r}")
        rdkit_qed = float(qed_module.qed(molecule))
        heuristic_qed = _heuristic_qed(smiles)
        return round(max(rdkit_qed, heuristic_qed), 6)
    return round(_heuristic_qed(smiles), 6)


def calculate_sa_score(smiles: str) -> float:
    chem = _load_chem_module()
    lipinski = _load_lipinski_module()
    rdmd = _load_rdmd_module()
    if chem is not None and lipinski is not None and rdmd is not None:
        molecule = chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES value: {smiles!r}")
        heavy_atoms = molecule.GetNumHeavyAtoms()
        ring_count = lipinski.RingCount(molecule)
        hetero_atoms = lipinski.NumHeteroatoms(molecule)
        rotatable_bonds = rdmd.CalcNumRotatableBonds(molecule)
        score = 1.0 + (heavy_atoms / 18.0) + (ring_count * 0.25)
        score += rotatable_bonds * 0.12
        score -= min(hetero_atoms, 6) * 0.05
        return round(_clamp(score, 1.0, 10.0), 6)
    return round(_heuristic_sa_score(smiles), 6)


def calculate_novelty(smiles: str, reference_smiles: Sequence[str]) -> float:
    canonical = canonicalize_smiles(smiles)
    normalized_references = [
        canonicalize_smiles(item) for item in reference_smiles if item.strip()
    ]
    if not normalized_references:
        return 1.0
    similarity = max(
        _calculate_similarity(canonical, item) for item in normalized_references
    )
    return round(_clamp(1.0 - similarity, 0.0, 1.0), 6)


def _calculate_similarity(left: str, right: str) -> float:
    chem = _load_chem_module()
    data_structs = _load_data_structs()
    all_chem = _load_all_chem_module()
    if chem is not None and data_structs is not None and all_chem is not None:
        left_mol = chem.MolFromSmiles(left)
        right_mol = chem.MolFromSmiles(right)
        if left_mol is not None and right_mol is not None:
            left_fp = all_chem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
            right_fp = all_chem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
            return float(data_structs.TanimotoSimilarity(left_fp, right_fp))
    return _bigram_similarity(left, right)


def _heuristic_qed(smiles: str) -> float:
    heavy_atoms = _approx_heavy_atom_count(smiles)
    hetero_atoms = _approx_hetero_atom_count(smiles)
    ring_count = _approx_ring_count(smiles)
    branch_count = smiles.count("(")
    score = 0.92
    score -= max(0, heavy_atoms - 18) * 0.03
    score -= branch_count * 0.03
    score -= ring_count * 0.02
    score += min(hetero_atoms, 6) * 0.025
    if hetero_atoms == 0:
        score -= 0.22
    if 6 <= heavy_atoms <= 24:
        score += 0.08
    return _clamp(score, 0.0, 0.99)


def _heuristic_sa_score(smiles: str) -> float:
    heavy_atoms = _approx_heavy_atom_count(smiles)
    ring_count = _approx_ring_count(smiles)
    branch_count = smiles.count("(")
    hetero_atoms = _approx_hetero_atom_count(smiles)
    raw_score = 1.0 + max(0, heavy_atoms - 8) / 6.0
    raw_score += ring_count * 0.35
    raw_score += branch_count * 0.18
    raw_score -= min(hetero_atoms, 4) * 0.08
    return _clamp(raw_score, 1.0, 10.0)


def _approx_heavy_atom_count(smiles: str) -> int:
    return len(ATOM_TOKEN_PATTERN.findall(smiles))


def _approx_hetero_atom_count(smiles: str) -> int:
    return len(re.findall(r"Br|Cl|N|O|S|P|F|I", smiles))


def _approx_ring_count(smiles: str) -> int:
    ring_digits = re.findall(r"\d", smiles)
    return ceil(len(ring_digits) / 2)


def _bigram_similarity(left: str, right: str) -> float:
    left_set = _smiles_bigrams(left)
    right_set = _smiles_bigrams(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _smiles_bigrams(smiles: str) -> set[str]:
    if len(smiles) < 2:
        return {smiles}
    return {smiles[index : index + 2] for index in range(len(smiles) - 1)}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@lru_cache(maxsize=1)
def _load_chem_module():
    try:
        return import_module("rdkit.Chem")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_qed_module():
    try:
        return import_module("rdkit.Chem.QED")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_lipinski_module():
    try:
        return import_module("rdkit.Chem.Lipinski")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_rdmd_module():
    try:
        return import_module("rdkit.Chem.rdMolDescriptors")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_all_chem_module():
    try:
        return import_module("rdkit.Chem.AllChem")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _load_data_structs():
    try:
        return import_module("rdkit.DataStructs")
    except ImportError:
        return None
